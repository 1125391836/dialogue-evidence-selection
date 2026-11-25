"""Frozen thresholds selected on dev-aa, applied once to previously unused dev-ad."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

import ticket_learning as study
from context_bundles.io import read_jsonl, write_json, write_jsonl


OUT = study.ROOT / "runs/order-risk-audit"
METHODS = ("recent", "bm25", "value_recent", "field_support")
OUTCOMES = ("correct", "wrong", "no_write")
GRID = [i / 1000 for i in range(1001)]


def threshold_counts(rows, method, cutoff):
    counts = Counter()
    for row in rows:
        ev = row["conditions"][method]
        if ev["confidence"] < cutoff or not study.REQUIRED <= ev["args"].keys():
            counts["no_write"] += 1
        else:
            counts["correct" if ev["args"] == row["gold"] else "wrong"] += 1
    return {label: counts[label] for label in OUTCOMES}


def choose_thresholds(rows):
    cap = threshold_counts(rows, "recent", .5)["wrong"]
    thresholds = {"recent": .5}
    for method in METHODS[1:]:
        thresholds[method] = next(t for t in GRID if threshold_counts(rows, method, t)["wrong"] <= cap)
    return cap, thresholds


def difference(diffs):
    n = len(diffs)
    freq = np.bincount(np.asarray(diffs) + 1, minlength=3)
    resamples = np.random.default_rng(42).multinomial(n, freq / n, size=20_000)
    lo, hi = np.quantile(100 * (resamples[:, 2] - resamples[:, 0])/n, [.025, .975])
    return dict(increases=int(freq[2]), decreases=int(freq[0]), unchanged=int(freq[1]),
                difference_pp=100*sum(diffs)/n, ci95_pp=[float(lo), float(hi)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    calibration_input = study.OUT / "dev-details.jsonl"
    development = list(read_jsonl(calibration_input))
    assert len(development) == 238
    cap, matched = choose_thresholds(development)
    policies = {"fixed": dict.fromkeys(METHODS, .5), "matched": matched}
    protocol = dict(
        cohort="official dev-ad, first eligible book_tickets per dialogue",
        note="Secondary audit designed after primary test-aa and replication dev-ab/ac; dev-ad untouched during design.",
        data_commit=study.COMMIT, history_tokens=128, methods=METHODS,
        calibrator="same training reader and pooled logistic calibrator as ticket_learning",
        criterion="Keep recent at .5. For each other method use lowest threshold in [0,1] in increments of .001 with no more dev-aa errors than recent at .5, out of all dialogues.",
        cutoff_development="238 dev-aa high-hash evaluation dialogues; low-hash calibration dialogues disjoint",
        development_errors_allowed=cap, policies=policies,
        development_counts={policy:{m:threshold_counts(development, m, c)
                                    for m,c in cuts.items()} for policy,cuts in policies.items()},
        exclusions="All training and all other official development/test shards; normalized dialogue fingerprint plus dialogue ID; intra-cohort duplicates removed.",
        primary_contrast="matched field_support minus recent; successes and wrong writes per dialogue",
        uncertainty="20,000 paired dialogue bootstrap draws, percentile CI, seed 42; template dependence unmodeled",
        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   (Path(__file__), Path(study.__file__), calibration_input)},
    )
    # The persisted policy cannot be silently changed after accessing holdout outcomes.
    frozen = OUT / "protocol.json"
    if frozen.exists():
        assert json.loads(frozen.read_text(encoding="utf8")) == json.loads(json.dumps(protocol))
    else:
        write_json(frozen, protocol)
    print("frozen", cap, matched, flush=True)
    training, used, texts, _ = study.load("train", ("aa","ab","ac","ad"))
    dev, used, texts, _ = study.load("dev", ("aa",), used, texts)
    # Include all historical test/replication shards in the ID and fingerprint exclusion.
    for split, parts in (("test", ("aa","ab","ac","ad")), ("dev", ("ab","ac"))):
        _, used, texts, _ = study.load(split, parts, used, texts)
    matcher = study.Mentions(training)
    vec, model, _ = study.fit(training, matcher)
    budget = study.Budget(128, study.TokenCounter("hf:" + str(study.ROOT / study.TOKENIZER)))
    calibration_rows = study.evaluate([r for r in dev if int(r[1][:2],16)<128],
                                      matcher, vec, model, budget, METHODS)
    calibration = study.calibrate(calibration_rows)
    # Check the reconstructed calibrator against the previously frozen development scores.
    check = study.evaluate([r for r in dev if r[0] == development[0]["case"]],
                           matcher, vec, model, budget, METHODS)
    study.apply_calibration(check, calibration)
    for m in METHODS:
        assert abs(check[0]["conditions"][m]["confidence"] - development[0]["conditions"][m]["confidence"]) < 1e-8
    holdout, _, _, coverage = study.load("dev", ("ad",), used, texts)
    scored = study.evaluate(holdout, matcher, vec, model, budget, METHODS)
    study.apply_calibration(scored, calibration)
    assert len(scored) == len({row["case"] for row in scored}) == len({row["fingerprint"] for row in scored})
    details = []
    for row in scored:
        outcomes = {}
        for policy, cuts in policies.items():
            outcomes[policy] = {m:study.outcome(ev["args"], row["gold"], ev["confidence"], cuts[m])
                                for m,ev in row["conditions"].items()}
        inputs = {m:{k:ev[k] for k in ("args","confidence","selected","tokens")}
                  for m,ev in row["conditions"].items()}
        details.append(dict(case=row["case"], fingerprint=row["fingerprint"], gold=row["gold"],
                            conditions=inputs, outcomes=outcomes))
    write_jsonl(OUT / "details.jsonl", details)
    result = dict(coverage=coverage, n=len(details), policies={})
    for policy in policies:
        tallies = {m:Counter(r["outcomes"][policy][m] for r in details) for m in METHODS}
        paired = {m:{s:difference([int(r["outcomes"][policy][m] == s) -
                                    int(r["outcomes"][policy]["recent"] == s) for r in details])
                     for s in OUTCOMES} for m in METHODS[1:]}
        result["policies"][policy] = dict(counts=tallies, paired=paired)
    write_json(OUT / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
