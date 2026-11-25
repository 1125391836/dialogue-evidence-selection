"""Remove the learned mention weighting only during history selection."""
from collections import Counter
import json
from pathlib import Path

import ticket_learning as study
from review_benchmarks import contexts
from context_bundles.core import Budget, Counter as TokenCounter
from context_bundles.io import sha256, write_json, write_jsonl

OUT = study.ROOT / "runs/review-selector-ablation"
ARM = "recency_scores"


def recency_scores(case, support):
    # The latest mention of a value wins. All values in the same turn tie.
    turn_ids = [e.id for e in case.evidence] + ["query"]
    return {
        tid: {pair: rank / len(turn_ids) for pair in support.get(tid, {})}
        for rank, tid in enumerate(turn_ids, start=1)
    }


def main():
    OUT.mkdir(exist_ok=True)
    write_json(OUT / "protocol.json", dict(
        design="Retrospective paired ablation; both evaluation cohorts previously inspected.",
        intervention="Replace only the selector's mention probabilities by (one-based chronological turn rank)/(number of turns including query). In a same-turn value tie use the existing lexicographic best_fields tie rule.",
        constants="Same training-only vocabulary, candidate pairs, field target selection algorithm, marginal coverage, most recent tie break, 128-token tokenizer budget, learned downstream reader, calibration and 0.5 threshold.",
        metrics="Official correct, wrong and no-write counts. Before the submission gate compare exact proposed order matches. Report paired gains/losses.",
        script_sha256=sha256(Path(__file__))))
    training, used, texts, _ = study.load("train", ["aa","ab","ac","ad"])
    dev, _, _, _ = study.load("dev", ["aa"], used, texts)
    calibration_rows = [r for r in dev if int(r[1][:2], 16) < 128]
    matcher = study.Mentions(training)
    vec, model, _ = study.fit(training, matcher)
    budget = Budget(128, TokenCounter("hf:" + str(study.ROOT / study.TOKENIZER)))
    calibration = study.calibrate(study.evaluate(calibration_rows, matcher, vec, model, budget))
    cohorts, baseline = contexts()
    results = {}
    for split, cases in cohorts.items():
        records = []
        counts = {arm: Counter() for arm in ("recent", ARM, "field_support")}
        paired = {}
        for key, (raw, gold) in cases.items():
            ref = baseline[split][key]
            case = study.render(raw)
            support = study.score(case, matcher, vec, model)
            arms = {}
            for arm in counts:
                selector = recency_scores(case, support) if arm == ARM else support
                method = "recent" if arm == "recent" else "field_support"
                selected = study.evidence(case, selector, budget, method)
                ids = [e.id for e in selected]
                assert budget.cost(selected) <= 128
                visible = ["query", *ids]
                best = study.best_fields(support, visible)
                args = study.predict(best)
                details = {"order_features":study.order_features(support, visible, best)}
                study.apply_calibration([{"conditions":{arm:details}}], calibration)
                confidence = details["confidence"]
                outcome = study.outcome(args, gold, confidence, .5)
                if arm != ARM:
                    before = ref["conditions"][arm]
                    assert (args, ids, outcome) == (before["args"], before["selected"], before["outcome"])
                arms[arm] = dict(selected=ids, args=args, confidence=confidence,
                                 outcome=outcome, exact_proposal=args == gold)
                counts[arm][outcome] += 1
                counts[arm]["exact_proposal"] += int(args == gold)
            records.append(dict(case=key, arms=arms))
        for metric in ("exact_proposal", "correct", "wrong"):
            def event(arm):
                return arm["exact_proposal"] if metric == "exact_proposal" else arm["outcome"] == metric
            pairs = [(event(r["arms"]["field_support"]), event(r["arms"][ARM])) for r in records]
            paired[metric] = dict(learned_only=pairs.count((True,False)), recency_scores_only=pairs.count((False,True)))
        results[split] = dict(n=len(records), counts=counts, paired=paired,
                              changed_selection=sum(r["arms"][ARM]["selected"] != r["arms"]["field_support"]["selected"] for r in records),
                              cohort_sha256=sha256(study.OUT / (split+"-details.jsonl")))
        write_jsonl(OUT / (split+"-details.jsonl"), records)
        print(split, json.dumps(results[split]), flush=True)
    write_json(OUT / "summary.json", results)


if __name__ == "__main__":
    main()
