"""Paired TicketTalk experiment with a shared, training-only field reader."""
import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from context_bundles.core import Budget, Counter as TokenCounter, Evidence
from context_bundles.io import write_json, write_jsonl
from context_bundles.selection import select
from ticket_execution import COMMIT, REQUIRED, ROOT, TOKENIZER, examples, execute, render


OUT = ROOT / "runs/review-chronology"
METHODS = ("recent", "bm25", "value_recent", "field_support", "full")
AFFIRM = re.compile(r"\b(yes|correct|confirmed|right|perfect|sounds good|that's it)\b", re.I)
CONFIRM = re.compile(r"\b(confirm|correct|total|summary|purchase|right|is that|so you|you(?:'d| would) like)\b", re.I)
CHANGE = re.compile(r"\b(instead|actually|rather|change|sorry|no longer)\b", re.I)


def load(split, parts, blocked_ids=None, blocked_text=None, skip_example=False):
    ids = set(blocked_ids or ())
    texts = set(blocked_text or ())
    result = []
    tally = Counter()
    for part in parts:
        for key, context, gold in examples(split, part):
            tally["book_rows"] += 1
            if skip_example:
                skip_example = False
                ids.add(key)
                tally["previously_viewed"] += 1
                continue
            if key in ids:
                tally["duplicate_or_exposed_id"] += 1
                continue
            ids.add(key)
            case = render(context)
            if case is None or not REQUIRED <= gold.keys():
                tally["missing_text_or_required_ref"] += 1
                continue
            visible = "\n".join([*(e.text for e in case.evidence), case.query])
            digest = hashlib.sha256(" ".join(visible.casefold().split()).encode()).hexdigest()
            if digest in texts:
                tally["duplicate_text"] += 1
                continue
            texts.add(digest)
            result.append((key, digest, case, gold))
    tally["scored"] = len(result)
    return result, ids, texts, tally


def words(text):
    return re.findall(r"\w+", text.casefold())


class Mentions:
    """Match only field values learned from the training purchase arguments."""

    def __init__(self, training):
        self.trie = {}
        for _, _, _, args in training:
            for field, value in args.items():
                tokens = words(value)
                if tokens:
                    node = self.trie
                    for token in tokens:
                        node = node.setdefault(token, {})
                    node.setdefault(None, set()).add((field, value))

    @lru_cache(maxsize=10000)
    def in_turn(self, text):
        tokens = words(text)
        found = []
        for begin in range(len(tokens)):
            node = self.trie
            for end in range(begin, len(tokens)):
                node = node.get(tokens[end])
                if node is None:
                    break
                found.extend((field, value, begin, end) for field, value in sorted(node.get(None, ())))
        return tokens, found


def featurize(case, matcher):
    # Reader features are local to one retained message and its original position.
    # Precomputing these per turn is equivalent to running it on the retained turns.
    turns = [*case.evidence, Evidence("query", "Q: " + case.query, len(case.evidence), "public")]
    mentions, features = [], []
    for t in turns:
        role, _, body = t.text.partition(": ")
        toks, matches = matcher.in_turn(body)
        nfields = len(set(m[0] for m in matches))
        alternatives = defaultdict(set)
        for f, value, _, _ in matches:
            alternatives[f].add(value)
        distance = len(case.evidence) - t.order
        for field, value, start, end in matches:
            base = {
                "field": field, "role": role, "field/role": field + "/" + role,
                "log_distance": math.log1p(distance), "nfields": min(8, nfields),
                "ambiguity": math.log1p(len(alternatives[field])), "span": min(12, end-start+1),
                "statement_position": start / max(1,len(toks)), "log_length": math.log1p(len(toks)),
                "confirm": int(bool(CONFIRM.search(body))), "corrects": int(bool(CHANGE.search(body))),
                "affirm": int(bool(AFFIRM.search(body))), "question": int("?" in body),
                "field_value": field + "=" + value,
            }
            for offset in range(1,4):
                left = toks[start-offset] if start >= offset else "<start>"
                right = toks[end+offset] if end+offset < len(toks) else "<end>"
                base[f"{field}/left{offset}"] = left
                base[f"{field}/right{offset}"] = right
            mentions.append((t.id, field, value))
            features.append(base)
    return mentions, features


def fit(training, matcher):
    rows, labels = [], []
    for _, _, case, gold in training:
        found, feats = featurize(case, matcher)
        rows.extend(feats)
        labels.extend(gold.get(f) == v for _, f, v in found)
    vec = DictVectorizer()
    model = LogisticRegression(C=1, max_iter=300, solver="liblinear", random_state=42)
    matrix = vec.fit_transform(rows)
    matrix.indices = matrix.indices.astype(np.int32)
    matrix.indptr = matrix.indptr.astype(np.int32)
    model.fit(matrix, labels)
    return vec, model, len(labels)


def score(case, matcher, vec, model):
    mentions, feats = featurize(case, matcher)
    result = defaultdict(dict)
    if feats:
        probs = model.predict_proba(vec.transform(feats))[:,1]
        for (tid, field, value), p in zip(mentions, probs):
            key = (field, value)
            result[tid][key] = max(result[tid].get(key, 0), float(p))
    return result


def best_fields(support, tids):
    scores = {}
    for tid in tids:
        for (field, value), probability in support.get(tid, {}).items():
            scores[field] = max(scores.get(field, (0., "")), (probability, value))
    return scores


def evidence(case, support, budget, method):
    if method in ("recent","bm25","full"):
        return select(case, method, budget)
    history = list(case.evidence)
    if method == "value_recent":
        ranked = sorted(history, key=lambda e: (bool(support.get(e.id)), e.order), reverse=True)
        chosen = []
        for e in ranked:
            if budget.fits(chosen + [e]):
                chosen.append(e)
        return sorted(chosen, key=lambda e:e.order)

    # Maximize per-field support for the full-history reader's most plausible values.
    field_values = best_fields(support, ["query", *(e.id for e in history)])
    target = {(field,value): p for field,(p,value) in field_values.items()}
    chosen = []
    covered = support.get("query", {}).copy()
    while history:
        def rank(e):
            local = support.get(e.id, {})
            gain = sum(max(0., local.get(k,0.) - covered.get(k,0.))*p for k,p in target.items())
            return gain, e.order
        proposal = max(history, key=rank)
        history.remove(proposal)
        if budget.fits(chosen+[proposal]):
            chosen.append(proposal)
            for k,p in support.get(proposal.id, {}).items():
                covered[k] = max(covered.get(k,0.),p)
    return sorted(chosen, key=lambda e:e.order)


def predict(best):
    return {f:v for f,(p,v) in best.items() if f in REQUIRED or p >= .5}


def order_features(support, ids, best):
    candidates = defaultdict(dict)
    for tid in ids:
        for (field, value), p in support.get(tid, {}).items():
            candidates[field][value] = max(candidates[field].get(value, 0.), p)
    features = {"fields_proposed": len(predict(best))}
    for field in sorted(set(candidates) | REQUIRED):
        scores = sorted(candidates[field].values(), reverse=True)
        top = scores[0] if scores else 0.
        runner_up = scores[1] if len(scores)>1 else 0.
        features[field+"/top"] = top
        features[field+"/gap"] = top - runner_up
        features[field+"/alternatives"] = math.log1p(len(scores))
    return features


def calibrate(rows):
    features, labels = [], []
    for row in rows:
        for method in ("recent","value_recent","field_support","bm25"):
            ev = row["conditions"][method]
            if REQUIRED <= ev["args"].keys():
                features.append(ev["order_features"])
                labels.append(ev["args"] == row["gold"])
    vectorizer = DictVectorizer(sparse=False)
    classifier = LogisticRegression(C=1, max_iter=300, solver="liblinear", random_state=42)
    classifier.fit(vectorizer.fit_transform(features), labels)
    return vectorizer, classifier


def apply_calibration(rows, calibration):
    vectorizer, classifier = calibration
    conditions = [ev for row in rows for ev in row["conditions"].values()]
    probs = classifier.predict_proba(vectorizer.transform([ev["order_features"] for ev in conditions]))[:,1]
    for ev, p in zip(conditions, probs):
        ev["confidence"] = float(p)


def outcome(args, gold, confidence, threshold):
    after = execute(args if confidence >= threshold else {})
    return "no_write" if not after else "correct" if after == [gold] else "wrong"


def evaluate(data, matcher, vec, model, budget, methods=METHODS):
    rows = []
    for key, fingerprint, case, gold in data:
        support = score(case, matcher, vec, model)
        conditions = {}
        for method in methods:
            items = evidence(case, support, budget, method)
            ids = [e.id for e in items]
            fields = best_fields(support, ["query",*ids])
            args = predict(fields)
            confidence = min((fields.get(k,(0.,))[0] for k in REQUIRED), default=0.)
            cost = budget.cost(items)
            if method != "full":
                assert cost <= budget.limit
            conditions[method] = dict(args=args, confidence=confidence, selected=ids, tokens=cost,
                                      order_features=order_features(support,["query",*ids],fields))
        rows.append(dict(case=key, fingerprint=fingerprint, gold=gold, conditions=conditions))
    return rows


def summarize(rows, threshold):
    total = defaultdict(Counter)
    for r in rows:
        for method, ev in r["conditions"].items():
            ev["outcome"] = outcome(ev["args"],r["gold"],ev["confidence"],threshold)
            total[method][ev["outcome"]] += 1
    paired = {}
    for label in ("correct","wrong","no_write"):
        d = np.array([int(r["conditions"]["field_support"]["outcome"] == label) -
                      int(r["conditions"]["recent"]["outcome"] == label) for r in rows])
        freq = [(d == v).sum() for v in (-1,0,1)]
        resamples = np.random.default_rng(42).multinomial(len(rows), np.array(freq)/len(rows), size=20000)
        interval = 100*np.quantile((resamples[:,2]-resamples[:,0])/len(rows), [.025,.975])
        paired[label] = dict(win=int(freq[2]), loss=int(freq[0]), diff_pp=100*float(d.mean()),
                             ci95_pp=interval.tolist())
    return dict(n=len(rows), counts={m:dict(v) for m,v in total.items()}, paired=paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Run frozen evaluation after development.")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    training, used, texts, train_log = load("train", ["aa","ab","ac","ad"])
    dev, used, texts, dev_log = load("dev", ["aa"], used, texts)
    matcher = Mentions(training)
    vec, model, n_labels = fit(training, matcher)
    print("fitted",len(training),n_labels,flush=True)
    budget = Budget(128, TokenCounter("hf:" + str(ROOT / TOKENIZER)))
    dev_rows = evaluate(dev, matcher, vec, model, budget)
    calibration_rows = [r for r in dev_rows if int(r["fingerprint"][:2], 16) < 128]
    evaluation_rows = [r for r in dev_rows if int(r["fingerprint"][:2], 16) >= 128]
    calibration = calibrate(calibration_rows)
    apply_calibration(evaluation_rows, calibration)
    threshold = .5
    summary = {"commit":COMMIT, "train":train_log, "train_mentions":n_labels, "development":dev_log,
               "calibration_dialogues":len(calibration_rows),
               "threshold":threshold, "dev_raw":summarize(evaluation_rows,0.), "dev":summarize(evaluation_rows,threshold)}
    write_json(OUT / "development.json",summary)
    write_jsonl(OUT / "dev-details.jsonl",evaluation_rows)
    print("development", json.dumps(summary), flush=True)
    if not args.test:
        return
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prior = ROOT / "runs/ticket-execution/details.jsonl"
    for row in prior.read_text(encoding="utf8").splitlines():
        used.add(json.loads(row)["case"])
    write_json(OUT / "protocol.json", dict(code=source_hash,
               adapter_sha256=hashlib.sha256((Path(__file__).parent/"ticket_execution.py").read_bytes()).hexdigest(),
               adapter="CSV unquoting; pre-<C> user utterance is current, post-<C> messages are history",
               threshold=threshold, seed=42,
               primary="field_support versus recent, 128 tokens, identical learned reader and execution threshold",
               primary_cohort="official test-aa excluding first inspected purchase and prior scored dialogue IDs",
               secondary="dev-ab and dev-ac, excluding training, tuned development, primary and duplicate texts",
               bootstrap="paired per-dialogue percentile bootstrap, 20000 resamples; template dependence unmodeled"))
    for name, split, parts, skip in [("test","test",["aa"],True),("replication","dev",["ab","ac"],False)]:
        sample, used, texts, log = load(split,parts,used,texts,skip)
        rows = evaluate(sample,matcher,vec,model,budget)
        apply_calibration(rows, calibration)
        summary[name] = dict(coverage=log, raw=summarize(rows,0.), primary=summarize(rows,threshold))
        write_jsonl(OUT / (name+"-details.jsonl"),rows)
        write_json(OUT / "summary.json",summary)
        print(name, json.dumps(summary[name]),flush=True)


if __name__ == "__main__":
    main()
