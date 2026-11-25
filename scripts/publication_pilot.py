"""Paired retention audit and a development-only categorical argument decoding pilot."""
from collections import Counter, defaultdict
import json
import os
import re

import numpy as np

from cross_dataset_retention import LIMITS, OUT, ROOT, SGD, TOKENIZER, sgd_cases
from context_bundles.core import Budget, Counter as TokenCounter
from context_bundles.io import read_jsonl, write_json, write_jsonl
from context_bundles.selection import pack, select


def paired():
    output = []
    for split in ("mem-test", "sgd-test"):
        groups = defaultdict(dict)
        for r in read_jsonl(OUT / split / "details.jsonl"):
            groups[r["budget"], r["group"], r["case"]][r["method"]] = int(r["success"])
        for limit in LIMITS:
            for baseline in ("bm25", "recent"):
                clusters = defaultdict(list)
                for (budget, group, _), scores in groups.items():
                    if budget == limit:
                        clusters[group].append(scores["schema_bm25"] - scores[baseline])
                sums = np.array([sum(v) for v in clusters.values()])
                sizes = np.array([len(v) for v in clusters.values()])
                rng = np.random.default_rng(42)
                samples = []
                for _ in range(100):
                    ix = rng.integers(len(sums), size=(100, len(sums)))
                    samples.extend(sums[ix].sum(axis=1) / sizes[ix].sum(axis=1))
                diff = [v for values in clusters.values() for v in values]
                output.append(dict(split=split, budget=limit, baseline=baseline, groups=len(sizes),
                                   cases=len(diff), wins=diff.count(1), losses=diff.count(-1),
                                   difference_pp=100*np.mean(diff),
                                   ci95_pp=list(100*np.quantile(samples, [.025, .975]))))
    return output


def decode(text, patterns):
    # Last explicit public enum mention per slot. Unknown values remain unanswered.
    prediction = {}
    for slot, candidates in patterns.items():
        found = [(m.end(), len(value), value) for value, pattern in candidates
                 for m in pattern.finditer(text)]
        if found:
            prediction[slot] = max(found)[2]
    return prediction


def value_recent(case, patterns, budget):
    def priority(piece):
        body = piece.text.partition(": ")[2]
        has_value = any(pattern.search(body) for candidates in patterns.values()
                        for _, pattern in candidates)
        return has_value, piece.order
    return pack(sorted(case.evidence, key=priority, reverse=True), budget)


def pilot(directory=SGD, methods=("query_only", "recent", "bm25", "schema_bm25", "half_recent", "full")):
    schemas = json.loads((directory / "schema.json").read_text(encoding="utf8"))
    # Identify a public service/intent signature from the adapter's properties.
    catalog = defaultdict(list)
    for service in schemas:
        for intent in service["intents"]:
            keys = set(intent["required_slots"]) | set(intent["optional_slots"])
            slots = [s for s in service["slots"] if s["name"] in keys]
            signature = tuple(sorted((s["name"], s["description"]) for s in slots))
            values = {s["name"]: s["possible_values"] for s in slots if s["is_categorical"]}
            catalog[signature].append(values)
    counter = TokenCounter("hf:" + str(TOKENIZER))
    budget = Budget(128, counter)
    rows = []
    coverage = Counter()
    for case, gold in sgd_cases(directory):
        coverage["examined"] += 1
        props = case.tool_schema["parameters"]["properties"]
        signature = tuple(sorted((k, v["description"]) for k, v in props.items()))
        catalogs = catalog[signature]
        assert catalogs and all(c == catalogs[0] for c in catalogs)
        values = catalogs[0]
        if not values:
            continue
        target = {k: str(v).casefold() for k, v in gold.items() if k in values}
        if not target:
            continue
        coverage["scored"] += 1
        patterns = {k: [(v.casefold(), re.compile(r"(?<!\w)" + re.escape(v) + r"(?!\w)", re.I))
                        for v in options if v] for k, options in values.items()}
        for method in methods:
            if method == "value_recent":
                chosen = value_recent(case, patterns, budget)
            else:
                chosen = [] if method == "query_only" else select(case, method, budget)
            text = "\n".join([*(e.text.partition(": ")[2] for e in chosen), case.query])
            pred = decode(text, patterns)
            rows.append(dict(case=case.id, method=method, expected=target, predicted=pred, exact=pred == target,
                             selected=[e.id for e in chosen], tokens=budget.cost(chosen)))
    return coverage, rows


if __name__ == "__main__":
    os.chdir(ROOT)
    root = ROOT / "runs/publication-pilot"
    root.mkdir(exist_ok=True)
    pairs = paired()
    write_json(root / "paired.json", pairs)
    coverage, rows = pilot()
    write_jsonl(root / "categorical.jsonl", rows)
    scores = Counter()
    for row in rows:
        scores[row["method"]] += int(row["exact"])
    write_json(root / "categorical-summary.json", dict(coverage=coverage, exact=scores))
    print(coverage, scores)
    for row in pairs:
        print(row)
