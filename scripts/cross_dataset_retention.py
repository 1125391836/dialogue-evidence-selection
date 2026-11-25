"""Same-budget, answer-blind history selection on Mem2Act and SGD development splits."""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from binding_pilot import DATA, TOKENIZER, message_case
from context_bundles.core import Budget, Case, Counter as TokenCounter, Evidence
from context_bundles.io import read_jsonl, sha256, write_json, write_jsonl
from context_bundles.selection import select

ROOT = Path(__file__).resolve().parents[1]
SGD = Path("data/external/sgd/dev")
OUT = Path("runs/cross-dataset-retention")
METHODS = ("recent", "bm25", "schema_bm25", "schema_history")
LIMITS = (128, 512)


def mem_cases(split="dev"):
    labels = {r["id"]: r["gold"]["arguments"] for r in read_jsonl(DATA / "labels.jsonl") if r["split"] == split}
    for row in read_jsonl(DATA / "cases.jsonl"):
        if row["split"] != split:
            continue
        case = message_case(Case.from_dict(row))
        evidence = []
        for e in case.evidence:
            m = json.loads(e.text)
            body = m.get("content") or ""
            if m.get("tool_calls"):
                body += " " + json.dumps(m["tool_calls"], ensure_ascii=False)
            evidence.append(Evidence(e.id, m["role"] + ": " + body, e.order, e.session_id))
        yield Case(case.id, case.query, case.tool_schema, tuple(evidence), case.group_id, split), labels[case.id]


def sgd_cases(directory):
    schema = {s["service_name"]: s for s in json.loads((directory / "schema.json").read_text(encoding="utf8"))}
    for path in sorted(directory.glob("dialogues_*.json")):
        for dialogue in json.loads(path.read_text(encoding="utf8")):
            turns = dialogue["turns"]
            calls = [(i, f) for i, t in enumerate(turns) for f in t["frames"] if f.get("service_call")]
            if not calls:
                continue
            i, frame = calls[-1]
            assert i and turns[i-1]["speaker"] == "USER"
            call = frame["service_call"]
            public = schema[frame["service"]]
            intent = next(x for x in public["intents"] if x["name"] == call["method"])
            keys = set(intent["required_slots"]) | set(intent["optional_slots"])
            props = {s["name"]: {"description": s["description"]} for s in public["slots"] if s["name"] in keys}
            history = tuple(Evidence(str(j), t["speaker"].lower() + ": " + t["utterance"], j, "dialogue")
                            for j, t in enumerate(turns[:i-1]))
            key = dialogue["dialogue_id"]
            yield Case(key, turns[i-1]["utterance"], {"parameters": {"properties": props}}, history, key, directory.name), call["parameters"]


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sgd-test", action="store_true")
    group.add_argument("--mem-test", action="store_true")
    args_cli = parser.parse_args()
    directory = SGD.with_name("test") if args_cli.sgd_test else SGD
    output = OUT / "sgd-test" if args_cli.sgd_test else OUT
    counter = TokenCounter("hf:" + str(TOKENIZER))
    rows = []
    summary = {}
    datasets = [("SGD", sgd_cases(directory))]
    if args_cli.mem_test:
        output = OUT / "mem-test"
        datasets = [("Mem2Act", mem_cases("test"))]
    elif not args_cli.sgd_test:
        datasets.insert(0, ("Mem2Act", mem_cases()))
    for dataset, cases in datasets:
        coverage = Counter()
        results = defaultdict(Counter)
        for case, args in cases:
            coverage["examined"] += 1
            targets = {}
            for key, value in args.items():
                if isinstance(value, bool) or not isinstance(value, (str, int, float)) or not str(value).strip():
                    continue
                pattern = re.compile(r"(?<!\w)" + re.escape(str(value).strip()) + r"(?!\w)", re.I)
                if pattern.search(case.query):
                    continue
                ids = {e.id for e in case.evidence if pattern.search(e.text.partition(": ")[2])}
                if ids:
                    targets[key] = ids
            if not targets:
                continue
            coverage["history_dependent_literal"] += 1
            for limit in LIMITS:
                budget = Budget(limit, counter)
                scores = {}
                for method in METHODS:
                    chosen = select(case, method, budget)
                    ids = {e.id for e in chosen}
                    kept = [k for k, hits in targets.items() if hits & ids]
                    scores[method] = len(kept) == len(targets)
                    rows.append(dict(dataset=dataset, case=case.id, group=case.group_id, budget=limit,
                                     method=method, selected=[e.id for e in chosen], tokens=budget.cost(chosen),
                                     fields=list(targets), kept=kept, success=scores[method]))
                    results[limit][method] += scores[method]
                for baseline in METHODS[:-1]:
                    results[limit][baseline+"_wins_over_fusion"] += scores[baseline] and not scores["schema_history"]
                    results[limit][baseline+"_losses_to_fusion"] += scores["schema_history"] and not scores[baseline]
        summary[dataset] = {"coverage": coverage, "all_fields_retained": results}
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "details.jsonl", rows)
    write_json(output / "summary.json", summary)
    files = [Path(__file__), Path("src/context_bundles/selection.py"), TOKENIZER]
    if not args_cli.mem_test:
        files.extend(directory.glob("*.json"))
    if not args_cli.sgd_test:
        files.extend([DATA / "cases.jsonl", DATA / "labels.jsonl"])
    write_json(output / "hashes.json", {p.name if p.is_absolute() else str(p): sha256(p) for p in files})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
