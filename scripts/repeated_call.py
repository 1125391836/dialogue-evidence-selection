"""Equal-budget recovery of a repeated tool call from observable interaction logs."""
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from context_bundles.core import Budget, Case, Counter as TokenCounter, Evidence, canonical
from context_bundles.io import sha256, write_json, write_jsonl
from context_bundles.selection import pack, select


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external/sgd-revision-audit/train"
OUT = ROOT / "runs/repeated-call"
METHODS = ("recent", "bm25", "previous_call_only", "previous_call_recent", "full")


def observations(turns, index, service, method):
    history = []
    receipts = {}
    for i, turn in enumerate(turns[:index-1]):
        for j, frame in enumerate(turn.get("frames", [])):
            call = frame.get("service_call")
            if call:
                key = f"{i}.{j}"
                text = "call " + canonical([frame["service"], call["method"], call["parameters"]])
                history.append(Evidence(key, text, len(history), "dialogue"))
                if frame["service"] == service and call["method"] == method:
                    receipts[key] = call["parameters"]
        history.append(Evidence(str(i), turn["speaker"] + ": " + turn["utterance"], len(history), "dialogue"))
    return tuple(history), receipts


def argument_reader(query, chosen, receipts, slots):
    result = {}
    for text, arguments in [(e.text, receipts.get(e.id)) for e in chosen] + [("USER: " + query, None)]:
        if arguments is not None:
            result = {k: str(v).casefold() for k, v in arguments.items()}
            continue
        if not text.startswith("USER: "):
            continue
        utterance = text[6:].casefold()
        for slot, choices in slots.items():
            last = None
            for value, pattern in choices:
                for m in pattern.finditer(utterance):
                    if last is None or m.end() > last[0]:
                        last = (m.end(), value)
            if last:
                result[slot] = last[1]
    return result


def summarize(rows):
    summary = {}
    for stratum in ("all", "changed", "identical"):
        subset = [r for r in rows if stratum == "all" or r["revision"] == stratum]
        if not subset:
            continue
        correct = {m: sum(r["arms"][m]["correct"] for r in subset) for m in METHODS}
        rng = np.random.default_rng(20261123)
        diff = np.array([int(r["arms"]["previous_call_recent"]["correct"]) - int(r["arms"]["recent"]["correct"])
                         for r in subset])
        samples = rng.multinomial(len(diff), [(diff==-1).mean(), (diff==0).mean(), (diff==1).mean()], size=10000)
        summary[stratum] = dict(n=len(diff), correct=correct, wins=int((diff>0).sum()), losses=int((diff<0).sum()),
                                difference_pp=100*diff.mean(), ci95_pp=list(100*np.quantile(
                                    (samples[:,2]-samples[:,0])/len(diff), [.025,.975])))
    return summary


def main():
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    paths = [DATA / f"dialogues_{i:03d}.json" for i in range(1,128,8)]
    rev = "e852981ae34990f4358979625854259302feaa78"
    for source in [DATA / "schema.json", *paths]:
        if not source.exists():
            blob = subprocess.check_output(["git", "-C", str(ROOT / "data/external/sgd"),
                                            "show", rev + ":train/" + source.name])
            source.write_bytes(blob)
    cat = json.loads((DATA / "schema.json").read_bytes())
    methods = {}
    for service in cat:
        for intent in service["intents"]:
            keys = set(intent["required_slots"]) | set(intent["optional_slots"])
            slots = {}
            for s in service["slots"]:
                if s["name"] not in keys or not s["is_categorical"]:
                    continue
                # Short strings and bare numbers are ambiguous across slots.
                options = [v.casefold() for v in s["possible_values"] if len(v) >= 3 and not v.isnumeric()]
                if options:
                    slots[s["name"]] = [(v,re.compile(r"(?<!\w)"+re.escape(v)+r"(?!\w)")) for v in options]
            methods[service["service_name"],intent["name"]] = slots
    token = TokenCounter("hf:" + str(ROOT / "data/tokenizers/deepseek-v4/tokenizer.json"))
    rows = []
    coverage = Counter()
    peeked = set()
    for split in ("dev", "test"):
        peeked.update(s["service_name"] for s in json.loads((ROOT / "data/external/sgd" / split / "schema.json").read_bytes()))
    for path in paths:
        for d in json.loads(path.read_bytes()):
            coverage["screened"] += 1
            turns = d["turns"]
            calls = [(i, f) for i,t in enumerate(turns) for f in t["frames"] if f.get("service_call")]
            if not calls:
                continue
            coverage["any_call"] += 1
            index, frame = calls[-1]
            assert turns[index - 1]["speaker"] == "USER"
            service = frame["service"]
            call = frame["service_call"]
            method = call["method"]
            history, receipts = observations(turns, index, service, method)
            if not receipts:
                continue
            coverage["previous_same_tool"] += 1
            current = turns[index-1]["utterance"]
            slots = methods[service,method]
            case = Case(d["dialogue_id"], current, {}, history, d["dialogue_id"], "audit")
            previous = next(e for e in reversed(history) if e.id in receipts)
            for limit in (64,128,256):
                budget = Budget(limit, token)
                arms = {}
                for name in METHODS:
                    if name == "previous_call_only":
                        selected = pack([previous], budget)
                    elif name == "previous_call_recent":
                        selected = pack([previous, *(e for e in reversed(history) if e.id != previous.id)], budget)
                    else:
                        selected = select(case, name, budget)
                    size = budget.cost(selected)
                    assert name == "full" or size <= limit
                    prediction = argument_reader(current, selected, receipts, slots)
                    arms[name] = dict(predicted=prediction, selected=[e.id for e in selected], tokens=size)
                gold = {k:str(v).casefold() for k,v in call["parameters"].items()}
                for arm in arms.values():
                    arm["correct"] = arm["predicted"] == gold
                prior = {k:str(v).casefold() for k,v in receipts[previous.id].items()}
                rows.append(dict(case=case.id, budget=limit, service=service, new_service=service not in peeked,
                                 method=method, cutoff=index, previous=prior, expected=gold, arms=arms,
                                 revision="identical" if prior==gold else "changed"))
    assert len({r["case"] for r in rows})*3 == len(rows)
    write_jsonl(OUT / "details.jsonl", rows)
    results = {}
    for limit in (64,128,256):
        cohort = [r for r in rows if r["budget"]==limit]
        results[str(limit)] = dict(total=summarize(cohort), unseen_service=summarize([r for r in cohort if r["new_service"]]))
    summary = dict(coverage=coverage, results=results, services=Counter(r["service"] for r in rows if r["budget"]==128),
                   hashes={p.name:sha256(p) for p in [*paths, DATA / "schema.json", Path(__file__)]})
    write_json(OUT / "summary.json", summary)
    print(json.dumps(dict(coverage=coverage, primary=results["128"]),ensure_ascii=False))


if __name__ == "__main__":
    main()
