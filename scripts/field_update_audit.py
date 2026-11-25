"""Replay two fixed-budget histories; score updates with SGD states after prediction."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from context_bundles.io import read_jsonl, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def choose(recent, retrieved, required):
    # Replace the recent context only when the candidate decoder preserves every
    # existing prediction and its additions are all required by the public API.
    extras = retrieved.keys() - recent.keys()
    if extras and extras <= required and recent.items() <= retrieved.items():
        return "value_recent"
    return "recent"


def audit(split):
    directory = ROOT / "data/external/sgd" / split
    schema = json.loads((directory / "schema.json").read_text(encoding="utf8"))
    required = {(service["service_name"], intent["name"]): set(intent["required_slots"])
                for service in schema for intent in service["intents"]}
    cached = defaultdict(dict)
    for r in read_jsonl(ROOT / "runs/recent-comparison" / split / "details.jsonl"):
        assert r["tokens"] <= 128
        cached[r["case"]][r["method"]] = r
    rows = []
    strata = defaultdict(Counter)
    totals = Counter()
    transitions = Counter()
    for path in sorted(directory.glob("dialogues_*.json")):
        for d in json.loads(path.read_text(encoding="utf8")):
            key = d["dialogue_id"]
            if key not in cached:
                continue
            turns = d["turns"]
            i, frame = [(i, f) for i,t in enumerate(turns) for f in t["frames"] if "service_call" in f][-1]
            service = frame["service"]
            call = frame["service_call"]
            method = call["method"]
            pair = cached[key]
            predictions = {k: v["predicted"] for k,v in pair.items()}
            choice = choose(predictions["recent"], predictions["value_recent"], required[service, method])
            predictions["required_recovery"] = predictions[choice]
            target = pair["recent"]["expected"]
            correct = {k: p == target for k,p in predictions.items()}
            totals.update(correct)
            totals["tasks"] += 1
            transitions[int(correct["required_recovery"]) - int(correct["recent"])] += 1
            # Consecutive user states in the same service and intent, ending at the call.
            states = []
            for turn in reversed(turns[:i]):
                if turn["speaker"] != "USER":
                    continue
                matches = [f["state"] for f in turn["frames"] if f["service"] == service
                           and f.get("state", {}).get("active_intent") == method]
                if len(matches) != 1:
                    break
                states.append(matches[0]["slot_values"])
            fields = {}
            for slot, value in target.items():
                trajectory = [{str(v).casefold() for v in s.get(slot, [])} for s in reversed(states)]
                if not trajectory or value not in trajectory[-1]:
                    label = "unaligned"
                elif any(a and b and a.isdisjoint(b) for a,b in zip(trajectory, trajectory[1:])):
                    label = "updated_in_segment"
                elif len(trajectory) >= 2 and value in trajectory[-2]:
                    label = "carried_from_previous"
                else:
                    label = "other"
                fields[slot] = label
                strata[label]["fields"] += 1
                for name, pred in predictions.items():
                    strata[label][name] += pred.get(slot) == value
            rows.append(dict(case=key, service=service, method=method, fields=fields, selected=choice,
                             predicted=predictions, expected=target, exact=correct))
    assert len(rows) == len(cached) == len({r["case"] for r in rows})
    out = ROOT / "runs/field-update-audit" / split
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "details.jsonl", rows)
    summary = dict(exact=totals, fields=strata, paired_vs_recent={str(k):v for k,v in transitions.items()})
    write_json(out / "summary.json", summary)
    print(split, json.dumps(summary))


if __name__ == "__main__":
    for split in ("dev", "test"):
        audit(split)
