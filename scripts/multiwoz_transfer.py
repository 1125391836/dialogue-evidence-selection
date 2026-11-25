"""Known-domain categorical state tracking at the final user turn in MultiWOZ 2.2."""
import json
import re
from collections import Counter
from pathlib import Path

from publication_pilot import ROOT, TOKENIZER, decode, value_recent
from context_bundles.core import Budget, Case, Counter as TokenCounter, Evidence
from context_bundles.io import sha256, write_json, write_jsonl
from context_bundles.selection import select


def main():
    data = ROOT / "data/external/multiwoz22"
    schema_path = data / "schema.json"
    schema = json.loads(schema_path.read_bytes())
    catalog = {s["service_name"]: {
        slot["name"]: [(v.casefold(), re.compile(r"(?<!\w)" + re.escape(v) + r"(?!\w)", re.I))
                       for v in slot["possible_values"] if v]
        for slot in s["slots"] if slot["is_categorical"]} for s in schema}
    paths = sorted((data / "test").glob("dialogues_*.json"))
    assert len(paths) == 2
    budget = Budget(128, TokenCounter("hf:" + str(ROOT / TOKENIZER)))
    counts = Counter()
    paired = Counter()
    rows = []
    seen = set()
    for path in paths:
        for d in json.loads(path.read_bytes()):
            key = d["dialogue_id"]
            assert key not in seen
            seen.add(key)
            i = max(i for i,t in enumerate(d["turns"]) if t["speaker"] == "USER")
            turn = d["turns"][i]
            history = tuple(Evidence(str(j), t["speaker"].lower() + ": " + t["utterance"], j, key)
                            for j,t in enumerate(d["turns"][:i]))
            case = Case(key, turn["utterance"], {}, history, key, "test")
            joint = dict.fromkeys(("recent", "value_recent", "bm25", "full"), True)
            domains = []
            for frame in turn["frames"]:
                domain = frame["service"]
                patterns = catalog[domain]
                gold = {k: [v.casefold() for v in vs] for k,vs in frame["state"]["slot_values"].items()
                        if k in patterns and vs}
                if not gold:
                    continue
                domains.append(domain)
                for method in joint:
                    chosen = (value_recent(case, patterns, budget) if method == "value_recent"
                              else select(case, method, budget))
                    text = "\n".join([*(e.text.partition(": ")[2] for e in chosen), case.query])
                    prediction = decode(text, patterns)
                    exact = prediction.keys() == gold.keys() and all(v in gold[k] for k,v in prediction.items())
                    joint[method] &= exact
                    cost = budget.cost(chosen)
                    assert method == "full" or cost <= 128
                    rows.append(dict(case=key, domain=domain, method=method, exact=exact, expected=gold,
                                     predicted=prediction, selected=[e.id for e in chosen], tokens=cost))
            if domains:
                counts["scored"] += 1
                counts.update(joint)
                paired[int(joint["value_recent"]) - int(joint["recent"])] += 1
    out = ROOT / "runs/multiwoz-transfer"
    out.mkdir(exist_ok=True)
    summary = dict(examined=len(seen), counts=counts, paired={str(k):v for k,v in paired.items()},
                   commit="fe0c8e65cfcd8462bd33c86e35f21addc84ca82b",
                   hashes={p.name: sha256(p) for p in [schema_path, *paths]})
    write_json(out / "summary.json", summary)
    write_jsonl(out / "details.jsonl", rows)
    print(summary)


if __name__ == "__main__":
    main()
