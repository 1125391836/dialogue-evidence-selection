"""Run the frozen recent-history comparison for SGD development and test dialogues."""
import os
from collections import Counter, defaultdict

from publication_pilot import ROOT, SGD, pilot, write_json, write_jsonl


def main():
    os.chdir(ROOT)
    for split in ("dev", "test"):
        coverage, rows = pilot(SGD.with_name(split), ("recent", "value_recent"))
        pairs = defaultdict(dict)
        exact = Counter()
        for row in rows:
            assert row["tokens"] <= 128
            pairs[row["case"]][row["method"]] = row["exact"]
            exact[row["method"]] += row["exact"]
        changes = Counter()
        for pair in pairs.values():
            change = int(pair["value_recent"]) - int(pair["recent"])
            changes[{-1: "loss", 0: "tie", 1: "win"}[change]] += 1
        assert len(pairs) == coverage["scored"]
        out = ROOT / "runs/recent-comparison" / split
        out.mkdir(parents=True, exist_ok=True)
        summary = dict(coverage=coverage, exact=exact, paired=changes)
        write_json(out / "summary.json", summary)
        write_jsonl(out / "details.jsonl", rows)
        print(split, summary, flush=True)


if __name__ == "__main__":
    main()
