"""Held-out TicketTalk purchase calls with a local single-order SQL simulator."""
import csv
import io
import json
import re
import sqlite3
import subprocess
from collections import Counter, defaultdict

from apibank_execution import difference
from publication_pilot import ROOT, TOKENIZER, decode, value_recent
from context_bundles.core import Budget, Case, Counter as TokenCounter, Evidence
from context_bundles.io import write_json, write_jsonl
from context_bundles.selection import select

COMMIT = "d92cb6af3005f1dc09c39e75e7daf4a04905e00b"
REPO = ROOT / "data/external/taskmaster"
REQUIRED = {"name.movie", "name.theater", "time.showing"}


def examples(split, part):
    name = f"TM-3-2020/splits/{split}/{split}-{part}.tsv"
    data = subprocess.check_output(["git", "-C", str(REPO), "show", COMMIT + ":" + name])
    for fields in csv.reader(io.StringIO(data.decode("utf8")), delimiter="\t"):
        if len(fields) == 3 and fields[1].startswith("[PN]book_tickets[PAN]"):
            args = dict(s.split("[PAV]", 1) for s in fields[1].split("[PAN]")[1:])
            yield fields[2], fields[0], {k: v.strip().casefold() for k,v in args.items()}


def render(context):
    # TicketTalk TSV serializes the current utterance first, followed by <C>
    # and the chronological history. Earlier API records are not utterances.
    current, separator, history = context.partition("<C>")
    if not separator:
        history = ""
    query = re.fullmatch(r"<U>([^<>]+)", current.strip(), flags=re.DOTALL)
    if not query:
        return None
    parts = re.split(r"<([A-Z]+)>", history)
    messages = [(parts[i], parts[i+1].strip()) for i in range(1, len(parts), 2)
                if parts[i] in {"U", "A"} and parts[i+1].strip()]
    hist = tuple(Evidence(str(i), role + ": " + text, i, "public")
                 for i,(role,text) in enumerate(messages))
    return Case("public", query.group(1).strip(), {}, hist, "public", "transfer")


def execute(args):
    with sqlite3.connect(":memory:") as conn:
        conn.execute("create table orders(payload text not null)")
        if REQUIRED <= args.keys():
            conn.execute("insert into orders values (?)", (json.dumps(args, sort_keys=True),))
        return [json.loads(row[0]) for row in conn.execute("select payload from orders")]


def main():
    values = defaultdict(set)
    training_ids = set()
    for key, _, args in examples("train","aa"):
        training_ids.add(key)
        for field, value in args.items():
            if value:
                values[field].add(value)
    # One compiled matcher per schema field, populated only from training purchases.
    compiled = {k: re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, sorted(v, key=lambda x:(-len(x), x))))
                              + r")(?!\w)", re.I) for k,v in values.items()}
    patterns = {k:[("", pattern)] for k,pattern in compiled.items()}
    budget = Budget(128, TokenCounter("hf:" + str(ROOT / TOKENIZER)))
    rows = []
    seen = set()
    for part in ("ab","ac","ad"):
        for key, context, gold in examples("test",part):
            if key in seen or key in training_ids:
                continue
            seen.add(key)
            case = render(context)
            if case is None or not REQUIRED <= gold.keys():
                continue
            ref = execute(gold)
            assert ref == [gold]
            outcomes = {}
            for method in ("recent","bm25","value_recent"):
                chosen = value_recent(case, patterns, budget) if method == "value_recent" else select(case, method, budget)
                text = "\n".join([*(e.text.partition(": ")[2] for e in chosen), case.query])
                args = {}
                for field, pattern in compiled.items():
                    matches = pattern.findall(text)
                    if matches:
                        args[field] = matches[-1].casefold()
                after = execute(args)
                assert budget.cost(chosen) <= 128
                outcomes[method] = dict(after=after, args=args, selected=[e.id for e in chosen],
                                        outcome="correct" if after == ref else "wrong" if after else "no_write")
            rows.append(dict(case=key, part=part, expected=gold, outcomes=outcomes))
    summary = dict(commit=COMMIT, training_dialogues=len(training_ids), screened=len(seen), scored=len(rows))
    summary["outcomes"] = {m:Counter(r["outcomes"][m]["outcome"] for r in rows) for m in outcomes}
    summary["differences"] = {m+"_"+label:difference([r["outcomes"]["value_recent"]["outcome"]==label for r in rows],
                                                      [r["outcomes"][m]["outcome"]==label for r in rows])
                              for m in ("recent","bm25") for label in ("correct","wrong","no_write")}
    out = ROOT / "runs/ticket-execution"
    out.mkdir(exist_ok=True)
    write_json(out / "summary.json", summary)
    write_jsonl(out / "details.jsonl", rows)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
