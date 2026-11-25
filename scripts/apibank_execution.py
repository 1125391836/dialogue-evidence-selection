"""Unmodified API-Bank dialogues, isolated upstream writes, paired state scoring."""
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from types import ModuleType

from dateutil.parser import parse
from scipy.stats import beta
from publication_pilot import ROOT, TOKENIZER, value_recent
from context_bundles.core import Budget, Case, Counter as TokenCounter, Evidence
from context_bundles.io import write_json, write_jsonl
from context_bundles.selection import select

SOURCE = ROOT / "data/external/damo-convai/api-bank"
OUT = ROOT / "runs/apibank-execution"
# Load only the local simulation tools used in this study.
package = ModuleType("apis")
package.__path__ = [str(SOURCE / "apis")]
sys.modules["apis"] = package
from apis.api import API
package.API = API
from apis.timed_switch import TimedSwitch
from apis.cancel_timed_switch import CancelTimedSwitch
from apis.book_hotel import BookHotel

TOOLS = {t.__name__: t for t in (BookHotel, TimedSwitch, CancelTimedSwitch)}
DATE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|(?:Jan\w*|Feb\w*|Mar\w*|Apr\w*|May|Jun\w*|Jul\w*|Aug\w*|Sep\w*|Oct\w*|Nov\w*|Dec\w*)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)\b", re.I)
CLOCK = re.compile(r"\b(?:\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]m)?|\d{1,2}\s*[ap]m)\b", re.I)
NUMBERS = dict(zip("zero one two three four five six seven eight nine ten".split(), range(11)))


def reader(text, tool, names):
    entities = [(m.end(), name) for name in names for m in
                re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I)]
    dates = []
    for raw in DATE.findall(text):
        try:
            dates.append(parse(raw, default=datetime(2023, 1, 1)).date().isoformat())
        except ValueError:
            continue
    if not dates or not entities:
        return None
    name = max(entities)[1]
    if tool is BookHotel:
        if len(dates) < 2:
            return None
        args = dict(hotel_name=name, check_in_time=dates[-2], check_out_time=dates[-1])
        for key, unit in (("adult_count", "adults?"), ("child_count", "child(?:ren)?|kids?"), ("room_count", "rooms?")):
            matches = re.findall(r"\b(\d+|" + "|".join(NUMBERS) + r")\s+(?:" + unit + r")\b", text.lower())
            if not matches:
                return None
            token = matches[-1]
            args[key] = int(token) if token.isdigit() else NUMBERS[token]
        return args
    times = CLOCK.findall(text)
    if not times:
        return None
    try:
        stamp = parse(dates[-1] + " " + times[-1]).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    args = dict(name=name, time=stamp)
    if tool is TimedSwitch:
        states = re.findall(r"\b(on|off)\b", text.lower())
        if not states:
            return None
        args["on"] = states[-1] == "on"
    return args


def run(tool, initial, args):
    database = copy.deepcopy(initial)
    result = tool(database).call(**args) if args is not None else {}
    return database, result.get("exception")


def digest(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def difference(a, b):
    # Simultaneous Bonferroni-adjusted exact binomial bounds for paired wins/losses.
    n = len(a)
    if not n:
        return dict(n=0, per_1000=None, ci95_per_1000=None)
    win = sum(x and not y for x,y in zip(a,b))
    loss = sum(y and not x for x,y in zip(a,b))
    def bounds(k):
        return (beta.ppf(.0125, k, n-k+1) if k else 0.,
                beta.ppf(.9875, k+1, n-k) if k<n else 1.)
    wl, wh = bounds(win)
    ll, lh = bounds(loss)
    return dict(n=n, win=win, loss=loss, per_1000=1000*(win-loss)/n,
                ci95_per_1000=[1000*(wl-lh), 1000*(wh-ll)])


def main():
    budget = Budget(128, TokenCounter("hf:" + str(ROOT / TOKENIZER)))
    rows = []
    for path in sorted((SOURCE / "lv1-lv2-samples/level-1-given-desc").glob("*.jsonl")):
        dialogue = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
        calls = [m for m in dialogue if m["role"] == "API"]
        if len(calls) != 1 or calls[0]["api_name"] not in TOOLS:
            continue
        call = calls[0]
        tool = TOOLS[call["api_name"]]
        row = dict(case=path.stem, service=tool.database_name)
        rows.append(row)
        initial = json.loads((SOURCE / "init_database" / (tool.database_name + ".json")).read_bytes())
        if set(call["param_dict"]) != set(tool.input_parameters):
            row["excluded"] = "schema_mismatch"
            continue
        expected, error = run(tool, initial, call["result"]["input"])
        if error or expected == initial or call["result"].get("exception"):
            row["excluded"] = "reference_error_or_no_change"
            continue
        prefix = dialogue[:dialogue.index(call)]
        last = max(i for i,m in enumerate(prefix) if m["role"] == "User")
        hist = tuple(Evidence(str(i), m["role"] + ": " + m["text"], i, "history")
                     for i,m in enumerate(prefix[:last]))
        case = Case(path.stem, prefix[last]["text"], {}, hist, path.stem, "external")
        names = sorted({r["hotel_name"] for r in initial.values()} if tool is BookHotel else set(initial))
        patterns = {"catalog": [(s, re.compile(r"(?<!\w)"+re.escape(s)+r"(?!\w)", re.I)) for s in names]}
        row.update(initial=digest(initial), expected_state=digest(expected), outcomes={}, query=case.query,
                   history_tokens=budget.cost(hist))
        for method in ("recent", "bm25", "value_recent"):
            chosen = value_recent(case, patterns, budget) if method == "value_recent" else select(case, method, budget)
            text = "\n".join([*(e.text.partition(": ")[2] for e in chosen), case.query])
            args = reader(text, tool, names)
            state, error = run(tool, initial, args)
            outcome = "correct" if state == expected else "no_write" if state == initial else "wrong"
            row["outcomes"][method] = dict(outcome=outcome, arguments=args, error=error,
                                           after=digest(state), selected=[e.id for e in chosen])
    scored = [r for r in rows if "outcomes" in r]
    out = {m:Counter(r["outcomes"][m]["outcome"] for r in scored) for m in ("recent","bm25","value_recent")}
    summary = dict(files=len(rows), scored=len(scored), exclusions=Counter(r["excluded"] for r in rows if "excluded" in r),
                   services=Counter(r["service"] for r in scored), over_budget=sum(r["history_tokens"]>128 for r in scored),
                   outcomes=out, differences={})
    for baseline in ("recent","bm25"):
        for label in ("correct","wrong","no_write"):
            summary["differences"][baseline+"_"+label] = difference(
                [r["outcomes"]["value_recent"]["outcome"]==label for r in scored],
                [r["outcomes"][baseline]["outcome"]==label for r in scored])
    OUT.mkdir(exist_ok=True)
    write_jsonl(OUT / "details.jsonl", rows)
    write_json(OUT / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

