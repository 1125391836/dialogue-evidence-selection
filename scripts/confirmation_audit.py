"""Post-hoc sensitivity to an explicitly confirmed, conflicting movie showtime."""
import re
from collections import Counter, defaultdict
import numpy as np

from ticket_execution import ROOT, examples, render
from context_bundles.io import read_jsonl, write_json, write_jsonl

TIME = re.compile(r"(?<!\w)(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*([ap])\.?m\b", re.I)
CONFIRM = re.compile(r"\b(?:confirm|is that (?:all )?(?:correct|right)|is this (?:correct|right))\b", re.I)
YES = re.compile(r"^\s*(?:yes|yep|yeah|correct|that(?:'s| is) (?:right|correct)|sounds good)\b", re.I)
REVISE = re.compile(r"\b(?:but|instead|except|actually|change|rather|wrong)\b", re.I)


def times(text):
    return {int(h) % 12 * 60 + int(m or 0) + 720 * (p.lower() == "p") for h,m,p in TIME.findall(text)}


def explicit_time(case):
    turns = [(e.text[0], e.text[3:].replace("’", "'")) for e in case.evidence]
    # render() exposes no speaker for the current request. Skip it as confirmation
    # evidence, but use any contradictory time in it to rule out attribution.
    for i in range(len(turns)-2, -1, -1):
        role, text = turns[i]
        next_role, reply = turns[i+1]
        ts = times(text)
        if role != "A" or not CONFIRM.search(text) or len(ts)!=1:
            continue
        if next_role != "U" or not YES.search(reply) or REVISE.search(reply):
            continue
        later = " ".join([*(s for _,s in turns[i+1:]), case.query])
        if times(later) - ts or REVISE.search(later):
            continue
        return ts.pop(), i
    return None, None


def main():
    root = ROOT / "runs/ticket-learning"
    OUT = ROOT / "runs/confirmation-audit"
    OUT.mkdir(exist_ok=True)
    summary = {}
    all_rows = []
    for split, source, parts in [("test","test",["aa"]), ("replication","dev",["ab","ac"])]:
        rows = {r["case"]:r for r in read_jsonl(root / (split+"-details.jsonl"))}
        by_group = defaultdict(Counter)
        for part in parts:
            for key,context,_ in examples(source,part):
                if key not in rows:
                    continue
                row = rows.pop(key)
                t, i = explicit_time(render(context))
                gold = row["gold"]
                gt = times(gold["time.showing"])
                label = ("unresolved" if t is None or len(gt)!=1 else
                         "agrees" if t in gt else "conflicts")
                count = by_group[label]
                count["tasks"] += 1
                results = {}
                for m in ("recent","field_support"):
                    arm = row["conditions"][m]
                    correct = arm["outcome"]=="correct"
                    exact = correct
                    if label=="conflicts" and arm["outcome"]!="no_write":
                        pred = dict(arm["args"])
                        exact = times(pred.pop("time.showing","")) == {t} and pred == {
                            k:v for k,v in gold.items() if k!="time.showing"}
                    count[m+"_official"] += correct
                    count[m+"_sensitivity"] += exact
                    results[m] = dict(official=correct,sensitivity=exact)
                all_rows.append(dict(split=split,case=key,label=label,source_turn=i,
                                     confirmed_minutes=t,reference=gold["time.showing"],results=results))
        assert not rows
        delta = [int(r["results"]["field_support"]["sensitivity"]) - int(r["results"]["recent"]["sensitivity"])
                 for r in all_rows if r["split"] == split]
        weights = np.array([delta.count(value) for value in (-1, 0, 1)]) / len(delta)
        samples = np.random.default_rng(42).multinomial(len(delta), weights, size=20000)
        summary[split] = dict(groups=by_group, difference_pp=100*np.mean(delta), ci95_pp=list(
            np.quantile(100*(samples[:,2]-samples[:,0])/len(delta), [.025,.975])))
    write_json(OUT / "summary.json",summary)
    write_jsonl(OUT / "details.jsonl",all_rows)
    print(summary)


if __name__=="__main__":
    main()
