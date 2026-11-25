"""Paired TicketTalk experiment with frozen decision evidence and Pivot selectors."""
from collections import Counter
from pathlib import Path

import numpy as np
import ticket_learning as study
from context_bundles.core import Budget, Counter as Tokens
from context_bundles.io import read_jsonl, write_json, write_jsonl
from decision_pivot import OUT, load_pivot
from decision_selection import choose
from review_benchmarks import contexts

ARMS = ("recent","field_support","adjacent","pivot","decision")


def coverage_threshold(rows, arm):
    scores = [r["arms"][arm]["confidence"] for r in rows if study.REQUIRED <= r["arms"][arm]["args"].keys()]
    candidates = [1.000001, *set(scores)]
    return min(candidates,key=lambda t:(abs(sum(s>=t for s in scores)-.2*len(rows)), -t))


def infer(key, case, gold, resources):
    matcher,vec,model,cal,budget,pivot = resources
    support = study.score(case,matcher,vec,model)
    arms = {}
    for arm in ARMS:
        selected = (study.evidence(case,support,budget,arm) if arm=="field_support" else choose(case,budget,arm,pivot))
        ids = [e.id for e in selected]
        cost = budget.cost(selected)
        assert cost <= budget.limit
        best = study.best_fields(support,["query",*ids])
        args = study.predict(best)
        ev = dict(order_features=study.order_features(support,["query",*ids],best))
        study.apply_calibration([{"conditions":{arm:ev}}],cal)
        p = ev["confidence"]
        arms[arm] = dict(selected=ids,tokens=cost,args=args,confidence=p,
                         fixed=study.outcome(args,gold,p,.5))
    return dict(case=key, gold=gold,arms=arms)


def summarize(rows, groups):
    summary = {}
    for target in ("fixed","dev20"):
        summary[target] = {a:dict(Counter(r["arms"][a][target] for r in rows)) for a in ARMS}
        unique = sorted({groups[r["case"]] for r in rows})
        # Resample the same entire scenarios for both arms.
        draw = np.random.default_rng(42).multinomial(len(unique),np.ones(len(unique))/len(unique),size=10000)
        deltas = {}
        for other in ("recent","adjacent","field_support","pivot"):
            table = np.zeros((len(unique),4))
            for r in rows:
                g = unique.index(groups[r["case"]])
                for j,a in enumerate(("decision",other)):
                    outcome=r["arms"][a][target]
                    table[g,2*j:2*j+2] += [outcome=="wrong", outcome!="no_write"]
            b = draw@table
            b = b[(b[:,1]>0)&(b[:,3]>0)]
            deltas[other] = (100*np.percentile(b[:,0]/b[:,1]-b[:,2]/b[:,3],[2.5,97.5])).tolist() if len(b) else None
        summary[target+"_conditional_risk_delta_ci95_pp"] = deltas
    return summary


def main():
    pivot = load_pivot()
    train,used,texts,_ = study.load("train",["aa","ab","ac","ad"])
    dev,_,_,_=study.load("dev",["aa"],used,texts)
    matcher = study.Mentions(train)
    vec,model,_=study.fit(train,matcher)
    budget=Budget(128,Tokens("hf:"+str(study.ROOT/study.TOKENIZER)))
    cal_rows = [r for r in dev if int(r[1][:2],16)<128]
    cal = study.calibrate(study.evaluate(cal_rows,matcher,vec,model,budget))
    resources = matcher,vec,model,cal,budget,pivot
    threshold_rows=[infer(k,c,g,resources) for k,fp,c,g in dev if int(fp[:2],16)>=128]
    thresholds={a:coverage_threshold(threshold_rows,a) for a in ARMS}
    write_json(OUT/"thresholds.json",dict(target=.2,dev_n=len(threshold_rows),thresholds=thresholds))
    cohorts,baseline=contexts()
    groups={r["case"]:r["scenario"] for r in read_jsonl(study.ROOT/"runs/review-generalization/case-groups.jsonl")}
    results={}
    for split,cases in cohorts.items():
        rows=[]
        for key,(raw,gold) in cases.items():
            row=infer(key,study.render(raw),gold,resources)
            for a,ev in row["arms"].items():
                ev["dev20"]=study.outcome(ev["args"],gold,ev["confidence"],thresholds[a])
                if a in ("recent","field_support"):
                    base=baseline[split][key]["conditions"][a]
                    assert (ev["selected"],ev["args"],ev["fixed"]) == (base["selected"],base["args"],base["outcome"])
            rows.append(row)
        write_jsonl(OUT/(split+"-details.jsonl"),rows)
        results[split]=summarize(rows,groups)
        print(split,results[split]["dev20"],flush=True)
    write_json(OUT/"ticket-summary.json",results)


if __name__=="__main__":
    main()
