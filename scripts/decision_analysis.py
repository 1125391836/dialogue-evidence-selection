"""Post-run paired checks for the decision-evidence experiments."""
import hashlib
import json
from collections import Counter

import numpy as np
from scipy.stats import binomtest

import review_labels
from context_bundles.io import read_jsonl,sha256,write_json
from decision_pivot import OUT
import decision_reader
import ticket_learning as study


def time_norm(obj):
    result=obj.copy()
    text=result.get("time","")
    minute=review_labels.gold_time(text)
    if minute is not None:
        result["time"]=f"{minute//60:02}:{minute%60:02}"
    return result


def reader_analysis(dataset):
    rows=read_jsonl(OUT/(dataset+"-7b-details.jsonl"))
    source="movie" if dataset=="movie_schema" else dataset
    prompts=read_jsonl(OUT/(source+"-prompts.jsonl"))
    assert len(rows)==len({r["case"] for r in rows})==len(prompts)
    cache={r["id"]:r["answer"] for r in read_jsonl(OUT/(dataset+"-7b-cache.jsonl"))}
    paired={}
    simr_clock=Counter()
    for task,r in zip(prompts,rows,strict=True):
        assert task["case"]==r["case"] and task["gold"]==r["gold"]
        for arm,s in task["arms"].items():
            result=r["arms"][arm]
            request=(decision_reader.request_key(decision_reader.movie.SYSTEM,s["prompt"],True)
                     if dataset=="movie_schema" else s["request"])
            assert request==result["request"] and request in cache
            if dataset=="simr":
                simr_clock[arm]+=int(time_norm(result["pred"])==time_norm(task["gold"]))
    for other in rows[0]["arms"]:
        if other=="decision":
            continue
        a=[r["arms"]["decision"]["outcome"]=="correct" for r in rows]
        b=[r["arms"][other]["outcome"]=="correct" for r in rows]
        plus=sum(x and not y for x,y in zip(a,b))
        minus=sum(y and not x for x,y in zip(a,b))
        n=len(rows)
        boot=np.random.default_rng(42).multinomial(n,[plus/n,minus/n,(n-plus-minus)/n],size=20000)
        paired[other]=dict(decision_only=plus,other_only=minus,
                           difference_pp=100*(plus-minus)/n,
                           dialogue_ci95_pp=(100*np.quantile((boot[:,0]-boot[:,1])/n,[.025,.975])).tolist(),
                           p=binomtest(plus,plus+minus).pvalue if plus+minus else 1)
    return dict(paired=paired,clock_normalized_exact=simr_clock)


def traditional(split):
    rows=read_jsonl(OUT/(split+"-details.jsonl"))
    gains,losses=Counter(),Counter()
    different=0
    for row in rows:
        a,b=[row["arms"][m] for m in ("recent","decision")]
        different+=a["selected"]!=b["selected"]
        for f in row["gold"].keys()|a["args"].keys()|b["args"].keys():
            old=a["args"].get(f)==row["gold"].get(f)
            new=b["args"].get(f)==row["gold"].get(f)
            gains[f]+=int(new and not old)
            losses[f]+=int(old and not new)
    # Diagnostic only: same top-k on each test cohort, hash-based tie breaking independent of gold.
    k=round(.2*len(rows))
    matched={}
    for arm in rows[0]["arms"]:
        valid=[r for r in rows if study.REQUIRED<=r["arms"][arm]["args"].keys()]
        ordered=sorted(valid,key=lambda r:(-r["arms"][arm]["confidence"],hashlib.sha256(r["case"].encode()).digest()))
        take=ordered[:k]
        matched[arm]=dict(submitted=len(take),wrong=sum(r["arms"][arm]["args"]!=r["gold"] for r in take))
    return dict(changed_selection=different,field_gains=gains,field_losses=losses,posthoc_same_coverage=matched)


if __name__=="__main__":
    results={s:traditional(s) for s in ("test","replication")}
    results.update({s:reader_analysis(s) for s in ("movie","movie_schema","restaurant","simr")})
    write_json(OUT/"paired-analysis.json",results)
    paths=sorted([p for p in OUT.iterdir() if p.suffix in {".json",".jsonl",".npz"} and p.name!="digests.json"])
    scripts=sorted((study.ROOT/"scripts").glob("decision_*.py"))
    write_json(OUT/"digests.json",{str(p.relative_to(study.ROOT)):sha256(p) for p in [*paths,*scripts,OUT/"design.md",OUT/"simr-design.md"]})
