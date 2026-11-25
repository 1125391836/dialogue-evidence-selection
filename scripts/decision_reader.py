"""7B reader on a frozen restaurant cohort and the existing 96-movie sample."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
import urllib.request
from collections import Counter

from context_bundles.core import Budget, Case, Counter as Tokens, Evidence
from context_bundles.io import read_jsonl, write_json, write_jsonl
from decision_pivot import ASSETS, OUT, load_pivot
from decision_selection import choose
from review_benchmarks import contexts
import review_generative as movie
import review_sgd_transfer as sgd
import decision_simr as simr
import ticket_learning as study

COMMIT = "e852981ae34990f4358979625854259302feaa78"
WEIGHTS = "2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730"
SERVICE = "Restaurants_1"
METHOD = "ReserveRestaurant"
REST_PROMPT = ("Recover the current ReserveRestaurant call from the visible dialogue. "
    "Return only one JSON object of supported parameters. Required: restaurant_name (restaurant name), "
    "city (restaurant city), time (local 24-hour HH:MM). Optional: date (YYYY-MM-DD), "
    "party_size (number as a string). The service defaults are date 2019-03-01 and party_size 2. "
    "Use 2019-03-01 as today for relative dates. Omit optional fields if not established. "
    "Omit unsupported required fields. Respect the latest user choice. Do not invent values.")
PORT = 18757


def blob(path):
    return subprocess.check_output(["git","-C",str(sgd.REPO),"show",COMMIT+":"+path])


def hashtext(s):
    return hashlib.sha256(s.encode()).hexdigest()


def restaurant():
    ids,seen,_=sgd.previous_inputs()
    old = json.loads((sgd.OUT/"sgd-protocol.json").read_bytes())["files"]
    for path in old:
        for d in json.loads(blob(path)):
            ids.add(("train",d["dialogue_id"]))
            for i,t in enumerate(d["turns"]):
                if any(f.get("service_call") for f in t["frames"]):
                    seen.add(sgd.digest(d["turns"][:i]))
    candidates=[]
    tally=Counter()
    for no in range(3,128,8):
        path=f"train/dialogues_{no:03d}.json"
        for d in json.loads(blob(path)):
            tally["screened"] += 1
            calls=[(i,f) for i,t in enumerate(d["turns"]) for f in t["frames"] if
                   f["service"]==SERVICE and f.get("service_call",{}).get("method")==METHOD]
            if not calls:
                continue
            i,f=calls[-1]
            turns=d["turns"][:i]
            if not turns or turns[-1]["speaker"]!="USER":
                continue
            key=d["dialogue_id"]
            fp=sgd.digest(turns)
            if ("train",key) in ids or fp in seen:
                tally["duplicates"]+=1
                continue
            ids.add(("train",key))
            seen.add(fp)
            gold={k:str(v).strip().casefold() for k,v in f["service_call"]["parameters"].items()}
            ev=tuple(Evidence(str(j),("U: " if t["speaker"]=="USER" else "A: ")+t["utterance"],j,key)
                     for j,t in enumerate(turns[:-1]))
            candidates.append((key,Case(key,turns[-1]["utterance"],{},ev,key,"restaurant"),gold,path,fp))
    candidates.sort(key=lambda x:hashtext("decision-restaurant-v1/"+x[0]))
    tally["eligible"]=len(candidates)
    tally["sampled"]=min(96,len(candidates))
    write_json(OUT/"restaurant-sampling.json",dict(commit=COMMIT, counts=tally,
               source=[f"train/dialogues_{i:03d}.json" for i in range(3,128,8)],old=old))
    return candidates[:96]


def prepare(dataset):
    arms=["recent","adjacent","pivot","decision","full"] if dataset!="movie" else ["recent","field_support","adjacent","pivot","decision"]
    if dataset!="movie":
        cohort=restaurant() if dataset=="restaurant" else simr.cases()
        system=REST_PROMPT if dataset=="restaurant" else simr.PROMPT
        pivot=load_pivot()
    else:
        primary=contexts()[0]["test"]
        retained={r["case"]:r["arms"] for r in read_jsonl(OUT/"test-details.jsonl")}
        keys=json.loads((movie.OUT/"protocol.json").read_bytes())["sample"]
        cohort=[(k,study.render(primary[k][0]),primary[k][1],"test-aa",None) for k in keys]
        system=movie.SYSTEM
    budget=Budget(128,Tokens("hf:"+str(study.ROOT/study.TOKENIZER)))
    rows=[]
    for key,case,gold,path,fp in cohort:
        sels={}
        for arm in arms:
            if dataset!="movie":
                selected=list(case.evidence) if arm=="full" else choose(case,budget,arm,pivot)
            else:
                keep=set(retained[key][arm]["selected"])
                selected=[e for e in case.evidence if e.id in keep]
            cost=budget.cost(selected)
            assert arm=="full" or cost<=128
            prompt="History:\n"+"\n".join(e.text for e in selected)+"\nCurrent user: "+case.query
            req=hashtext(json.dumps([WEIGHTS,system,prompt,42,0,128],ensure_ascii=False))
            sels[arm]=dict(prompt=prompt,request=req,selected=[e.id for e in selected],tokens=cost)
        rows.append(dict(case=key,source=path,fingerprint=fp,gold=gold,arms=sels))
    write_jsonl(OUT/(dataset+"-prompts.jsonl"),rows)
    return system,rows


MOVIE_SCHEMA = {"type":"json_schema","json_schema":{"name":"order","schema":{
    "type":"object","additionalProperties":False,
    "properties":{k:{"type":"string"} for k in
                  ("name.movie","name.theater","time.showing","date.showing","num.tickets","seating")}
}}}


def request_key(system,prompt,constrained=False):
    parts=[WEIGHTS,system,prompt,42,0,128]
    if constrained:
        parts.append(MOVIE_SCHEMA)
    return hashtext(json.dumps(parts,ensure_ascii=False))


def generate(system,prompt,constrained=False):
    payload=dict(messages=[dict(role="system",content=system),dict(role="user",content=prompt)],
                 temperature=0,seed=42,max_tokens=128,stream=False,response_format=MOVIE_SCHEMA if constrained else {"type":"json_object"})
    url=f"http://127.0.0.1:{PORT}/v1/chat/completions"
    req=urllib.request.Request(url,json.dumps(payload).encode(),{"Content-Type":"application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req,timeout=180) as resp:
                choice=json.load(resp)["choices"][0]
            return dict(raw=choice["message"]["content"],finish=choice["finish_reason"])
        except OSError:
            if attempt==2:
                raise
            time.sleep(2)


def classify(raw,gold,required,defaults):
    try:
        obj=json.loads(raw)
        if not isinstance(obj,dict) or any(not isinstance(v,(str,int,float)) or isinstance(v,bool) for v in obj.values()):
            raise ValueError
        pred={k:str(v).strip().casefold() for k,v in obj.items() if str(v).strip()}
        malformed=False
    except ValueError:
        pred={}
        malformed=True
    written=required<=pred.keys()
    return dict(pred=pred,malformed=malformed,outcome="no_write" if not written else "correct" if pred==gold else "wrong",
                required_exact=written and all(pred[k]==gold.get(k) for k in required),
                default_exact=written and (defaults|pred)==(defaults|gold))


def evaluate(dataset):
    constrained = dataset=="movie_schema"
    path=OUT/(("movie" if constrained else dataset)+"-prompts.jsonl")
    rows=read_jsonl(path)
    system={"restaurant":REST_PROMPT,"movie":movie.SYSTEM,"movie_schema":movie.SYSTEM,"simr":simr.PROMPT}[dataset]
    required={"restaurant_name","city","time"} if dataset=="restaurant" else simr.FIELDS if dataset=="simr" else study.REQUIRED
    defaults={"date":"2019-03-01","party_size":"2"} if dataset=="restaurant" else {}
    cache_file=OUT/(dataset+"-7b-cache.jsonl")
    cache={r["id"]:r["answer"] for r in read_jsonl(cache_file)} if cache_file.exists() else {}
    pending={}
    for row in rows:
        for ev in row["arms"].values():
            assert ev["request"]==request_key(system,ev["prompt"])
            key=request_key(system,ev["prompt"],constrained)
            ev["request"]=key
            if key not in cache:
                pending[key]=ev["prompt"]
    if pending:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models") as resp:
            available=json.load(resp)["data"]
        if WEIGHTS not in [m["id"] for m in available]:
            raise ValueError("Expected pinned local reader alias")
        with concurrent.futures.ThreadPoolExecutor(4) as pool, cache_file.open("a",encoding="utf8") as out:
            tasks={pool.submit(generate,system,p,constrained):k for k,p in pending.items()}
            for task in concurrent.futures.as_completed(tasks):
                key=tasks[task]
                cache[key]=task.result()
                out.write(json.dumps(dict(id=key,answer=cache[key]))+"\n")
                out.flush()
    results=[]
    counts={}
    for row in rows:
        arms={}
        for arm,ev in row["arms"].items():
            answer=cache[ev["request"]]
            result=classify(answer["raw"],row["gold"],required,defaults)
            arms[arm]=dict(**result,request=ev["request"],finish=answer["finish"])
            total=counts.setdefault(arm,Counter())
            total[result["outcome"]]+=1
            for metric in ("required_exact","default_exact","malformed"):
                total[metric]+=result[metric]
        results.append(dict(case=row["case"],gold=row["gold"],arms=arms))
    write_jsonl(OUT/(dataset+"-7b-details.jsonl"),results)
    write_json(OUT/(dataset+"-7b-summary.json"),dict(n=len(rows),weights=WEIGHTS,counts=counts,unique=len(cache)))
    print(dataset,counts,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset",choices=["movie","restaurant","simr","movie_schema"])
    ap.add_argument("stage",choices=["prepare","score"])
    args=ap.parse_args()
    if args.dataset=="movie_schema" and args.stage=="prepare":
        ap.error("movie_schema uses existing movie prompts; use movie prepare")
    (prepare if args.stage=="prepare" else evaluate)(args.dataset)
