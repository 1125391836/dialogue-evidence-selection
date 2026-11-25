"""Fixed second-reader audit on a hash sample; local loopback llama.cpp server."""
import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from collections import Counter

from ticket_execution import ROOT, REQUIRED, execute, examples, render
from context_bundles.io import read_jsonl, write_json, write_jsonl

OUT = ROOT / "runs/review-generative"
SYSTEM = ("Recover the current movie ticket order from visible dialogue. Return one JSON object with "
          "string fields name.movie, name.theater, time.showing, and, only when established, "
          "date.showing, num.tickets, seating. Copy the user's chosen values, using words as spoken, "
          "not offered alternatives. Do not invent values. Omit unsupported fields. Output JSON only.")
WEIGHTS = "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
METHODS = ("recent", "field_support")


def complete(prompt, port):
    body = dict(messages=[dict(role="system",content=SYSTEM), dict(role="user",content=prompt)],
                temperature=0, seed=42, max_tokens=128, stream=False, response_format={"type":"json_object"})
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as response:
        result = json.loads(response.read())
    answer = result["choices"][0]["message"]["content"]
    try:
        raw = json.loads(answer)
        if not isinstance(raw,dict):
            raise ValueError("expected argument object")
        pred = {k:str(v).casefold().strip() for k,v in raw.items()
                if isinstance(v,(str,float,int)) and str(v).strip()}
        malformed = False
    except (ValueError,TypeError):
        pred, malformed = {}, True
    return dict(predicted=pred, raw=answer, malformed=malformed,
                seconds=time.perf_counter()-start, finish=result["choices"][0]["finish_reason"])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=18756)
    ap.add_argument("--refresh", action="store_true", help="Request fresh responses and store them in a separate file.")
    args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    index = {r["case"]:r for r in read_jsonl(ROOT/"runs/review-chronology/test-details.jsonl")}
    sample = sorted(index,key=lambda k:hashlib.sha256(("reader-v1/"+k).encode()).digest())[:96]
    prompts = {}
    gold = {}
    for key,ctx,target in examples("test","aa"):
        if key not in sample or key in gold:
            continue
        case=render(ctx)
        if not case or not REQUIRED<=target.keys():
            continue
        row=index[key]
        assert row["gold"]==target
        gold[key]=target
        for method in METHODS:
            selected=set(row["conditions"][method]["selected"])
            turns=[e.text for e in case.evidence if e.id in selected]
            prompts[key, method]="History:\n"+"\n".join(turns)+"\nCurrent user: "+case.query
    assert len(gold)==len(sample)
    write_json(OUT/"protocol.json",dict(source="review-chronology/test-aa", sample=sample, system=SYSTEM,
                 model="Qwen/Qwen2.5-1.5B-Instruct-GGUF Q4_K_M", sha256=WEIGHTS,
                 llama_cpp="b6330 Vulkan", temperature=0, seed=42, budget=128,
                 scoring="exact lowercased fields and values; no learned submission threshold"))
    prefix=OUT/("cache-refresh.jsonl" if args.refresh else "cache.jsonl")
    cache=({r["prompt_sha256"]:r["answer"] for r in read_jsonl(prefix)}
           if prefix.exists() and not args.refresh else {})
    hashes={k:hashlib.sha256((WEIGHTS+SYSTEM+v).encode()).hexdigest() for k,v in prompts.items()}
    pending={hashes[k]:v for k,v in prompts.items() if hashes[k] not in cache}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool, prefix.open("a",encoding="utf8") as f:
        tasks={pool.submit(complete,p,args.port):h for h,p in pending.items()}
        for future in concurrent.futures.as_completed(tasks):
            h=tasks[future]
            cache[h]=future.result()
            f.write(json.dumps(dict(prompt_sha256=h,answer=cache[h]),ensure_ascii=False)+"\n")
            f.flush()
    rows=[]
    counts={m:Counter() for m in METHODS}
    for key in sample:
        arms={}
        for m in METHODS:
            result=cache[hashes[key,m]]
            after=execute(result["predicted"])
            category="correct" if after==[gold[key]] else "wrong" if after else "no_write"
            counts[m][category]+=1
            arms[m]=dict(**result,outcome=category)
        rows.append(dict(case=key,expected=gold[key],arms=arms))
    suffix="-refresh" if args.refresh else ""
    write_jsonl(OUT/f"details{suffix}.jsonl",rows)
    required_exact = {m:sum(all(r["arms"][m]["predicted"].get(k)==r["expected"][k] for k in REQUIRED)
                            for r in rows) for m in METHODS}
    write_json(OUT/f"summary{suffix}.json",dict(n=len(rows),unique_prompts=len(set(hashes.values())),
               counts=counts, required_fields_exact=required_exact))
    print(counts)


if __name__=="__main__":
    main()
