"""Paired comparisons with a pinned MiniLM encoder and measured offline CPU cost."""
import argparse
from collections import Counter
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
from time import perf_counter

import numpy as np
from scipy.stats import binomtest
from tokenizers import Tokenizer
import ticket_learning as ticket
import ticket_execution
from context_bundles.core import Budget, Counter as TokenCounter
from context_bundles.io import read_jsonl, sha256, write_json, write_jsonl
from context_bundles.selection import pack, select
from ticket_execution import examples, execute, render

OUT = ticket.ROOT / "runs/review-benchmarks"
REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
WEIGHTS = {"model_quint8_avx2.onnx":"b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21",
           "tokenizer.json":"be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"}
METHODS = ("recent", "bm25", "dense", "dense_recent_rrf", "field_support", "no_marginal", "tie_oldest", "optional_0.3", "optional_0.7")
SCHEMA_QUERY = "Book movie tickets: movie title, theater, showtime, date, number of tickets and seats. "


class Encoder:
    def __init__(self):
        sys.path.insert(0, str(OUT / "runtime"))
        import onnxruntime as ort
        self.version = ort.__version__
        for f,expected in WEIGHTS.items():
            assert sha256(OUT / "encoder" / f) == expected, f
        self.tokenizer = Tokenizer.from_file(str(OUT / "encoder/tokenizer.json"))
        self.tokenizer.enable_truncation(256)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(OUT / "encoder/model_quint8_avx2.onnx"),
                                            sess_options=opts, providers=["CPUExecutionProvider"])

    @lru_cache(maxsize=32000)
    def embed(self, text):
        ids = np.asarray(self.tokenizer.encode(text).ids, dtype=np.int64)[None,:]
        data = {"input_ids":ids, "attention_mask":np.ones_like(ids), "token_type_ids":np.zeros_like(ids)}
        tokens = self.session.run(["last_hidden_state"], data)[0]
        # Single sentence, all positions have mask=1. Same embedding in cached/uncached mode.
        vector = tokens[0].mean(axis=0)
        return vector / max(float(np.linalg.norm(vector)),1e-12)

    def choose(self, case, budget, hybrid):
        q = self.embed(SCHEMA_QUERY + case.query)
        dense = sorted(case.evidence, key=lambda e:(-float(q @ self.embed(e.text.partition(": ")[2])), -e.order))
        if hybrid:
            recent = sorted(case.evidence, key=lambda e:-e.order)
            rank = {e.id:1/(60+i) for i,e in enumerate(recent, start=1)}
            for i,e in enumerate(dense, start=1):
                rank[e.id] += 1/(60+i)
            dense.sort(key=lambda e:(-rank[e.id], -e.order))
        return pack(dense, budget)


def ablated(case, support, budget, mode):
    fields = ticket.best_fields(support, ["query", *(e.id for e in case.evidence)])
    goals = {(f,v):p for f,(p,v) in fields.items()}
    covered = dict(support.get("query", {}))
    available = list(case.evidence)
    chosen = []
    while available:
        def key(e):
            local = support.get(e.id,{})
            gain = sum(p*max(0.,local.get(fv,0.)-covered.get(fv,0.)) for fv,p in goals.items())
            return gain, -e.order if mode == "tie_oldest" else e.order
        best = max(available,key=key)
        available.remove(best)
        if budget.fits(chosen+[best]):
            chosen.append(best)
            if mode != "no_marginal":
                for fv,p in support.get(best.id,{}).items():
                    covered[fv] = max(p,covered.get(fv,0.))
    return sorted(chosen,key=lambda e:e.order)


def visible_scores(case, selected, matcher, vectorizer, model):
    # Re-index a sparse slice so the reader's original distance-to-query feature
    # is preserved. Source IDs, text and role stay exactly as in the public case.
    offset = len(case.evidence) - len(selected)
    subset = replace(case,evidence=tuple(replace(e,order=e.order-offset) for e in selected))
    return ticket.score(subset,matcher,vectorizer,model)


def infer(case, method, resources, precomputed=None):
    matcher, vec, model, calibration, encoder, budget = resources
    support = precomputed
    if method in ("recent", "bm25"):
        selected = select(case,method,budget)
    elif method in ("dense", "dense_recent_rrf"):
        selected = encoder.choose(case,budget,method=="dense_recent_rrf")
    else:
        if support is None:
            support = ticket.score(case,matcher,vec,model)
        if method in ("no_marginal","tie_oldest"):
            selected = ablated(case,support,budget,method)
        else:
            selected = ticket.evidence(case,support,budget,"field_support")
    if support is None:
        support = visible_scores(case,selected,matcher,vec,model)
    ids = [e.id for e in selected]
    best = ticket.best_fields(support,["query",*ids])
    optional_threshold = {"optional_0.3":.3,"optional_0.7":.7}.get(method,.5)
    args = {f:v for f,(p,v) in best.items() if f in ticket.REQUIRED or p>=optional_threshold}
    features = ticket.order_features(support,["query",*ids],best)
    features["fields_proposed"] = len(args)
    mapper,lr = calibration
    confidence = float(lr.predict_proba(mapper.transform([features]))[0,1])
    # Includes sqlite connection, schema creation, write and read-back.
    state = execute(args if confidence>=.5 else {})
    return dict(state=state, selected=ids, tokens=budget.cost(selected), confidence=confidence)


def contexts():
    cohorts = {}
    old = {}
    for name,split,parts in [("test","test",["aa"]),("replication","dev",["ab","ac"])]:
        prev = {r['case']:r for r in read_jsonl(ticket.OUT / (name+"-details.jsonl"))}
        old[name] = prev
        cases = {}
        for part in parts:
            for key, context, gold in examples(split,part):
                if key in prev and key not in cases:
                    assert gold == prev[key]['gold']
                    cases[key] = (context,gold)
        assert cases.keys() == prev.keys()
        cohorts[name] = cases
    return cohorts,old


def intervals(rows):
    out = {}
    for method in METHODS:
        if method == "field_support":
            continue
        for outcome in ("correct","wrong"):
            diff = [int(r['arms']['field_support']['outcome']==outcome)-int(r['arms'][method]['outcome']==outcome) for r in rows]
            wins,loss = diff.count(1),diff.count(-1)
            n=len(rows)
            samples=np.random.default_rng(42).multinomial(n,[loss/n,1-(wins+loss)/n,wins/n],size=20000)
            ci=100*np.quantile((samples[:,2]-samples[:,0])/n,[.025,.975])
            out[method+"_"+outcome]=dict(wins=wins, losses=loss, diff_pp=100*(wins-loss)/n,ci95_pp=ci.tolist(),
                                           raw_p=binomtest(wins,wins+loss,.5).pvalue if wins+loss else 1.)
    return out


def timing(raw_cases, resources, ids):
    matcher, _, _, _, encoder, _ = resources
    methods = ("recent","bm25","field_support","dense","dense_recent_rrf")
    times=[]
    rng=random.Random(20261123)
    for key in ids:
        context=raw_cases[key][0]
        order=list(methods)
        rng.shuffle(order)
        for method in order:
            # Request-cold: model and process loaded, no prior text/embedding cache.
            matcher.in_turn.cache_clear()
            encoder.embed.cache_clear()
            for phase in ("request_cold","immediate_repeat_warm"):
                t=perf_counter()
                state=infer(render(context),method,resources)
                ms=(perf_counter()-t)*1000
                times.append(dict(case=key,method=method,phase=phase,ms=ms,tokens=state['tokens']))
    group={}
    for method in methods:
        group[method]={}
        for phase in ("request_cold","immediate_repeat_warm"):
            ms=[r['ms'] for r in times if r['method']==method and r['phase']==phase]
            group[method][phase]=dict(n=len(ms),p50_ms=float(np.median(ms)),p95_ms=float(np.percentile(ms,95)))
    return times,group


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-only",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    protocol=dict(ticket_commit=ticket.COMMIT, encoder="sentence-transformers/all-MiniLM-L6-v2", encoder_commit=REV,
                  encoder_hashes=WEIGHTS, similarity="unit-normalized mean-pooled vectors, cosine; fixed public schema plus current turn",
                  rrf="equal-weight reciprocal rank fusion with recency, k=60", budget=128,threshold=.5,seed=42,
                  methods=METHODS, cohorts="reuse exact review-chronology test and replication dialogue IDs",
                  adapter_sha256=sha256(Path(ticket_execution.__file__)),
                  timing="64 primary dialogues with smallest SHA256(ID), cache-cold and immediate repeat, RNG 20261123")
    write_json(OUT / "protocol.json",protocol)
    training,used,texts,_=ticket.load('train',['aa','ab','ac','ad'])
    dev,*_=ticket.load('dev',['aa'],used,texts)
    matcher=ticket.Mentions(training)
    vec,model,_=ticket.fit(training,matcher)
    budget=Budget(128,TokenCounter('hf:'+str(ticket.ROOT/ticket.TOKENIZER)))
    calib_d=[r for r in dev if int(r[1][:2],16)<128]
    calibration=ticket.calibrate(ticket.evaluate(calib_d,matcher,vec,model,budget))
    start=perf_counter()
    encoder=Encoder()
    load_ms=(perf_counter()-start)*1000
    resources=matcher,vec,model,calibration,encoder,budget
    cohorts,old=contexts()
    summary={}
    if not args.timing_only:
        for split, raw_cases in cohorts.items():
            report=[]
            for key,(context,gold) in raw_cases.items():
                case=render(context)
                sup=ticket.score(case,matcher,vec,model)
                arms={}
                for method in METHODS:
                    ev=infer(case,method,resources,sup)
                    ev['outcome']='correct' if ev['state']==[gold] else 'wrong' if ev['state'] else 'no_write'
                    assert ev['tokens']<=128
                    if method in ("recent","bm25","field_support"):
                        assert ev['outcome']==old[split][key]['conditions'][method]['outcome']
                    arms[method]=ev
                report.append(dict(case=key,arms=arms))
            summary[split]=dict(n=len(report),counts={m:dict(Counter(r['arms'][m]['outcome'] for r in report)) for m in METHODS},
                                paired=intervals(report))
            write_jsonl(OUT/(split+'-details.jsonl'),report)
            write_json(OUT/'quality.json',summary)
            print(split,summary[split]['counts'],flush=True)
    ids=sorted(cohorts['test'],key=lambda k:hashlib.sha256(k.encode()).digest())[:64]
    timeline,latency=timing(cohorts['test'],resources,ids)
    write_jsonl(OUT/'timings.jsonl',timeline)
    write_json(OUT/'timing.json',dict(cpu=platform.processor(), cpu_name=subprocess.check_output(
        ['powershell','-NoProfile','-Command','(Get-CimInstance Win32_Processor).Name'],text=True).strip() if os.name=='nt' else platform.processor(),
        python=sys.version,onnxruntime=encoder.version,encoder_load_ms=load_ms,encoder_threads=2,methods=latency,
        sources={str(p.relative_to(ticket.ROOT)):sha256(p) for p in [Path(__file__).resolve(),Path(ticket.__file__).resolve(),
                   Path(ticket_execution.__file__).resolve(),ticket.OUT/'protocol.json',OUT/'protocol.json']}))
    print(latency,flush=True)


if __name__=='__main__':
    main()
