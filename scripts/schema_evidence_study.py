"""Paired public-schema evidence selection, known intent/domain categorical outcomes."""
import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from binding_pilot import TOKENIZER
from context_bundles.core import Budget, Case, Counter as TokenCounter, Evidence
from context_bundles.io import sha256, write_json, write_jsonl
from context_bundles.selection import pack, select

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/schema-evidence-study"
MW_VERSION = "fe0c8e65cfcd8462bd33c86e35f21addc84ca82b"
METHODS = ("recent", "bm25", "value_recent", "pivot", "coverage", "full")
WORDS = "zero one two three four five six seven eight nine ten".split()
IGNORE = set("the with are for number whether service type what which where from whose what requested user name this has must and how does can would domain one their desired need want specify specific".split())
PIVOT = re.compile(r"\b(confirm|correct|instead|change|actually|please book|reservation for|you would like|you'd like|just to check|just to verify)\b", re.I)
CORRECTION = re.compile(r"\b(instead|actually|change|rather)\b", re.I)


def stem(word):
    if len(word) > 4 and word.endswith("s"):
        return word[:-1]
    return word


def terms(text):
    return {stem(w) for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in IGNORE}


@dataclass
class Field:
    terms: set
    pattern: re.Pattern
    aliases: dict
    required: bool


def field_defs(slots, keys, required):
    fields = {}
    for s in slots:
        if not s["is_categorical"] or s["name"] not in keys:
            continue
        mapping = {}
        for v in s["possible_values"]:
            value = v.casefold().strip()
            if value:
                mapping[value] = value
                if value.isdigit() and int(value) < len(WORDS):
                    mapping[WORDS[int(value)]] = value
        if not mapping:
            continue
        pattern = "|".join(map(re.escape, sorted(mapping, key=lambda v: (-len(v), v))))
        words = terms(s["name"].split("-")[-1] + " " + s["description"])
        fields[s["name"]] = Field(words, re.compile(r"(?<!\w)(?:" + pattern + r")(?!\w)", re.I),
                                  mapping, s["name"] in required)
    return fields


def cases(dataset):
    base = ROOT / "data/external"
    if dataset == "sgd-dev":
        paths = sorted((base / "sgd/dev").glob("dialogues_*.json"))
        schemas = json.loads((base / "sgd/dev/schema.json").read_bytes())
        catalogs = {}
        for schema in schemas:
            for intent in schema["intents"]:
                required = set(intent["required_slots"])
                keys = required | set(intent["optional_slots"])
                catalogs[schema["service_name"], intent["name"]] = field_defs(schema["slots"], keys, required)
    else:
        paths = []
        for k in ("001", "002"):
            source = f"data/MultiWOZ_2.2/dev/dialogues_{k}.json"
            path = OUT / "source" / f"multiwoz-dev-{k}.json"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(subprocess.check_output(["git", "-C", str(base / "multiwoz"), "show", MW_VERSION + ":" + source]))
            data = path.read_bytes()
            git_hash = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            expected = subprocess.check_output(["git", "-C", str(base/"multiwoz"), "rev-parse", MW_VERSION+":"+source], text=True).strip()
            assert git_hash == expected
            paths.append(path)
        schemas = json.loads((base / "multiwoz22/schema.json").read_bytes())
        catalogs = {s["service_name"]: field_defs(s["slots"], {v["name"] for v in s["slots"]}, set()) for s in schemas}
    for path in paths:
        for d in json.loads(path.read_bytes()):
            turns = d["turns"]
            labels = []
            if dataset == "sgd-dev":
                calls = [(i, f) for i,t in enumerate(turns) for f in t["frames"] if f.get("service_call")]
                if not calls:
                    continue
                i, f = calls[-1]
                j = i-1
                tool = f["service_call"]
                fields = catalogs[f["service"], tool["method"]]
                gold = {k: [str(v).casefold()] for k,v in tool["parameters"].items() if k in fields}
                labels.append((f["service"], fields, gold))
            else:
                j = max(i for i,t in enumerate(turns) if t["speaker"]=="USER")
                for f in turns[j]["frames"]:
                    fields = catalogs[f["service"]]
                    gold = {k: [str(v).casefold() for v in vs] for k,vs in f["state"]["slot_values"].items() if k in fields and vs}
                    labels.append((f["service"], fields, gold))
            assert turns[j]["speaker"]=="USER"
            key = d["dialogue_id"]
            hist = tuple(Evidence(str(i), t["speaker"].lower()+": "+t["utterance"], i, key) for i,t in enumerate(turns[:j]))
            case = Case(key, turns[j]["utterance"], {}, hist, key, dataset)
            yield case, labels


def mentions(piece, fields):
    role, _, text = piece.text.partition(": ")
    words = terms(text)
    raw = defaultdict(list)
    for slot,f in fields.items():
        for m in f.pattern.finditer(text):
            value = f.aliases[m[0].casefold()]
            raw[slot].append((value, m.end()))
    counts = Counter(value for matches in raw.values() for value in set(v for v,_ in matches))
    result = {}
    for slot, hits in raw.items():
        f = fields[slot]
        anchor = bool(f.terms & words)
        for value, end in hits:
            # Short shared enum values (e.g., "1" under several fields) require context.
            generic = value in {"yes","no","true","false"} or value.isdigit()
            supported = anchor or (not generic and counts[value] == 1) or f.required
            if not supported:
                continue
            score = (1.0 + 2.0*anchor + (role=="user") + bool(PIVOT.search(text))
                     + 1.0*(role=="user" and bool(CORRECTION.search(text)))
                     - 1.0*("?" in text and role!="user"))
            key = score, end
            if slot not in result or key > result[slot][0]:
                result[slot] = (key, value)
    return result, bool(raw)


def evaluate(case, fields, budget):
    cur = Evidence("current", "user: "+case.query, len(case.evidence), case.group_id)
    annotated = {e.id: mentions(e, fields) for e in (*case.evidence, cur)}
    # Each selector can inspect history. The common reader processes only selected raw
    # turns and the current query. Its per-turn scores never use neighboring turns.
    def scores(e):
        return {k: (local[0] + e.order/(len(case.evidence)+1), value)
                for k,(local,value) in annotated[e.id][0].items()}
    def predict(chosen):
        best = {}
        for e in [*chosen, cur]:
            for k,pair in scores(e).items():
                if k not in best or pair[0]>=best[k][0]:
                    best[k] = pair
        return {k:v for k,(_,v) in best.items()}
    # Rank source turns by the number/strength of best-supported field mentions
    # they contain. A value already better supported in the current turn costs zero.
    candidates = {k:(v[0],cur.id) for k,v in scores(cur).items()}
    for e in case.evidence:
        for k,(score,_) in scores(e).items():
            if k not in candidates or score > candidates[k][0]:
                candidates[k] = (score,e.id)
    support = Counter()
    for slot,(score,source) in candidates.items():
        support[source] += (2 if fields[slot].required else 1) * (1 + max(0,score))
    choices = {method:select(case,method,budget) for method in ("recent","bm25","full")}
    for method, key in (
        ("coverage", lambda e:(support[e.id],e.order)),
        ("value_recent", lambda e:(annotated[e.id][1],e.order)),
        ("pivot", lambda e:(bool(PIVOT.search(e.text)),e.order)),
    ):
        choices[method] = pack(sorted(case.evidence,key=key,reverse=True), budget)
    return {m:dict(pred=predict(ch), ids=[e.id for e in ch], tokens=budget.cost(ch)) for m,ch in choices.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sgd-dev","multiwoz-dev"), required=True)
    args = parser.parse_args()
    budget = Budget(128, TokenCounter("hf:"+str(ROOT/TOKENIZER)))
    rows = []
    screened = 0
    for case, labels in cases(args.dataset):
        screened += 1
        for domain, fields, gold in labels:
            if not gold:
                continue
            results = evaluate(case, fields, budget)
            for m,r in results.items():
                assert m=="full" or r["tokens"]<=128
                pred = r["pred"]
                r["exact"] = pred.keys()==gold.keys() and all(v in gold[k] for k,v in pred.items())
            rows.append(dict(dialogue=case.id, domain=domain, gold=gold, results=results))
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["dialogue"]].append(r["results"])
    joint = {m:np.array([all(v[m]["exact"] for v in group) for group in grouped.values()],int) for m in METHODS}
    summary = dict(screened=screened, dialogues=len(grouped), domain_frames=len(rows),
                   exact={m:int(v.sum()) for m,v in joint.items()}, comparisons={})
    for m in ("recent","bm25","pivot","value_recent"):
        delta = joint["coverage"]-joint[m]
        n=len(delta)
        wins,losses = (int((delta==sign).sum()) for sign in (1,-1))
        rng=np.random.default_rng(7331)
        boot=rng.multinomial(n,[wins/n,losses/n,1-(wins+losses)/n],size=20000)
        ci=np.quantile((boot[:,0]-boot[:,1])*100/n,[.025,.975]).tolist()
        summary["comparisons"][m] = dict(wins=wins, losses=losses, delta_pp=100*(wins-losses)/n, ci95_pp=ci,
                mcnemar_p=binomtest(wins,wins+losses).pvalue if wins+losses else 1.)
    out=OUT / args.dataset
    out.mkdir(parents=True,exist_ok=True)
    summary["script_sha256"]=sha256(Path(__file__))
    write_json(out/"summary.json",summary)
    write_jsonl(out/"details.jsonl",rows)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
