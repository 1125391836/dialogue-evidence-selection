"""Counterbalanced compressed-only judgments scored against independent full-text returns."""
import hashlib
import json
from collections import Counter

import review_human as human
from context_bundles.core import Budget, Case, Counter as Tokens, Evidence
from decision_pivot import OUT
from decision_selection import choose
from ticket_learning import ROOT, TOKENIZER

ARMS=("recent","field_support","decision")
LETTERS="DEF"
INSTRUCTIONS = """# 仅根据所给材料判断电影场次

你会看到英文对话的部分消息。A 是助手，U 是用户。每题最后一句为当前用户话语。请独立填写包内 CSV，按显示的局部轮次编号记录证据。即使语句中间缺少上下文，也只根据看到的内容作答。用户明确选择场次、或唯一的报价得到接受时，填 resolved 和 24 小时时间 HH:MM。场次未确定、指代不清、拒绝、更改后未落实或上午下午不清楚时，填 unresolved 并留空时间。evidence_turns 填判断所依据的行号，多项用英文逗号分隔。用 note 说明理由。请勿更改其他编号。只阅读你的压缩包，完成后将 CSV 交回协调人。
"""


def shuffled(items, salt):
    return sorted(items,key=lambda k:hashlib.sha256((salt+"|"+k).encode()).digest())


def build(dialogues, cases):
    budget=Budget(128,Tokens("hf:"+str(ROOT/TOKENIZER)))
    displays, assignment = {},[]
    order=shuffled(dialogues,"decision-blind-v1")
    for li,letter in enumerate(LETTERS):
        public={}
        ids=shuffled(order,"decision-private-"+letter)
        for i,qid in enumerate(order):
            arm=ARMS[(i+li)%len(ARMS)]
            turns=dialogues[qid]
            _,row=cases[qid]
            ev=tuple(Evidence(str(j),t["role"]+": "+t["text"],j,qid) for j,t in enumerate(turns[:-1]))
            case=Case(qid,turns[-1]["text"],{},ev,qid,"ticket")
            selected=([e.id for e in choose(case,budget,arm)] if arm=="decision" else row["conditions"][arm]["selected"])
            assert budget.fits([ev[int(j)] for j in selected])
            blind_id=f"V{ids.index(qid)+1:02d}"
            public[blind_id]=[turns[int(j)] for j in selected]+[turns[-1]]
            assignment.append(dict(reader=letter,blind_id=blind_id,audit_id=qid,arm=arm,selected=selected))
        displays[letter]=dict(sorted(public.items()))
    return displays,assignment


def main():
    folder=OUT/"compressed-blind"
    dialogues,cases,_=human.source()
    displays,mapping=build(dialogues,cases)
    human.preserve(folder/"coordinator-mapping.json",human.encoded(mapping))
    inbox=folder/"returns"
    inbox.mkdir(exist_ok=True)
    vals={}
    hashes={}
    for letter,items in displays.items():
        human.preserve(folder/f"reviewer_{letter}.zip",human.packet(items,letter,INSTRUCTIONS.encode()))
        file=inbox/f"reviewer_{letter}.csv"
        if file.exists():
            data=file.read_bytes()
            vals[letter]=human.form(data,items,letter)
            hashes[letter]=human.digest(data)
    progress={l:sum(x is not None for x in vals.get(l,{}).values()) for l in LETTERS}
    print("compressed completed",progress)
    if min(progress.values())!=len(dialogues):
        return
    full=human.collect(human.ROOT,dialogues,cases,human.source()[2],full_instructions())
    if full["stage"]!="complete":
        print("awaiting full-text judgments",full["stage"])
        return
    from pathlib import Path
    reference=Path(full["result"]).read_bytes()
    gold={r["audit_id"]:r["label"] for r in json.loads(reference)["cases"]}
    summary={a:Counter() for a in ARMS}
    rows=[]
    for item in mapping:
        q=item["audit_id"]
        answer=vals[item["reader"]][item["blind_id"]]
        true=gold[q]
        outcome=("full_unresolved" if true=="unresolved" else "unresolved" if answer=="unresolved" else
                 "same_time" if answer==true else "different_time")
        summary[item["arm"]][outcome]+=1
        rows.append(dict(**item,answer=answer,full=true,outcome=outcome))
    inputs=dict(compressed=hashes,full=human.digest(reference))
    path=folder/"results"/human.digest(human.encoded(inputs))/"scores.json"
    human.preserve(path,human.encoded(dict(inputs=inputs,n_each=len(gold),counts=summary,details=rows)))
    print(path)


def full_instructions():
    # Reuse the exact A/B packet instructions even after the original source is edited.
    import zipfile
    with zipfile.ZipFile(human.ROOT/"reviewer_A.zip") as z:
        return z.read("README.md")


if __name__=="__main__":
    main()
