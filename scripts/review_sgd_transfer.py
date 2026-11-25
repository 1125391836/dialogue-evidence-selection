"""Evaluate frozen schema-based categorical retrieval on additional SGD conversations."""
import hashlib
import json
import subprocess
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest

from context_bundles.core import Budget, Case, Counter as TokenCounter, Evidence
from context_bundles.io import sha256, write_json, write_jsonl
from schema_evidence_study import ROOT, TOKENIZER, evaluate, field_defs


OUT = ROOT / "runs/review-generalization"
REPO = ROOT / "data/external/sgd"
AUDIT = ROOT / "data/external/sgd-revision-audit"


def final_call(dialogue):
    turns = dialogue["turns"]
    for index in reversed(range(1, len(turns))):
        for frame in reversed(turns[index]["frames"]):
            if frame.get("service_call"):
                return index, frame
    return None


def digest(history):
    text = "\n".join(t["speaker"].lower() + ":" + " ".join(t["utterance"].casefold().split()) for t in history)
    return hashlib.sha256(text.encode()).hexdigest()


def previous_inputs():
    # SGD dialogue IDs are qualified by split. For text overlap compare across
    # all splits, including utterance sequences leading into any earlier call.
    ids, fingerprints, exposed_train_services = set(), set(), set()
    for split, folder in (("train", AUDIT/"train"), ("dev", REPO/"dev"), ("test", REPO/"test")):
        paths = sorted(folder.glob("dialogues_*.json"))
        if not paths:
            raise FileNotFoundError(folder)
        for path in paths:
            for d in json.loads(path.read_bytes()):
                ids.add((split, d["dialogue_id"]))
                history = d["turns"]
                for i, turn in enumerate(history):
                    if any(f.get("service_call") for f in turn["frames"]):
                        fingerprints.add(digest(history[:i]))
                    if split == "train":
                        exposed_train_services.update(f["service"] for f in turn["frames"])
    return ids, fingerprints, exposed_train_services


def summarize(rows):
    n = len(rows)
    a = sum(r["results"]["recent"]["exact"] for r in rows)
    b = sum(r["results"]["coverage"]["exact"] for r in rows)
    wins = sum(r["results"]["coverage"]["exact"] and not r["results"]["recent"]["exact"] for r in rows)
    losses = wins - (b - a)
    rng = np.random.default_rng(20261123)
    pairs = rng.multinomial(n, [wins/n, losses/n, (n-wins-losses)/n], size=20000)
    return dict(n=n, recent=a, coverage=b, wins=wins, losses=losses, difference_pp=100*(b-a)/n,
                ci95_pp=np.percentile(100*(pairs[:, 0]-pairs[:, 1])/n, [2.5, 97.5]).tolist(),
                p=binomtest(wins, wins + losses, .5).pvalue if wins + losses else 1.)


def main():
    protocol = json.loads((OUT/"sgd-protocol.json").read_bytes())
    adapter = ROOT / "scripts/schema_evidence_study.py"
    if sha256(adapter) != protocol["frozen_adapter_sha256"]:
        raise RuntimeError("Adapter has changed since protocol freeze")
    commit = protocol["source_commit"]
    def blob(path):
        return subprocess.check_output(["git", "-C", str(REPO), "show", f"{commit}:{path}"])
    schemas = json.loads(blob("train/schema.json"))
    excluded = {s["service_name"] for split in ("dev", "test") for s in json.loads(blob(split+"/schema.json"))}
    allowed = {s["service_name"] for s in schemas} - excluded
    catalogs = {}
    for s in schemas:
        for intent in s["intents"]:
            required = set(intent["required_slots"])
            keys = required | set(intent["optional_slots"])
            catalogs[s["service_name"], intent["name"]] = field_defs(s["slots"], keys, required)
    historical_ids, historical_text, exposed = previous_inputs()
    seen_ids, seen_text, rows, sources = set(), set(), [], {}
    counts = Counter()
    budget = Budget(protocol["tokens"], TokenCounter("hf:" + str(ROOT / TOKENIZER)))
    for path in protocol["files"]:
        data = blob(path)
        sources[path] = hashlib.sha256(data).hexdigest()
        for d in json.loads(data):
            counts["screened"] += 1
            key = d["dialogue_id"]
            call = final_call(d)
            if not call:
                counts["no_call"] += 1
                continue
            i, frame = call
            history = d["turns"][:i]
            if history[-1]["speaker"] != "USER":
                counts["no_preceding_user"] += 1
                continue
            fp = digest(history)
            if ("train", key) in historical_ids or fp in historical_text or key in seen_ids or fp in seen_text:
                counts["duplicate"] += 1
                continue
            seen_ids.add(key)
            seen_text.add(fp)
            service = frame["service"]
            if service not in allowed:
                counts["service_excluded"] += 1
                continue
            tool = frame["service_call"]
            fields = catalogs[service, tool["method"]]
            gold = {k: str(v).casefold() for k, v in tool["parameters"].items() if k in fields}
            if not gold:
                counts["no_categorical_target"] += 1
                continue
            evidence = tuple(Evidence(str(j), t["speaker"].lower()+": "+t["utterance"], j, key)
                             for j, t in enumerate(history[:-1]))
            case = Case(key, history[-1]["utterance"], {}, evidence, key, "sgd-transfer")
            outputs = evaluate(case, fields, budget)
            results = {arm: outputs[arm] for arm in protocol["arms"]}
            for value in results.values():
                assert value["tokens"] <= budget.limit
                value["exact"] = value["pred"] == gold
            rows.append(dict(dialogue=key, source=path, fingerprint=fp, service=service,
                             method=tool["method"], gold=gold, results=results))
    counts["eligible"] = len(rows)
    assert counts["screened"] == sum(v for k, v in counts.items() if k != "screened")
    groups = defaultdict(list)
    for row in rows:
        groups[row["service"]].append(row)
    summary = dict(protocol_sha256=sha256(OUT/"sgd-protocol.json"), counts=counts, file_sha256=sources,
                   schema_eligible_services=sorted(allowed), previous_train_frame_services=sorted(exposed),
                   overall=summarize(rows), by_service={s:summarize(v) for s,v in sorted(groups.items())})
    write_json(OUT/"sgd-results.json", summary)
    write_jsonl(OUT/"sgd-details.jsonl", rows)
    s = summary["overall"]
    lines = ["# 冻结规则在新增 SGD 对话上的分类字段匹配", "",
             f"预先固定 8 个 SGD 官方训练分片以及原有模式规则、比较方法和 128 token 预算。一共检查 {counts['screened']} 段对话，在不属于官方开发与测试服务目录、最后一次调用有分类参数的 {s['n']} 段上评分。此前训练分片中已经出现过全部候选服务的对话框架。运行时已知目标服务及意图。任务只要求匹配模式中有枚举值的调用参数，完整订单、交易成功、学习型购票模型的迁移均未在本项中测量。", "",
             f"最近消息正确 {s['recent']}/{s['n']}，字段选择正确 {s['coverage']}/{s['n']}，相差 {s['difference_pp']:+.2f} 个百分点，逐对话配对 95% 区间 [{s['ci95_pp'][0]:+.2f}, {s['ci95_pp'][1]:+.2f}]，双侧精确 McNemar p={s['p']:.4g}。其中 {s['wins']} 段只在字段选择下正确，{s['losses']} 段只在最近消息下正确。", "",
             "|最终服务|对话|最近消息正确|字段选择正确|字段独对 / 最近独对|", "|---|---:|---:|---:|---:|"]
    for service, v in summary["by_service"].items():
        lines.append(f"|{service}|{v['n']}|{v['recent']}|{v['coverage']}|{v['wins']} / {v['losses']}|")
    lines.extend(["", "按服务分组的数字反映样本组成。样本虽与旧对话文件分离，但共享数据生成过程；逐对话区间也没有计入服务或模板聚类。", "",
                  "固定协议与适配器哈希见 [sgd-protocol.json](sgd-protocol.json)，逐步筛选和按服务统计见 [sgd-results.json](sgd-results.json)，逐题预测见 [sgd-details.jsonl](sgd-details.jsonl)。", ""])
    (OUT/"sgd-recap.md").write_text("\n".join(lines), encoding="utf8")
    print(json.dumps(summary["overall"]))


if __name__ == "__main__":
    main()
