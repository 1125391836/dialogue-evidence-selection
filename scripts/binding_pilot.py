"""Offline Mem2ActBench identifier retention and controlled cart-write pilot."""
import json
import re
from collections import Counter as Counts
from dataclasses import replace
from pathlib import Path

from context_bundles.core import Budget, Case, Counter, Evidence, canonical
from context_bundles.io import read_jsonl, sha256, write_json, write_jsonl
from context_bundles.selection import select


DATA = Path("data/processed/mem2act-v1")
TOKENIZER = Path("data/tokenizers/deepseek-v4/tokenizer.json")
OUT = Path("runs/binding-pilot")
LIMITS = (512, 1024, 2048)
PRODUCT = re.compile(r"\bB[A-Z0-9]{9}\b")


def id_fields(arguments):
    fields = {}
    for key, value in arguments.items():
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower()
        if "id" not in re.split(r"[^a-z0-9]+", name):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            continue
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if len(str(value).strip()) >= 4:
            fields[key] = str(value).strip()
    return fields


def message_case(case):
    pieces = []
    for block in case.evidence:
        for index, message in enumerate(json.loads(block.text)):
            pieces.append(Evidence(f"{block.id}:m{index}", canonical(message),
                                   len(pieces), block.session_id))
    return replace(case, evidence=tuple(pieces))


def hits(evidence, fields):
    # Inspect visible content only; provenance IDs and hidden labels cannot score a hit.
    found = []
    for key, value in fields.items():
        pattern = re.compile(r"(?<![a-z0-9])" + re.escape(value) + r"(?![a-z0-9])", re.I | re.ASCII)
        if any(pattern.search(piece.text) for piece in evidence):
            found.append(key)
    return found


def retention(cases, labels, counter):
    rows = []
    for case in cases:
        fields = id_fields(labels[case.id]["gold"]["arguments"])
        if not fields:
            continue
        full = hits(case.evidence, fields)
        messages = message_case(case)
        for limit in LIMITS:
            budget = Budget(limit, counter)
            for method, inputs, ranker in [
                ("recent_block", case, "recent"), ("bm25_block", case, "bm25"),
                ("bm25_message", messages, "bm25"),
            ]:
                visible = select(inputs, ranker, budget)
                rows.append(dict(case=case.id, group=case.group_id, split=case.split,
                                 fields=list(fields), full=full, kept=hits(visible, fields),
                                 method=method, budget=limit, tokens=budget.cost(visible),
                                 selected=[piece.id for piece in visible]))
    return rows


def cart_replay(case):
    # One adapted source, not independent benchmark tasks. Archive supplied to
    # the reader contains only the corresponding variant's public messages.
    messages = [json.loads(p.text) for p in message_case(case).evidence]
    lists = [m["content"] for m in messages if m.get("role") == "user"
             and len(PRODUCT.findall(str(m.get("content", "")))) == 4]
    if len(lists) != 1 or "first" not in case.query.lower():
        raise ValueError("Source list or ordinal changed")
    ids = PRODUCT.findall(lists[0])
    if len(set(ids)) != 4:
        raise ValueError("Expected four distinct products")
    catalog = sorted(ids)
    rows = []
    for variant, order in [("A", ids), ("B", [ids[1], ids[0], *ids[2:]]), ("unspecified", [])]:
        # Unordered candidate inventory exists in every world. Only the user
        # list has ordinal semantics. Remove all alternative copies by design.
        archive = []
        if order:
            iterator = iter(order)
            archive = [PRODUCT.sub(lambda match: next(iterator), lists[0])]
        for policy in ("full", "ids_only_ask", "ids_only_guess", "ids_then_reread", "ordered_summary"):
            visible = archive if policy == "full" else []
            reads = int(policy == "ids_then_reread")
            if reads:
                visible = archive
            if policy == "ordered_summary":
                visible = [", ".join(PRODUCT.findall(m)) for m in archive]
            ordered = [pid for message in visible for pid in PRODUCT.findall(message)]
            chosen = ordered[0] if ordered else None
            if chosen is None and policy == "ids_only_guess":
                chosen = catalog[0]
            before = dict.fromkeys(catalog, 0)
            after = before.copy()
            if chosen is not None:
                after[chosen] += 1
            changed = [pid for pid in catalog if before[pid] != after[pid]]
            target = order[0] if order else None
            if not changed:
                outcome = "redundant_question" if target else "necessary_question"
            elif not target:
                outcome = "unsupported_write"
            else:
                outcome = "correct_write" if changed == [target] else "wrong_write"
            rows.append(dict(source=case.id, variant=variant, policy=policy, target=target,
                             current_query=case.query, before=before, after=after,
                             reads=reads, outcome=outcome))
    return rows


def main():
    cases = [Case.from_dict(row) for row in read_jsonl(DATA / "cases.jsonl")
             if row["split"] in {"train", "dev"}]
    labels = {row["id"]: row for row in read_jsonl(DATA / "labels.jsonl")
              if row["split"] in {"train", "dev"}}
    counter = Counter("hf:" + str(TOKENIZER))
    rows = retention(cases, labels, counter)
    replay = cart_replay(next(case for case in cases if case.id == "qa_443"))
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "retention.jsonl", rows)
    write_jsonl(OUT / "cart.jsonl", replay)
    snapshot = json.loads(Path("data/raw/mem2act-full/snapshot.json").read_text(encoding="utf8"))
    write_json(OUT / "manifest.json", dict(upstream=snapshot["repository"], commit=snapshot["commit"],
               splits=dict(Counts(case.split for case in cases)), limits=LIMITS,
               hashes={str(p): sha256(p) for p in (DATA / "cases.jsonl", DATA / "labels.jsonl", TOKENIZER, Path(__file__))}))
    eligible = {r["case"] for r in rows}
    visible_full = {r["case"] for r in rows if r["full"] == r["fields"]}
    report = ["# Mem2ActBench 指代证据先导实验", "",
              f"公开数据固定版本 `{snapshot['commit']}`。共筛查 {len(cases)} 道训练和开发题，"
              f"{len(eligible)} 题有符合规则的标量 ID 字段，{len(visible_full)} 题在完整历史中可直接匹配全部入选 ID。"
              "测试集未评分。字段名按下划线和驼峰拆分，含独立 id 且值至少四字符。", "",
              "## 同预算历史选择", "",
              "表中每格是完整历史可匹配且压缩后仍保留全部目标 ID 的题数。仅匹配提供的历史正文，"
              "允许大小写变化。出现目标字符串并不证明用户选择了它。三种方法共享当前请求、"
              "候选历史和 DeepSeek 分词器，不读取答案进行排序。预算包含来源 ID 和 JSON 包装。", "",
              "| 历史预算 | 最近交互块 | BM25 交互块 | BM25 单条消息 | 分母 |",
              "|---:|---:|---:|---:|---:|"]
    for limit in LIMITS:
        values = [sum(r["case"] in visible_full and r["kept"] == r["fields"] for r in rows
                      if r["budget"] == limit and r["method"] == method)
                  for method in ("recent_block", "bm25_block", "bm25_message")]
        report.append("| " + " | ".join(map(str, [limit, *values, len(visible_full)])) + " |")
    report += ["", "## 可重置购物车", "",
               "由训练题 qa_443 的四个商品和当前请求构造三份历史。A 保留原始清单，B 交换前两项，第三份从未提供排序。"
               "每次重置相同的四个购物车数量，分别执行增加一件或询问，按前后状态判分。固定规则读取第一项。"
               "这是一次单来源机制检查。猜测基线选字典序首项；原文回读可直接访问该历史，尚未测试全库自动检索。", "",
               "| 策略 | A | B | 从未指定 |", "|---|---|---|---|"]
    for policy in dict.fromkeys(r["policy"] for r in replay):
        report.append("| " + " | ".join([policy] + [r["outcome"] for r in replay if r["policy"] == policy]) + " |")
    report += ["", "正确写入、错写、无依据写入、重复询问和必要询问分别编码。询问后未模拟用户回复。"
               "没有新增大模型调用、线上交易或真实用户数据。"
               "两部分分别测原始题的 ID 保留和构造题的操作后果，不能将两表连乘估计实际错写率。", "",
               "逐题结果见 [retention.jsonl](retention.jsonl)，写入轨迹见 [cart.jsonl](cart.jsonl)，"
               "版本见 [manifest.json](manifest.json)。可用 `python run.py binding` 复现。"]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report[:12]))


if __name__ == "__main__":
    main()
