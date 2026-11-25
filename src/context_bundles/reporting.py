"""Traceable aggregate tables, paired comparisons, cost accounting and local report."""

from collections import defaultdict
import csv
import html
import json
from pathlib import Path

from .deepseek import cost_interval
from .evaluation import score
from .io import read_jsonl, write_json, write_jsonl
from .statistics import paired_cluster_bootstrap


def make_report(campaign):
    root, config = campaign.root, campaign.config
    jobs = campaign.ledger.rows()
    labels_path = Path(config["labels_path"])
    labels = {r["id"]: r for r in read_jsonl(labels_path)} if labels_path.exists() else {}
    scored, cost_rows, interactions = [], [], []
    for row in jobs:
        spec, result = row["spec"], row["result"] or {}
        costs = cost_interval(result.get("usage", {}), config["pricing"])
        cost_rows.append({"request_id": row["request_id"], "stage": row["stage"],
                          "method": spec.get("method"), "usage": result.get("usage", {}),
                          "status": row["job_status"], **costs})
        if spec.get("case_id") not in labels or row["stage"] == "summary_generation":
            continue
        metrics = score(result.get("output", ""), labels[spec["case_id"]]["gold"])
        if result.get("status") != "ok" or result.get("finish_reason") != "stop":
            metrics = {"exact_match": 0.0, "argument_f1": 0.0, "tool_name_correct": 0.0,
                       "valid_json_call": False, "abstained": False}
        scored.append({k: v for k, v in {**spec, **metrics, **result, **costs,
                      "request_id": row["request_id"], "stage": row["stage"],
                      "job_status": row["job_status"]}.items() if k != "messages"})
    by_method = defaultdict(list)
    for row in scored:
        if row["stage"] in ("baseline", "selection", "summary"):
            by_method[row["method"]].append(row)
    tables = []
    for method, rows in sorted(by_method.items()):
        ids = {r["case_id"] for r in rows}
        related_costs = [c for c in cost_rows if c["request_id"] in {r["request_id"] for r in rows}]
        if method == "summary":
            generated_ids = {r["generation_request_id"] for r in rows}
            related_costs.extend(c for c in cost_rows if c["request_id"] in generated_ids)
        known_low = [c["usd_low"] for c in related_costs if c["usd_low"] is not None]
        known_high = [c["usd_high"] for c in related_costs if c["usd_high"] is not None]
        tables.append({"method": method, "cases": len(ids), "groups": len({r["group_id"] for r in rows}),
                       "calls": len(rows), "exact_match": sum(r["exact_match"] for r in rows) / len(rows),
                       "argument_f1": sum(r["argument_f1"] for r in rows) / len(rows),
                       "mean_history_tokens": sum(r["history_cost"] for r in rows) / len(rows),
                       "errors": sum(r.get("status") != "ok" for r in rows),
                       "fallback_calls": sum(bool(r.get("guard", {}).get("fallback")) for r in rows),
                       "truncated": sum(r.get("finish_reason") == "length" for r in rows),
                       "observed_cost_usd_low": sum(known_low), "observed_cost_usd_high": sum(known_high),
                       "missing_usage_calls": len(related_costs) - len(known_low),
                       "cost_scope": "execution plus summary generation; predictor training reported separately"})
    probe_groups = defaultdict(lambda: defaultdict(list))
    for row in scored:
        if row["stage"] == "diagnostics":
            probe_groups[(row["case_id"], row["pair_id"], row["split"])][row["condition"]].append(row)
    for (case_id, pair_id, split), conditions in sorted(probe_groups.items()):
        if not all(k in conditions for k in ("base", "a", "b", "ab", "random")):
            continue
        entry = {"case_id": case_id, "pair_id": pair_id, "split": split,
                 "repeats": len(conditions["base"]), "interpretation": "exploratory_not_verified_complementarity"}
        for metric in ("exact_match", "argument_f1"):
            means = {k: sum(r[metric] for r in v) / len(v) for k, v in conditions.items()}
            entry[metric] = {"means": means, "interaction": means["ab"] - means["a"] - means["b"] + means["base"],
                             "joint_gain": means["ab"] - means["base"],
                             "vs_random": means["ab"] - means["random"]}
        entry["random_length_delta"] = conditions["random"][0].get("random_length_delta")
        entry["infrastructure_or_truncation_failures"] = sum(
            r.get("status") != "ok" or r.get("finish_reason") != "stop" for values in conditions.values() for r in values)
        interactions.append(entry)
    pairs = [paired_cluster_bootstrap(scored, a, b, seed=config["seed"])
             for a, b in [("learned_bundle", "learned_greedy"), ("learned_bundle", "bm25"),
                          ("rule_pairs", "bm25"), ("summary", "bm25")]]
    totals = {"prompt_tokens": sum(r["usage"].get("prompt_tokens", 0) for r in cost_rows),
              "completion_tokens": sum(r["usage"].get("completion_tokens", 0) for r in cost_rows),
              "known_usd_low": sum(r["usd_low"] for r in cost_rows if r["usd_low"] is not None),
              "known_usd_high": sum(r["usd_high"] for r in cost_rows if r["usd_high"] is not None),
              "missing_usage_calls": sum(r["usd_low"] is None and r["status"] != "planned" for r in cost_rows),
              "pricing": config["pricing"]}
    report = {"campaign": config["name"], "status": campaign.ledger.stats(), "methods": tables,
              "paired_comparisons": pairs, "interactions": interactions, "costs": totals,
              "limitations": ["Small exploratory cohort; not publication evidence or a new-method performance claim.",
                              "Development cohort only; final test set is not evaluated.",
                              "One repeat per condition in the 60-call schedule cannot establish stable interaction.",
                              "Summary uses a smaller paired subset; compare on shared cases only.",
                              "Random controls are approximately, not exactly, length-matched.",
                              "Training cost is included in campaign totals, not hidden in per-method execution cost."]}
    report["limitations"].append("Guarded selector results may include BM25 fallback; inspect fallback_calls and per-call guard reasons.")
    if (root / "utility_model.json").exists():
        model = json.loads((root / "utility_model.json").read_text())
        report["predictor"] = {k: model[k] for k in ["training_rows", "training_groups", "group_cv", "gate", "constant_target"]}
    out = root / "report"
    out.mkdir(exist_ok=True)
    write_json(out / "report.json", report)
    write_jsonl(out / "scored.jsonl", scored)
    write_jsonl(out / "costs.jsonl", cost_rows)
    if tables:
        with (out / "methods.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
            writer.writeheader();writer.writerows(tables)
    lines = ["# DeepSeek 分阶段实验报告", "", "这是小样本探索运行，不构成方法有效性或论文结论。", "",
             f"调用：{report['status']['reserved_calls']} / {report['status']['call_cap']}；"
             f"已记录输入 token {totals['prompt_tokens']}，输出 token {totals['completion_tokens']}。", "",
             f"按核对价格估算的已知费用区间：${totals['known_usd_low']:.6f}—${totals['known_usd_high']:.6f}；"
             f"缺失用量 {totals['missing_usage_calls']} 次。不是账单。", "",
             "| 方法 | 问题数 | 完整调用正确率 | 参数 F1 | 平均历史 token |", "|---|---:|---:|---:|---:|"]
    for r in tables:
        lines.append(f"| {r['method']} | {r['cases']} | {r['exact_match']:.3f} | {r['argument_f1']:.3f} | {r['mean_history_tokens']:.0f} |")
    lines += ["", "完整历史参考不受相同预算约束。摘要只在相同问题交集上与其他方法比较。", "",
              "学习器由训练组的干预结果拟合；当前训练组极少，交叉验证、恒定标签和样本门槛均记录在 report.json。",
              "测试组保持未使用。不能根据开发集这一次运行判断论文新方法成立。", "",
              "四条件诊断与随机控制的原始结果见 report.json / scored.jsonl；已有合作归因公式不是本项目原创。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    body_rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in
        [r["method"], r["cases"], f"{r['exact_match']:.3f}", f"{r['argument_f1']:.3f}", f"{r['mean_history_tokens']:.0f}"]) + "</tr>" for r in tables)
    (out / "index.html").write_text(f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>DeepSeek 分阶段实验报告</title><style>body{{font:17px system-ui;max-width:1000px;margin:45px auto;padding:24px;color:#17212b}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd3da;padding:12px;text-align:left}}th{{background:#eaf0f6}}.note{{color:#655318}}a{{color:#155bad}}</style>
<h1>DeepSeek 分阶段实验报告</h1><p class="note">小样本探索结果，不能用作论文效果结论。完整历史不受相同预算约束；摘要仅覆盖部分共同问题。</p>
<p>调用 {report['status']['reserved_calls']} / {report['status']['call_cap']} · 输入 {totals['prompt_tokens']} tokens · 输出 {totals['completion_tokens']} tokens</p>
<p>已知费用估计 ${totals['known_usd_low']:.6f}—${totals['known_usd_high']:.6f}，缺失用量 {totals['missing_usage_calls']} 次。费用包含诊断、训练数据采集与摘要生成。</p>
<table><thead><tr><th>方法</th><th>问题数</th><th>完整调用正确率</th><th>参数 F1</th><th>平均历史 token</th></tr></thead><tbody>{body_rows}</tbody></table>
<p><a href="report.json">完整结构化报告</a> · <a href="methods.csv">方法表</a> · <a href="report.md">文字报告</a></p>
<p>训练与开发分组隔离，最终测试集未使用。候选组合、预测器与搜索均有日志；单次互补信号需要重复实验和人工审核。</p></html>""", encoding="utf-8")
    print(json.dumps({"stage": "report", "output": str(out), **report["status"]}), flush=True)
    return report
