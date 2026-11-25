"""Offline postmortem. Does not retrain, change original results, or call a model."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from context_bundles.campaign import Campaign
from context_bundles.guards import guarded_choice, promotion_gate
from context_bundles.io import read_jsonl, sha256, write_json, write_jsonl
from context_bundles.sensitivity import default_sensitivity
from context_bundles.utility import FEATURES, SetFeatures

p = argparse.ArgumentParser()
p.add_argument("--config", type=Path, default=Path("configs/deepseek-60.json"))
p.add_argument("--out", type=Path, default=Path("runs/deepseek-progressive-60"))
args = p.parse_args()
c = Campaign(args.config, args.out)
c.load_data()
before = c.ledger.stats()
model = json.loads((c.root / "utility_model.json").read_text())
examples = read_jsonl(c.root / "training_examples.jsonl")
if any(e["split"] != "train" for e in examples):
    raise ValueError("Support can only be estimated from training examples")
if sorted({e["group_id"] for e in examples}) != model["training_groups"]:
    raise ValueError("Support examples do not match frozen model")
# Augment a separate in-memory copy with train-only coordinate ranges; keep coefficients unchanged.
model["feature_min"] = [min(e["features"][i] for e in examples) for i in range(len(FEATURES))]
model["feature_max"] = [max(e["features"][i] for e in examples) for i in range(len(FEATURES))]
sensitivities, guarded = [], []
methods = defaultdict(list)
for row in c.ledger.rows():
    spec, result = row["spec"], row["result"] or {}
    if spec.get("case_id") not in c.cases or row["stage"] == "summary_generation":
        continue
    case = c.cases[spec["case_id"]]
    if case.split == "test":
        raise ValueError("Postmortem must not inspect final-test outcomes")
    sensitivity = default_sensitivity(result.get("output", ""), c.labels[case.id]["gold"], case.tool_schema)
    if result.get("status") != "ok" or result.get("finish_reason") != "stop":
        sensitivity["default_aware_exact_match"] = 0.0
        sensitivity["strict_exact_match"] = 0.0
    item = {"request_id": row["request_id"], "case_id": case.id, "split": case.split,
            "method": spec["method"], **sensitivity}
    sensitivities.append(item)
    if row["stage"] in ("baseline", "selection", "summary"):
        methods[spec["method"]].append(item)
    if row["stage"] == "selection":
        features = SetFeatures(case, c.budget)(frozenset(spec["selected_ids"]))
        selected, guard = guarded_choice(case, c.budget, model, spec["selected_ids"], features)
        guarded.append({"case_id": case.id, "method": spec["method"],
                        "original_selected_ids": spec["selected_ids"], "guard": guard,
                        "effective_selected_ids": [e.id for e in selected],
                        "history_cost": c.budget.cost(selected),
                        "executed": False, "performance": None})
table = [{"method": name, "calls": len(rows),
          "strict_exact_match": sum(r["strict_exact_match"] for r in rows) / len(rows),
          "default_aware_exact_match": sum(r["default_aware_exact_match"] for r in rows) / len(rows)}
         for name, rows in sorted(methods.items())]
summary = {"kind": "posthoc_offline_diagnostic_not_new_experiment", "calls_added": 0,
           "status": before, "promotion": promotion_gate(model), "methods": table,
           "empty_original_selections": sum(not r["original_selected_ids"] for r in guarded),
           "fallback_proposals": sum(r["guard"]["fallback"] is not None for r in guarded),
           "frozen_model_sha256": sha256(c.root / "utility_model.json"),
           "training_examples_sha256": sha256(c.root / "training_examples.jsonl"),
           "next_stage": "blocked_for_scaling_pending_metric_audit_and_better_training_support"}
out = c.root / "postmortem"
out.mkdir(exist_ok=True)
write_json(out / "review.json", summary)
write_jsonl(out / "default_sensitivity.jsonl", sensitivities)
write_jsonl(out / "guarded_proposals_not_executed.jsonl", guarded)
lines = ["# 首轮实验诊断与下一阶段入口", "",
         "本分析没有新增 API 调用，没有重训或覆盖原实验模型与严格评分。默认值分析是看到结果后增加的敏感性检查。", "",
         "## 当前结论", "",
         "工程流程跑通，方法有效性尚未建立。当前不应直接扩大正式测试或据此写论文效果结论。", "",
         f"原实验 {before['reserved_calls']} / {before['call_cap']} 次调用；6 次学习选择都选择了空历史。",
         "25 条训练观测来自 5 个独立组。Ridge 把训练范围之外的空历史外推为高价值，搜索因而停止。",
         f"分组交叉验证均方误差 {summary['promotion']['group_cv_mse']:.6f}，常数基线 {summary['promotion']['constant_cv_mse']:.6f}。当前预测器没有优于常数基线。", "",
         "## 修复和边界", "",
         "新实验默认检查训练组数、标签变化、交叉验证收益及训练特征范围，拒绝不可靠候选后回退同预算 BM25。",
         "30 个训练组只是可配置的工程下限，不是统计充分性或论文录用门槛。坐标范围检查也不能保证组合分布内泛化。",
         "旧实验按冻结计划恢复；不会用新策略冒充旧方法重新跑分。此处 6 个回退方案仅离线生成，尚未执行，无修复后准确率。", "",
         "## 默认值敏感性检查", "",
         "只补工具 schema 显式 default，或严格匹配 ‘The default value is’ 的字面值；必填字段、未知字段和含糊描述不补。双方对称归一化，保留类型区别。",
         "这不等于真实工具的执行语义验证，也不替代主指标。未使用 gold 推断默认值。", "",
         "| 方法 | 调用数 | 原严格 EM | 默认值归一化 EM |", "|---|---:|---:|---:|"]
for row in table:
    lines.append(f"| {row['method']} | {row['calls']} | {row['strict_exact_match']:.3f} | {row['default_aware_exact_match']:.3f} |")
lines += ["", "## 下一轮顺序", "",
          "1. 人工审核 train/dev 的任务、历史依据和默认参数约定，先定义评分协议；最终 test 99 题保持未使用。",
          "2. 训练采样覆盖空集、低预算集、单项/组合、替代证据和随机控制，避免只用非空高预算集合。",
          "3. 对候选做实体和参数依赖校验，重复干预并匹配长度、位置，区分评分门槛效应与真正的组合依赖。",
          "4. 按独立历史组收集更多训练数据，只有预测器通过预设检查才扩大选择比较；保留负结果。",
          "5. 静态诊断支持假设后再接 AppWorld 的可重放执行与统一恢复；尚未实现该阶段。", ""]
(out / "review.md").write_text("\n".join(lines), encoding="utf-8")
if c.ledger.stats() != before:
    raise RuntimeError("Offline review unexpectedly changed ledger")
print(json.dumps(summary, ensure_ascii=False, indent=2))
