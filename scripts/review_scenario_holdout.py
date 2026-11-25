"""Five refits with disjoint native scenario labels in fitting and evaluation."""
import hashlib

from context_bundles.core import Budget, Counter as TokenCounter
from context_bundles.io import read_jsonl, write_json, write_jsonl
from review_generalization import ARMS, COHORTS, OUT, SOURCE, cluster_analysis, scenarios
from ticket_learning import (ROOT, TOKENIZER, Mentions, apply_calibration, calibrate, evaluate,
                             fit, load, outcome)


def fold(scenario):
    return int.from_bytes(hashlib.sha256(scenario.encode()).digest(), "big") % 5


def main():
    OUT.mkdir(exist_ok=True)
    by_id, _ = scenarios()
    train, used, texts, _ = load("train", ["aa", "ab", "ac", "ad"])
    dev, used, texts, _ = load("dev", ["aa"], used, texts)
    calibration = [r for r in dev if int(r[1][:2],16) < 128]
    for row in read_jsonl(ROOT / "runs/ticket-execution/details.jsonl"):
        used.add(row["case"])
    samples = {}
    for name, (split, parts) in COHORTS.items():
        samples[name], used, texts, _ = load(split, parts, used, texts, name == "test")
        expected = {(r["case"],r["fingerprint"]) for r in read_jsonl(SOURCE/f"{name}-details.jsonl")}
        assert expected == {(r[0],r[1]) for r in samples[name]}
    budget = Budget(128, TokenCounter("hf:" + str(ROOT / TOKENIZER)))
    foldmap = {s:fold(s) for s in sorted({by_id[r[0]] for sample in [train, dev,*samples.values()] for r in sample})}
    summary = dict(fold_rule="SHA256(exact native scenario UTF-8) interpreted as big-endian integer modulo 5",
                   native_scenarios=foldmap, status="post hoc; cohort outcomes previously inspected; conceptually related scenarios may have different labels",
                   budget=128, threshold=.5, folds={}, cohorts={})
    details = []
    for i in range(5):
        fitting = [r for r in train if foldmap[by_id[r[0]]] != i]
        tuning = [r for r in calibration if foldmap[by_id[r[0]]] != i]
        fit_groups = {by_id[r[0]] for r in fitting+tuning}
        target_groups = {s for s, idx in foldmap.items() if idx == i}
        assert fit_groups.isdisjoint(target_groups)
        matcher = Mentions(fitting)
        vec, model, _ = fit(fitting, matcher)
        dev_rows = evaluate(tuning, matcher, vec, model, budget, ("recent","value_recent","field_support","bm25"))
        cal = calibrate(dev_rows)
        fold_summary = dict(train_n=len(fitting), calibration_n=len(tuning), target_scenarios=sorted(target_groups),
                            training_scenarios=sorted(fit_groups), scored={})
        for name, original in samples.items():
            held = [r for r in original if by_id[r[0]] in target_groups]
            predictions = evaluate(held, matcher, vec, model, budget, ARMS)
            apply_calibration(predictions, cal)
            fold_summary["scored"][name] = len(predictions)
            for row in predictions:
                row["scenario"] = by_id[row["case"]]
                row["fold"] = i
                row["cohort"] = name
                for arm in row["conditions"].values():
                    arm["outcome"] = outcome(arm["args"], row["gold"], arm["confidence"], .5)
                    del arm["order_features"]
                details.append(row)
        summary["folds"][str(i)] = fold_summary
        print("Fold", i, "scored", fold_summary["scored"], flush=True)
    for j, (name, expected) in enumerate(samples.items()):
        group = [r for r in details if r["cohort"] == name]
        assert len(group) == len(expected) == len({r["case"] for r in group})
        strata = [dict(scenario=r["scenario"],arms=r["conditions"]) for r in group]
        summary["cohorts"][name] = cluster_analysis(strata, 20261123+j)
    write_json(OUT/"scenario-refit-results.json", summary)
    write_jsonl(OUT/"scenario-refit-details.jsonl", details)
    lines = ["# 按采集场景隔离训练、校准与评测", "",
             "将原始 scenario 字符串的 SHA-256 哈希对 5 取余，分为五折。每折从训练词表、候选值逻辑回归和开发集整单校准中排除该折全部场景，然后对该折对话预测一次。阈值继续使用 0.5，输入预算为 128 token。汇总仍按原来的主测及复验对话分别进行。", "",
             "|队列|对话|最近消息正确 / 错单|字段选择正确 / 错单|正确率差 [场景 95% 区间]（百分点）|错单率差 [场景 95% 区间]（百分点）|", "|---|---:|---:|---:|---:|---:|"]
    for name, s in summary["cohorts"].items():
        a, b = (s["arms"][m] for m in ARMS)
        def interval(label):
            v=s["outcomes"][label]
            lo, hi=v["scenario_ci95_pp"]
            return f"{v['delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        lines.append(f"|{name}|{s['n']}|{a['correct']} / {a['wrong']}|{b['correct']} / {b['wrong']}|{interval('correct')}|{interval('wrong')}|")
    lines.extend(["", "该补充实验是在既有对话结果已知之后进行的。评测单位为采集场景的原生标签。不同标签可能共享电影、剧院、情境或话术。五个模型的训练集有重叠，场景区间条件于本次拟合参数。少用了部分训练样本，和原结果的差别同时受样本量和场景隔离影响。", "",
                  "逐折训练数、校准数与场景名单见 [scenario-refit-results.json](scenario-refit-results.json)，逐题字段与提交通道见 [scenario-refit-details.jsonl](scenario-refit-details.jsonl)。", ""])
    (OUT/"scenario-refit.md").write_text("\n".join(lines),encoding="utf8")


if __name__ == "__main__":
    main()
