"""Retrospective uncertainty and subgroup checks on frozen TicketTalk predictions."""
import hashlib
import json
import subprocess
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import binomtest

from context_bundles.io import read_jsonl, sha256, write_json, write_jsonl
from ticket_execution import COMMIT, REPO, ROOT, examples, render


SOURCE = ROOT / "runs/review-chronology"
OUT = ROOT / "runs/review-generalization"
ARMS = ("recent", "field_support")
OUTCOMES = ("correct", "wrong")
COHORTS = {"test": ("test", ("aa",)), "replication": ("dev", ("ab", "ac"))}
SEED = 20261123
REPLICATES = 20000


def fetch_scenarios(item):
    name, oid = item
    data = subprocess.check_output(["git", "-C", str(REPO), "cat-file", "blob", oid])
    digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if digest != oid:
        raise RuntimeError(f"Unexpected Git object hash for {name}")
    return [(row["conversation_id"], row["scenario"]) for row in json.loads(data)]


def scenarios():
    command = ["git", "-C", str(REPO), "ls-tree", "-r", COMMIT, "TM-3-2020/data/"]
    files = []
    for line in subprocess.check_output(command, text=True).splitlines():
        prefix, name = line.split("\t")
        if name.endswith(".json"):
            files.append((name, prefix.split()[2]))
    assert len(files) == 20
    by_id = {}
    for item in files:
        for case_id, scenario in fetch_scenarios(item):
            assert scenario and (case_id not in by_id or by_id[case_id] == scenario)
            by_id[case_id] = scenario
    return by_id, dict(files)


def analysis_rows(cohort, scenario_by_id):
    source = SOURCE / f"{cohort}-details.jsonl"
    cached = {r["case"]: r for r in read_jsonl(source)}
    rows = []
    split, parts = COHORTS[cohort]
    for part in parts:
        for case_id, context, gold in examples(split, part):
            if case_id not in cached:
                continue
            row = cached[case_id]
            case = render(context)
            if case is None:
                continue
            visible = "\n".join([*(e.text for e in case.evidence), case.query])
            digest = hashlib.sha256(" ".join(visible.casefold().split()).encode()).hexdigest()
            if digest != row["fingerprint"]:
                continue
            assert gold == row["gold"]
            arms = {m: row["conditions"][m] for m in ARMS}
            for arm in arms.values():
                assert arm["outcome"] in ("correct", "wrong", "no_write")
                if arm["outcome"] != "no_write":
                    assert (arm["args"] == gold) == (arm["outcome"] == "correct")
            rows.append(dict(case=case_id, fingerprint=digest, scenario=scenario_by_id[case_id],
                             public_messages=len(case.evidence) + 1, gold=gold, arms=arms))
            del cached[case_id]
    assert not cached, "Could not align frozen predictions on both dialogue ID and visible text"
    return rows


def sign_flip(differences):
    # Exact two-sided null distribution by convolution of integer cluster totals.
    # Equivalent to jointly swapping the two method labels inside each group.
    distribution = Counter({0: 1})
    for diff in differences:
        if diff:
            next_distribution = Counter()
            for total, count in distribution.items():
                next_distribution[total + diff] += count
                next_distribution[total - diff] += count
            distribution = next_distribution
    observed = abs(sum(differences))
    return sum(v for k, v in distribution.items() if abs(k) >= observed) / sum(distribution.values())


def holm(pvalues):
    adjusted = {}
    previous = 0.0
    for i, (key, pvalue) in enumerate(sorted(pvalues.items(), key=lambda pair: pair[1])):
        previous = min(1.0, max(previous, (len(pvalues) - i) * pvalue))
        adjusted[key] = previous
    return adjusted


def outcome_counts(rows):
    return {arm: {outcome: sum(r["arms"][arm]["outcome"] == outcome for r in rows)
                  for outcome in (*OUTCOMES, "no_write")} for arm in ARMS}


def cluster_analysis(rows, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scenario"]].append(row)
    table = {k: dict(n=len(v), arms=outcome_counts(v)) for k, v in sorted(groups.items())}
    sizes = np.array([g["n"] for g in table.values()])
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(table), size=(REPLICATES, len(table)))
    total = len(rows)
    stats = {}
    for outcome in OUTCOMES:
        diffs = [s["arms"]["field_support"][outcome] - s["arms"]["recent"][outcome] for s in table.values()]
        delta = sum(diffs)
        bootstrap = 100 * np.asarray(diffs)[indices].sum(axis=1) / sizes[indices].sum(axis=1)
        leaveout = {k: 100 * (delta - d) / (total - v["n"]) for (k, v), d in zip(table.items(), diffs)}
        wins = sum(r["arms"]["field_support"]["outcome"] == outcome and r["arms"]["recent"]["outcome"] != outcome for r in rows)
        losses = wins - delta
        stats[outcome] = dict(delta_pp=100 * delta / total, scenario_ci95_pp=np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                              scenario_sign_flip_p=sign_flip(diffs), n_informative_groups=sum(bool(d) for d in diffs),
                              dialogue_wins=wins, dialogue_losses=losses,
                              dialogue_mcnemar_p=binomtest(wins, wins + losses, .5).pvalue if wins + losses else 1.,
                              leave_one_scenario_out_min_pp=min(leaveout.values()),
                              leave_one_scenario_out_max_pp=max(leaveout.values()), leave_one_scenario_out=leaveout)
    return dict(n=total, n_scenarios=len(groups), min_group=int(sizes.min()), median_group=float(np.median(sizes)),
                max_group=int(sizes.max()), arms=outcome_counts(rows), outcomes=stats, scenarios=table)


def strata(rows):
    lengths = {}
    for name, lo, hi in (("1-12", 1, 12), ("13-20", 13, 20), ("21+", 21, float("inf"))):
        group = [r for r in rows if lo <= r["public_messages"] <= hi]
        lengths[name] = dict(n=len(group), arms=outcome_counts(group))
    assert sum(g["n"] for g in lengths.values()) == len(rows)
    fields = sorted({k for r in rows for args in (r["gold"], *(r["arms"][m]["args"] for m in ARMS)) for k in args})
    errors = {}
    for arm in ARMS:
        submitted = [r for r in rows if r["arms"][arm]["outcome"] != "no_write"]
        field_counts = {}
        for field in fields:
            counts = dict(omitted=0, spurious=0, wrong_value=0, correct_present=0, correct_absent=0)
            for row in submitted:
                pred, gold = row["arms"][arm]["args"], row["gold"]
                if field in gold and field not in pred:
                    key = "omitted"
                elif field in pred and field not in gold:
                    key = "spurious"
                elif gold.get(field) != pred.get(field):
                    key = "wrong_value"
                else:
                    key = "correct_present" if field in gold else "correct_absent"
                counts[key] += 1
            assert sum(counts.values()) == len(submitted)
            field_counts[field] = counts
        errors[arm] = dict(n_submitted=len(submitted), fields=field_counts)
    return dict(public_messages=lengths, field_errors_of_submitted=errors)


def report(result):
    lines = ["# 原生采集场景聚类和分组误差", "", "冻结原实验阈值 0.5 和预测。所有差值均为字段选择减最近消息，单位为百分点。按每条对话在原始 JSON 的完整 scenario 字符串聚类。", "",
             "|批次|对话 / 场景|指标|差值|场景重采样 95% 区间|分别删除一个场景后的差值范围|场景符号检验 Holm p|",
             "|---|---:|---|---:|---:|---:|---:|"]
    for cohort, data in result["cohorts"].items():
        for label in OUTCOMES:
            s = data["outcomes"][label]
            low, high = s["scenario_ci95_pp"]
            lines.append(f"|{cohort}|{data['n']} / {data['n_scenarios']}|{label}|{s['delta_pp']:+.2f}|[{low:+.2f}, {high:+.2f}]|[{s['leave_one_scenario_out_min_pp']:+.2f}, {s['leave_one_scenario_out_max_pp']:+.2f}]|{s['scenario_holm_p']:.4g}|")
    lines.extend(["", "20,000 次场景整组有放回抽样，每次以抽到的全部对话为分母。区间是单项百分位区间。另在每个场景内整体交换方法标签，精确计算双侧 p 值，以两个批次的两项结局组成四项 Holm 检验族。该检验依赖场景间独立性和零假设下方法标签在场景内整体可交换；它是事后敏感性分析。少数场景贡献差异时有效信息有限。逐一删除场景考察已训练模型的评测构成敏感性；模型训练时未按场景隔离。", "",
                  "## 对话长度", "", "输入长度计入当前用户消息及之前所有公开的用户、助手消息。分箱用于描述。表中为各箱内的正确 / 错单数，以该箱全部对话为分母。", "", "|批次|公开消息数|对话|最近消息正确 / 错单|字段选择正确 / 错单|", "|---|---|---:|---:|---:|"])
    for cohort, data in result["cohorts"].items():
        for length, g in data["strata"]["public_messages"].items():
            a, b = [g["arms"][m] for m in ARMS]
            lines.append(f"|{cohort}|{length}|{g['n']}|{a['correct']} / {a['wrong']}|{b['correct']} / {b['wrong']}|")
    lines.extend(["", "## 已提交订单的字段错误", "", "漏填表示参考有值但预测缺失，多填表示参考缺失但预测有值，错值表示双方都有值且不相等。同一订单可以有多处字段错误。只列出现错误的字段。分母为该方法在该批次的提交数。", "", "|批次|方法|提交数|字段|漏填|多填|错值|", "|---|---|---:|---|---:|---:|---:|"])
    for cohort, data in result["cohorts"].items():
        for arm, group in data["strata"]["field_errors_of_submitted"].items():
            for field, errors in group["fields"].items():
                if errors['omitted'] + errors['spurious'] + errors['wrong_value']:
                    lines.append(f"|{cohort}|{arm}|{group['n_submitted']}|{field}|{errors['omitted']}|{errors['spurious']}|{errors['wrong_value']}|")
    lines.extend(["", "## 正确率差受哪些场景影响", ""])
    for cohort, data in result["cohorts"].items():
        leaveout = data["outcomes"]["correct"]["leave_one_scenario_out"]
        scene = min(leaveout, key=leaveout.get)
        group = data["scenarios"][scene]
        gain = group["arms"]["field_support"]["correct"] - group["arms"]["recent"]["correct"]
        lines.append(f"{cohort} 中，去除 {scene} 后正确率差降幅最大。该场景有 {group['n']} 段，字段选择多匹配 {gain} 单；其余 {data['n']-group['n']} 段的正确率差为 {leaveout[scene]:+.2f} 个百分点。")
        lines.append("")
    lines.extend(["逐场景结局、删组结果、普通 McNemar 对照、文件哈希见 [results.json](results.json)。逐对话分组见 [case-groups.jsonl](case-groups.jsonl)。按场景隔离并重新训练模型的实验见 [五折重训](scenario-refit.md)，模式规则跨服务结果见 [SGD 新对话](sgd-recap.md)。", ""])
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by_id, files = scenarios()
    result = dict(taskmaster_commit=COMMIT, raw_git_blobs=files, seed=SEED, replicates=REPLICATES,
                  unit="exact upstream scenario; independent scene groups assumed", cohorts={})
    evidence = []
    pvals = {}
    seen = set()
    for i, cohort in enumerate(COHORTS):
        rows = analysis_rows(cohort, by_id)
        fingerprints = {r["fingerprint"] for r in rows}
        assert len(fingerprints) == len(rows) and not fingerprints & seen
        seen.update(fingerprints)
        analysis = cluster_analysis(rows, SEED + i)
        analysis["prediction_sha256"] = sha256(SOURCE / f"{cohort}-details.jsonl")
        analysis["strata"] = strata(rows)
        result["cohorts"][cohort] = analysis
        for outcome in OUTCOMES:
            pvals[cohort, outcome] = analysis["outcomes"][outcome]["scenario_sign_flip_p"]
        for r in rows:
            evidence.append(dict(cohort=cohort, case=r["case"], scenario=r["scenario"], public_messages=r["public_messages"],
                                 outcomes={m: r["arms"][m]["outcome"] for m in ARMS}))
    for (cohort, outcome), p in holm(pvals).items():
        result["cohorts"][cohort]["outcomes"][outcome]["scenario_holm_p"] = p
    write_json(OUT / "results.json", result)
    write_jsonl(OUT / "case-groups.jsonl", evidence)
    (OUT / "recap.md").write_text(report(result), encoding="utf8")
    print("Wrote", OUT, "with", len(evidence), "joined dialogues")


if __name__ == "__main__":
    main()
