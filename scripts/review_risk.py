"""Retrospective TicketTalk risk audit with development-only policy selection."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from ticket_execution import ROOT, REQUIRED, examples, render


SOURCE = ROOT / "runs/review-chronology"
OUT = ROOT / "runs/review-risk"
METHODS = ("recent", "field_support")
COSTS = (2, 5, 10)
CUTS = tuple(round(x / 20, 2) for x in range(21)) + (1.01,)
SEAT_CUTS = (.50, .65, .80, .95, 1.01)
ANCHOR = re.compile(r"\b(?:seats?|seating|rows?|aisle|balcony|recliners?|center|middle|front|back row)\b", re.I)
AFFIRM = re.compile(r"^\s*(?:yes|yeah|yep|correct|perfect|sounds good|that(?:'s| is) (?:right|correct))\b", re.I)
NEGATION = re.compile(r"\b(?:but|rather|instead|actually|except|wrong|change)\b", re.I)
OPTIONS = ("raw", "threshold", "evidence")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def words(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold().replace("’", "'")))


def supported_seat(value, turns):
    """Require a seat value in a seat-related user statement or an affirmed proposal."""
    needle = " " + words(value) + " "
    for i, (role, text) in enumerate(turns):
        if not ANCHOR.search(text) or needle not in " " + words(text) + " ":
            continue
        if role == "U":
            return True
        if i+1 < len(turns):
            # Keep original chronology and require consecutiveness so omission of
            # interceding messages cannot manufacture an acknowledgment.
            next_role, reply = turns[i+1]
            if next_role == "U" and AFFIRM.search(reply) and not NEGATION.search(reply):
                return True
    return False


def add_evidence(rows, split, parts):
    by_id = {r["case"]:r for r in rows}
    assert len(by_id) == len(rows)
    for part in parts:
        for key, text, gold in examples(split, part):
            if key not in by_id:
                continue
            r = by_id[key]
            case = render(text)
            if case is None:
                continue
            visible = "\n".join([*(e.text for e in case.evidence), case.query])
            fingerprint = hashlib.sha256(" ".join(visible.casefold().split()).encode()).hexdigest()
            if fingerprint != r["fingerprint"]:
                continue
            assert gold == r["gold"]
            del by_id[key]
            all_turns = [(e.text.partition(": ")[0], e.text.partition(": ")[2]) for e in case.evidence]
            all_turns.append(("U", case.query))
            for method in METHODS:
                arm = r["conditions"][method]
                assert arm["tokens"] <= 128
                indices = {int(i) for i in arm["selected"]} | {len(all_turns)-1}
                # Mask all unavailable turns and retain positions for adjacency.
                observed = [(who,body) if i in indices else ("", "")
                            for i,(who,body) in enumerate(all_turns)]
                seat = arm["args"].get("seating")
                arm["seat_supported"] = bool(seat and supported_seat(seat, observed))
    assert not by_id, "Missing public dialogue text for a cached row"


def proposed(arm, option, seat_cut):
    args = dict(arm["args"])
    reject = ((option == "evidence" and not arm["seat_supported"]) or
              (option == "threshold" and arm["order_features"].get("seating/top",0.) < seat_cut))
    if reject:
        args.pop("seating", None)
    return args


def arrays(rows, method, option, seat_cut):
    scores, correct = [], []
    for r in rows:
        arm = r["conditions"][method]
        args = proposed(arm, option, seat_cut)
        scores.append(arm["confidence"] if REQUIRED <= args.keys() else -1.)
        correct.append(args == r["gold"])
    return np.asarray(scores), np.asarray(correct,dtype=bool)


def stats(scores, correct, cutoff):
    submit = scores >= cutoff
    n = len(scores)
    c = int(np.sum(submit & correct))
    w = int(np.sum(submit & ~correct))
    nwrite = n-c-w
    return dict(n=n, submitted=c+w, correct=c, wrong=w, no_write=nwrite, coverage=(c+w)/n,
                conditional_error=w/(c+w) if c+w else None, unconditional_error=w/n)


def risks(scores, correct):
    eligible = np.flatnonzero(scores>=0)
    ids = sorted(eligible, key=lambda i: -scores[i])
    n = len(scores)
    points = [dict(threshold=None,coverage=0.,conditional_error=None,unconditional_error=0.,
                   correct=0,wrong=0,submitted=0)]
    c = w = 0
    for cutoff, group in groupby(ids,key=lambda i: scores[i]):
        indexes = list(group)
        right = int(correct[indexes].sum())
        c += right
        w += len(indexes)-right
        points.append(dict(threshold=float(cutoff),coverage=(w+c)/n,conditional_error=w/(w+c),
                           unconditional_error=w/n, correct=c,wrong=w,submitted=c+w))
    assert c+w == len(eligible)
    return points


def choose_seat_cut(rows):
    dev = {}
    for t in SEAT_CUTS:
        right = 0
        # All methods, all dev requests, same candidate thresholds and exact
        # optional-field inclusion/value objective independent of submission.
        for r in rows:
            for method in METHODS:
                a = proposed(r["conditions"][method],"threshold",t)
                right += a.get("seating") == r["gold"].get("seating")
        dev[t] = right
    return max(SEAT_CUTS, key=lambda t: (dev[t], -t)), dev


def tune(rows, seat_cut):
    policies = {}
    for option in OPTIONS:
        policies[option] = {}
        for method in METHODS:
            scores, correct = arrays(rows,method,option,seat_cut)
            candidates = [(t,stats(scores,correct,t)) for t in CUTS]
            policies[option][method] = {}
            for cost in COSTS:
                # Conservative tie break: fewer errors, then a higher threshold.
                def key(pair):
                    t, s = pair
                    return cost*s["wrong"] + s["no_write"], s["wrong"], -t
                t,s = min(candidates, key=key)
                policies[option][method][str(cost)] = dict(cutoff=t, dev=s)
    return policies


def per_row_loss(scores, correct, cutoff, cost):
    return np.where(scores<cutoff, 1, np.where(correct,0,cost))


def loss_difference(a, b):
    diff = a - b
    values, counts = np.unique(diff, return_counts=True)
    rng = np.random.default_rng(20261123)
    sims = rng.multinomial(len(diff),counts/len(diff),size=10000) @ values / len(diff)
    return dict(per_request=float(diff.mean()), ci95=np.quantile(sims,[.025,.975]).tolist(),
                cheaper=int(np.sum(diff<0)), dearer=int(np.sum(diff>0)))


def retuned_bootstrap(dev, cohorts, seat_cut, draws=3000):
    """Independently resample dev and evaluation dialogues; keep pairing across arms."""
    rng = np.random.default_rng(20261124)
    n_dev = len(dev)
    weights = rng.multinomial(n_dev, [1/n_dev]*n_dev, size=draws).astype(float)
    evaluation_weights = {split: rng.multinomial(len(rows), [1/len(rows)]*len(rows), size=draws).astype(float)
                          for split, rows in cohorts.items()}
    result = dict(replicates=draws, unit="dialogue; dev and evaluation drawn independently, paired within cohort",
                  note="field and order models held fixed", policies={})
    cuts = np.array(CUTS)
    for option in ("raw", "evidence"):
        result["policies"][option] = {}
        for cost in COSTS:
            frequencies = {}
            losses = defaultdict(dict)
            cutoffs = {}
            for arm in METHODS:
                scores, correct = arrays(dev, arm, option, seat_cut)
                submitted = scores[:, None] >= cuts[None, :]
                errors = weights @ (submitted & ~correct[:, None])
                rejected = n_dev - weights @ submitted
                risk = cost * errors + rejected
                # Exactly match tune(): minimize loss, then wrong count,
                # then choose the largest cutoff among equal choices.
                tied_loss = risk == risk.min(axis=1, keepdims=True)
                errors_of_best = np.where(tied_loss, errors, n_dev+1)
                best = tied_loss & (errors == errors_of_best.min(axis=1, keepdims=True))
                indices = len(cuts) - 1 - np.argmax(best[:, ::-1], axis=1)
                cutoffs[arm] = indices
                frequencies[arm] = {f"{cuts[k]:.2f}": int((indices == k).sum()) for k in np.unique(indices)}
                for split, rows in cohorts.items():
                    scores, correct = arrays(rows, arm, option, seat_cut)
                    matrix = np.stack([per_row_loss(scores, correct, t, cost) for t in cuts], axis=1)
                    sampled = evaluation_weights[split] @ matrix / len(rows)
                    losses[split][arm] = sampled[np.arange(draws), indices]
            joint = Counter((f"{cuts[i]:.2f}", f"{cuts[j]:.2f}") for i,j in zip(cutoffs["recent"], cutoffs["field_support"]))
            result["policies"][option][str(cost)] = dict(threshold_frequencies=frequencies,
                joint_threshold_frequencies={" / ".join(k):v for k,v in sorted(joint.items())},
                evaluation={s: np.quantile(v["field_support"]-v["recent"], [.025,.975]).tolist() for s,v in losses.items()})
    return result


def seat_counts(rows, method, option, seat_cut, cutoff):
    tally = Counter()
    for r in rows:
        arm = r["conditions"][method]
        arg = proposed(arm, option, seat_cut)
        if not REQUIRED <= arg.keys() or arm["confidence"] < cutoff:
            continue
        seat = arg.get("seating")
        gold = r["gold"].get("seating")
        if seat == gold:
            tally["match"] += 1
        elif seat is None:
            tally["omission"] += 1
        elif gold is None:
            tally["spurious"] += 1
        else:
            tally["wrong_value"] += 1
    return dict(tally)


def plot(curves, points):
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, ax = plt.subplots(2,2,figsize=(9.4,6.6), sharex=True, layout="constrained")
    for j,(split,title) in enumerate((("test","主测"),("replication","复验"))):
        ax[0,j].set_title(f"{title}（{points[split]['recent']['n']} 段）")
        for method, color, label in (("recent","#505776","最近消息"),("field_support","#007E74","字段证据")):
            data = curves[split][method][1:]
            for i,key in enumerate(("conditional_error","unconditional_error")):
                axes = ax[i,j]
                axes.step([100*p['coverage'] for p in data], [100*p[key] for p in data],
                          where="post", color=color,label=label,lw=1.7)
                p = points[split][method]
                if p["submitted"]:
                    axes.scatter([100*p['coverage']],[100*p[key]], s=48, marker="D", color=color,
                                 zorder=4, edgecolors="white",linewidths=.7)
                axes.grid(alpha=.2)
        ax[1,j].set_xlabel("提交覆盖率（%）")
    ax[0,0].set_ylabel("已提交订单中的错单率（%）")
    ax[1,0].set_ylabel("全部请求中的错单率（%）")
    ax[0,0].legend(frameon=False,loc="upper left")
    fig.suptitle("提交更多订单如何改变风险",fontsize=15)
    fig.savefig(OUT / "risk-coverage.png",dpi=250)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dev = read_rows(SOURCE / "dev-details.jsonl")
    add_evidence(dev,"dev",["aa"])
    seat_cut, selection = choose_seat_cut(dev)
    policies = tune(dev,seat_cut)
    protocol = dict(status="retrospective; corrected chronological input; cohorts previously inspected",
                    prediction_source=str(SOURCE.relative_to(ROOT)),
                    wrong_cost=COSTS, no_write_cost=1, correct_cost=0, seat_cut=float(seat_cut),
                    seat_candidates=selection, order_thresholds=CUTS, policies=policies,
                    dev_n=len(dev), dev_sha256=hashlib.sha256((SOURCE/"dev-details.jsonl").read_bytes()).hexdigest())
    save(OUT/"policy.json",protocol)
    results, curves, markers, cohorts = {}, {}, {}, {}
    seen = {r["fingerprint"] for r in dev}
    for split,source,parts in (("test","test",["aa"]),("replication","dev",["ab","ac"])):
        rows = read_rows(SOURCE/(split+"-details.jsonl"))
        fingerprints = {r["fingerprint"] for r in rows}
        assert len(fingerprints) == len(rows) and not fingerprints & seen
        seen |= fingerprints
        add_evidence(rows,source,parts)
        cohorts[split] = rows
        result = dict(n=len(rows), source_sha256=hashlib.sha256((SOURCE/(split+"-details.jsonl")).read_bytes()).hexdigest(),
                      controls={},cost_policies={})
        markers[split], curves[split] = {}, {}
        for option in OPTIONS:
            result["controls"][option] = {}
            result["cost_policies"][option] = {}
            losses = defaultdict(dict)
            for method in METHODS:
                score, correct = arrays(rows,method,option,seat_cut)
                control = stats(score,correct,.5)
                control["seat_disagreement"] = seat_counts(rows,method,option,seat_cut,.5)
                if option == "raw":
                    curves[split][method] = risks(score,correct)
                    assert all(control[k] == sum(r["conditions"][method]["outcome"]==k for r in rows)
                               for k in ("correct","wrong","no_write"))
                result["controls"][option][method] = control
                arms = {}
                for cost in COSTS:
                    t = policies[option][method][str(cost)]["cutoff"]
                    s = stats(score,correct,t)
                    loss = per_row_loss(score,correct,t,cost)
                    arms[str(cost)] = dict(cutoff=t, statistics=s, cost_per_request=float(loss.mean()))
                    losses[cost][method] = loss
                    if option == "raw" and cost == 5:
                        markers[split][method] = s
                result["cost_policies"][option][method] = arms
            result["cost_policies"][option]["field_minus_recent"] = {
                str(cost): loss_difference(losses[cost]["field_support"],losses[cost]["recent"]) for cost in COSTS}
        results[split] = result
    save(OUT / "results.json",results)
    save(OUT / "curve.json",curves)
    stability = retuned_bootstrap(dev, cohorts, seat_cut)
    save(OUT / "threshold-stability.json", stability)
    plot(curves, markers)
    write_recap(results, protocol, stability)
    print("seat cutoff",seat_cut)
    for split,r in results.items():
        for opt in OPTIONS:
            print(split,opt,r['controls'][opt])
            print({c:(round(policies[opt]['recent'][c]['cutoff'],2), round(policies[opt]['field_support'][c]['cutoff'],2),
                      r['cost_policies'][opt]['field_minus_recent'][c]) for c in map(str,COSTS)})


def write_recap(results, policy, stability):
    lines = ["# 错单代价、覆盖率与座位字段证据", "",
             f"使用修正对话时序后的逐题预测。{policy['dev_n']} 个开发对话参与阈值选择。两批评测的对话曾用于之前的探索。正确提交、未提交、错单的损失分别设为 0、1、λ，λ 取 2、5、10。这些权重为假设的业务偏好。阈值候选为 0.00 至 1.00（间隔 .05）和全部拒绝，在开发集选择损失最小的阈值。并列时先选错单少的阈值，再选较高阈值。", "",
             "## 风险与提交覆盖率", "", "曲线按整组置信度并列值移动阈值，菱形为开发集按 λ=5 选定的操作点。条件错单率以已提交订单为分母，总体错单率以全部请求为分母。零提交时条件风险无定义。", "", "![条件及总体错单率](risk-coverage.png)", "",
             "|批次|方法|正确|错单|提交 / 全部|条件错单率|总体错单率|", "|---|---|---:|---:|---:|---:|---:|"]
    for split, r in results.items():
        for method, s in r["controls"]["raw"].items():
            cond = f"{s['conditional_error']:.2%}" if s["submitted"] else "未定义"
            lines.append(f"|{split}|{method}|{s['correct']}|{s['wrong']}|{s['submitted']} / {r['n']}|{cond}|{s['unconditional_error']:.2%}|")
    lines.extend(["", "上表取 .50 整单阈值。下表在开发集对每个方法、每项代价分别选择阈值。每请求损失差小于零表示字段选择损失较低。区间为基于对话的配对重采样 10,000 次，按每项单独计算。实际错单代价、有限开发集的阈值波动及模板相关性尚有额外不确定性。", "",
                  "|座位策略|λ|最近 / 字段阈值|主测每请求损失差 [95% 区间]|复验每请求损失差 [95% 区间]|", "|---|---:|---:|---:|---:|"])
    for option in ("raw", "evidence"):
        for cost in map(str, COSTS):
            thresholds = [policy["policies"][option][m][cost]["cutoff"] for m in METHODS]
            cells = []
            for r in results.values():
                d = r["cost_policies"][option]["field_minus_recent"][cost]
                low, high = d["ci95"]
                cells.append(f"{d['per_request']:+.3f} [{low:+.3f}, {high:+.3f}]")
            lines.append(f"|{option}|{cost}|{thresholds[0]:.2f} / {thresholds[1]:.2f}|{'|'.join(cells)}|")
    lines.extend(["", "## 阈值选择的不确定性", "", "在 3,000 次重采样中分别抽取开发和评测对话。每次重新在开发样本选取阈值，然后到对应评测样本计算每请求损失差。两种方法共享每次抽取的对话。以下百分位区间含开发集阈值选择波动，仍假设对话独立，字段和整单模型权重保持固定。",
                  "", "|座位策略|λ|主测 95% 区间|复验 95% 区间|", "|---|---:|---:|---:|"])
    for mode, selection in stability["policies"].items():
        for cost, s in selection.items():
            cells = [f"[{low:+.3f}, {high:+.3f}]" for low,high in s["evaluation"].values()]
            lines.append(f"|{mode}|{cost}|{'|'.join(cells)}|")
    lines.extend(["", "各阈值及两臂阈值组合的入选频数见 [threshold-stability.json](threshold-stability.json)。", "", "## 可选座位字段", "", f"两臂使用相同的座位规则。只有选中历史或当前用户话语含座位用词及待填值，或助手的座位提议与可见的紧邻用户肯定构成一对，才保留原拟填座位。否定、改口和隐含表达仍可能造成误判。整单阈值固定 .50。只提高座位分数阈值的消融在开发集选为 {policy['seat_cut']:.2f}。", "",
                  "|批次|方法|提交数|原正确 / 错单|门控正确 / 错单|座位多填（原 → 门控）|座位漏填（原 → 门控）|", "|---|---|---:|---:|---:|---:|---:|"])
    for split, r in results.items():
        for m in METHODS:
            a, b = [r["controls"][op][m] for op in ("raw", "evidence")]
            assert a["submitted"] == b["submitted"]
            def change(key):
                return f"{a['seat_disagreement'].get(key,0)} → {b['seat_disagreement'].get(key,0)}"
            lines.append(f"|{split}|{m}|{a['submitted']}|{a['correct']} / {a['wrong']}|{b['correct']} / {b['wrong']}|{change('spurious')}|{change('omission')}|")
    lines.extend(["", "整单匹配使用官方参考。曲线和操作点用于离线比较，部署前仍需测量真实风险。", "",
                  "项目根目录可运行 python scripts/review_risk.py。开发选择见 [policy.json](policy.json)，逐阈值曲线见 [curve.json](curve.json)，全部计数及区间见 [results.json](results.json)。", ""])
    (OUT / "recap.md").write_text("\n".join(lines), encoding="utf8")


if __name__ == "__main__":
    main()
