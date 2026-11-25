"""Plot paired effects and the exploratory coverage comparison from saved orders."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/论文初稿"
REQUIRED = {"name.movie", "name.theater", "time.showing"}


def main():
    directory = ROOT / "runs/ticket-learning"
    summary = json.loads((directory / "summary.json").read_bytes())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False})
    cn_font = Path("C:/Windows/Fonts/msyh.ttc")
    if cn_font.is_file():
        plt.rcParams["font.family"] = FontProperties(fname=cn_font).get_name()
    fig, (left, right) = plt.subplots(1,2,figsize=(9.2,3.55), layout="constrained")
    for j, (split, label, color) in enumerate([("test","主测","#166777"),("replication","复验","#a04828")]):
        means = []
        low, high = [], []
        for outcome in ("correct","wrong","no_write"):
            pair = summary[split]["primary"]["paired"][outcome]
            point = pair["diff_pp"]
            a, b = pair["ci95_pp"]
            means.append(point)
            low.append(point-a)
            high.append(b-point)
        left.errorbar(means, np.arange(3)+(j-.5)*.16, xerr=[low,high], fmt=("o","s")[j],
                      color=color, capsize=3, label=label)
        rows = [json.loads(x) for x in (directory / f"{split}-details.jsonl").read_text(encoding="utf8").splitlines()]
        def top_correct(method, k):
            valid = [(r["conditions"][method]["confidence"], r["case"], r["conditions"][method]["args"]==r["gold"])
                     for r in rows if REQUIRED <= r["conditions"][method]["args"].keys()]
            return sum(correct for _,_,correct in sorted(valid,key=lambda v:(-v[0],v[1]))[:k])
        coverages = [10,20,30,50]
        difference = [100*(top_correct("field_support",len(rows)*c//100)-top_correct("recent",len(rows)*c//100))/len(rows)
                      for c in coverages]
        right.plot(coverages,difference,marker=("o","s")[j],color=color,label=label)
    left.axvline(0,color="0.6",lw=.8)
    left.set(yticks=[0,1,2],yticklabels=["正确","错写","未写入"],ylim=(2.5,-.5),xlabel="相对最近消息的变化（百分点）")
    left.set_title("A    相同提交阈值",loc="left")
    left.legend(loc="lower right",frameon=False)
    right.axhline(0,color="0.6",lw=.8)
    right.set(xticks=coverages,xlabel="各方法提交的对话比例（%）",ylabel="正确订单率差（百分点）")
    right.set_title("B    相同提交数量",loc="left")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "订单率与提交覆盖率.png",dpi=260,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
