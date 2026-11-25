"""Offline reference replays and paired target-selection controls on two benchmarks."""
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data/external"
OUT = ROOT / "runs/external-benchmarks"
TAU = EXT / "tau-bench-main/tau_bench"
os.environ["APPWORLD_ROOT"] = str(EXT / "appworld")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(name, rows):
    (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def tau_tools():
    # Import upstream tool implementations without initializing paid user simulators.
    sys.path.insert(0, str(TAU.parent))
    for package, directory in [("tau_bench", TAU), ("tau_bench.envs", TAU / "envs"),
                               ("tau_bench.envs.retail", TAU / "envs/retail")]:
        module = ModuleType(package)
        module.__path__ = [str(directory)]
        sys.modules[package] = module
    from tau_bench.envs.retail.tools import ALL_TOOLS
    from tau_bench.envs.retail.tasks_dev import TASKS_DEV
    return {tool.get_info()["function"]["name"]: tool for tool in ALL_TOOLS}, TASKS_DEV


def decide(ids, target, policy):
    # Same inventory and current request across the three histories. Only the
    # earlier confirmation varies. Truncation retains the last two events.
    archive = [{"chosen": target}] if target is not None else []
    archive += [{"candidates": ids}, {"request": "Apply the change to my selected record."}]
    visible = archive if policy == "full" else archive[-2:]
    recovered = []
    if policy == "reread":
        recovered = [event for event in archive if "chosen" in event]
    if policy == "structured":
        visible = [{"chosen": e["chosen"]} for e in archive if "chosen" in e] + visible
    decision = next((e["chosen"] for e in visible + recovered if "chosen" in e), None)
    return decision, visible, recovered


def paired(dataset, case, ids, write):
    rows = []
    for variant, target in zip(("A", "B", "unknown"), [*ids, None], strict=True):
        for policy in ("full", "truncated", "structured", "reread"):
            decision, visible, recovered = decide(ids, target, policy)
            before, after = write(decision)
            changed = [key for key in before if before[key] != after[key]]
            if decision is None:
                assert not changed
                result = "necessary_question" if target is None else "redundant_question"
            else:
                result = "correct_write" if changed == [str(target)] else "wrong_write"
            rows.append(dict(dataset=dataset, source=case, variant=variant, policy=policy,
                             target=target, decision=decision, visible=visible, recovered=recovered,
                             archive_reads=int(policy=="reread"), before=before, after=after, outcome=result))
    return rows


def retail():
    tools, tasks = tau_tools()
    data = {p.stem: load(p) for p in (TAU / "envs/retail/data").glob("*.json")}
    refs, controls = [], []
    for index, task in enumerate(tasks):
        state = copy.deepcopy(data)
        actions = []
        for action in task.actions:
            try:
                output = tools[action.name].invoke(state, **action.kwargs)
                error = output if isinstance(output, str) and output.startswith("Error") else None
            except Exception as exc:
                error = repr(exc)
            actions.append(dict(name=action.name, arguments=action.kwargs, error=error))
        refs.append(dict(task=index, actions=actions, success=all(not a["error"] for a in actions)
                         if actions else None, changed_orders=[k for k in data["orders"]
                         if data["orders"][k] != state["orders"][k]]))
    # Adapt all distinct retail-dev users with two pending orders, regardless of
    # their original task type. This is a controlled scenario, not the task reward.
    for user in sorted({t.user_id for t in tasks}):
        ids = sorted(k for k,v in data["orders"].items() if v["user_id"]==user and v["status"]=="pending")[:2]
        if len(ids) != 2:
            continue
        def write(decision):
            state = dict(orders={k:copy.deepcopy(data["orders"][k]) for k in ids},
                         users={user:copy.deepcopy(data["users"][user])})
            before = {k:state["orders"][k]["status"] for k in ids}
            if decision is not None:
                result = tools["cancel_pending_order"].invoke(state, decision, "no longer needed")
                if result.startswith("Error"):
                    raise RuntimeError(result)
            return before, {k:state["orders"][k]["status"] for k in ids}
        controls.extend(paired("tau-retail",user,ids,write))
    return refs, controls


def appworld():
    from appworld import AppWorld, load_task_ids
    ids = load_task_ids("dev")
    families = {}
    for tid in ids:
        families.setdefault(tid.split("_")[0],tid)
    sample = list(families.values())[:6]
    refs, controls = [], []
    for tid in sample:
        with AppWorld(task_id=tid, experiment_name="external-reference", ground_truth_mode="full") as world:
            initial = world.evaluate().success
            world.execute(world.task.ground_truth.compiled_solution_code + "\nsolution(apis, requester)")
            result = world.evaluate().to_dict()
            refs.append(dict(task=tid, baseline_initial=initial, **result))
    # Use one actual dev world; reset it before each independent note edit.
    tid = sample[0]
    def notes(world):
        a = world.apis
        email = a.supervisor.show_profile()["email"]
        password = next(p["password"] for p in a.supervisor.show_account_passwords() if p["account_name"]=="simple_note")
        token = a.simple_note.login(username=email,password=password)["access_token"]
        candidates = [n["note_id"] for n in a.simple_note.search_notes(access_token=token,page_limit=2)]
        return a.simple_note, token, candidates
    with AppWorld(task_id=tid,experiment_name="external-notes") as world:
        _, _, candidates = notes(world)
    def write(decision):
        with AppWorld(task_id=tid,experiment_name="external-notes") as world:
            api, token, current = notes(world)
            assert current == candidates
            def state():
                return {str(i):api.show_note(note_id=i,access_token=token)["title"] for i in current}
            before = state()
            if decision is not None:
                api.update_note(note_id=decision,title="Controlled pilot revision",access_token=token)
            return before, state()
    controls.extend(paired("appworld-notes",tid,candidates,write))
    return refs, controls


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tau, tr = retail()
    save("tau-dev.jsonl",tau)
    aw, ar = appworld()
    save("appworld-dev.jsonl",aw)
    save("paired.jsonl",tr+ar)
    counts = Counter((r["dataset"],r["policy"],r["outcome"]) for r in tr+ar)
    lines = ["# 两个外部基准的离线实验", "",
             f"τ-bench 原版零售 dev 共 {len(tau)} 题。参考动作全部无工具报错 {sum(r['success'] is True for r in tau)} 题，"
             f"至少一次报错 {sum(r['success'] is False for r in tau)} 题，无参考动作 {sum(r['success'] is None for r in tau)} 题。"
             "这项检查直接执行作者给出的工具参数，不评价对话、策略合规或官方奖励。", "",
             f"AppWorld 0.2.0 dev 按清单取前六个不同任务族各一题。初态通过 {sum(r['baseline_initial'] for r in aw)}/{len(aw)}，"
             f"执行官方参考解后通过 {sum(r['success'] for r in aw)}/{len(aw)}。每条记录含官方断言详情。", "",
             "## 历史压缩与恢复对照", "",
             "三组模板分别提前确认 A、确认 B、从未确认。截断只保留候选清单和最终请求。结构化摘要直接保留选择字段，"
             "回读扫描归档中的该字段。规则遇到歧义即询问，故不追问到底、不模拟猜测模型。每个分支独立重置业务状态。"
             "τ-bench 使用开发用户的两笔待取消订单；AppWorld 使用一个开发世界中同一账户的两条实际笔记。"
             "场景复用了官方状态与写接口，模板和判分为本项目新增。未调用大模型，不估计自然歧义发生率。", "",
             "|环境|策略|正确写入|错写|重复询问|必要询问|","|---|---|---:|---:|---:|---:|"]
    for dataset in ("tau-retail","appworld-notes"):
        for policy in ("full","truncated","structured","reread"):
            values=[counts[dataset,policy,outcome] for outcome in ("correct_write","wrong_write","redundant_question","necessary_question")]
            lines.append("|"+"|".join(map(str,[dataset,policy,*values]))+"|")
    lines += ["","[零售参考调用](tau-dev.jsonl) · [AppWorld 评分详情](appworld-dev.jsonl) · [逐分支可见历史、读回证据及前后状态](paired.jsonl)",
              "", "在 context-bundles 目录执行：", "", "```powershell",
              ".\\data\\external\\appworld-env\\Scripts\\python.exe .\\scripts\\external_benchmarks.py","```",
              "", "数据来源：[原版 τ-bench](https://github.com/sierra-research/tau-bench) 与 [AppWorld](https://github.com/StonyBrookNLP/appworld)。本地输入包 SHA256：",""]
    for p in [EXT/"tau-bench.zip",EXT/"appworld-source.zip",EXT/"appworld/data-0.2.0.bundle"]:
        lines.append(f"- {p.name}：{hashlib.sha256(p.read_bytes()).hexdigest()}")
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(OUT/"report.md")


if __name__=="__main__":
    main()
