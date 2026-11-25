"""Observable-only sampling of empty, sparse, dense and paired context sets.

The plan is an intervention design, not observed performance or a novelty claim.
No answer file is opened while planning or validating.
"""

from collections import Counter as Counts, defaultdict
from itertools import combinations
import hashlib
import html
import json
from pathlib import Path
import random

from .core import Budget, Case, Counter, canonical, messages
from .io import read_jsonl, sha256, write_json, write_jsonl
from .selection import bm25, lexical_pairs, pack, select


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_public(config):
    path = Path(config["cases_path"])
    audit_path = path.parent / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("engineering_only"):
        raise ValueError("Curriculum needs complete data, not a partial smoke dataset")
    rows = read_jsonl(path)
    if len(rows) != len({r["id"] for r in rows}):
        raise ValueError("Duplicate case IDs")
    group_split = {}
    for row in rows:
        if row["group_id"] in group_split and group_split[row["group_id"]] != row["split"]:
            raise ValueError("A history group crosses partitions")
        group_split[row["group_id"]] = row["split"]
    counter = Counter(config["encoding"])
    provenance = {"cases_sha256": sha256(path), "audit_sha256": sha256(audit_path),
                  "tokenizer_sha256": sha256(config["encoding"][3:]) if config["encoding"].startswith("hf:") else None}
    return {r["id"]: Case.from_dict(r) for r in rows}, counter, provenance


def sample_groups(cases, counts, seed):
    if set(counts) != {"train", "dev"} or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError("Specify positive train/dev group counts; final test is forbidden")
    selected = []
    for split, count in sorted(counts.items()):
        grouped = defaultdict(list)
        for case in cases.values():
            if case.split == split:
                grouped[case.group_id].append(case)
        groups = sorted(grouped)
        random.Random(f"{seed}:{split}:groups").shuffle(groups)
        if len(groups) < count:
            raise ValueError(f"Not enough independent {split} groups")
        for group in groups[:count]:
            choices = sorted(grouped[group], key=lambda c: c.id)
            selected.append(random.Random(f"{seed}:{group}:case").choice(choices))
    return selected


def case_design(case, config, counter):
    budget = Budget(config["history_budget"], counter)
    scores = bm25(case.query, case.evidence)
    ranked = sorted(case.evidence, key=lambda e: (-scores[e.id], e.id))
    by_id = {e.id: e for e in case.evidence}
    coalitions, probes, exclusions = {}, [], []

    def add(selected, family, **extra):
        selected = sorted(selected, key=lambda e: (e.order, e.id))
        ids = [e.id for e in selected]
        cost = budget.cost(selected)
        if family != "reference_full" and cost > budget.limit:
            raise ValueError("Sampling produced an over-budget set")
        role = {"family": family, **extra}
        key = digest([case.id, ids])
        if key not in coalitions:
            prompt = messages(case, selected)
            prompt_cost = sum(counter(m["content"]) for m in prompt)
            if prompt_cost > config["max_prompt_tokens"]:
                exclusions.append({"case_id": case.id, "role": role, "reason": "prompt_cap"})
                return None
            coalitions[key] = {"case_id": case.id, "group_id": case.group_id, "split": case.split,
                "coalition_id": key, "selected_ids": ids, "history_cost": cost,
                "budget": budget.limit if cost <= budget.limit else -1,
                "unit": counter.unit, "encoding": counter.name,
                "prompt_content_tokens": prompt_cost, "messages": prompt,
                "provider": config["provider"], "roles": []}
        if role not in coalitions[key]["roles"]:
            coalitions[key]["roles"].append(role)
        return key

    add([], "empty")
    feasible = [e for e in ranked if budget.fits([e])]
    if feasible:
        add([feasible[0]], "singleton_relevant")
        add([random.Random(f"{config['seed']}:{case.id}:single").choice(feasible)], "singleton_random")
    for target in config["sample_budgets"]:
        subbudget = Budget(target, counter)
        for method in ("bm25", "recent"):
            add(select(case, method, subbudget), method, target_budget=target)
        shuffled = list(case.evidence)
        random.Random(f"{config['seed']}:{case.id}:{target}:uniform").shuffle(shuffled)
        add(pack(shuffled, subbudget), "uniform", target_budget=target)
    add(list(case.evidence), "reference_full")

    pairs = lexical_pairs(case.evidence, config["candidate_limit"])
    pairs.sort(key=lambda pair: (-sum(scores[i] for i in pair), pair))
    pair = next((p for p in pairs if budget.fits([by_id[i] for i in p])), None)
    if pair is None:
        exclusions.append({"case_id": case.id, "reason": "no_feasible_lexical_pair"})
    else:
        a, b = pair
        group = [by_id[a], by_id[b]]
        base = []
        for e in ranked:
            if e.id not in pair and budget.cost(base + [e]) <= budget.limit // 2 and budget.fits(base + group + [e]):
                base.append(e)
        pair_id = digest([case.id, [e.id for e in base], pair])
        conditions = {}
        for condition, extra in [("base", []), ("a", [by_id[a]]), ("b", [by_id[b]]), ("ab", group)]:
            conditions[condition] = add(base + extra, "pair_probe", pair_id=pair_id, condition=condition)
        # Bounded random candidate pool; accept only a declared length tolerance.
        others = [e for e in case.evidence if e.id not in {x.id for x in base + group}]
        random.Random(f"{config['seed']}:{case.id}:control").shuffle(others)
        others = others[:config["control_pool_items"]]
        controls = [list(p) for p in combinations(others, 2) if budget.fits(base + list(p))]
        target_cost = budget.cost(base + group)
        control = min(controls, key=lambda p: (abs(budget.cost(base + p) - target_cost), tuple(e.id for e in p))) if controls else None
        delta = budget.cost(base + control) - target_cost if control else None
        matched = delta is not None and abs(delta) <= config["control_tolerance_tokens"]
        if matched:
            conditions["random"] = add(base + control, "pair_probe", pair_id=pair_id, condition="random")
        else:
            exclusions.append({"case_id": case.id, "reason": "no_control_within_length_tolerance", "nearest_delta": delta})
        probes.append({"pair_id": pair_id, "case_id": case.id, "split": case.split,
            "base_ids": [e.id for e in base], "members": list(pair), "conditions": conditions,
            "control_available": matched and conditions.get("random") is not None,
            "control_length_delta": delta,
            "candidate_kind": "lexical_overlap_not_verified_dependency",
            "position_controlled": False, "semantic_irrelevance_verified": False})
    return list(coalitions.values()), probes, exclusions


def check_config(config):
    for key in ("repeats", "history_budget", "max_prompt_tokens", "candidate_limit", "control_pool_items"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"Positive integer required: {key}")
    if type(config.get("control_tolerance_tokens")) is not int or config["control_tolerance_tokens"] < 0:
        raise ValueError("Nonnegative control tolerance required")
    budgets = config.get("sample_budgets", [])
    if not budgets or len(set(budgets)) != len(budgets) or any(type(b) is not int or b < 1 or b > config["history_budget"] for b in budgets):
        raise ValueError("Sample budgets must be unique positive values within history budget")


def build_plan(config, out):
    out = Path(out)
    check_config(config)
    cases, counter, provenance = load_public(config)
    if out.exists():
        existing = validate_plan(out)
        if existing["config"] != config:
            raise ValueError("Plan is frozen; select a new output directory for changed sampling")
        return existing
    cohort = sample_groups(cases, config["groups"], config["seed"])
    coalitions, probes, exclusions = [], [], []
    for case in cohort:
        chosen, diagnostics, missing = case_design(case, config, counter)
        coalitions.extend(chosen); probes.extend(diagnostics); exclusions.extend(missing)
    # Repeated calls are intentional. Identical contexts serving multiple roles are shared.
    rows = []
    for repeat in range(config["repeats"]):
        shuffled = sorted(coalitions, key=lambda s: s["coalition_id"])
        random.Random(f"{config['seed']}:{repeat}:schedule").shuffle(shuffled)
        for coalition in shuffled:
            spec = {**coalition, "method": "curriculum_v2", "repeat": repeat}
            rows.append({"observation_id": digest(spec), **spec})
    out.mkdir(parents=True)
    write_jsonl(out / "requests.jsonl", rows)
    write_json(out / "pairs.json", probes)
    write_json(out / "exclusions.json", exclusions)
    write_json(out / "cohort.json", [{"case_id": c.id, "group_id": c.group_id, "split": c.split} for c in cohort])
    manifest = {"schema_version": 2, "kind": "offline_curriculum_plan_not_results", "config": config,
        "provenance": provenance, "requests": len(rows), "unique_contexts": len(coalitions),
        "group_counts": dict(Counts(c.split for c in cohort)), "test_cases": 0,
        "empty_contexts": sum(not s["selected_ids"] for s in coalitions),
        "available_pair_controls": sum(p["control_available"] for p in probes),
        "pairs": len(probes), "actual_model_calls": 0,
        "prompt_content_tokens": sum(r["prompt_content_tokens"] for r in rows),
        "max_completion_tokens": len(rows) * config["provider"]["max_output_tokens"],
        "limits": ["Plan only, no model performance or fee charged.",
                   "Near-length controls are not guaranteed semantically irrelevant or position matched.",
                   "Empty/sparse sampling improves coverage but does not establish generalization.",
                   "Two repeats are exploratory, not a power calculation.",
                   "Reference-full observations are excluded from predictor fitting."]}
    manifest["files_sha256"] = {name: sha256(out / name) for name in ["requests.jsonl", "pairs.json", "exclusions.json", "cohort.json"]}
    write_json(out / "manifest.json", manifest)
    validate_plan(out)
    write_audit_view(out, cases, manifest)
    return manifest


def validate_plan(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    check_config(config)
    cases, counter, provenance = load_public(config)
    if provenance != manifest["provenance"]:
        raise ValueError("Source data or tokenizer changed")
    expected_files = {"requests.jsonl", "pairs.json", "exclusions.json", "cohort.json"}
    if set(manifest["files_sha256"]) != expected_files:
        raise ValueError("Unexpected plan files")
    for name, expected in manifest["files_sha256"].items():
        if sha256(out / name) != expected:
            raise ValueError(f"Frozen file changed: {name}")
    rows = read_jsonl(out / "requests.jsonl")
    if len(rows) != manifest["requests"] or len(rows) != len({r["observation_id"] for r in rows}):
        raise ValueError("Observation count/identity mismatch")
    budget = Budget(config["history_budget"], counter)
    groups, observed, coalition_rows = {}, defaultdict(set), {}
    for row in rows:
        spec = {k: v for k, v in row.items() if k != "observation_id"}
        if digest(spec) != row["observation_id"]:
            raise ValueError("Observation content changed")
        case = cases[row["case_id"]]
        if case.split not in ("train", "dev") or (row["split"], row["group_id"]) != (case.split, case.group_id):
            raise ValueError("Final-test or group leakage")
        if case.group_id in groups and groups[case.group_id] != case.id:
            raise ValueError("Multiple sampled cases in a group")
        groups[case.group_id] = case.id
        by_id = {e.id: e for e in case.evidence}
        ids = row["selected_ids"]
        if len(ids) != len(set(ids)) or not set(ids) <= by_id.keys():
            raise ValueError("Invalid selected evidence")
        selected = [by_id[i] for i in ids]
        if row["messages"] != messages(case, selected):
            raise ValueError("Prompt does not match public inputs")
        cost = budget.cost(selected)
        if row["history_cost"] != cost or (cost > budget.limit and any(r["family"] != "reference_full" for r in row["roles"])):
            raise ValueError("History cost or budget mismatch")
        expected_budget = budget.limit if cost <= budget.limit else -1
        if row["budget"] != expected_budget or row["encoding"] != counter.name or row["unit"] != counter.unit:
            raise ValueError("Budget encoding mismatch")
        prompt_cost = sum(counter(m["content"]) for m in row["messages"])
        if prompt_cost != row["prompt_content_tokens"] or prompt_cost > config["max_prompt_tokens"]:
            raise ValueError("Prompt cap mismatch")
        if row["provider"] != config["provider"]:
            raise ValueError("Provider changed")
        if row["coalition_id"] != digest([case.id, ids]):
            raise ValueError("Coalition identity mismatch")
        prior = coalition_rows.setdefault(row["coalition_id"], row)
        if {k: v for k, v in prior.items() if k not in ("repeat", "observation_id")} != {k: v for k, v in row.items() if k not in ("repeat", "observation_id")}:
            raise ValueError("Repeated observations changed context or roles")
        for role in row["roles"]:
            if role["family"] == "empty" and ids:
                raise ValueError("Empty condition contains evidence")
            if role["family"].startswith("singleton_") and len(ids) != 1:
                raise ValueError("Singleton condition must have one block")
            if "target_budget" in role and (role["target_budget"] not in config["sample_budgets"] or cost > role["target_budget"]):
                raise ValueError("Sampling target exceeded")
        if row["repeat"] not in range(config["repeats"]):
            raise ValueError("Invalid repeat")
        observed[row["coalition_id"]].add(row["repeat"])
    if any(repeats != set(range(config["repeats"])) for repeats in observed.values()):
        raise ValueError("Incomplete repeat design")
    cohort = json.loads((out / "cohort.json").read_text(encoding="utf-8"))
    actual = [{"case_id": cid, "group_id": gid, "split": cases[cid].split} for gid, cid in groups.items()]
    if sorted(cohort, key=canonical) != sorted(actual, key=canonical):
        raise ValueError("Cohort is missing observations")
    if dict(Counts(c["split"] for c in cohort)) != config["groups"] or manifest["group_counts"] != config["groups"]:
        raise ValueError("Independent group counts changed")
    empty_cases = {r["case_id"] for r in rows if any(role["family"] == "empty" for role in r["roles"])}
    if empty_cases != {c["case_id"] for c in cohort}:
        raise ValueError("Each sampled case must have an empty-context anchor")
    for pair in json.loads((out / "pairs.json").read_text(encoding="utf-8")):
        base, members = set(pair["base_ids"]), pair["members"]
        if len(members) != 2 or base & set(members):
            raise ValueError("Pair overlaps its fixed background")
        for condition, extra in [("base", set()), ("a", {members[0]}), ("b", {members[1]}), ("ab", set(members))]:
            key = pair["conditions"].get(condition)
            if key is not None:
                row = coalition_rows[key]
                if row["case_id"] != pair["case_id"] or set(row["selected_ids"]) != base | extra:
                    raise ValueError("Pair intervention changed background")
        if pair["control_available"]:
            random_row = coalition_rows[pair["conditions"]["random"]]
            ids = set(random_row["selected_ids"])
            joint = coalition_rows[pair["conditions"]["ab"]]
            delta = random_row["history_cost"] - joint["history_cost"]
            if not base <= ids or ids & set(members) or len(ids - base) != 2 or delta != pair["control_length_delta"] or abs(delta) > config["control_tolerance_tokens"]:
                raise ValueError("Invalid length control")
    return manifest


def write_audit_view(out, cases, manifest):
    """Human review shows raw evidence and hypotheses; no hidden answer access."""
    cohort = json.loads((out / "cohort.json").read_text(encoding="utf-8"))
    probes = {p["case_id"]: p for p in json.loads((out / "pairs.json").read_text(encoding="utf-8"))}
    cards, annotations = [], []
    for entry in cohort:
        case = cases[entry["case_id"]]
        pair = probes.get(case.id, {})
        blocks = "".join(f"<details><summary>{html.escape(e.id)} · {'候选成员' if e.id in pair.get('members', []) else '历史'}</summary><pre>{html.escape(e.text)}</pre></details>" for e in case.evidence)
        cards.append(f"<section><h2>{html.escape(case.id)} · {case.split}</h2><p>{html.escape(case.query)}</p>"
                     f"<details><summary>工具定义</summary><pre>{html.escape(json.dumps(case.tool_schema,ensure_ascii=False,indent=2))}</pre></details>"
                     f"<details><summary>组合假设与控制条件</summary><pre>{html.escape(json.dumps(pair,ensure_ascii=False,indent=2))}</pre></details>{blocks}</section>")
        annotations.append({**entry, "reviewed": False, "necessary_evidence_ids": [],
                            "pair_dependency_verified": None, "control_irrelevant_verified": None,
                            "failure_category": None, "notes": ""})
    write_jsonl(out / "annotations-template.jsonl", annotations)
    (out / "audit.html").write_text("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>下一阶段证据审核</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:20px}section{border-top:2px solid #ddd;margin-top:32px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:14px}summary{cursor:pointer;padding:8px}</style>"
        f"<h1>下一阶段证据审核</h1><p>{len(cohort)} 个独立组，计划 {manifest['requests']} 次调用；尚未执行。"
        "只呈现 train/dev 可见输入；词面关联不是已验证依赖。填写审核副本，不覆盖原始模板。</p>"
        + "".join(cards) + "</html>", encoding="utf-8")
