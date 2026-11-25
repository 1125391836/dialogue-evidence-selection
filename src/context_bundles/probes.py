"""Four-condition diagnostics on fixed backgrounds, separate from main evaluation."""

import hashlib
import json
from pathlib import Path
import random

from .core import Budget, Case, Counter, canonical, messages
from .io import read_jsonl, sha256, write_json, write_jsonl
from .selection import bm25, lexical_pairs


def make_probes(cases_path: Path, config: dict, out: Path) -> dict:
    if out.exists():
        raise FileExistsError("Choose a new probe directory")
    if config["split"] == "test":
        raise ValueError("Use train/dev for exploratory probes; freeze a separate final evaluation protocol first")
    if config.get("max_cases", 0) < 1 or config.get("repeats", 0) < 1 or config.get("pairs_per_case", 0) < 1:
        raise ValueError("Positive case, repeat and pair limits are required")
    counter = Counter(config["encoding"])
    budget = Budget(config["history_budget"], counter)
    cases = [Case.from_dict(v) for v in read_jsonl(cases_path) if v["split"] == config["split"]]
    random.Random(config.get("seed", 42)).shuffle(cases)
    rows, pair_count, infeasible = [], 0, 0
    audit_path = cases_path.parent / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    engineering_only = audit.get("engineering_only", False)
    for case in cases[:config["max_cases"]]:
        by_id = {e.id: e for e in case.evidence}
        relevance = bm25(case.query, case.evidence)
        pairs = lexical_pairs(case.evidence, config.get("max_candidate_pairs", 100))
        pairs.sort(key=lambda p: (-sum(relevance[i] for i in p), p))
        for a, b in pairs[:config["pairs_per_case"]]:
            group = [by_id[a], by_id[b]]
            if not budget.fits(group):
                infeasible += 1
                continue
            # Construct ONE shared base; reserve room for both a and b even in singleton probes.
            base = []
            remainder = sorted((e for e in case.evidence if e.id not in (a, b)),
                               key=lambda e: (-relevance[e.id], e.id))
            for e in remainder:
                if budget.fits(base + group + [e]):
                    base.append(e)
            pair_id = hashlib.sha256(canonical([case.id, a, b]).encode()).hexdigest()[:20]
            for condition, additions in [("base", []), ("a", [by_id[a]]), ("b", [by_id[b]]), ("ab", group)]:
                chosen = sorted(base + additions, key=lambda e: (e.order, e.id))
                for repeat in range(config["repeats"]):
                    rid = hashlib.sha256(canonical([pair_id, condition, repeat, config]).encode()).hexdigest()[:24]
                    rows.append({
                        "request_id": rid, "case_id": case.id, "group_id": case.group_id,
                        "split": case.split, "method": "probe_" + condition,
                        "pair_id": pair_id, "pair_members": [a, b], "condition": condition,
                        "repeat": repeat, "budget": budget.limit, "unit": counter.unit,
                        "encoding": counter.name, "history_cost": budget.cost(chosen),
                        "selected_ids": [e.id for e in chosen], "base_ids": [e.id for e in base],
                        "messages": messages(case, chosen),
                        "engineering_only": engineering_only,
                    })
            pair_count += 1
    if not rows:
        raise ValueError("No feasible lexical candidate pairs; inspect the dataset/budget before changing the protocol")
    random.Random(config.get("seed", 42)).shuffle(rows)
    out.mkdir(parents=True)
    write_jsonl(out / "requests.jsonl", rows)
    meta = {"kind": "offline_four_condition_probe_plan", "pairs": pair_count, "requests": len(rows),
            "engineering_only": engineering_only,
            "infeasible_pairs": infeasible, "config": config, "cases_sha256": sha256(cases_path),
            "limitations": [
                "Lexical candidates are not verified complementary groups.",
                "These probes estimate candidate behavior, not prevalence over all possible groups.",
                "Lengths differ across the four conditions; matched-length controls must precede causal claims.",
                "Background may contain redundant evidence. Inspect sources on train/dev, never test labels.",
                "Positive interaction in all-or-nothing exact match alone is insufficient evidence.",
            ]}
    write_json(out / "manifest.json", meta)
    return meta
