import hashlib
import json
from pathlib import Path
import random
import time

from .core import Budget, Case, Counter, canonical, messages
from .io import read_jsonl, sha256, write_json, write_jsonl
from .selection import select


def make_plan(cases_path: Path, config: dict, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"Use a new plan directory: {out}")
    counter = Counter(config["encoding"])
    methods = config["methods"]
    budgets = config["history_budgets"]
    repeats = config.get("repeats", 1)
    if repeats < 1 or not methods or len(set(methods)) != len(methods):
        raise ValueError("Require positive repeats and unique, nonempty methods")
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("Require unique, nonempty budgets")
    cases = [Case.from_dict(v) for v in read_jsonl(cases_path) if v["split"] == config["split"]]
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    random.Random(config.get("seed", 42)).shuffle(cases)
    limit = config.get("max_cases")
    if limit is not None and limit < 1:
        raise ValueError("max_cases must be positive")
    cases = cases[:limit]
    if not cases:
        raise ValueError("No cases in selected split")
    rows = []
    audit_path = cases_path.parent / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    engineering_only = audit.get("engineering_only", False)
    for case in cases:
        for method in methods:
            # Full-history reference runs once per repetition, not once per budget.
            for limit in (budgets[:1] if method == "full" else budgets):
                budget = Budget(limit, counter)
                started = time.perf_counter()
                selected = select(case, method, budget, config.get("max_pairs", 100))
                selection_seconds = time.perf_counter() - started
                if method != "full" and not budget.fits(selected):
                    raise AssertionError("Selection exceeded declared history budget")
                prompt = messages(case, selected)
                for repeat in range(repeats):
                    rid = hashlib.sha256(canonical([case.id, method, limit if method != "full" else None, repeat, config]).encode()).hexdigest()[:24]
                    rows.append({
                        "request_id": rid, "case_id": case.id, "group_id": case.group_id,
                        "split": case.split, "method": method,
                        "budget": limit if method != "full" else -1,
                        "unit": counter.unit, "encoding": counter.name,
                        "repeat": repeat, "selected_ids": [e.id for e in selected],
                        "selected_count": len(selected), "candidate_count": len(case.evidence),
                        "history_cost": budget.cost(selected),
                        "prompt_content_cost": sum(counter(m["content"]) for m in prompt),
                        "prompt_count_note": "Content only; excludes provider chat framing. API usage is authoritative for billing.",
                        "selection_seconds": selection_seconds, "messages": prompt,
                        "engineering_only": engineering_only,
                    })
    random.Random(config.get("seed", 42)).shuffle(rows)
    out.mkdir(parents=True)
    write_jsonl(out / "requests.jsonl", rows)
    manifest = {
        "kind": "offline_plan_no_model_calls", "cases": len(cases), "requests": len(rows),
        "independent_groups": len({c.group_id for c in cases}), "config": config,
        "cases_sha256": sha256(cases_path), "requests_sha256": sha256(out / "requests.jsonl"),
        "budget_scope": "serialized history only; fixed prompt and schema measured separately",
        "formal_token_budget": counter.unit == "tokens",
        "engineering_only": engineering_only,
        "warning": "utf8_bytes is a smoke-test unit, not an LLM token count" if counter.unit == "bytes" else None,
    }
    write_json(out / "manifest.json", manifest)
    return manifest
