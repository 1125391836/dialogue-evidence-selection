"""Execute a complete frozen design within an existing ledger, then fit train-only.

This module cannot create a fresh quota or increase an existing quota.
"""

from collections import defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from .core import Budget, Case, Counter
from .curriculum import digest, validate_plan
from .deepseek import DeepSeekClient
from .evaluation import score
from .executor import execute_stage
from .guards import promotion_gate, support_violations
from .io import read_jsonl, sha256, write_json, write_jsonl
from .ledger import Ledger
from .utility import SetFeatures, predict, train_predictor


def open_existing_ledger(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("An existing campaign ledger is required; this command never creates a quota")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        row = db.execute("SELECT value FROM metadata WHERE key='call_cap'").fetchone()
    if row is None:
        raise ValueError("Ledger has no call cap")
    return Ledger(path, int(row[0]))


def stage_name(manifest):
    return "curriculum_v2_" + digest([manifest["config"], manifest["files_sha256"]])[:20]


def plan_status(plan_dir, ledger):
    plan_dir = Path(plan_dir)
    manifest = validate_plan(plan_dir)
    stage = stage_name(manifest)
    specs = read_jsonl(plan_dir / "requests.jsonl")
    saved = {r["request_id"]: r for r in ledger.rows(stage)}
    pending, uncertain, completed, errors = [], [], 0, 0
    expected = {digest([stage, s]) for s in specs}
    if set(saved) - expected:
        raise ValueError("Unexpected jobs in curriculum stage")
    for spec in specs:
        key = digest([stage, spec])
        row = saved.get(key)
        if row is None or row["job_status"] == "planned":
            pending.append(key)
        elif row["job_status"] == "running":
            uncertain.append(key)
        else:
            completed += 1
            errors += row["job_status"] == "error"
    return {"stage": stage, "planned_calls": len(specs), "new_calls_needed": len(pending),
            "completed_calls": completed, "error_calls": errors, "uncertain_calls": len(uncertain),
            "ledger": ledger.stats()}


def execute_plan(plan_dir, ledger_path, max_new_calls, client=None):
    if type(max_new_calls) is not int or max_new_calls < 0:
        raise ValueError("Explicit nonnegative max_new_calls is required")
    ledger = open_existing_ledger(ledger_path)
    state = plan_status(plan_dir, ledger)
    if state["uncertain_calls"]:
        raise ValueError("Uncertain calls exist; inspect before any retry")
    needed = state["new_calls_needed"]
    if needed > max_new_calls or needed > state["ledger"]["remaining_calls"]:
        raise ValueError(f"Frozen design requires {needed} new calls; invocation cap={max_new_calls}, "
                         f"ledger remaining={state['ledger']['remaining_calls']}. No requests sent.")
    if needed == 0:
        return state
    manifest = json.loads((Path(plan_dir) / "manifest.json").read_text(encoding="utf-8"))
    specs = read_jsonl(Path(plan_dir) / "requests.jsonl")
    if client is None:
        client = DeepSeekClient(manifest["config"]["provider"])
    execute_stage(ledger, state["stage"], specs, client, concurrency=2)
    return plan_status(plan_dir, ledger)


def fit_plan(plan_dir, ledger_path, labels_path, out, alpha=10.0):
    """Average repeats per context; never inflate n by expanding shared roles."""
    plan_dir, labels_path, out = Path(plan_dir), Path(labels_path), Path(out)
    if out.exists():
        raise FileExistsError("Choose a new fit directory; never overwrite a frozen model")
    manifest = validate_plan(plan_dir)
    ledger = open_existing_ledger(ledger_path)
    state = plan_status(plan_dir, ledger)
    if state["new_calls_needed"] or state["uncertain_calls"]:
        raise ValueError("Complete the frozen design before fitting; partial fitting is forbidden")
    config = manifest["config"]
    cases = {r["id"]: Case.from_dict(r) for r in read_jsonl(config["cases_path"])}
    label_rows = read_jsonl(labels_path)
    labels = {r["id"]: r for r in label_rows}
    if len(labels) != len(label_rows):
        raise ValueError("Duplicate label IDs")
    budget = Budget(config["history_budget"], Counter(config["encoding"]))
    observations = defaultdict(list)
    expected_contexts = set()
    excluded = []
    for row in ledger.rows(state["stage"]):
        spec, result = row["spec"], row["result"] or {}
        case = cases[spec["case_id"]]
        label = labels[case.id]
        if (label["split"], label["group_id"]) != (case.split, case.group_id):
            raise ValueError("Case/label partition mismatch")
        if all(r["family"] == "reference_full" for r in spec["roles"]):
            continue
        if spec["history_cost"] > budget.limit:
            raise ValueError("Reference-only context entered fitting")
        expected_contexts.add(spec["coalition_id"])
        if result.get("status") != "ok" or result.get("finish_reason") != "stop":
            excluded.append({"observation_id": spec["observation_id"], "reason": "infrastructure_or_truncation"})
            continue
        target = score(result["output"], label["gold"])["exact_match"]
        observations[spec["coalition_id"]].append((spec, target, row["request_id"]))
    examples, incomplete = [], []
    for coalition in sorted(expected_contexts):
        rows = observations[coalition]
        if len(rows) != config["repeats"]:
            incomplete.append(coalition)
            continue  # No averaging over only the successful repeats.
        spec = rows[0][0]
        case = cases[spec["case_id"]]
        examples.append({"case_id": case.id, "group_id": case.group_id, "split": case.split,
            "coalition_id": coalition, "selected_ids": spec["selected_ids"],
            "features": SetFeatures(case, budget)(frozenset(spec["selected_ids"])),
            "target": sum(r[1] for r in rows) / len(rows), "repeat_count": len(rows),
            "request_ids": [r[2] for r in rows]})
    train = [e for e in examples if e["split"] == "train"]
    dev = [e for e in examples if e["split"] == "dev"]
    model = train_predictor(train, alpha)
    model["target"] = "mean_strict_exact_match_over_complete_repeats"
    dev_rows = [{"case_id": e["case_id"], "group_id": e["group_id"],
                 "coalition_id": e["coalition_id"], "target": e["target"],
                 "prediction": predict(model, e["features"]),
                 "support_violations": support_violations(model, e["features"])} for e in dev]
    dev_group_errors = defaultdict(list)
    for row in dev_rows:
        dev_group_errors[row["group_id"]].append((row["prediction"] - row["target"]) ** 2)
    model.update({"plan_manifest_sha256": sha256(plan_dir / "manifest.json"),
                  "labels_sha256": sha256(labels_path), "examples_sha256": digest(examples)})
    report = {"training_groups": len(model["training_groups"]), "training_contexts": len(train),
        "development_contexts": len(dev), "development_groups": len(dev_group_errors),
        "dev_group_macro_mse": (sum(sum(v) / len(v) for v in dev_group_errors.values()) / len(dev_group_errors)) if dev_group_errors else None,
        "gate": promotion_gate(model), "excluded_calls": excluded, "incomplete_contexts": incomplete,
        "note": "Predictor diagnostics only; no selector success or publication claim. Development labels are not fit."}
    out.mkdir(parents=True)
    write_jsonl(out / "training_examples.jsonl", train)
    write_jsonl(out / "development_predictions.jsonl", dev_rows)
    write_json(out / "utility_model.json", model)
    write_json(out / "report.json", report)
    return report
