"""Default action is offline planning; execution needs an explicit call cap."""
import argparse
import json
from pathlib import Path

from context_bundles.curriculum import build_plan, validate_plan
from context_bundles.curriculum_run import execute_plan, fit_plan, open_existing_ledger, plan_status

p = argparse.ArgumentParser(description="Next-stage sampling, validation, quota-bound execution and fitting")
p.add_argument("--action", choices=["plan", "validate", "status", "run", "train"], default="plan")
p.add_argument("--config", type=Path, default=Path("configs/curriculum-v2.json"))
p.add_argument("--out", type=Path, default=Path("runs/curriculum-v2-plan"))
p.add_argument("--ledger", type=Path, default=Path("runs/deepseek-progressive-60/ledger.sqlite3"))
p.add_argument("--max-new-calls", type=int)
p.add_argument("--labels", type=Path, default=Path("data/processed/mem2act-v1/labels.jsonl"))
p.add_argument("--fit-out", type=Path, default=Path("runs/curriculum-v2-fit"))
args = p.parse_args()
try:
    if args.action == "plan":
        result = build_plan(json.loads(args.config.read_text(encoding="utf-8-sig")), args.out)
        result = {k: result[k] for k in ["kind", "requests", "unique_contexts", "group_counts", "empty_contexts", "pairs", "available_pair_controls", "actual_model_calls"]}
    elif args.action == "validate":
        result = {"passed": True, "requests": validate_plan(args.out)["requests"], "model_calls": 0}
    elif args.action == "status":
        result = plan_status(args.out, open_existing_ledger(args.ledger))
    elif args.action == "run":
        if args.max_new_calls is None:
            p.error("--action run requires --max-new-calls; default action does not send requests")
        result = execute_plan(args.out, args.ledger, args.max_new_calls)
    else:
        result = fit_plan(args.out, args.ledger, args.labels, args.fit_out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
except (ValueError, FileNotFoundError, FileExistsError) as exc:
    p.exit(2, f"{exc}\n")
