import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from .data import prepare
from .evaluation import summarize
from .experiment import make_plan
from .io import read_jsonl, sha256, write_json, write_jsonl
from .provider import ChatClient
from .probes import make_probes
from .search import synthetic_demo


def main():
    parser = argparse.ArgumentParser(description="Context selection pilot (offline unless run-api is requested)")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Known synthetic utility, no LLM")
    demo.add_argument("--out", type=Path)
    prep = commands.add_parser("prepare", help="Separate observed inputs from labels and make group splits")
    prep.add_argument("--qa", type=Path, required=True)
    prep.add_argument("--conversations", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--quarantine-empty-sources", action="store_true",
                      help="Explicitly exclude and log records that have no source history IDs")
    prep.add_argument("--quarantine-missing-sources", action="store_true",
                      help="Explicitly exclude and log records referencing unavailable source histories")
    plan = commands.add_parser("plan", help="Write prompts and selections without network calls")
    plan.add_argument("--cases", type=Path, required=True)
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--out", type=Path, required=True)
    probe = commands.add_parser("probe-plan", help="Offline four-condition exploratory diagnosis on train/dev")
    probe.add_argument("--cases", type=Path, required=True)
    probe.add_argument("--config", type=Path, required=True)
    probe.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run-api", help="Send precomputed requests to a model endpoint")
    run.add_argument("--requests", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--model", default=os.environ.get("MODEL_NAME"))
    run.add_argument("--base-url", default=os.environ.get("MODEL_BASE_URL"))
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-output-tokens", type=int, default=512)
    run.add_argument("--max-requests", type=int, default=10)
    run.add_argument("--allow-byte-smoke", action="store_true")
    run.add_argument("--allow-partial-smoke", action="store_true")
    score = commands.add_parser("score", help="Evaluate completed predictions offline")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "demo":
        result = synthetic_demo()
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.out, result)
    elif args.command == "prepare":
        result = prepare(args.qa, args.conversations, args.out, args.seed, args.quarantine_empty_sources,
                         args.quarantine_missing_sources)
    elif args.command == "plan":
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        result = make_plan(args.cases, config, args.out)
    elif args.command == "probe-plan":
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        result = make_probes(args.cases, config, args.out)
    elif args.command == "run-api":
        if not args.model or not args.base_url:
            parser.error("Set --model and --base-url (or MODEL_NAME and MODEL_BASE_URL)")
        if args.max_requests < 1 or args.max_output_tokens < 1:
            parser.error("Request and output caps must be positive")
        if args.out.exists():
            parser.error("Choose a new output directory to preserve previous results")
        rows = read_jsonl(args.requests)
        if len({r["request_id"] for r in rows}) != len(rows):
            parser.error("Duplicate request IDs in plan")
        if not args.allow_byte_smoke and any(r["unit"] != "tokens" for r in rows):
            parser.error("Rebuild with a real tokenizer, or use --allow-byte-smoke for an engineering smoke test")
        if not args.allow_partial_smoke and any(r.get("engineering_only") for r in rows):
            parser.error("Partial source data is engineering-only; fetch complete data or use --allow-partial-smoke")
        client = ChatClient(args.model, args.base_url, args.temperature, args.max_output_tokens)
        chosen = rows[:args.max_requests]
        args.out.mkdir(parents=True)
        run_meta = {"model": args.model, "base_url": args.base_url, "temperature": args.temperature,
                    "max_output_tokens": args.max_output_tokens, "requested_calls": len(chosen),
                    "total_plan_calls": len(rows), "complete_plan": len(chosen) == len(rows),
                    "requests_sha256": sha256(args.requests),
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "warning": "A capped subset may be unbalanced; do not treat it as the full comparison."}
        write_json(args.out / "manifest.json", run_meta)
        with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for i, row in enumerate(chosen):
                value = client.complete(row["messages"])
                record = {k: v for k, v in row.items() if k != "messages"}
                record.update(value)
                record["requested_model"] = args.model
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{i + 1}/{len(chosen)} {value['status']}", flush=True)
        result = {"output": str(args.out), "calls": len(chosen), "complete_plan": len(chosen) == len(rows)}
    else:
        if args.out.exists():
            parser.error("Choose a new score output directory")
        report = summarize(read_jsonl(args.predictions), read_jsonl(args.labels))
        args.out.mkdir(parents=True)
        write_json(args.out / "summary.json", report["summary"])
        write_jsonl(args.out / "scored.jsonl", report["scored"])
        result = report["summary"]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
