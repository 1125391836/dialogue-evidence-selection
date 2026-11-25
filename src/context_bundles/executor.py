from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from .deepseek import DeepSeekClient
from .io import write_json, write_jsonl
from .ledger import Ledger


def execute_stage(ledger: Ledger, stage: str, specs: list[dict], client: DeepSeekClient,
                  concurrency: int = 2) -> list[dict]:
    ids = [ledger.register(stage, spec) for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate jobs within a stage: repeat IDs must be explicit")
    def work(pair):
        key, spec = pair
        if not ledger.claim(key):
            return
        result = client.complete(spec["messages"], spec.get("max_output_tokens"))
        ledger.finish(key, result)
        print(json.dumps({"stage": stage, "case": spec.get("case_id"),
                          "method": spec.get("method"), "status": result["status"]}), flush=True)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(work, zip(ids, specs)))
    rows = ledger.rows(stage)
    if any(r["job_status"] in ("planned", "running") for r in rows):
        raise RuntimeError("Stage incomplete: quota reached or a previous call is in an uncertain state; inspect status")
    if any(r["result"].get("error_type") in ("HTTP_401", "HTTP_402", "HTTP_403") for r in rows):
        raise RuntimeError("Authentication/balance/access error; stopped without automatic retries")
    return rows


def export_ledger(ledger: Ledger, root: Path):
    write_jsonl(root / "jobs.jsonl", ledger.rows())
    write_json(root / "status.json", ledger.stats())
