"""One-command offline acceptance; copies the campaign before replaying stages."""
import argparse
from contextlib import closing, redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile

from context_bundles.campaign import Campaign
from context_bundles.io import sha256, write_json

p = argparse.ArgumentParser(description="Verify local setup, data, tests and zero-call resume")
p.add_argument("--campaign", type=Path, default=Path("runs/deepseek-progressive-60"))
p.add_argument("--config", type=Path, default=Path("configs/deepseek-60.json"))
p.add_argument("--out", type=Path, default=Path("runs/verification/latest.json"))
args = p.parse_args()
checks = []


def check(name, function):
    try:
        details = function()
        checks.append({"name": name, "passed": True, "details": details})
        print(f"[PASS] {name}", flush=True)
    except Exception as exc:
        checks.append({"name": name, "passed": False, "error_type": type(exc).__name__, "details": str(exc)})
        print(f"[FAIL] {name}: {type(exc).__name__}", flush=True)


def command(*command_args):
    result = subprocess.run([sys.executable, *command_args], capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError((result.stdout + result.stderr)[-6000:])
    return (result.stdout + result.stderr).strip()[-6000:]


def environment():
    packages = {}
    for module, distribution in [("numpy", "numpy"), ("sklearn", "scikit-learn"),
                                 ("tokenizers", "tokenizers"), ("tiktoken", "tiktoken")]:
        __import__(module)
        packages[distribution] = importlib.metadata.version(distribution)
    return {"python": sys.version.split()[0], "executable": sys.executable, "packages": packages}


def replay():
    if not args.campaign.is_dir():
        raise FileNotFoundError("Existing real campaign is missing")
    protected = [p for p in args.campaign.rglob("*") if p.is_file()]
    hashes = {str(path): sha256(path) for path in protected}
    ledger_path = args.campaign / "ledger.sqlite3"
    with closing(sqlite3.connect(ledger_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity check failed")
        original_count = db.execute("SELECT count(*) FROM attempts").fetchone()[0]
    class NoNetwork:
        def complete(self, *a, **kw):
            raise AssertionError("Offline replay attempted a new model request")
    with tempfile.TemporaryDirectory(prefix="context-bundles-verify-") as td:
        copy = Path(td) / "campaign"
        copy.mkdir()
        # Only inputs to stage replay; no need to copy the potentially large reports.
        for name in ["config.json", "data_manifest.json", "baseline_cohort.json", "diagnostic_cohort.json",
                     "utility_model.json", "training_examples.jsonl"]:
            shutil.copy2(args.campaign / name, copy / name)
        shutil.copytree(args.campaign / "plans", copy / "plans")
        with closing(sqlite3.connect(ledger_path.resolve().as_uri() + "?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(copy / "ledger.sqlite3")) as destination:
                source.backup(destination)
        campaign = Campaign(args.config, copy)
        campaign._client = NoNetwork()
        with redirect_stdout(io.StringIO()):
            campaign.run("all")
        status = campaign.ledger.stats()
        if status["reserved_calls"] != original_count:
            raise ValueError("Replay changed the call count")
        if hashes != {str(path): sha256(path) for path in protected}:
            raise ValueError("Original campaign artifacts changed")
        return {"status": status, "model_calls": 0, "original_files_unchanged": len(protected)}


check("Environment and imports", environment)
check("Raw data and history split audit", lambda: command("scripts/project.py", "audit"))
check("Unit and integration tests", lambda: command("scripts/project.py", "test"))
check("Frozen campaign replay without model access", replay)
plan_path = Path("runs/curriculum-v2-plan")
if plan_path.exists():
    check("Next-stage frozen plan", lambda: command("scripts/project.py", "next", "--action", "validate"))
result = {"checked_utc": datetime.now(timezone.utc).isoformat(),
          "engineering_passed": all(c["passed"] for c in checks), "actual_model_calls": 0,
          "scientific_effectiveness_established": False, "checks": checks,
          "note": "Passing verifies execution and invariants; it does not establish method effectiveness."}
args.out.parent.mkdir(parents=True, exist_ok=True)
write_json(args.out, result)
lines = ["# 项目离线验收", "",
         f"工程验收：{'PASS' if result['engineering_passed'] else 'FAIL'}。实际新增模型调用：0。",
         "此结论不代表方法有效，科学效果仍需独立实验验证。", "",
         "| 检查项 | 状态 |", "|---|---|"]
for item in checks:
    lines.append(f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} |")
lines += ["", "完整日志、依赖版本和失败原因见同目录 JSON。原实验在临时副本恢复，未重新发送模型请求。", ""]
args.out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
print(f"{'PASS' if result['engineering_passed'] else 'FAIL'}: {sum(c['passed'] for c in checks)}/{len(checks)} checks; 0 model calls; {args.out}")
raise SystemExit(0 if result["engineering_passed"] else 1)
