"""Capture explicit source files and frozen artifacts, without credentials or raw data."""
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import sys
import zipfile

from context_bundles.io import sha256, write_json

root = Path(__file__).resolve().parents[1]
out = root / "runs/deepseek-progressive-60/reproducibility"
out.mkdir(exist_ok=True)
files = [root / p for p in ["README.md", "pyproject.toml", "requirements-lock.txt", ".gitignore"]]
for folder, pattern in [("src", "*.py"), ("scripts", "*.py"), ("tests", "*.py"), ("configs", "*.json"), ("docs", "*.md")]:
    files.extend((root / folder).rglob(pattern))
files = sorted(set(files))
archive = out / "source-after-pilot.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for path in files:
        z.write(path, path.relative_to(root).as_posix())
frozen = root / "runs/deepseek-progressive-60"
artifacts = [frozen / p for p in ["config.json", "data_manifest.json", "jobs.jsonl", "utility_model.json",
             "training_examples.jsonl", "report/report.json", "postmortem/review.json"]]
artifacts.extend((frozen / "plans").glob("*.jsonl"))
manifest = {"captured_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Source after first real pilot and engineering fixes; not the exact pre-pilot source revision.",
            "python": sys.version, "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ["context-bundles", "numpy", "scikit-learn", "tokenizers", "tiktoken", "matplotlib"]},
            "source_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in files},
            "artifact_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in artifacts},
            "archive_sha256": sha256(archive)}
write_json(out / "manifest.json", manifest)
print(f"Captured {len(files)} source files and {len(artifacts)} experiment artifacts; no API calls.")
