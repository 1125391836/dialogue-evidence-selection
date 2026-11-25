"""Recover complete JSONL records from a contiguous downloaded prefix.

Engineering test only. Never use the selected subset for scientific comparisons.
The complete file's Git checksum remains unverified until download finishes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("data/raw/mem2act-small"))
    p.add_argument("--out", type=Path, default=Path("data/raw/mem2act-partial-smoke"))
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("Choose a new engineering subset directory")
    parts = []
    for file in args.raw.glob("*.part"):
        match = re.fullmatch(r"\.([0-9a-f]{40})\.(\d+)-(\d+)\.part", file.name)
        if match:
            parts.append((int(match[2]), int(match[3]), match[1], file))
    data = b""
    blob = None
    for start, end, sha, file in sorted(parts):
        if start != len(data):
            break
        if blob is not None and sha != blob:
            raise ValueError("Mixed source blobs")
        blob = sha
        raw = file.read_bytes()
        if len(raw) != end - start + 1:
            break
        data += raw
    sessions = []
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        sessions.append(json.loads(line))
    if not sessions:
        raise ValueError("No complete prefix records")
    sources = {src for s in sessions for src in s["original_conversation_ids"]}
    qas = [json.loads(l) for l in (args.raw / "qa_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    usable = [q for q in qas if set(q["source_conversation_ids"]) <= sources]
    args.out.mkdir(parents=True)
    for name, rows in [("qa_dataset.jsonl", usable), ("Mem2ACT_conversation.jsonl", sessions)]:
        (args.out / name).write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in rows), encoding="utf-8")
    metadata = {
        "kind": "partial_download_engineering_smoke_only", "scientific_evaluation_allowed": False,
        "complete_file_checksum_verified": False, "source_blob": blob,
        "prefix_bytes": len(data), "prefix_sha256": hashlib.sha256(data).hexdigest(),
        "complete_sessions": len(sessions), "eligible_qa": len(usable),
        "warning": "Source-prefix availability selection is biased. Full data must replace it before research runs.",
    }
    (args.out / "snapshot.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
