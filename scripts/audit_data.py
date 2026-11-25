"""Verify raw blobs, processed isolation and source grouping without model calls."""
import argparse
import hashlib
import json
from pathlib import Path

from context_bundles.io import read_jsonl, sha256, write_json

p = argparse.ArgumentParser()
p.add_argument("--raw", type=Path, default=Path("data/raw/mem2act-full"))
p.add_argument("--processed", type=Path, default=Path("data/processed/mem2act-v1"))
p.add_argument("--out", type=Path, default=Path("data/processed/mem2act-v1/integrity.json"))
args = p.parse_args()
snapshot = json.loads((args.raw / "snapshot.json").read_text())
if not snapshot.get("complete"):
    raise ValueError("Snapshot is incomplete")
for name, info in snapshot["files"].items():
    raw = (args.raw / name).read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != info["git_blob_sha1"] or hashlib.sha256(raw).hexdigest() != info["sha256"]:
        raise ValueError(f"Checksum failed: {name}")
audit = json.loads((args.processed / "audit.json").read_text())
if audit["qa_sha256"] != sha256(args.raw / "qa_dataset.jsonl") or audit["conversations_sha256"] != sha256(args.raw / "Mem2ACT_conversation.jsonl"):
    raise ValueError("Processed audit does not match raw files")
cases = read_jsonl(args.processed / "cases.jsonl")
labels = read_jsonl(args.processed / "labels.jsonl")
if {c["id"] for c in cases} != {l["id"] for l in labels}:
    raise ValueError("Case/label mismatch")
sessions = {s["session_id"]: s for s in read_jsonl(args.raw / "Mem2ACT_conversation.jsonl")}
owner = {}
for c in cases:
    if set(c) != {"id", "query", "tool_schema", "evidence", "group_id", "split"}:
        raise ValueError("Unexpected public case fields")
    if len(c["evidence"]) != len({e["id"] for e in c["evidence"]}):
        raise ValueError("Duplicate evidence IDs")
    sources = {source for e in c["evidence"] for source in sessions[e["session_id"]]["original_conversation_ids"]}
    for source in sources:
        identity = (c["group_id"], c["split"])
        if source in owner and owner[source] != identity:
            raise ValueError("Shared source crosses independent groups or partitions")
        owner[source] = identity
result = {"passed": True, "raw_files_verified": len(snapshot["files"]), "cases": len(cases),
          "unique_observed_sources": len(owner), "excluded_records": len(audit["excluded_records"]),
          "partition_counts": audit["partition_counts"], "commit": snapshot["commit"]}
write_json(args.out, result)
print(json.dumps(result, ensure_ascii=False, indent=2))
