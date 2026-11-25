"""Mem2ActBench adapter for a declared provided-context diagnostic setting."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .core import canonical
from .io import read_jsonl, sha256, write_json, write_jsonl


def session_blocks(session: dict, start_order: int) -> list[dict]:
    blocks = []
    pending = []
    start_turn = 0
    def flush():
        if pending:
            blocks.append({
                "id": f"{session['session_id']}:turn{start_turn:04d}",
                "text": canonical(pending), "order": start_order + len(blocks),
                "session_id": session["session_id"],
            })
    for i, turn in enumerate(session["turns"]):
        if turn.get("role") == "user" and pending:
            flush()
            pending = []
            start_turn = i
        # Strip source_id and any dataset annotations; retain actual observed content.
        clean = {k: turn[k] for k in ("role", "content", "tool_calls", "tool_call_id", "name") if k in turn}
        if "role" not in clean:
            raise ValueError(f"Turn without role in {session['session_id']}")
        pending.append(clean)
    flush()
    return blocks


def prepare(qa_path: Path, conversations_path: Path, out: Path, seed: int = 42,
            quarantine_empty_sources: bool = False, quarantine_missing_sources: bool = False) -> dict:
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    qas, sessions = read_jsonl(qa_path), read_jsonl(conversations_path)
    qids = [q["qa_id"] for q in qas]
    sids = [s["session_id"] for s in sessions]
    if len(qids) != len(set(qids)) or len(sids) != len(set(sids)):
        raise ValueError("Duplicate QA or session IDs")
    excluded = []
    if quarantine_empty_sources:
        excluded = [{"id": q["qa_id"], "reason": "empty_source_conversation_ids"}
                    for q in qas if not q["source_conversation_ids"]]
        qas = [q for q in qas if q["source_conversation_ids"]]
    source_sessions = defaultdict(set)
    for n, session in enumerate(sessions):
        for src in session["original_conversation_ids"]:
            source_sessions[src].add(n)
    if quarantine_missing_sources:
        retained = []
        for qa in qas:
            missing = set(qa["source_conversation_ids"]) - source_sessions.keys()
            if missing:
                excluded.append({"id": qa["qa_id"], "reason": "missing_source_histories", "missing_ids": sorted(missing)})
            else:
                retained.append(qa)
        qas = retained
    selected_sessions = []
    footprints = []
    for qa in qas:
        sources = qa["source_conversation_ids"]
        missing = set(sources) - source_sessions.keys()
        if missing or not sources:
            raise ValueError(f"Missing source histories for {qa['qa_id']}: {sorted(missing)}")
        indices = sorted(set().union(*(source_sessions[s] for s in sources)))
        selected_sessions.append(indices)
        # Include incidental histories too: a distractor in one case may be a target in another.
        footprints.append({src for i in indices for src in sessions[i]["original_conversation_ids"]})
    parent = list(range(len(qas)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, footprint in enumerate(footprints):
        for src in footprint:
            if src in owners:
                parent[find(i)] = find(owners[src])
            else:
                owners[src] = i
    members = defaultdict(list)
    for i, qa in enumerate(qas):
        members[find(i)].append(qa["qa_id"])
    group_ids = {root: "g_" + hashlib.sha256(canonical(sorted(ids)).encode()).hexdigest()[:16]
                 for root, ids in members.items()}
    def split(group_id):
        v = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16) / 2**32
        return "train" if v < 0.7 else ("dev" if v < 0.85 else "test")
    cases, labels = [], []
    for i, qa in enumerate(qas):
        evidence = []
        for idx in selected_sessions[i]:
            evidence.extend(session_blocks(sessions[idx], len(evidence)))
        gid = group_ids[find(i)]
        partition = split(gid)
        cases.append({"id": qa["qa_id"], "query": qa["query"],
                      "tool_schema": qa["target_tool_schema"], "evidence": evidence,
                      "group_id": gid, "split": partition})
        labels.append({"id": qa["qa_id"], "group_id": gid, "split": partition,
                       "gold": {"name": qa["tool_call"]["name"], "arguments": qa["tool_call"]["arguments"]},
                       "diagnostic_only": {
                           "grounding_info": qa["tool_call"].get("grounding_info", {}),
                           "complexity_metadata": qa.get("complexity_metadata", {}),
                           "source_conversation_ids": qa["source_conversation_ids"],
                       }})
    partition_counts = dict(Counter(c["split"] for c in cases))
    warnings = []
    if len(partition_counts) < 3:
        warnings.append("Some partitions are empty; do not split connected groups to force target ratios.")
    if members and max(map(len, members.values())) > len(qas) / 3:
        warnings.append("A large connected component limits effective independent sample size.")
    audit = {
        "setting": "provided_target_tool_and_source_session_union_diagnostic",
        "not_official_split": True, "seed": seed, "cases": len(cases), "sessions": len(sessions),
        "independent_groups": len(members), "partition_counts": partition_counts,
        "largest_group_cases": max(map(len, members.values()), default=0),
        "excluded_records": excluded,
        "group_sizes": sorted(map(len, members.values()), reverse=True),
        "qa_sha256": sha256(qa_path), "conversations_sha256": sha256(conversations_path),
        "warnings": warnings,
        "limitations": [
            "Gold source IDs construct the supplied context only; this is NOT global retrieval evaluation.",
            "Whole associated sessions, including distractors, are retained; selectors never see gold source IDs.",
            "Cross-session order follows the released file, not inferred real-world timestamps.",
            "No gold fact chain, answer, parameter source, or difficulty label is in cases.jsonl.",
            "One user turn plus following assistant/tool turns is an indivisible evidence block.",
        ],
    }
    provenance = qa_path.parent / "snapshot.json"
    audit["source_snapshot"] = json.loads(provenance.read_text(encoding="utf-8")) if provenance.exists() else None
    audit["engineering_only"] = bool(audit["source_snapshot"] and
                                     audit["source_snapshot"].get("scientific_evaluation_allowed") is False)
    out.mkdir(parents=True)
    write_jsonl(out / "cases.jsonl", cases)
    write_jsonl(out / "labels.jsonl", labels)
    write_json(out / "audit.json", audit)
    return audit
