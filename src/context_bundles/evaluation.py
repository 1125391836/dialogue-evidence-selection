"""Strict tool-call scoring; no LLM judge and no gold exposure to selectors."""

import json
import re
from collections import defaultdict

from .core import canonical


def parse_prediction(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Malformed fenced JSON")
        text = match.group(1)
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f"Non-JSON number: {value}")
    value = json.loads(text, object_pairs_hook=no_duplicates, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("Prediction must be an object")
    if value == {"abstain": True}:
        return value
    if set(value) != {"name", "arguments"} or not isinstance(value["name"], str) or not isinstance(value["arguments"], dict):
        raise ValueError("Expected exactly name (string) and arguments (object)")
    return value


def score(text: str, gold: dict) -> dict:
    try:
        pred = parse_prediction(text)
    except (ValueError, TypeError):
        return {"valid_json_call": False, "abstained": False, "exact_match": 0.0,
                "tool_name_correct": 0.0, "argument_f1": 0.0}
    abstained = pred.get("abstain") is True
    true_args = {(k, canonical(v)) for k, v in gold["arguments"].items()}
    pred_args = {(k, canonical(v)) for k, v in pred.get("arguments", {}).items()}
    f1 = 2 * len(true_args & pred_args) / (len(true_args) + len(pred_args)) if true_args or pred_args else 1.0
    name_ok = pred.get("name") == gold["name"]
    return {
        "valid_json_call": not abstained, "abstained": abstained,
        "exact_match": float(not abstained and name_ok and pred_args == true_args),
        "tool_name_correct": float(name_ok),
        "argument_f1": 0.0 if abstained else f1,
    }


def summarize(predictions: list[dict], labels: list[dict]) -> dict:
    label_map = {v["id"]: v for v in labels}
    if len(label_map) != len(labels):
        raise ValueError("Duplicate label IDs")
    seen = set()
    groups = defaultdict(list)
    scored = []
    for row in predictions:
        identity = row["request_id"]
        if identity in seen:
            raise ValueError(f"Duplicate prediction request: {identity}")
        seen.add(identity)
        label = label_map[row["case_id"]]
        metrics = score(row.get("output", ""), label["gold"])
        if row.get("status") != "ok":
            metrics = {"valid_json_call": False, "abstained": False, "exact_match": 0.0,
                       "tool_name_correct": 0.0, "argument_f1": 0.0}
        result = {**row, **metrics, "group_id": label["group_id"]}
        scored.append(result)
        groups[(row["method"], row["budget"], row["unit"])].append(result)
    summary = []
    for (method, budget, unit), rows in sorted(groups.items()):
        unique_cases = {v["case_id"] for v in rows}
        summary.append({
            "method": method, "budget": budget, "unit": unit,
            "calls": len(rows), "cases": len(unique_cases),
            "errors": sum(v.get("status") != "ok" for v in rows),
            "exact_match": sum(v["exact_match"] for v in rows) / len(rows),
            "argument_f1": sum(v["argument_f1"] for v in rows) / len(rows),
            "abstentions": sum(v["abstained"] for v in rows),
            "usage_reported_calls": sum(bool(v.get("usage")) for v in rows),
            "note": "Call-average descriptive pilot statistics; not an independent-sample significance test.",
        })
    return {"summary": summary, "scored": scored}
