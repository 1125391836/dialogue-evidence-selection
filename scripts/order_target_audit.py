"""Offline order-ID predictions at the first order mutation in tau-bench retail."""
import json
import re
from collections import Counter
from pathlib import Path
from context_bundles.core import Budget, Case, Counter as TokenCounter, Evidence
from context_bundles.selection import pack, select

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/tau-bench-main/historical_trajectories/gpt-4o-retail.json"
OUT = ROOT / "runs/order-target-audit"
ORDER = re.compile(r"\bW\d{7}\b", re.I)


def is_order_write(name):
    return "order" in name and name.startswith(("cancel_", "modify_", "return_", "exchange_"))


def predict(user_messages, budget):
    mentions = [order.upper() for message in user_messages for order in ORDER.findall(message)]
    recent = ORDER.findall(user_messages[-1]) if user_messages else []
    predictions = {
        "latest_turn_unique": recent[0].upper() if len(set(x.upper() for x in recent)) == 1 else None,
        "last_mention": mentions[-1] if mentions else None,
        "history_unique": mentions[0] if len(set(mentions)) == 1 else None,
    }
    prior = tuple(Evidence(str(i), msg, i, "conversation") for i, msg in enumerate(user_messages[:-1]))
    current = user_messages[-1] if user_messages else ""
    case = Case("public", current, {}, prior, "public", "external")
    choices = {
        "bm25_128": select(case, "bm25", budget),
        "id_first_128": pack(sorted(prior, key=lambda e: (bool(ORDER.search(e.text)), e.order),
                                    reverse=True), budget),
    }
    for name, selected in choices.items():
        candidates = {order.upper() for msg in [*(e.text for e in selected), current]
                      for order in ORDER.findall(msg)}
        predictions[name] = next(iter(candidates)) if len(candidates) == 1 else None
    return predictions


def main():
    records = json.loads(SOURCE.read_text(encoding="utf8"))
    rows = []
    seen = set()
    budget = Budget(128, TokenCounter("hf:" + str(ROOT / "data/tokenizers/deepseek-v4/tokenizer.json")))
    for record in records:
        if record["trial"] != 0:
            continue
        task = record["task_id"]
        assert task not in seen
        seen.add(task)
        user_messages = []
        for index, msg in enumerate(record["traj"]):
            if any(is_order_write(call["function"]["name"]) for call in msg.get("tool_calls") or []):
                break
            if msg["role"] == "user":
                user_messages.append(msg["content"])
        else:
            rows.append(dict(task=task, status="no_observed_write"))
            continue
        predictions = predict(user_messages, budget)
        reference = {action["kwargs"]["order_id"].upper().lstrip("#")
                     for action in record["info"]["task"]["actions"]
                     if is_order_write(action["name"]) and "order_id" in action["kwargs"]}
        if len(reference) != 1:
            rows.append(dict(task=task, status="no_unique_reference", targets=sorted(reference)))
            continue
        target = reference.pop()
        outcomes = {method: ("abstain" if guess is None else "correct" if guess == target else "wrong")
                    for method, guess in predictions.items()}
        rows.append(dict(task=task, status="scored", cutoff=index, user_messages=user_messages,
                         target=target, predictions=predictions, outcomes=outcomes))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "details.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
    scored = [row for row in rows if row["status"] == "scored"]
    summary = dict(source=SOURCE.name, trial=0, tasks=len(seen), coverage=Counter(r["status"] for r in rows),
                   results={method: Counter(r["outcomes"][method] for r in scored) for method in predict([], budget)})
    payload = json.dumps(summary, indent=2)
    (OUT / "summary.json").write_text(payload + "\n", encoding="utf8")
    print(payload)


if __name__ == "__main__":
    main()
