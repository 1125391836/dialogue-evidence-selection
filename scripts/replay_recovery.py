"""Replay the same compressed window before and after the current request."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from context_bundles.core import Budget, Case, Counter
from context_bundles.recovery import HistoryWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("data/processed/mem2act-v1/cases.jsonl"))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--case-id")
    parser.add_argument("--query", help="Override the follow-up request for a single case")
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--read-limit", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("runs/recovery/latest.json"))
    args = parser.parse_args()
    if args.query is not None and not args.case_id:
        parser.error("--query requires --case-id")
    counter = Counter("hf:data/tokenizers/deepseek-v4/tokenizer.json")
    budget = Budget(args.budget, counter)
    traces = []
    with args.cases.open(encoding="utf-8") as source:
        for line in source:
            data = json.loads(line)
            if data["split"] != args.split or (args.case_id and data["id"] != args.case_id):
                continue
            case = Case.from_dict(data)
            before = HistoryWindow.compress(case, budget)
            query = case.query if args.query is None else args.query
            after, trace = before.recover(query, args.top_k, args.read_limit)
            traces.append(dict(trace, case_id=case.id, query=query, messages=after.prompt()))
    if not traces:
        parser.error("No matching cases")
    changed = sum(bool(t["recovered"]) for t in traces)
    maximum = max(t["after_cost"] for t in traces)
    report = dict(cases=str(args.cases), split=args.split, history_limit=args.budget,
                  read_limit=args.read_limit, top_k=args.top_k, tokenizer=counter.name,
                  count=len(traces), changed=changed, max_history_tokens=maximum, traces=traces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent, delete=False) as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
    Path(out.name).replace(args.output)
    print(f"回放 {len(traces)} 个任务，其中 {changed} 个找回了原文。")
    print(f"历史上下文最大用量 {maximum}/{args.budget} tokens。模型调用 0 次。")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
