import json
from pathlib import Path
import tempfile
import unittest

from context_bundles.core import Budget, Case, Counter, Evidence, messages, render_history
from context_bundles.data import prepare
from context_bundles.evaluation import parse_prediction, score, summarize
from context_bundles.experiment import make_plan
from context_bundles.io import read_jsonl, write_jsonl
from context_bundles.probes import make_probes
from context_bundles.search import bundle_beam, exhaustive, synthetic_demo
from context_bundles.selection import select


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.counter = Counter()
        self.evidence = (
            Evidence("a", 'Project Alpha owner is Alice.', 0, "s1"),
            Evidence("b", 'Alice has contact ID C17.', 1, "s1"),
            Evidence("c", 'Other notes ' * 100, 2, "s1"),
        )
        self.case = Case("q", "Contact Project Alpha owner", {}, self.evidence, "g", "dev")

    def test_serialization_overhead_and_long_block(self):
        cost = self.counter(render_history(list(self.evidence[:2])))
        for method in ("recent", "bm25", "rule_pairs"):
            selected = select(self.case, method, Budget(cost, self.counter))
            self.assertLessEqual(self.counter(render_history(selected)), cost)
            self.assertNotIn("c", [e.id for e in selected])
            self.assertEqual(len(selected), len({e.id for e in selected}))

    def test_history_in_prompt_is_same_serialization(self):
        selected = list(self.evidence[:2])
        self.assertTrue(messages(self.case, selected)[1]["content"].endswith(render_history(selected)))

    def test_full_reference_may_exceed_budget(self):
        budget = Budget(2, self.counter)
        self.assertEqual(select(self.case, "bm25", budget), [])
        self.assertGreater(budget.cost(select(self.case, "full", budget)), budget.limit)


class ScoringTests(unittest.TestCase):
    gold = {"name": "notify", "arguments": {"id": "C17", "urgent": True}}

    def test_exact_and_extra_arguments(self):
        self.assertEqual(score(json.dumps(self.gold), self.gold)["exact_match"], 1)
        bad = {"name": "notify", "arguments": {"id": "C17", "urgent": True, "extra": 3}}
        result = score(json.dumps(bad), self.gold)
        self.assertEqual(result["exact_match"], 0)
        self.assertAlmostEqual(result["argument_f1"], 0.8)

    def test_bool_is_not_integer_and_duplicate_keys_fail(self):
        bad = '{"name":"notify","arguments":{"id":"C17","urgent":1}}'
        self.assertEqual(score(bad, self.gold)["exact_match"], 0)
        with self.assertRaises(ValueError):
            parse_prediction('{"name":"x","name":"notify","arguments":{}}')
        with self.assertRaises(ValueError):
            parse_prediction('{"name":"x","arguments":{"n":NaN}}')

    def test_invalid_abstention_and_api_errors_are_not_dropped(self):
        self.assertTrue(score('{"abstain":true}', self.gold)["abstained"])
        self.assertEqual(score("not json", self.gold)["exact_match"], 0)
        labels = [{"id": "q", "group_id": "g", "gold": self.gold}]
        predictions = [{"request_id": "r", "case_id": "q", "method": "bm25", "budget": 100,
                        "unit": "bytes", "status": "api_error", "output": json.dumps(self.gold)}]
        report = summarize(predictions, labels)
        self.assertEqual(report["summary"][0]["errors"], 1)
        self.assertEqual(report["summary"][0]["exact_match"], 0)
        with self.assertRaises(ValueError):
            summarize(predictions * 2, labels)


class DatasetTests(unittest.TestCase):
    def fixtures(self, root):
        sessions = [
            {"session_id": "s1", "original_conversation_ids": ["x", "incidental"], "turns": [
                {"role": "user", "content": "Project Alpha owner is Alice.", "source_id": "x"},
                {"role": "assistant", "content": "Recorded."},
                {"role": "user", "content": "Alice has contact ID C17.", "source_id": "incidental"}]},
            {"session_id": "s2", "original_conversation_ids": ["y", "incidental"], "turns": [
                {"role": "user", "content": "Unrelated history."}]},
        ]
        qas = [{"qa_id": f"q{i}", "source_conversation_ids": [src], "query": "Contact Project Alpha owner",
                "target_tool_schema": {"name": "notify"},
                "tool_call": {"name": "notify", "arguments": {"id": "GOLD_CANARY"},
                              "grounding_info": {"id": "GOLD_ANNOTATION"}},
                "evolution_chain": "GOLD_CHAIN", "complexity_metadata": {"level": "L4"}}
               for i, src in enumerate(["x", "y"])]
        write_jsonl(root / "qa.jsonl", qas)
        write_jsonl(root / "sessions.jsonl", sessions)
        return qas, sessions

    def test_incidental_history_groups_and_label_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixtures(root)
            audit = prepare(root / "qa.jsonl", root / "sessions.jsonl", root / "prepared")
            cases = read_jsonl(root / "prepared/cases.jsonl")
            self.assertEqual(audit["independent_groups"], 1)
            self.assertEqual(cases[0]["split"], cases[1]["split"])
            text = (root / "prepared/cases.jsonl").read_text()
            self.assertNotIn("GOLD_", text)
            self.assertNotIn("grounding_info", text)
            self.assertNotIn("source_id", text)
            self.assertIn("Alice has contact ID C17", text)
            self.assertEqual(len(cases[0]["evidence"]), 2)

    def test_missing_source_fails_instead_of_inventing_history(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qas, _ = self.fixtures(root)
            qas[0]["source_conversation_ids"] = ["absent"]
            write_jsonl(root / "qa.jsonl", qas)
            with self.assertRaises(ValueError):
                prepare(root / "qa.jsonl", root / "sessions.jsonl", root / "prepared")

    def test_empty_source_requires_explicit_quarantine(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qas, _ = self.fixtures(root)
            qas[0]["source_conversation_ids"] = []
            write_jsonl(root / "qa.jsonl", qas)
            with self.assertRaises(ValueError):
                prepare(root / "qa.jsonl", root / "sessions.jsonl", root / "strict")
            audit = prepare(root / "qa.jsonl", root / "sessions.jsonl", root / "quarantined",
                            quarantine_empty_sources=True)
            self.assertEqual(audit["cases"], 1)
            self.assertEqual(audit["excluded_records"], [{"id": "q0", "reason": "empty_source_conversation_ids"}])

    def test_plan_and_probe_fixed_background(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.fixtures(root)
            prepare(root / "qa.jsonl", root / "sessions.jsonl", root / "prepared")
            cases = read_jsonl(root / "prepared/cases.jsonl")
            for c in cases:
                c["split"] = "dev"  # Test fixture only; production split is never rewritten.
            write_jsonl(root / "cases.jsonl", cases)
            config = {"encoding": "utf8_bytes", "split": "dev", "seed": 42,
                      "max_cases": 2, "history_budgets": [200, 500],
                      "methods": ["bm25", "full"], "repeats": 2}
            manifest = make_plan(root / "cases.jsonl", config, root / "plan")
            self.assertEqual(manifest["requests"], 12)
            self.assertNotIn("GOLD_", (root / "plan/requests.jsonl").read_text())
            with self.assertRaises(FileExistsError):
                make_plan(root / "cases.jsonl", config, root / "plan")
            pc = {"encoding": "utf8_bytes", "split": "dev", "seed": 42,
                  "max_cases": 2, "history_budget": 2000, "pairs_per_case": 1, "repeats": 2}
            make_probes(root / "cases.jsonl", pc, root / "probes")
            rows = read_jsonl(root / "probes/requests.jsonl")
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({tuple(r["base_ids"]) for r in rows}), 1)
            for r in rows:
                self.assertLessEqual(r["history_cost"], r["budget"])
                additions = set(r["selected_ids"]) - set(r["base_ids"])
                self.assertEqual(len(additions), {"base": 0, "a": 1, "b": 1, "ab": 2}[r["condition"]])


class SearchTests(unittest.TestCase):
    def test_synthetic_gap_and_search_limit(self):
        result = synthetic_demo()
        self.assertEqual(result["greedy"]["utility"], 0.1)
        self.assertEqual(result["bundle_beam"]["utility"], 1.0)
        self.assertEqual(result["exhaustive"]["utility"], 1.0)
        selected, calls = bundle_beam(["a", "b"], [("a", "b")], lambda s: len(s) <= 2,
                                      lambda s: float(len(s)), max_evaluations=2)
        self.assertLessEqual(calls, 2)
        self.assertLessEqual(len(selected), 2)

    def test_invalid_oracle_scope_and_bundle(self):
        with self.assertRaises(ValueError):
            exhaustive([str(i) for i in range(21)], lambda s: True, lambda s: 0)
        with self.assertRaises(ValueError):
            bundle_beam(["a"], [("missing",)], lambda s: True, lambda s: 0)


if __name__ == "__main__":
    unittest.main()
