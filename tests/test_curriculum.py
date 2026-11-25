from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from context_bundles.curriculum import build_plan, sample_groups, validate_plan
from context_bundles.curriculum_run import execute_plan, fit_plan, stage_name
from context_bundles.io import read_jsonl, write_json, write_jsonl
from context_bundles.ledger import Ledger


class FakeEvidenceClient:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, max_tokens=None):
        self.calls += 1
        value = "C17" if "Alice contact is C17" in messages[-1]["content"] else "missing"
        return {"status": "ok", "finish_reason": "stop", "output": json.dumps({"name": "find", "arguments": {"id": value}}),
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class CurriculumTests(unittest.TestCase):
    def fixture(self, root):
        rows, labels = [], []
        for split, n in [("train", 2), ("dev", 1), ("test", 1)]:
            for index in range(n):
                cid = f"{split}{index}"
                rows.append({"id": cid, "group_id": cid, "split": split,
                    "query": "Find the Alpha project owner contact",
                    "tool_schema": {"name": "find", "parameters": {"properties": {"id": {"type": "string"}}}},
                    "evidence": [{"id": f"e{i}", "text": text, "order": i, "session_id": cid} for i, text in enumerate([
                        "Alpha owner is Alice.", "Alice contact is C17.", "Beta owner is Bob.",
                        "Bob contact is C18.", "A short unrelated fact.", "Other historical context. " * 25])],
                    "hidden_gold_marker": "NEVER_COPY_THIS"})
                labels.append({"id": cid, "group_id": cid, "split": split,
                               "gold": {"name": "find", "arguments": {"id": "C17"}}})
        write_jsonl(root / "cases.jsonl", rows)
        write_jsonl(root / "labels.jsonl", labels)
        write_json(root / "audit.json", {"engineering_only": False})
        config = json.loads(Path("configs/curriculum-v2.json").read_text())
        config.update({"cases_path": str(root / "cases.jsonl"), "groups": {"train": 2, "dev": 1},
                       "encoding": "utf8_bytes", "history_budget": 600, "sample_budgets": [140, 300, 600],
                       "max_prompt_tokens": 5000})
        return config

    def test_coverage_deduplication_determinism_and_no_answer_access(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self.fixture(root)
            # Planner cannot depend on labels: remove the fixture's answer file.
            (root / "labels.jsonl").unlink()
            first = build_plan(config, root / "p1")
            second = build_plan(config, root / "p2")
            self.assertEqual(first, second)
            self.assertEqual(first, build_plan(config, root / "p1"))
            rows = read_jsonl(root / "p1/requests.jsonl")
            self.assertEqual(first["empty_contexts"], 3)
            self.assertEqual({r["split"] for r in rows}, {"train", "dev"})
            self.assertNotIn("NEVER_COPY_THIS", json.dumps(rows))
            unique = {(r["case_id"], tuple(r["selected_ids"]), r["repeat"]) for r in rows}
            self.assertEqual(len(rows), len(unique))
            self.assertTrue(any(len(r["roles"]) > 1 for r in rows))
            self.assertTrue(any(not r["selected_ids"] for r in rows))
            self.assertTrue(any(len(r["selected_ids"]) == 1 for r in rows))
            self.assertTrue(any(r["history_cost"] > 600 for r in rows))  # full reference, not fitting data
            validate_plan(root / "p1")

    def test_freeze_rejects_tampering_configuration_changes_and_test_sampling(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self.fixture(root)
            build_plan(config, root / "p")
            changed = deepcopy(config); changed["seed"] += 1
            with self.assertRaises(ValueError):
                build_plan(changed, root / "p")
            with self.assertRaises(ValueError):
                sample_groups({}, {"train": 1, "test": 1}, 42)
            with (root / "p/requests.jsonl").open("a", encoding="utf-8") as stream:
                stream.write('{}\n')
            with self.assertRaises(ValueError):
                validate_plan(root / "p")

    def test_missing_pair_is_retained_and_not_fabricated(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self.fixture(root)
            cases = read_jsonl(root / "cases.jsonl")
            for case in cases:
                case["evidence"] = [{"id": "a", "text": "plain unrelated words", "order": 0, "session_id": case["id"]}]
            write_jsonl(root / "cases.jsonl", cases)
            manifest = build_plan(config, root / "p")
            self.assertEqual(manifest["pairs"], 0)
            self.assertEqual(manifest["empty_contexts"], 3)
            exclusions = json.loads((root / "p/exclusions.json").read_text())
            self.assertEqual(sum(e["reason"] == "no_feasible_lexical_pair" for e in exclusions), 3)

    def test_insufficient_quota_sends_nothing_and_does_not_create_jobs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self.fixture(root)
            build_plan(config, root / "p")
            ledger = Ledger(root / "ledger.sqlite3", 2)
            fake = FakeEvidenceClient()
            with self.assertRaises(ValueError):
                execute_plan(root / "p", ledger.path, 2, fake)
            self.assertEqual(fake.calls, 0)
            self.assertEqual(ledger.stats()["reserved_calls"], 0)
            self.assertEqual(ledger.rows(), [])
            with self.assertRaises(ValueError):
                execute_plan(root / "p", root / "missing.sqlite3", 2, fake)
            self.assertFalse((root / "missing.sqlite3").exists())

    @patch("builtins.print")
    def test_failed_repeats_are_not_silently_averaged_or_omitted(self, _):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self.fixture(root)
            manifest = build_plan(config, root / "p")
            rows = read_jsonl(root / "p/requests.jsonl")
            train_contexts = sorted({r["coalition_id"] for r in rows if r["split"] == "train" and any(role["family"] != "reference_full" for role in r["roles"])})
            all_failed, partly_failed = train_contexts[:2]
            ledger = Ledger(root / "ledger.sqlite3", manifest["requests"])
            fake = FakeEvidenceClient()
            for spec in rows:
                key = ledger.register(stage_name(manifest), spec)
                self.assertTrue(ledger.claim(key))
                fail = spec["coalition_id"] == all_failed or (spec["coalition_id"] == partly_failed and spec["repeat"] == 0)
                result = {"status": "api_error", "output": "", "usage": {}} if fail else fake.complete(spec["messages"])
                ledger.finish(key, result)
            result = fit_plan(root / "p", ledger.path, root / "labels.jsonl", root / "fit")
            self.assertEqual(set(result["incomplete_contexts"]), {all_failed, partly_failed})
            self.assertEqual(len(result["excluded_calls"]), 3)
            examples = read_jsonl(root / "fit/training_examples.jsonl")
            self.assertFalse({all_failed, partly_failed} & {e["coalition_id"] for e in examples})

    @patch("builtins.print")
    def test_complete_execution_resume_and_train_dev_isolation(self, _):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = self.fixture(root)
            manifest = build_plan(config, root / "p")
            ledger = Ledger(root / "ledger.sqlite3", manifest["requests"])
            fake = FakeEvidenceClient()
            with self.assertRaises(ValueError):
                fit_plan(root / "p", ledger.path, root / "labels.jsonl", root / "fit")
            execute_plan(root / "p", ledger.path, manifest["requests"], fake)
            execute_plan(root / "p", ledger.path, 0, fake)
            self.assertEqual(fake.calls, manifest["requests"])
            result = fit_plan(root / "p", ledger.path, root / "labels.jsonl", root / "fit")
            examples = read_jsonl(root / "fit/training_examples.jsonl")
            self.assertTrue(all(e["split"] == "train" and e["repeat_count"] == 2 for e in examples))
            self.assertTrue(any(not e["selected_ids"] for e in examples))
            self.assertEqual(result["training_groups"], 2)
            self.assertEqual(result["development_groups"], 1)
            labels = read_jsonl(root / "labels.jsonl")
            for label in labels:
                if label["split"] == "dev":
                    label["gold"]["arguments"]["id"] = "different development gold"
            write_jsonl(root / "changed-labels.jsonl", labels)
            fit_plan(root / "p", ledger.path, root / "changed-labels.jsonl", root / "fit2")
            one = json.loads((root / "fit/utility_model.json").read_text())
            two = json.loads((root / "fit2/utility_model.json").read_text())
            self.assertEqual(one["coefficient"], two["coefficient"])
            self.assertEqual(one["intercept"], two["intercept"])


if __name__ == "__main__":
    unittest.main()
