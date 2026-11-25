from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from context_bundles.campaign import Campaign
from context_bundles.core import Budget, Case, Counter, Evidence
from context_bundles.deepseek import DeepSeekClient, cost_interval
from context_bundles.io import read_jsonl, write_json, write_jsonl
from context_bundles.ledger import Ledger
from context_bundles.statistics import paired_cluster_bootstrap
from context_bundles.utility import FEATURES, SetFeatures, predict, train_predictor


class LedgerTests(unittest.TestCase):
    def test_concurrent_cap_resume_and_uncertain_calls(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(Path(td) / "ledger.sqlite3", 4)
            keys = [ledger.register("s", {"i": i}) for i in range(10)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                claimed = list(pool.map(ledger.claim, keys))
            self.assertEqual(sum(claimed), 4)
            self.assertEqual(ledger.stats()["reserved_calls"], 4)
            for key, yes in zip(keys, claimed):
                if yes:
                    self.assertFalse(ledger.claim(key))
            first = next(k for k, yes in zip(keys, claimed) if yes)
            ledger.finish(first, {"status": "ok", "usage": {}})
            resumed = Ledger(Path(td) / "ledger.sqlite3", 4)
            self.assertEqual(resumed.stats()["reserved_calls"], 4)
            self.assertFalse(resumed.claim(first))
            with self.assertRaises(ValueError):
                Ledger(Path(td) / "ledger.sqlite3", 5)


class UtilityTests(unittest.TestCase):
    def test_feature_invariance_and_train_only_guard(self):
        case = Case("q", "Find Alice account", {"parameters": {"properties": {"account_id": {}}}},
                    (Evidence("a", "Alice is owner", 0, "s"), Evidence("b", "Alice account C17", 1, "s")), "g", "train")
        extractor = SetFeatures(case, Budget(2000, Counter()))
        features = extractor(frozenset(["a", "b"]))
        self.assertEqual(features, extractor(frozenset(["b", "a"])))
        self.assertEqual(len(features), len(FEATURES))
        examples = [{"case_id": f"q{i}", "group_id": f"g{i // 2}", "split": "train",
                     "features": [float(i % 2)] * len(FEATURES), "target": float(i % 2)} for i in range(8)]
        model = train_predictor(examples)
        self.assertEqual(model["training_rows"], 8)
        self.assertGreaterEqual(predict(model, features), 0)
        self.assertLessEqual(predict(model, features), 1)
        examples[0]["split"] = "dev"
        with self.assertRaises(ValueError):
            train_predictor(examples)

    def test_paired_statistics_use_shared_cases_and_groups(self):
        rows = [{"case_id": "q1", "group_id": "g", "method": m, "exact_match": v}
                for m, v in [("a", 1), ("b", 0)]]
        rows.append({"case_id": "unpaired", "group_id": "g2", "method": "a", "exact_match": 1})
        stats = paired_cluster_bootstrap(rows, "a", "b", repetitions=20)
        self.assertEqual(stats["paired_cases"], 1)
        self.assertEqual(stats["independent_groups"], 1)
        self.assertEqual(stats["mean_difference"], 1)
        self.assertEqual(stats["missing_cases"], ["unpaired"])


class DeepSeekTests(unittest.TestCase):
    def test_credential_destination_and_explicit_payload(self):
        for url in ["https://api.deepseek.com.evil.invalid", "http://api.deepseek.com", "https://key@api.deepseek.com"]:
            with self.assertRaises(ValueError):
                DeepSeekClient({"base_url": url, "model": "x"}, key="test-only")
        client = DeepSeekClient({"base_url": "https://api.deepseek.com", "model": "deepseek-flash"}, key="test-only")
        payload = client.payload([{"role": "user", "content": "JSON"}])
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertNotIn("test-only", json.dumps(payload))

    def test_cost_missing_not_zero(self):
        price = {"input_hit_low": .003, "input_hit_high": .006, "input_miss_low": .15,
                 "input_miss_high": .3, "output_low": .6, "output_high": 1.2}
        self.assertIsNone(cost_interval({}, price)["usd_high"])
        result = cost_interval({"prompt_tokens": 100, "completion_tokens": 50}, price)
        self.assertLess(result["usd_low"], result["usd_high"])


class FakeClient:
    def __init__(self):self.calls = 0
    def complete(self, messages, max_tokens=None):
        self.calls += 1
        if "ready" in messages[0]["content"]:
            text = '{"ready":true}'
        elif "Compress" in messages[0]["content"]:
            text = '{"summary":"Alice owns Project Alpha. Alice has contact ID C17."}'
        else:
            text = '{"name":"notify","arguments":{"id":"C17"}}'
        return {"status": "ok", "output": text, "finish_reason": "stop", "returned_model": "fake",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class CampaignTests(unittest.TestCase):
    @patch("builtins.print")
    def test_all_stages_resume_without_extra_calls_and_report(self, mock_print):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cases, labels = [], []
            for split, count in [("train", 5), ("dev", 3)]:
                for n in range(count):
                    cid = f"{split}{n}"
                    evidence = [{"id": f"{cid}{i}", "text": text, "order": i, "session_id": cid} for i, text in enumerate([
                        "Project Alpha owner is Alice.", "Alice contact ID is C17.",
                        "Other person Bob has ID C18.", "Bob works on Project Beta."])]
                    cases.append({"id": cid, "query": "Contact Project Alpha owner", "tool_schema": {"name": "notify"},
                                  "evidence": evidence, "group_id": cid, "split": split})
                    labels.append({"id": cid, "group_id": cid, "split": split,
                                   "gold": {"name": "notify", "arguments": {"id": "C17"}}})
            write_jsonl(root / "cases.jsonl", cases)
            write_jsonl(root / "labels.jsonl", labels)
            write_json(root / "audit.json", {"engineering_only": False})
            config = json.loads(Path("configs/deepseek-60.json").read_text())
            config.update({"cases_path": str(root / "cases.jsonl"), "labels_path": str(root / "labels.jsonl"),
                           "encoding": "utf8_bytes", "history_budget": 300})
            write_json(root / "config.json", config)
            campaign = Campaign(root / "config.json", root / "run")
            fake = FakeClient();campaign._client = fake
            campaign.run("all")
            self.assertEqual(campaign.ledger.stats()["reserved_calls"], 58)
            self.assertEqual(fake.calls, 58)
            campaign.run("all")
            self.assertEqual(fake.calls, 58)
            model = json.loads((root / "run/utility_model.json").read_text())
            self.assertEqual(model["training_rows"], 25)
            self.assertTrue(all(g.startswith("train") for g in model["training_groups"]))
            report = json.loads((root / "run/report/report.json").read_text())
            self.assertEqual(report["status"]["reserved_calls"], 58)
            self.assertEqual(len(report["methods"]), 7)
            self.assertEqual(report["costs"]["missing_usage_calls"], 0)
            config["call_cap"] = 59
            write_json(root / "changed.json", config)
            with self.assertRaises(ValueError):Campaign(root / "changed.json", root / "run")


if __name__ == "__main__":
    unittest.main()
