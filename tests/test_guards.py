import unittest
import os
import subprocess
import sys
from copy import deepcopy

from context_bundles.core import Budget, Case, Counter, Evidence
from context_bundles.guards import guarded_choice, promotion_gate, support_violations
from context_bundles.sensitivity import default_sensitivity, documented_defaults
from context_bundles.utility import FEATURES, training_examples_match


class ReproducibilityTests(unittest.TestCase):
    def test_training_resume_tolerates_roundoff_not_changed_labels_or_features(self):
        a = [{"target": 1, "case_id": "q", "features": [.4] * len(FEATURES)}]
        b = deepcopy(a)
        b[0]["features"][0] += 1e-15
        self.assertTrue(training_examples_match(a, b))
        b[0]["target"] = 0
        self.assertFalse(training_examples_match(a, b))
        b = deepcopy(a)
        b[0]["features"][0] += 1e-6
        self.assertFalse(training_examples_match(a, b))

    def test_bm25_is_identical_across_python_hash_seeds(self):
        code = ("import json; from context_bundles.selection import bm25; "
                "from context_bundles.core import Evidence; "
                "print(json.dumps(bm25('alpha beta gamma delta epsilon', "
                "(Evidence('a','alpha beta gamma gamma delta epsilon',0,'s'),"
                "Evidence('b','alpha alpha beta epsilon',1,'s'))),sort_keys=True))")
        outputs = [subprocess.check_output([sys.executable, "-c", code],
                   env={**os.environ, "PYTHONHASHSEED": seed}) for seed in ["1", "2", "3"]]
        self.assertEqual(len(set(outputs)), 1)


class GuardTests(unittest.TestCase):
    def model(self):
        return {"features": FEATURES, "training_groups": [str(i) for i in range(30)],
                "constant_target": False, "group_cv": [{"mse": .1, "constant_mse": .2}],
                "feature_min": [.2] * len(FEATURES), "feature_max": [.8] * len(FEATURES)}

    def test_gate_fails_closed_and_requires_predictive_gain(self):
        model = self.model()
        self.assertTrue(promotion_gate(model)["allowed_for_guarded_pilot"])
        model["group_cv"][0]["mse"] = .3
        self.assertFalse(promotion_gate(model)["allowed_for_guarded_pilot"])
        model["training_groups"] = ["one"]
        self.assertIn("insufficient_independent_training_groups", promotion_gate(model)["reasons"])
        del model["feature_min"]
        self.assertIn("missing_training_support", promotion_gate(model)["reasons"])

    def test_outside_support_and_empty_proposal_fall_back_within_budget(self):
        case = Case("q", "Alice contact", {"name": "find"},
                    (Evidence("a", "Alice contact C17", 0, "s"), Evidence("b", "unrelated " * 200, 1, "s")), "g", "dev")
        budget = Budget(200, Counter())
        chosen, audit = guarded_choice(case, budget, self.model(), [], [0.] * len(FEATURES))
        self.assertEqual([e.id for e in chosen], ["a"])
        self.assertTrue(budget.fits(chosen))
        self.assertEqual(audit["fallback"], "bm25")
        self.assertIn("outside_training_support", audit["reasons"])
        self.assertEqual(len(support_violations(self.model(), [float("nan")] * len(FEATURES))), len(FEATURES))


class DefaultSensitivityTests(unittest.TestCase):
    def test_only_documented_optional_type_correct_defaults(self):
        schema = {"name": "search", "parameters": {"required": ["text"], "properties": {
            "text": {"type": "string", "default": "never fill"},
            "flag": {"type": "boolean", "description": "The default value is 'false'"},
            "page": {"type": "int", "description": "The default value is 1"},
            "source": {"type": "string", "description": "The default value is 'global', i.e. all"},
            "unknown": {"type": "string", "description": "Usually English"},
            "invalid": {"type": "boolean", "default": 0},
            "ambiguous": {"type": "string", "description": "The default value is 'a'. The default value is 'b'."}}}}
        defaults = documented_defaults(schema)
        self.assertEqual(set(defaults), {"flag", "page", "source"})
        gold = {"name": "search", "arguments": {"text": "term", "flag": False, "page": 1, "source": "global"}}
        outcome = default_sensitivity('{"name":"search","arguments":{"text":"term"}}', gold, schema)
        self.assertEqual(outcome["strict_exact_match"], 0)
        self.assertEqual(outcome["default_aware_exact_match"], 1)
        missing_required = default_sensitivity('{"name":"search","arguments":{}}', gold, schema)
        self.assertEqual(missing_required["default_aware_exact_match"], 0)

    def test_no_gold_inference_no_bool_number_conflation_no_invalid_calls(self):
        schema = {"name": "s", "parameters": {"properties": {"flag": {"type": "boolean", "default": False}}}}
        gold = {"name": "s", "arguments": {"flag": False, "lang": "en"}}
        for text in ['{"name":"s","arguments":{}}', '{"name":"s","arguments":{"flag":0,"lang":"en"}}',
                     '{"abstain":true}', '{"name":"wrong","arguments":{}}', 'not json']:
            self.assertEqual(default_sensitivity(text, gold, schema)["default_aware_exact_match"], 0)


if __name__ == "__main__":
    unittest.main()
