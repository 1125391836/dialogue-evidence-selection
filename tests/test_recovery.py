import unittest

from context_bundles.core import Budget, Case, Counter, Evidence
from context_bundles.recovery import HistoryWindow


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        evidence = (Evidence("a", "order number 1248", 0, "s"),
                    Evidence("b", "meeting on Friday", 1, "s"))
        case = Case("c", "status", {}, evidence, "g", "dev")
        counter = Counter()
        limit = Budget(100, counter).cost([evidence[1]])
        self.window = HistoryWindow.compress(case, Budget(limit, counter))
        self.assertEqual(self.window.visible, (evidence[1],))

    def test_recovery_swaps_and_can_revisit_original(self):
        first, trace = self.window.recover("order number", 1, 100)
        self.assertEqual((trace["recovered"], trace["evicted"]), (["a"], ["b"]))
        self.assertLessEqual(trace["after_cost"], first.budget.limit)
        self.assertIn("1248", first.prompt()[1]["content"])
        second, trace = first.recover("meeting", 1, 100)
        self.assertEqual(second.visible, self.window.visible)
        self.assertEqual(trace["recovered"], ["b"])

    def test_failed_reads_leave_window_intact(self):
        for query, k, cap in [("absent", 1, 100), ("order", 0, 100), ("order", 1, 2)]:
            with self.subTest(query=query, k=k, cap=cap):
                result, trace = self.window.recover(query, k, cap)
                self.assertEqual(result.visible, self.window.visible)
                self.assertEqual(trace["evicted"], [])
                self.assertEqual(trace["read_cost"], 0)
