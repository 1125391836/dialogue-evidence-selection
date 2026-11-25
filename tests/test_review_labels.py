import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import review_labels as audit
from ticket_execution import render


class ReviewLabelsTests(unittest.TestCase):
    def test_clock_parser_preserves_meridiem_uncertainty(self):
        for text, minute in [('12:00AM', 0), ('12:00 p.m.', 720), ('7.30 pm', 1170),
                             ('21:10', 1270), ('00:05', 5), ('7pm', 1140), ('8:30', None),
                             ('12:01', None), ('27:00', None), ('9:77pm', None)]:
            with self.subTest(text=text):
                self.assertEqual(audit.gold_time(text), minute)

    def test_current_reply_follows_chronological_history(self):
        raw = '<U>Yes!<C><U>I want tickets<A>So, 7:30pm, is that correct?'
        turns = audit.chronological_turns(raw)
        self.assertEqual([r['role'] for r in turns], ['U', 'A', 'U'])
        self.assertEqual(turns[-1]['text'], 'Yes!')
        self.assertEqual(audit.current_confirmation(render(raw)), (1170, 1))
        changed = raw.replace('<U>Yes!', '<U>Yes, but change the time to 8pm')
        self.assertEqual(audit.current_confirmation(render(changed)), (None, None))

    def test_rescoring_retains_commit_and_other_fields(self):
        gold = {'name.movie':'x', 'time.showing':'4:30pm'}
        pred = {'name.movie':'x', 'time.showing':'9:10pm'}
        outcomes = [('wrong',pred,'correct'),('no_write',pred,'no_write'),
                    ('correct',dict(pred, **{'name.movie':'y'}),'wrong'),
                    ('wrong',dict(pred, **{'time.showing':'9:10'}),'indeterminate'),
                    ('wrong',dict(pred, **{'time.showing':'4:30'}),'wrong')]
        for status, proposal, expected in outcomes:
            with self.subTest(proposal=proposal):
                self.assertEqual(audit.normalized_outcome(status, proposal, gold, 1270), expected)

    def test_identification_interval_contains_observed_bounds(self):
        left = audit.wilson(7, 48, .025)[0]
        right = audit.wilson(10, 48, .025)[1]
        self.assertLess(left, 7/48)
        self.assertGreater(right, 10/48)
        self.assertEqual(audit.wilson(0, 0), [None, None])

