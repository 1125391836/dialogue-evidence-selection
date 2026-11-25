import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import review_benchmarks as study
from context_bundles.core import Case, Evidence


class ReviewBenchmarkTests(unittest.TestCase):
    def test_local_reader_preserves_original_position(self):
        pieces = tuple(Evidence(str(i), f'U: sample {i}', i, 's') for i in range(7))
        case = Case('id','sample 5',{},pieces,'id','test')
        class Matcher:
            def in_turn(self, text):
                return text.split(), [('field',text,0,0)]
        class Vec:
            def transform(self, features):
                return features
        class Classifier:
            def predict_proba(self, features):
                import numpy as np
                return np.array([[0., f['log_distance']] for f in features])
        model = Matcher(), Vec(), Classifier()
        selected = [pieces[1],pieces[5]]
        full = study.ticket.score(case,*model)
        local = study.visible_scores(case,selected,*model)
        self.assertEqual(local,{key:full[key] for key in ('1','5','query')})

    def test_ablations_remove_specified_information(self):
        pieces = tuple(Evidence(str(i), str(i), i, 's') for i in range(3))
        case = Case('id','',{},pieces,'id','test')
        class TwoTurns:
            def fits(self, selected):
                return len(selected) <= 2
        scores = {'0':{('x','a'):.9},'1':{('x','a'):.9},'2':{('y','b'):.4}}
        ids = lambda pieces: [e.id for e in pieces]
        self.assertEqual(ids(study.ticket.evidence(case,scores,TwoTurns(),'field_support')),['1','2'])
        self.assertEqual(ids(study.ablated(case,scores,TwoTurns(),'no_marginal')),['0','1'])
        self.assertEqual(ids(study.ablated(case,scores,TwoTurns(),'tie_oldest')),['0','2'])
