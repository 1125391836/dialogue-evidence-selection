import io
import sys
import unittest
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from context_bundles.core import Budget, Case, Evidence
import decision_blind as blind
import decision_reader as reader
import decision_selection as selector


class DecisionTests(unittest.TestCase):
    def test_only_current_confirmation_included_once(self):
        hist=(Evidence("0","U: 5 pm",0,"x"),Evidence("1","A: 7 pm instead?",1,"x"))
        case=Case("x","Yes",{},hist,"x","test")
        for arm in ("adjacent","decision","recent"):
            got=selector.choose(case,Budget(500,len),arm)
            self.assertEqual(got,list(hist))

    def test_all_three_blinded_versions_use_distinct_readers(self):
        import review_human as human
        dialogues,cases,_=human.source()
        displays,rows=blind.build(dialogues,cases)
        self.assertEqual(len(rows),144)
        self.assertEqual(set(Counter((r["reader"],r["arm"]) for r in rows).values()),{16})
        self.assertEqual(len({(r["audit_id"],r["reader"]) for r in rows}),144)
        self.assertEqual(len({(r["audit_id"],r["arm"]) for r in rows}),144)
        for letter,display in displays.items():
            with zipfile.ZipFile(io.BytesIO(human.packet(display,letter,blind.INSTRUCTIONS.encode()))) as z:
                self.assertEqual(len(z.namelist()),3)
                blank=human.form(z.read(f"reviewer_{letter}.csv"),display,letter)
                self.assertEqual(len(blank),48)
                self.assertEqual(set(blank.values()),{None})

    def test_attempt_and_submission_requirements(self):
        required={"city","time"}
        gold={"city":"x","time":"19:00","party_size":"2"}
        r=reader.classify('{"city":"x","time":"19:00"}',gold,required,{"party_size":"2"})
        self.assertTrue(r["default_exact"] and r["required_exact"])
        self.assertEqual(r["outcome"],"wrong")
        self.assertEqual(reader.classify('{"city":"x"}',gold,required,{})["outcome"],"no_write")
        self.assertTrue(reader.classify('{"city":null}',gold,required,{})["malformed"])
