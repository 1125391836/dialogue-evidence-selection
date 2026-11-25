import csv
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import review_human as workflow


class HumanAuditTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.dialogues = {k:[{"role":"U","text":"7pm please"}] for k in ("X","Y","Z")}
        gold = {"time.showing":"6pm","movie":"m"}
        self.cases = {k:("test",dict(case=k,gold=gold,conditions={
            "recent":{"args":gold,"outcome":"no_write"},
            "field_support":{"args":dict(gold, **{"time.showing":"7pm"}),"outcome":"wrong"}
        })) for k in self.dialogues}
        self.run_stage()

    def run_stage(self):
        return workflow.collect(self.root,self.dialogues,self.cases,{"fixture":True},b"blind instructions")

    def filled(self, labels, letter="A"):
        buf = io.StringIO()
        writer = csv.DictWriter(buf,fieldnames=[*workflow.audit.FORM, "packet_id"])
        writer.writeheader()
        for key, value in labels.items():
            dialogues = {k:self.dialogues[k] for k in labels}
            row = {"audit_id":key, "packet_id":workflow.packet_id(dialogues, letter)}
            if value:
                row.update(status="unresolved" if value=="?" else "resolved",
                           time_24h="" if value=="?" else value, evidence_turns="0")
            writer.writerow(row)
        return buf.getvalue().encode("utf-8-sig")

    def put(self, letter, vals):
        (self.root/"returns"/f"reviewer_{letter}.csv").write_bytes(self.filled(dict(zip(self.dialogues,vals)), letter))

    def test_dispute_replay_and_changed_forms(self):
        self.put("A",["19:00","?","19:00"])
        self.put("B",["20:00","?",None])
        self.assertEqual(self.run_stage()["completed"], {"A":3,"B":2})
        self.assertFalse((self.root/"rounds").exists())
        self.put("B",["20:00","?","19:00"])
        stage = self.run_stage()
        where = Path(stage["directory"])
        with zipfile.ZipFile(where/"reviewer_C.zip") as packet:
            self.assertEqual(set(packet.namelist()),{"reviewer_C.csv","README.md","dialogues.md"})
            self.assertEqual(packet.read("reviewer_C.csv"),workflow.blank_form({"X":self.dialogues["X"]},"C"))
            self.assertNotIn(b"## Y",packet.read("dialogues.md"))
        self.assertEqual(json.loads((where/"agreement.json").read_bytes())["agreed"],2)
        third = where/"reviewer_C.csv"
        third.write_bytes(self.filled({"X":"19:00"},"C"))
        result = Path(self.run_stage()["result"])
        scored = json.loads(result.read_bytes())
        self.assertEqual(scored["counts"],{"conflicts":2,"human_unresolved":1})
        self.assertEqual(scored["orders"]["test"]["field_support"]["human_time"],{"correct":2,"indeterminate":1})
        self.assertEqual(scored["orders"]["test"]["recent"]["human_time"],{"no_write":3})
        self.assertEqual(Path(self.run_stage()["result"]),result)
        third.write_bytes(self.filled({"X":"?"},"C"))
        self.assertNotEqual(Path(self.run_stage()["result"]), result)
        self.put("A",["20:00","?","19:00"])
        final = Path(self.run_stage()["result"])
        self.assertFalse((final.parent/"reviewer_C.zip").exists())
        self.assertTrue(result.exists())
        self.assertEqual(json.loads(result.read_bytes()),scored)

    def test_form_rejects_wrong_scope_and_invalid_evidence(self):
        valid = self.filled({"X":"19:00","Y":"?","Z":"19:00"})
        for data in [self.filled({"X":"19:00"}), valid.replace(b",0,",b",1,"),
                     valid.replace(b"19:00",b"25:00"), valid.replace(b"Y,unresolved",b"X,unresolved")]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                workflow.form(data,self.dialogues,"A")
        with self.assertRaises(ValueError):
            workflow.collect(self.root,self.dialogues,self.cases,{"changed":True},b"blind instructions")

    def test_unresolved_other_field_error_and_degenerate_kappa(self):
        ev = {"outcome":"wrong","args":{"time.showing":"7pm","movie":"other"}}
        self.assertEqual(workflow.label_outcome(ev,{"movie":"m"},None),"wrong")
        self.assertIsNone(workflow.agreement({"X":"unresolved"},{"X":"unresolved"})["cohen_kappa"])
