"""Prepare corrected TicketTalk blind packets and score independent human returns."""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import review_labels as audit
from context_bundles.io import read_jsonl, sha256

ROOT = audit.OUT / "human-v2"
COHORT = audit.study.OUT
METHODS = ("recent", "field_support")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def preserve(path, data):
    """Publish the complete file once, checking byte identity on replay."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(data)
        try:
            try:
                os.link(tmp, path)
            except FileExistsError:
                pass
        finally:
            tmp.unlink()
    if path.read_bytes() != data:
        raise ValueError(f"Frozen file differs: {path}")


def source():
    lock = json.loads((audit.OUT / "blind-lock.json").read_bytes())
    old_file = audit.OUT / "blinded/dialogues.jsonl"
    if sha256(old_file) != lock["dialogues_sha256"] or sha256(audit.OUT/"protocol.json") != lock["protocol_sha256"]:
        raise ValueError("Original audit lock mismatch")
    old = {r["audit_id"]:r["turns"] for r in read_jsonl(old_file)}
    links = read_jsonl(audit.OUT / "linkage.jsonl")
    original = {r["case"]:name for name, _, _ in audit.PARTS
                for r in read_jsonl(audit.OLD / (name+"-details.jsonl"))}
    sampled = sorted(original, key=lambda k:digest(("review-label-v1-20261123|"+k).encode()))[:48]
    expected = [dict(audit_id=f"Q{i:02d}",case=k,split=original[k]) for i,k in enumerate(sampled,1)]
    if links != expected or {r["audit_id"] for r in links} != old.keys() or len(original) != 1391:
        raise ValueError("Sample differs from the original 48-of-1391 draw")

    dialogues, cases, changes, inputs = {}, {}, {}, {}
    for split, source_name, parts in audit.PARTS:
        path = COHORT / (split+"-details.jsonl")
        inputs[split] = sha256(path)
        rows = read_jsonl(path)
        results = {r["case"]:r for r in rows}
        if len(results) != len(rows):
            raise ValueError("Duplicate evaluation case")
        selected = {r["case"]:r["audit_id"] for r in links if r["split"] == split}
        for part in parts:
            for key, raw, gold in audit.study.examples(source_name, part):
                if key not in selected:
                    continue
                qid = selected[key]
                case = audit.study.render(raw)
                if case is None or key not in results or qid in cases:
                    continue
                turns = audit.chronological_turns(raw)
                public = [*(e.text for e in case.evidence), case.query]
                fingerprint = digest(" ".join(" ".join(public).casefold().split()).encode())
                row = results[key]
                if fingerprint != row["fingerprint"] or gold != row["gold"]:
                    continue
                if public != [*(t["role"]+": "+t["text"] for t in turns[:-1]), turns[-1]["text"]] or turns[-1]["role"] != "U":
                    raise ValueError(f"Packet render disagrees with evaluation: {qid}")
                dialogues[qid] = turns
                cases[qid] = (split, row)
                if old[qid] != turns:
                    changes[qid] = split
    if cases.keys() != old.keys():
        raise ValueError("Incomplete case/fingerprint/reference alignment")
    study_lock = dict(original_dialogues_sha256=lock["dialogues_sha256"],
                      protocol_sha256=lock["protocol_sha256"], linkage_sha256=sha256(audit.OUT/"linkage.jsonl"),
                      corrected_details_sha256=inputs, corrected_dialogues_sha256=digest(encoded(dict(sorted(dialogues.items())))),
                      original_population=len(original), matched=Counter(s for s,_ in cases.values()),
                      revised_display=changes, data_commit=audit.study.COMMIT)
    return dict(sorted(dialogues.items())), cases, study_lock


def packet_id(dialogues, letter):
    return digest(encoded([letter, sorted(dialogues.items())]))


def blank_form(dialogues, letter):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=[*audit.FORM, "packet_id"])
    writer.writeheader()
    token = packet_id(dialogues, letter)
    writer.writerows({"audit_id":qid, "packet_id":token} for qid in dialogues)
    return buf.getvalue().encode("utf-8-sig")


def packet(dialogues, letter, instructions):
    blocks = ["# TicketTalk 场次盲审"]
    for key, turns in dialogues.items():
        blocks.append("## "+key)
        blocks.extend(f'{i:02d} {t["role"]}  {t["text"]}' for i,t in enumerate(turns))
    files = {"README.md":instructions, "dialogues.md":"\n\n".join(blocks).encode(),
             f"reviewer_{letter}.csv":blank_form(dialogues, letter)}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(zipfile.ZipInfo(name), data, compress_type=zipfile.ZIP_DEFLATED)
    return buf.getvalue()


def form(data, dialogues, letter):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    token = packet_id(dialogues, letter)
    if reader.fieldnames != [*audit.FORM, "packet_id"]:
        raise ValueError(f"{letter}: invalid CSV header")
    values = {}
    for row in reader:
        key = row["audit_id"]
        if key not in dialogues or key in values or None in row or None in row.values():
            raise ValueError(f"{letter}: malformed or duplicate row {key}")
        if row["packet_id"] != token:
            raise ValueError(f"{letter} {key}: form belongs to another packet")
        status, stamp, turns = (row[k].strip() for k in ("status","time_24h","evidence_turns"))
        if all(not row[k].strip() for k in audit.FORM[1:]):
            values[key] = None
            continue
        valid = status == "unresolved" and not stamp
        if status == "resolved":
            valid = re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", stamp)
        if not valid or not all(i.strip().isascii() and i.strip().isdigit() and int(i) < len(dialogues[key])
                                for i in turns.split(",")):
            raise ValueError(f"{letter} {key}: check status, HH:MM and evidence turn indices")
        values[key] = stamp or "unresolved"
    if values.keys() != dialogues.keys():
        raise ValueError(f"{letter}: missing audit IDs")
    return values


def agreement(a, b):
    n = len(a)
    disputes = sorted(k for k in a if a[k] != b[k])
    agreed = n - len(disputes)
    ac, bc = Counter(a.values()), Counter(b.values())
    expected_pairs = sum(count*bc[v] for v,count in ac.items())
    both = sum(a[k] != "unresolved" and b[k] != "unresolved" for k in a)
    return dict(n=n, agreed=agreed, exact_agreement=agreed/n,
                cohen_kappa=(agreed*n-expected_pairs)/(n*n-expected_pairs) if expected_pairs != n*n else None,
                jointly_resolved=both, same_minute=sum(a[k]==b[k]!="unresolved" for k in a),
                disputes=disputes)


def label_outcome(ev, gold, target):
    if target is not None:
        return audit.normalized_outcome(ev["outcome"], ev["args"], gold, target)
    if ev["outcome"] == "no_write":
        return "no_write"
    others = lambda d: {k:v for k,v in d.items() if k!="time.showing"}
    return "wrong" if others(ev["args"]) != others(gold) else "indeterminate"


def results(cases, final):
    counts, statuses, details = {}, Counter(), []
    for key, label in sorted(final.items()):
        split, r = cases[key]
        target = None if label == "unresolved" else int(label[:2])*60 + int(label[3:])
        gold = r["gold"]
        ref = audit.gold_time(gold["time.showing"])
        status = ("human_unresolved" if target is None else "reference_ambiguous" if ref is None
                  else "agrees" if ref == target else "conflicts")
        statuses[status] += 1
        fields = {}
        for method in METHODS:
            ev = r["conditions"][method]
            fields[method] = {"official_exact":ev["outcome"], "human_time":label_outcome(ev,gold,target)}
            if ref is not None and target is not None:
                fields[method]["official_normalized_comparable"] = label_outcome(ev,gold,ref)
                fields[method]["human_normalized_comparable"] = fields[method]["human_time"]
            for view, outcome in fields[method].items():
                counts.setdefault(split,{}).setdefault(method,{}).setdefault(view,Counter())[outcome] += 1
        details.append(dict(audit_id=key, case=r["case"], split=split, label=label,
                            official_time=gold["time.showing"], comparison=status, orders=fields))
    n = len(final)
    conflicts, comparable = statuses["conflicts"], statuses["agrees"]+statuses["conflicts"]
    return dict(n=n, n_by_split=Counter(split for split,_ in cases.values()), counts=statuses,
                comparable_n=comparable, conflict_fraction=conflicts/n, conflict_wilson95=audit.wilson(conflicts,n),
                conflict_fraction_comparable=conflicts/comparable if comparable else None,
                identification_bounds=[conflicts/n,(conflicts+n-comparable)/n],
                simultaneous_sampling_bounds95=[audit.wilson(conflicts,n,.025)[0],audit.wilson(conflicts+n-comparable,n,.025)[1]],
                orders=counts, cases=details)


def collect(root, dialogues, cases, lock, instructions):
    """Only completed stages publish results. Returned originals are archived per version."""
    preserve(root/"source-lock.json", encoded(lock))
    for letter in "AB":
        preserve(root/f"reviewer_{letter}.zip", packet(dialogues,letter,instructions))
    inbox = root/"returns"
    inbox.mkdir(exist_ok=True)
    raw, values = {}, {}
    for letter in "AB":
        path = inbox/f"reviewer_{letter}.csv"
        if path.exists():
            raw[letter] = path.read_bytes()
            values[letter] = form(raw[letter], dialogues, letter)
    done = {l:sum(v is not None for v in values.get(l,{}).values()) for l in "AB"}
    if min(done.values()) != len(dialogues):
        return dict(stage="awaiting_independent_forms", completed=done, required_each=len(dialogues))
    hashes = {l:digest(data) for l,data in raw.items()}
    version = root/"rounds"/digest(encoded(dict(source=lock, forms=hashes)))
    for l,data in raw.items():
        preserve(version/f"original_{l}.csv", data)
    stats = agreement(values["A"], values["B"])
    preserve(version/"agreement.json", encoded(dict(**stats, form_sha256=hashes)))
    disputed = {k:dialogues[k] for k in stats["disputes"]}
    final = dict(values["A"])
    report_dir = version
    if disputed:
        preserve(version/"reviewer_C.zip", packet(disputed,"C",instructions))
        third = version/"reviewer_C.csv"
        third_values = {}
        if third.exists():
            data = third.read_bytes()
            third_values = form(data, disputed, "C")
        completed = sum(v is not None for v in third_values.values())
        if completed < len(disputed):
            return dict(stage="awaiting_adjudication", directory=str(version),
                        completed=completed, required=len(disputed))
        report_dir = version/digest(data)
        preserve(report_dir/"original_C.csv", data)
        final.update(third_values)
    preserve(report_dir/"human-results.json", encoded(results(cases, final)))
    return dict(stage="complete", result=str(report_dir/"human-results.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    dialogues, cases, lock = source()
    source_text = (audit.OUT/"blinded/README.md").read_text(encoding="utf-8-sig")
    instructions = source_text[:source_text.index("1. ")] + "1. 请独立阅读 dialogues.md，在随包 CSV 按 Q 编号逐行记录判断。完成前请只使用本包材料。\n" + source_text[source_text.index("2. "):source_text.index("编码完成后")] + "完成后请仅将 CSV 交回协调人。\n"
    print(json.dumps(collect(ROOT, dialogues, cases, lock, instructions.encode()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
