"""Blinded sample, comparison and training-label ablation for TicketTalk."""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
from collections import Counter
from dataclasses import replace
from statistics import NormalDist

import ticket_learning as study
from ticket_execution import REPO
from confirmation_audit import explicit_time, times
from context_bundles.core import Budget, Counter as TokenCounter, Evidence
from context_bundles.io import read_jsonl, sha256, write_json, write_jsonl

OUT = study.ROOT / 'runs/review-labels'
OLD = study.ROOT / 'runs/ticket-learning'
PARTS = [('test','test',['aa']), ('replication','dev',['ab','ac'])]
FORM = ['audit_id','status','time_24h','evidence_turns','interpretation','note']


def packet_contexts(source, part):
    """Reproduce the frozen packet's raw TSV quoting and sampling IDs."""
    path = f'TM-3-2020/splits/{source}/{source}-{part}.tsv'
    data = subprocess.check_output(['git', '-C', str(REPO), 'show', study.COMMIT + ':' + path])
    for line in data.decode('utf8').splitlines():
        fields = line.split('\t')
        if len(fields) == 3 and fields[1].startswith('[PN]book_tickets[PAN]'):
            yield fields[2], fields[0]


def chronological_turns(raw_context):
    current, separator, history = raw_context.partition('<C>')
    if not separator or '<C>' in history:
        raise ValueError('Expected one current/history separator')
    chunks = re.split(r'<([A-Z]+)>', history + current)
    return [dict(role=chunks[i],text=chunks[i + 1].strip()) for i in range(1, len(chunks), 2)
            if chunks[i] in {'U','A'} and chunks[i + 1].strip()]


def prepare():
    cohort = {r['case']:split for split,_,_ in PARTS for r in read_jsonl(OLD/(split+'-details.jsonl'))}
    assert len(cohort)==1391
    take = sorted(cohort,key=lambda k:hashlib.sha256(('review-label-v1-20261123|'+k).encode()).digest())[:48]
    ids = {key:f'Q{i+1:02d}' for i,key in enumerate(take)}
    pack, link = [], []
    for split,source,shards in PARTS:
        seen=set()
        for part in shards:
            for key,context in packet_contexts(source,part):
                if key not in ids or key in seen or cohort[key]!=split:
                    continue
                seen.add(key)
                turns = chronological_turns(context)
                pack.append(dict(audit_id=ids[key],turns=turns))
                link.append(dict(audit_id=ids[key],case=key,split=split))
    assert len(pack)==48
    pack.sort(key=lambda r:r['audit_id'])
    blind = OUT/'blinded'; blind.mkdir(exist_ok=True)
    packet_path = blind/'dialogues.jsonl'
    lock_path = OUT/'blind-lock.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_bytes())
        if sha256(packet_path) != lock['dialogues_sha256'] or read_jsonl(packet_path) != pack:
            raise ValueError('Source or previously locked dialogue display changed')
    else:
        write_jsonl(packet_path,pack)
    blocks=[]
    for r in pack:
        blocks.append('## '+r['audit_id']+'\n\n'+'\n\n'.join(f"{i:02d} {t['role']}  {t['text']}" for i,t in enumerate(r['turns'])))
    (blind/'dialogues.md').write_text('# TicketTalk 场次盲审包\n\n'+'\n\n'.join(blocks)+'\n',encoding='utf8')
    for reviewer in ('A','B'):
        path=blind/f'reviewer_{reviewer}.csv'
        if not path.exists():
            with path.open('w',newline='',encoding='utf-8-sig') as f:
                writer=csv.DictWriter(f,fieldnames=FORM);writer.writeheader()
                writer.writerows({'audit_id':r['audit_id']} for r in pack)
    write_jsonl(OUT/'linkage.jsonl',sorted(link,key=lambda r:r['audit_id']))
    print('blinded sample ready',len(pack),sha256(blind/'dialogues.jsonl'))


def gold_time(value):
    value = value.casefold().strip().replace('.m.','m').replace('.m','m')
    m=re.fullmatch(r'(\d{1,2})(?:[:.](\d\d))?\s*([ap])\.?m\.?',value)
    if m and 1<=int(m[1])<=12 and int(m[2] or 0)<60:
        return int(m[1])%12*60+int(m[2] or 0)+720*(m[3]=='p')
    m=re.fullmatch(r'(\d{1,2}):(\d\d)',value)
    if m and (12<int(m[1])<24 or int(m[1])==0) and int(m[2])<60:
        return int(m[1])*60+int(m[2])
    return None


def wilson(k,n,alpha=.05):
    if n==0: return [None,None]
    z=NormalDist().inv_cdf(1-alpha/2);p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;rad=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return [max(0.,center-rad),min(1.,center+rad)]


def compare():
    ann={r['audit_id']:r for r in read_jsonl(OUT/'model-blind.jsonl')}
    source=list(read_jsonl(OUT/'linkage.jsonl'))
    assert set(ann)=={r['audit_id'] for r in source} and len(ann)==48
    lock = json.loads((OUT/'blind-lock.json').read_bytes())
    for field, path in [('sha256','model-blind.jsonl'),('dialogues_sha256','blinded/dialogues.jsonl'),('protocol_sha256','protocol.json')]:
        if lock[field] != sha256(OUT/path):
            raise ValueError('Blind lock mismatch: ' + path)
    official={r['case']:r['gold'] for split,_,_ in PARTS for r in read_jsonl(OLD/(split+'-details.jsonl'))}
    dialogues={r['audit_id']:r for r in read_jsonl(OUT/'blinded/dialogues.jsonl')}
    results=[]
    for key in source:
        a=ann[key['audit_id']];turns=dialogues[key['audit_id']]['turns']
        assert all(0<=i<len(turns) for i in a['evidence_turns'])
        raw=official[key['case']]['time.showing'];official_time=gold_time(raw)
        gt=None if a['time_24h'] is None else int(a['time_24h'][:2])*60+int(a['time_24h'][3:])
        label=('review_uncertain' if gt is None else 'reference_uncertain' if official_time is None else 'agrees' if gt==official_time else 'conflicts')
        results.append(dict(**key,**{k:v for k,v in a.items() if k!='audit_id'},reference=raw,label=label))
    counts=Counter(r['label'] for r in results);n=len(results);resolved=counts['agrees']+counts['conflicts'];c=counts['conflicts'];u=n-resolved
    out=dict(population=1391,sample=n,counts=counts,evidence=Counter(r['interpretation'] for r in results),clearly_disagreed_fraction=c/n,
             clearly_disagreed_ci95=wilson(c,n),among_comparable=c/resolved if resolved else None,among_comparable_ci95=wilson(c,resolved),
             observed_bounds=[c/n,(c+u)/n],sampling_and_identification_bounds95=[wilson(c,n,.025)[0],wilson(c+u,n,.025)[1]],annotation_sha256=sha256(OUT/'model-blind.jsonl'),protocol_sha256=sha256(OUT/'protocol.json'))
    write_json(OUT/'blind-summary.json',out);write_jsonl(OUT/'comparison.jsonl',results);print(out)


def current_confirmation(case):
    index = len(case.evidence)
    last = Evidence(str(index), 'U: ' + case.query, index, 'public')
    return explicit_time(replace(case, evidence=(*case.evidence, last), query=''))


def normalized_outcome(outcome, arguments, gold, target):
    if outcome == 'no_write':
        return outcome
    others = lambda fields: {k:v for k,v in fields.items() if k != 'time.showing'}
    if others(arguments) != others(gold):
        return 'wrong'
    value = arguments.get('time.showing', '')
    minute = gold_time(value)
    if minute is not None:
        return 'correct' if minute == target else 'wrong'
    bare = re.fullmatch(r'(0?[1-9]|1[0-2])[:.]([0-5]\d)', value.strip())
    if bare and (int(bare[1]) % 12) * 60 + int(bare[2]) == target % 720:
        return 'indeterminate'
    return 'wrong'


def rescore():
    """Keep each write decision and all non-time reference fields fixed."""
    compare()
    blind = {r['case']: r for r in read_jsonl(OUT/'comparison.jsonl')}
    historical = {r['case']: r for r in read_jsonl(study.ROOT/'runs/confirmation-audit/details.jsonl')}
    for old_order in (True, False):
        root = OUT if old_order else OUT/'chronological'
        original = OLD if old_order else study.ROOT/'runs/review-chronology'
        official = {r['case']:r['gold'] for name,_,_ in PARTS
                    for r in read_jsonl(original/(name + '-details.jsonl'))}
        if old_order:
            conflicts = {k:r['confirmed_minutes'] for k,r in historical.items() if r['label'] == 'conflicts'}
            comparable = {k for k,r in historical.items() if r['label'] != 'unresolved'}
        else:
            detected = json.loads((root/'confirmation-minutes.json').read_bytes())
            comparable = {k for k,minute in detected.items() if minute is not None and gold_time(official[k]['time.showing']) is not None}
            conflicts = {k:detected[k] for k in comparable if detected[k] != gold_time(official[k]['time.showing'])}
        cross_tab = {}
        for k in blind.keys() & official.keys():
            rule = 'conflicts' if k in conflicts else 'agrees' if k in comparable else 'unresolved'
            cross_tab.setdefault(blind[k]['label'], Counter())[rule] += 1
        scores = {}
        rows = read_jsonl(root/'train-details.jsonl')
        assert {r['arm'] for r in rows} == {'unchanged','exclude_conflicts','random_drop'}
        assert all(sum(r['arm'] == arm for r in rows) == len(official) for arm in {r['arm'] for r in rows})
        for r in rows:
            k = r['case']
            for m, outcome in r['outcomes'].items():
                pred = r['predictions'][m]
                adjusted = outcome
                if k in conflicts and outcome != 'no_write':
                    if old_order:
                        others = {f:v for f,v in pred.items() if f != 'time.showing'}
                        match = times(pred.get('time.showing','')) == {conflicts[k]} and others == {
                            f:v for f,v in official[k].items() if f != 'time.showing'}
                        adjusted = 'correct' if match else 'wrong'
                    else:
                        adjusted = normalized_outcome(outcome, pred, official[k], conflicts[k])
                def tally(split, target, result):
                    group = scores.setdefault(r['arm'],{}).setdefault(split,{}).setdefault(target,{})
                    group.setdefault(m, Counter())[result] += 1
                tally(r['split'],'official',outcome)
                tally(r['split'],'confirmation_sensitivity',adjusted)
                if k in blind:
                    time = blind[k]['time_24h']
                    tally('audit_sample','official',outcome)
                    if time is not None:
                        hour, minute = map(int,time.split(':'))
                        adjusted = normalized_outcome(outcome, pred, official[k], 60*hour+minute)
                        tally('audit_sample','blind_time_with_official_other_fields',adjusted)
        write_json(root/'target-sensitivity.json', dict(confirmation_conflicts=len(conflicts),audit_overlap=len(blind.keys() & official.keys()),
                                                       audit_label_by_rule=cross_tab,results=scores))


def human_agreement():
    dialogues = {r['audit_id']:r['turns'] for r in read_jsonl(OUT/'blinded/dialogues.jsonl')}
    forms = []
    for letter in 'AB':
        with (OUT/'blinded'/f'reviewer_{letter}.csv').open(encoding='utf-8-sig',newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != len(dialogues) or {r['audit_id'] for r in rows} != set(dialogues):
            raise ValueError('The form must contain all 48 distinct audit IDs')
        values = {}
        for row in rows:
            key = row['audit_id']
            stamp = row['time_24h'].strip()
            status = row['status'].strip()
            if status not in {'resolved','unresolved'}:
                raise ValueError(f'{letter} {key}: enter resolved/unresolved before comparing forms')
            if status == 'resolved' and not re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d',stamp):
                raise ValueError(f'{letter} {key}: resolved judgments require HH:MM')
            if status == 'unresolved' and stamp:
                raise ValueError(f'{letter} {key}: unresolved judgments require an empty time')
            indices = row['evidence_turns'].strip().split(',')
            if not all(ix.strip().isdigit() and 0 <= int(ix) < len(dialogues[key]) for ix in indices):
                raise ValueError(f'{letter} {key}: cite the supporting turns')
            values[key] = stamp or 'unresolved'
        forms.append(values)
    a,b = forms
    both = [k for k in a if 'unresolved' not in (a[k],b[k])]
    disagree = [dict(audit_id=k,reviewer_A=a[k],reviewer_B=b[k]) for k in sorted(a) if a[k] != b[k]]
    observed = 1 - len(disagree)/len(a)
    counts_a,counts_b = Counter(a.values()),Counter(b.values())
    expected = sum(n*counts_b[v] for v,n in counts_a.items()) / len(a)**2
    write_json(OUT/'human-agreement.json',dict(n=len(a),exact_agreement=observed,cohen_kappa=(observed-expected)/(1-expected)
               if expected < 1 else None,jointly_resolved=len(both),same_minute=sum(a[k]==b[k] for k in both),
               disagreements_for_adjudication=disagree))


def retrain():
    out = OUT/'chronological'
    out.mkdir(exist_ok=True)
    baseline = study.ROOT/'runs/review-chronology'
    training,used,texts,_=study.load('train',['aa','ab','ac','ad'])
    dev,used,texts,_=study.load('dev',['aa'],used,texts)
    calibration=[r for r in dev if int(r[1][:2],16)<128]
    for r in read_jsonl(study.ROOT/'runs/ticket-execution/details.jsonl'): used.add(r['case'])
    sample={}
    for split,source,shards in PARTS:
        sample[split],used,texts,_=study.load(source,shards,used,texts,split=='test')
    conflicts=set();comparable=0
    for key,_,case,gold in training:
        confirmed,_=current_confirmation(case)
        gt={gold_time(gold['time.showing'])} - {None}
        if confirmed is not None and len(gt)==1:
            comparable+=1
            if confirmed not in gt: conflicts.add(key)
    control=set(random.Random(45127).sample(sorted(r[0] for r in training),len(conflicts)))
    budget=Budget(128,TokenCounter('hf:'+str(study.ROOT/study.TOKENIZER)))
    details=[]
    summary={'training':len(training),'calibration':len(calibration),'comparable':comparable,'excluded':len(conflicts),'arms':{},'seed':45127,'calibration_labels':'unchanged official','adapter':'chronological'}
    confirmations = {k: current_confirmation(case)[0] for rows in sample.values() for k,_,case,_ in rows}
    write_json(out/'confirmation-minutes.json', confirmations)
    for name,drop in [('unchanged',set()),('exclude_conflicts',conflicts),('random_drop',control)]:
        train=[r for r in training if r[0] not in drop]
        matcher=study.Mentions(train); vec,model,n_labels=study.fit(train,matcher)
        cal=study.calibrate(study.evaluate(calibration,matcher,vec,model,budget))
        summary['arms'][name]={}
        for split,rows in sample.items():
            results=study.evaluate(rows,matcher,vec,model,budget,methods=('recent','field_support'))
            study.apply_calibration(results,cal)
            scores=study.summarize(results,.5)
            summary['arms'][name][split]=scores
            for r in results:
                details.append(dict(arm=name,split=split,case=r['case'],outcomes={m:v['outcome'] for m,v in r['conditions'].items()},predictions={m:v['args'] for m,v in r['conditions'].items()}))
            if name=='unchanged':
                old={r['case']:r for r in read_jsonl(baseline/(split+'-details.jsonl'))}
                assert {r['case'] for r in results} == set(old)
                assert all(r['conditions'][m]['outcome']==old[r['case']]['conditions'][m]['outcome'] for r in results for m in ('recent','field_support'))
        write_json(out/'train-summary.json',summary);write_jsonl(out/'train-details.jsonl',details)
        print(name,len(train),n_labels,flush=True)
    write_json(out/'training-drop-ids.json',dict(exclude_conflicts=sorted(conflicts),random_drop=sorted(control)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    actions = {'prepare':prepare,'compare':compare,'retrain':retrain,'rescore':rescore,'human-agreement':human_agreement}
    parser.add_argument('action',choices=actions)
    args=parser.parse_args()
    actions[args.action]()


