"""Offline train/dev audit; --collect runs at most the two frozen diagnostic requests."""
import argparse
from collections import Counter
import json
from pathlib import Path

from context_bundles.core import Case, Budget, Counter as TokenCounter, render_history
from context_bundles.io import read_jsonl, write_json, write_jsonl
from context_bundles.curriculum_run import open_existing_ledger
from context_bundles.executor import execute_stage
from context_bundles.deepseek import DeepSeekClient

ROOT = Path(__file__).resolve().parents[1]
import os
os.chdir(ROOT)
out = Path('runs/evidence-audit-522')
out.mkdir(exist_ok=True, parents=True)
p = argparse.ArgumentParser()
p.add_argument('--collect', action='store_true')
args = p.parse_args()
cfg = json.loads(Path('configs/deepseek-60.json').read_text())
cases = {r['id']: r for r in read_jsonl(cfg['cases_path']) if r['split'] in ('dev','train')}
labels = {r['id']: r['gold'] for r in read_jsonl(cfg['labels_path']) if r['split'] in ('dev','train')}
pairs = json.loads(Path('runs/curriculum-v2-plan/pairs.json').read_text())
def leaves(value):
    if isinstance(value, str) and len(value.strip()) >= 4:
        yield value.casefold()
    elif isinstance(value, dict):
        for v in value.values():
            yield from leaves(v)
    elif isinstance(value, list):
        for v in value:
            yield from leaves(v)

audit = []
for pair in pairs:
    cid = pair['case_id']
    c = cases[cid]
    # Literal overlap is a screening proxy. It also catches defaults and common words.
    targets = set(leaves(labels[cid]['arguments']))
    content = {e['id']: e['text'].casefold() for e in c['evidence']}
    hits = [{t for t in targets if t in content[i]} for i in pair['members']]
    a, b = hits
    if not targets:
        kind = 'no_eligible_string'
    elif not (a | b):
        kind = 'no_literal_hit'
    elif a - b and b - a:
        kind = 'each_adds_a_literal'
    elif a == b:
        kind = 'identical_nonempty_hits'
    else:
        kind = 'one_subsumes_other'
    audit.append({'case_id': cid, 'members': pair['members'], 'kind':kind,
                  'hits': [sorted(h) for h in hits]})
write_jsonl(out/'literal_audit.jsonl', audit)

cid = 'qa_522'
case = Case.from_dict(cases[cid])
values = [m['value'] for m in labels[cid]['arguments']['messages']]
by_id = {e.id:e for e in case.evidence}
ids = ['session_01663:turn0024','session_01664:turn0014','session_01664:turn0016']
a,b,d = [by_id[i] for i in ids]
for val, block in zip(values, [a,b], strict=True):
    assert [e.id for e in case.evidence if val in e.text] == [block.id]
budget = Budget(cfg['history_budget'], TokenCounter(cfg['encoding']))
stage = 'qa522_ordered_recall_v1'
specs = []
for condition, selected in [('A_plus_control',[a,d]),('AB',[a,b])]:
    prompt = [{'role':'system','content':'Extract user questions from the provided history. Return JSON with a questions array of verbatim user question strings in chronological order. Include only earlier questions about Python that are present. If fewer than two are present return only those found.'},
              {'role':'user','content':case.query+'\nHistorical evidence JSON:\n'+render_history(selected)}]
    specs.append({'case_id':cid,'split':'train','group_id':case.group_id,'method':'ordered_recall',
                  'condition':condition,'messages':prompt,'provider':cfg['provider'],
                  'selected_ids':[e.id for e in selected], 'history_cost':budget.cost(selected)})
assert all(r['history_cost']<=budget.limit for r in specs)
assert abs(specs[0]['history_cost']-specs[1]['history_cost'])<=64
plan_path = out/'plan.json'
protocol = {'endpoint':'questions array equals two verbatim user questions in gold order',
            'scope':'post-hoc train case, evidence supplied using labels; one response per condition; extraction subtask',
            'requests':specs}
if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8'))!=protocol:
    raise ValueError('Frozen diagnostic changed')
write_json(plan_path,protocol)
ledger=open_existing_ledger('runs/deepseek-progressive-60/ledger.sqlite3')
if args.collect:
    execute_stage(ledger,stage,specs,DeepSeekClient(cfg['provider']),concurrency=1)
rows=ledger.rows(stage)
scores=[]
for row in rows:
    result=row['result'] or {}
    try:
        obj=json.loads(result['output'])
        returned=obj.get('questions',[]) if isinstance(obj,dict) else []
    except (KeyError,ValueError):
        returned=[]
    ok=result.get('status')=='ok' and result.get('finish_reason')=='stop'
    scores.append({'condition':row['spec']['condition'],'request_id':row['request_id'],
                   'usable':ok,'output':result.get('output'),'usage':result.get('usage'),
                   'exact_ordered_pair':(returned==values) if ok else None,
                   'verbatim_hits':sum(v in returned for v in values) if ok and isinstance(returned,list) else None})
result={'literal_counts':dict(Counter(r['kind'] for r in audit)), 'gold_questions':values,
        'topic_name_literal_in_history':any(labels[cid]['arguments']['topic_name'] in e.text for e in case.evidence),
        'scores':scores,'ledger':ledger.stats()}
write_json(out/'results.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
