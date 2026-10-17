import hashlib
import json
from pathlib import Path

path = Path('evals/questions.jsonl')
if not path.exists():
    raise SystemExit('Create evals/questions.jsonl from the template first.')
records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
ids = [r['id'] for r in records]
assert len(ids) == len(set(ids)), 'Duplicate question IDs'
required = {'id','question','expected_answer','source_article','source_quote','question_type','split','verified'}
for r in records:
    assert required <= r.keys(), f'Missing fields: {r.get("id")}'
    assert r['verified'] is True, f'Unverified: {r["id"]}'
    assert r['split'] in {'dev', 'test'}
    assert r['question_type'] in {'single_hop','multi_hop','unanswerable'}
print(f'Validated {len(records)} questions')
print('SHA256:', hashlib.sha256(path.read_bytes()).hexdigest())
