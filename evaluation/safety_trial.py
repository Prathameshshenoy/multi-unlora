"""Bounded private raw safety diagnostic, never an automatic release gate."""
import csv
from difflib import SequenceMatcher
import hashlib
import io
import json
from pathlib import Path
import re
import threading
import urllib.request

from evaluation.freeform_review import packet_hash

PROTOCOL = 'advbench-direct-raw-development-v1'
SOURCE_REVISION = '098262edf85f807224e70ecd87b9d83716bf6b73'
SOURCE_SHA256 = '6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1'
SOURCE_URL = f'https://raw.githubusercontent.com/llm-attacks/llm-attacks/{SOURCE_REVISION}/data/advbench/harmful_behaviors.csv'
TRAIN_SHA256 = '28ed509a1ae2b61571cd8d15e06783c1fbf8f28280a8bdeb9d761a39f3e0fb60'
HOLDOUT_SHA256 = '22a3415afbc7c49134ad756c9cb3a6dd059e162bd8a104ffe3e7317122d0a1a7'
RUN_NAME = 'raw_safety_development_v1'


def normalized(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def possible_duplicate(left, right):
    a, b = normalized(left), normalized(right)
    if a == b:
        return True
    aa, bb = set(a.split()), set(b.split())
    if not aa or not bb:
        return False
    overlap = len(aa & bb)
    return (overlap/len(aa | bb) >= .65
            or (min(len(aa), len(bb)) >= 8 and overlap/min(len(aa), len(bb)) >= .9)
            or SequenceMatcher(None, a, b, autojunk=False).ratio() >= .8)


def allocate_cases(goals, excluded_prompts, *, development_n=20, confirmation_n=60):
    """Source-only allocation. Heuristic deduplication is not a semantic proof."""
    excluded = tuple(excluded_prompts)
    if not excluded or any(not isinstance(v, str) or not v.strip() for v in (*goals, *excluded)):
        raise ValueError('Explicit source goals and excluded training/validation prompts are required')
    if any(type(n) is not int or n < 1 for n in (development_n, confirmation_n)):
        raise ValueError('Positive frozen split sizes are required')
    unique, exact, near = {}, 0, 0
    excluded_keys = {normalized(p) for p in excluded}
    for prompt in goals:
        key = normalized(prompt)
        if key in unique:
            continue
        if key in excluded_keys:
            exact += 1
            continue
        if any(possible_duplicate(prompt, old) for old in excluded):
            near += 1
            continue
        unique[key] = {'case_id': hashlib.sha256(key.encode()).hexdigest(), 'prompt': prompt}
    ordered = sorted(unique.values(), key=lambda row: hashlib.sha256(('raw-safety-v1:'+row['case_id']).encode()).hexdigest())
    if len(ordered) < development_n+confirmation_n:
        raise ValueError('Not enough disjoint candidates; do not silently shrink or reuse training cases')
    # Also remove near duplicates across and within the two allocated splits.
    selected = []
    for row in ordered:
        if not any(possible_duplicate(row['prompt'], prior['prompt']) for prior in selected):
            selected.append(row)
        if len(selected) == development_n+confirmation_n:
            break
    if len(selected) < development_n+confirmation_n:
        raise ValueError('Not enough distinct cases after cross-split duplicate checks')
    return {'development': selected[:development_n], 'confirmation': selected[development_n:],
            'audit': {'source_rows': len(goals), 'excluded_training_validation_prompts': len(excluded),
                      'exact_overlap_rows': exact, 'heuristic_near_overlap_rows': near,
                      'remaining_exact_distinct_candidates': len(ordered),
                      'semantic_nonoverlap_proven': False}}


def run_safety_development(*, checkpoint=lambda: None):
    import config
    from demo.backends import GPUBackend
    from evaluation.behavior_cases import CASES
    from evaluation.freeform_trial import MANIFEST
    from inference.runtime import sha256_file
    from inference.prompts import PROMPT_PROTOCOL
    from training.pilot import load_pairs, pairs_fingerprint

    root = Path(config.RUNS_DIR)/RUN_NAME
    root.mkdir(parents=True, exist_ok=True)
    output = root/'generation_report.json'
    if output.exists() or (root/'allocation.json').exists():
        raise RuntimeError('Immutable safety diagnostic already allocated or executed')
    train, heldout, dataset_id = load_pairs('jailbreak')
    if pairs_fingerprint(train) != TRAIN_SHA256 or pairs_fingerprint(heldout) != HOLDOUT_SHA256:
        raise ValueError('Original jailbreak training/validation split changed')
    data = urllib.request.urlopen(SOURCE_URL, timeout=30).read()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('Pinned AdvBench content changed')
    goals = [row['goal'] for row in csv.DictReader(io.StringIO(data.decode()))]
    allocation = allocate_cases(goals, [p for p, _ in train+heldout])
    allocation.update(protocol=PROTOCOL, source_revision=SOURCE_REVISION, source_sha256=SOURCE_SHA256,
                      jailbreak_dataset_fingerprint=dataset_id, train_pairs_sha256=TRAIN_SHA256,
                      holdout_pairs_sha256=HOLDOUT_SHA256)
    (root/'allocation.json').write_text(json.dumps(allocation, indent=2))
    checkpoint()  # Freeze source-only allocation before loading/generating.

    report = {'protocol': PROTOCOL, 'phase': 'development', 'status': 'running',
              'scope': '20 direct harmful behaviors after exact/heuristic training/validation exclusions, plus four benign sensitive controls. Not adversarial-attack confirmation or semantic nonoverlap proof.',
              'allocation_sha256': packet_hash(allocation), 'data_audit': allocation['audit'],
              'source_revision': SOURCE_REVISION, 'source_sha256': SOURCE_SHA256,
              'manifest_sha256': sha256_file(MANIFEST), 'prompt_protocol': PROMPT_PROTOCOL,
              'max_new_tokens': 128, 'sentence_limit': None, 'prompt_style': 'standard',
              'production_quality_verified': False, 'records': [], 'benign_records': []}
    # Keep all assigned cases, including those a timeout never reaches.
    for case in allocation['development']:
        report['records'].append({**case, 'outputs': {
            arm: {'text': '', 'source': 'missing', 'simulated': False, 'finish_reason': 'not_started'}
            for arm in ('baseline', 'unlora')}})
    for case in CASES:
        if case.category == 'benign_sensitive':
            report['benign_records'].append({'case_id': case.id, 'prompt': case.prompt,
                'review_checks': list(case.review_checks), 'outputs': {
                    arm: {'text': '', 'source': 'missing', 'simulated': False, 'finish_reason': 'not_started'}
                    for arm in ('baseline', 'unlora')}})

    def persist():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(output)
        checkpoint()

    persist()
    backend = GPUBackend(manifest_path=MANIFEST, use_cache=False)
    for row in report['records']+report['benign_records']:
        def capture(value):
            row['outputs'][value['branch']] = value
        complete = False
        for event, payload in backend.stream(row['prompt'], 128, threading.Event(),
                _branches=('baseline', 'unlora'), _raw_output_callback=capture):
            if event == 'error':
                raise RuntimeError('Safety generation failed; preserve private partial report')
            if event in ('routing', 'metadata'):
                row[event] = payload
            if event == 'complete':
                complete = payload.get('simulated') is False
                row['presentation_sources'] = {arm: payload[arm]['source'] for arm in ('baseline', 'unlora')}
        if not complete:
            persist()
            raise RuntimeError('Incomplete safety generation')
        persist()
    report['status'] = 'complete'
    persist()
    return report
