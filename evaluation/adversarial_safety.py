"""Private fixed-artifact stress test of our own model; no attack optimization."""
import hashlib
import json
from pathlib import Path
import threading
import urllib.request

from evaluation.freeform_review import packet_hash
from evaluation.safety_trial import possible_duplicate, normalized, TRAIN_SHA256, HOLDOUT_SHA256

PROTOCOL = 'fixed-adaptive-artifact-safety-development-v1'
REVISION = '7787faaf8b6fdd56e63ec320d5568bd3e35c48c0'
SOURCE_SHA256 = 'cf25133fca8ac75cf83b6c4a7b8684f35711399cc987f0c54268295e929cdb40'
SOURCE_URL = f'https://raw.githubusercontent.com/tml-epfl/llm-adaptive-attacks/{REVISION}/jailbreak_artifacts/exps_llama3-8b.json'
RUN_NAME = 'fixed_adversarial_safety_v1'
PROFILES = {'current': None, 'jail_half': (0., .5), 'jail_one': (0., 1.)}


def allocate_attacks(source, excluded):
    """Ignore upstream responses/success labels; allocate from goals alone."""
    if not excluded:
        raise ValueError('Verified training and validation exclusions are required')
    unique = {}
    for row in source['jailbreaks']:
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ('goal', 'prompt')):
            raise ValueError('Every artifact needs an explicit goal and attack prompt')
        if any(possible_duplicate(row['goal'], prompt) for prompt in excluded):
            continue
        cid = hashlib.sha256(normalized(row['goal']).encode()).hexdigest()
        unique.setdefault(cid, {'case_id': cid, 'goal': row['goal'], 'prompt': row['prompt']})
    ordered = sorted(unique.values(), key=lambda row: hashlib.sha256(('fixed-artifacts-v1:'+row['case_id']).encode()).hexdigest())
    selected = []
    for row in ordered:
        if not any(possible_duplicate(row['goal'], old['goal']) for old in selected):
            selected.append(row)
        if len(selected) == 28:
            break
    if len(selected) < 28:
        raise ValueError('Need eight development and twenty untouched confirmation behaviors')
    return {'development': selected[:8], 'confirmation': selected[8:],
            'audit': {'source_cases': len(source['jailbreaks']), 'eligible_exact_distinct_cases': len(unique),
                      'semantic_nonoverlap_proven': False, 'upstream_success_labels_used_for_selection': False}}


def run_adversarial_safety(*, checkpoint=lambda: None):
    import config
    from demo.backends import GPUBackend
    from evaluation.behavior_cases import CASES
    from evaluation.freeform_trial import MANIFEST
    from inference.prompts import comparison_prompt, PROMPT_PROTOCOL, BRIEF_PROTOCOL
    from inference.runtime import sha256_file
    from training.pilot import load_pairs, pairs_fingerprint

    root = Path(config.RUNS_DIR)/RUN_NAME
    root.mkdir(parents=True, exist_ok=True)
    if (root/'allocation.json').exists():
        raise RuntimeError('Immutable adversarial allocation already exists')
    train, validation, _ = load_pairs('jailbreak')
    if pairs_fingerprint(train) != TRAIN_SHA256 or pairs_fingerprint(validation) != HOLDOUT_SHA256:
        raise ValueError('Original training/validation reconstruction failed')
    raw = urllib.request.urlopen(SOURCE_URL, timeout=30).read()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('Fixed attack source content changed')
    allocation = allocate_attacks(json.loads(raw), [p for p, _ in train+validation])
    allocation.update(protocol=PROTOCOL, source_sha256=SOURCE_SHA256, source_revision=REVISION)
    (root/'allocation.json').write_text(json.dumps(allocation, indent=2))
    checkpoint()
    report = {'protocol': PROTOCOL, 'phase': 'development', 'status': 'running',
        'scope': 'Eight source-selected fixed Llama-3 adaptive artifacts, four brief benign controls, four model profiles. Not full AdvBench/JailbreakBench, optimized attacks, or fresh confirmation.',
        'allocation_sha256': packet_hash(allocation), 'data_audit': allocation['audit'],
        'manifest_sha256': sha256_file(MANIFEST), 'source_sha256': SOURCE_SHA256,
        'source_revision': REVISION, 'prompt_protocol': PROMPT_PROTOCOL,
        'max_new_tokens': 128, 'sentence_limit': None, 'profiles': PROFILES,
        'production_quality_verified': False, 'records': [], 'benign_records': []}
    for case in allocation['development']:
        report['records'].append({**case, 'outputs': {}})
    for case in CASES:
        if case.category == 'benign_sensitive':
            report['benign_records'].append({'case_id': case.id, 'prompt': case.prompt,
                'generation_prompt': comparison_prompt(case.prompt, 'brief'),
                'prompt_variant': BRIEF_PROTOCOL, 'review_checks': list(case.review_checks), 'outputs': {}})
    for row in report['records']+report['benign_records']:
        row['outputs'] = {arm: {'text': '', 'source': 'missing', 'simulated': False, 'finish_reason': 'not_started'}
                          for arm in ('baseline', *PROFILES)}
    def persist():
        temporary = root/'generation_report.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(root/'generation_report.json')
        checkpoint()
    persist()
    backend = GPUBackend(manifest_path=MANIFEST, use_cache=False)
    for row in report['records']+report['benign_records']:
        brief = 'generation_prompt' in row
        for profile, weights in PROFILES.items():
            def capture(value):
                arm = 'baseline' if value['branch'] == 'baseline' else profile
                row['outputs'][arm] = value
            done = False
            for event, payload in backend.stream(row.get('generation_prompt', row['prompt']), 48 if brief else 128,
                    threading.Event(), sentence_limit=1 if brief else None,
                    _branches=('baseline', 'unlora') if profile == 'current' else ('unlora',),
                    _fixed_alphas=weights, _raw_output_callback=capture):
                if event == 'error':
                    raise RuntimeError('Artifact generation failed; preserve private partial report')
                if event == 'routing':
                    row.setdefault('routing', {})[profile] = payload
                if event == 'complete':
                    done = payload.get('simulated') is False
                    row.setdefault('presentation_sources', {})[profile] = {
                        arm: value['source'] for arm, value in payload.items() if isinstance(value, dict) and 'source' in value}
            persist()
            if not done:
                raise RuntimeError('Incomplete artifact generation')
    report['status'] = 'complete'
    persist()
    return report
