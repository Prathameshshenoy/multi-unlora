"""Immutable evaluation of one isolated paired hallucination training trial.

Specificity is a paired teacher-forced diagnostic, not generation accuracy.
Conditional-label validation cannot promote the free-form serving policy.
This module never starts paid work, training, or another model deployment.
"""
import json
import math
from pathlib import Path
import random
import statistics

from evaluation.quality import fingerprint, paired_summary
from evaluation.specificity import specificity_margin, summarize
from evaluation.tuning import (candidates, choose_candidate, confirmation_checks,
                               _score_partition, VALIDATION_N, CONFIRMATION_N)
from inference.prompts import PROMPT_PROTOCOL, training_tokens

PAIRED_QUALITY_PROTOCOL = 'paired-hall-specificity-and-conditional-confirmation-v1'
SPECIFICITY_N = 16
MAX_RESPONSE_CONTEXT = 512
TRIAL_NAME = 'paired_hall_v1'


class ResponseTooLong(ValueError):
    """A complete native response cannot fit the diagnostic context."""


def teacher_forced_score(model, tokenizer, prompt, response, adapter=None, *, max_length=MAX_RESPONSE_CONTEXT):
    """Return sum/count/mean causal response logps, with no silent truncation."""
    import torch
    from inference.adapters import active_adapter
    from training.paired_objective import response_log_probabilities
    from training.pilot import ResponseCollator

    if not isinstance(response, str) or not response.strip():
        raise ValueError('A nonempty reference response is required')
    extended = training_tokens(tokenizer, prompt, response, max_length+1)
    if extended is None or len(extended['input_ids']) > max_length:
        raise ResponseTooLong('Full native response exceeds diagnostic context')
    inputs = {key: value.to(next(model.parameters()).device)
              for key, value in ResponseCollator(tokenizer.pad_token_id)([extended]).items()}
    with active_adapter(model, adapter), torch.inference_mode():
        output = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                       use_cache=False, return_dict=True)
        total = float(response_log_probabilities(output.logits, inputs['labels'])[0])
    count = int((inputs['labels'][:, 1:] != -100).sum())
    if count < 1 or not math.isfinite(total):
        raise RuntimeError('Invalid teacher-forced response score')
    return {'sum_log_probability': total, 'response_tokens': count,
            'mean_log_probability': total/count, 'mean_nll': -total/count}


def specificity_ci(rows, *, seed=42, draws=2000):
    if not rows:
        raise ValueError('Specificity summary requires paired observations')
    rng, size = random.Random(seed), len(rows)
    samples = sorted(sum(rows[rng.randrange(size)]['specificity_margin'] for _ in rows)/size
                     for _ in range(draws))
    return [samples[int(draws*.025)], samples[min(draws-1, int(draws*.975))]]


def paired_specificity(model, tokenizer, records, *, adapter='hallucination', reference_adapter='pilot_hallucination', additional_adapters=()):
    """Compare original and candidate on the SAME complete response pairs."""
    if model.training:
        raise ValueError('Specificity measurements require eval mode')
    names=(reference_adapter,adapter,*additional_adapters)
    if len(set(names))!=len(names):
        raise ValueError('Specificity adapter names must be distinct')
    rows = {name:[] for name in names}
    excluded = []
    for index, item in enumerate(records):
        pair_id = fingerprint(item)
        try:
            base_bad = teacher_forced_score(model, tokenizer, item['prompt'], item['bad'])
            base_good = teacher_forced_score(model, tokenizer, item['prompt'], item['good'])
            scored = {}
            for name in rows:
                bad = teacher_forced_score(model, tokenizer, item['prompt'], item['bad'], name)
                good = teacher_forced_score(model, tokenizer, item['prompt'], item['good'], name)
                values = specificity_margin(base_bad['mean_nll'], bad['mean_nll'],
                                            base_good['mean_nll'], good['mean_nll'])
                scored[name] = {**values, 'pair_id': pair_id, 'pair_index': index,
                    'base_bad': base_bad, 'expert_bad': bad, 'base_good': base_good, 'expert_good': good,
                    'sum_logp_preference_gain': (bad['sum_log_probability']-good['sum_log_probability'])
                        -(base_bad['sum_log_probability']-base_good['sum_log_probability'])}
        except ResponseTooLong:
            excluded.append({'pair_index': index, 'pair_id': pair_id, 'reason': 'full_response_context_length'})
            continue
        for name in rows:
            rows[name].append(scored[name])
    if not rows[adapter]:
        raise RuntimeError('No complete paired responses fit the diagnostic context')
    summaries = {name: {**summarize(values), 'specificity_margin_bootstrap_95_ci': specificity_ci(values)}
                 for name, values in rows.items()}
    changes = [new['specificity_margin']-old['specificity_margin']
               for new, old in zip(rows[adapter], rows[reference_adapter])]
    return {'n_requested': len(records), 'n_complete_pairs': len(changes), 'excluded': excluded,
            'adapters': {name: {'summary': summaries[name], 'rows': values} for name, values in rows.items()},
            'mean_candidate_minus_original_specificity': statistics.mean(changes),
            'scope': 'Mean response-token logps include assistant end-of-turn tokens. Sum-logp controls are retained. No generation accuracy or safety claim.'}


def report_usage(value):
    """Collect every selected, excluded or merely visited prior benchmark ID/index."""
    ids, indices = set(), set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'question_id' and isinstance(item, str):
                ids.add(item)
            elif key == 'question_ids' and isinstance(item, list):
                ids.update(entry for entry in item if isinstance(entry, str))
            elif key == 'dataset_index' and isinstance(item, int):
                indices.add(item)
            elif key == 'touched_indices' and isinstance(item, list):
                indices.update(entry for entry in item if isinstance(entry, int))
            nested_ids, nested_indices = report_usage(item)
            ids.update(nested_ids)
            indices.update(nested_indices)
    elif isinstance(value, list):
        for item in value:
            nested_ids, nested_indices = report_usage(item)
            ids.update(nested_ids)
            indices.update(nested_indices)
    return ids, indices


def trial_usage(initial, conditional, task, dataset_size):
    original = initial['benchmarks'][task]
    all_ids, all_indices = report_usage(original)
    if not all_indices or len(original.get('rows', [])) < CONFIRMATION_N:
        raise RuntimeError('Initial report does not identify the original held-out questions')
    # Previously used validation questions are permitted for validation only.
    validation_blocked = set(all_ids)
    previous_validation = conditional.get('validation', {}).get(task, {})
    previous_confirmation = conditional.get('confirmation', {}).get(task, {})
    confirmed_ids, _ = report_usage(previous_confirmation)
    validation_blocked.update(confirmed_ids)
    for section in (previous_validation, previous_confirmation):
        ids, indices = report_usage(section)
        all_ids.update(ids)
        all_indices.update(indices)
    half = dataset_size//2
    start = max(half+CONFIRMATION_N, max(all_indices)+1)
    return validation_blocked, all_ids, start


def _read_json(path):
    if not Path(path).is_file():
        raise RuntimeError(f'Required prior evaluation artifact is missing: {Path(path).name}')
    return json.loads(Path(path).read_text())


def evaluate_paired_trial():
    """Evaluate one candidate bundle without overwriting any canonical artifact."""
    import config
    from datasets import load_dataset
    from inference.operator import OPERATOR_VERSION
    from inference.runtime import load_inference_model, validate_artifacts, sha256_file
    from training.paired import load_paired_source

    directory = Path(config.RUNS_DIR)/TRIAL_NAME
    output = directory/'quality_report.json'
    if output.exists():
        raise RuntimeError('This paired trial already has a quality report; do not retune on confirmation')
    canonical_path = Path(config.ADAPTER_DIR)/'manifest.json'
    trial_path = directory/'manifest.json'
    canonical = validate_artifacts()
    trial = validate_artifacts(manifest_path=trial_path)
    if trial.get('source_manifest_sha256') != sha256_file(canonical_path):
        raise RuntimeError('Candidate is not bound to the unchanged original pilot')
    initial = _read_json(Path(config.RESULTS_DIR)/'pilot_quality.json')
    conditional = _read_json(Path(config.RESULTS_DIR)/'conditional_quality.json')
    if initial.get('manifest_fingerprint') != fingerprint(canonical) or conditional.get('manifest_fingerprint') != fingerprint(canonical):
        raise RuntimeError('Prior benchmark reports do not belong to the canonical source pilot')
    train, validation, fresh, split = load_paired_source(canonical, initial)
    trial_split = trial['concepts']['hallucination']['paired_trial']['source_split']
    if trial_split != split:
        raise RuntimeError('Paired candidate data provenance differs from its source reconstruction')
    if len(validation) < SPECIFICITY_N or len(fresh) < SPECIFICITY_N:
        raise RuntimeError('Insufficient original validation or untouched specificity records')
    # Dataset identity and prior usage are checked before loading an 8B GPU model.
    datasets = {
        'mmlu': load_dataset('cais/mmlu', 'all', split='validation').shuffle(seed=config.EVAL_SEED),
        'truthfulqa_mc1': load_dataset('truthful_qa', 'multiple_choice', split='validation').shuffle(seed=config.EVAL_SEED)}
    usage = {}
    for task, dataset in datasets.items():
        if dataset._fingerprint != initial['benchmarks'][task]['dataset_fingerprint']:
            raise RuntimeError(f'{task} source fingerprint changed')
        if dataset._fingerprint != conditional['validation'][task]['dataset_fingerprint']:
            raise RuntimeError(f'{task} conditional validation fingerprint changed')
        for previous in (initial['benchmarks'][task], conditional['validation'][task]):
            if previous.get('selection_seed') != config.EVAL_SEED:
                raise RuntimeError(f'{task} prior benchmark selection seed differs or is missing')
        if task in conditional.get('confirmation', {}) and conditional['confirmation'][task].get('selection_seed') != config.EVAL_SEED:
            raise RuntimeError(f'{task} prior confirmation selection seed differs or is missing')
        usage[task] = trial_usage(initial, conditional, task, len(dataset))
    report = {'protocol': PAIRED_QUALITY_PROTOCOL, 'operator': OPERATOR_VERSION,
        'prompt_protocol': PROMPT_PROTOCOL, 'manifest_fingerprint': fingerprint(trial),
        'source_manifest_fingerprint': fingerprint(canonical), 'model_revision': trial['model_revision'],
        'prior_report_fingerprints': {'pilot': fingerprint(initial), 'conditional': fingerprint(conditional)},
        'predeclared_grid': candidates(), 'specificity_n_per_partition': SPECIFICITY_N,
        'validation_n_per_benchmark': VALIDATION_N, 'confirmation_n_per_benchmark': CONFIRMATION_N,
        'specificity_max_context': MAX_RESPONSE_CONTEXT, 'source_split': split,
        'validation': {}, 'confirmation': {}, 'specificity': {},
        'production_quality_verified': False, 'freeform_serving_policy_validated': False,
        'scope': 'Paired likelihood specificity and separately conditioned-label accuracy only; no raw harmful answers or reference text are written.'}
    model, tokenizer, v_hall, v_jail = load_inference_model(manifest_path=trial_path, calibrate_vectors=False)
    # This third LoRA is small; it reuses the same frozen base and permits a
    # directly paired original-versus-candidate diagnostic without a second model.
    model.load_adapter(config.HALL_ADAPTER_PATH, adapter_name='pilot_hallucination')
    model.eval()
    model.disable_adapter_layers()
    report['specificity']['validation'] = paired_specificity(model, tokenizer, validation[:SPECIFICITY_N])
    report['specificity']['fresh_confirmation'] = paired_specificity(model, tokenizer, fresh[:SPECIFICITY_N])
    validation_summaries = {candidate['id']: {} for candidate in candidates()}
    blocked_fresh = {}
    for task, dataset in datasets.items():
        validation_blocked, prior_ids, start = usage[task]
        measured = _score_partition(model, tokenizer, v_hall, v_jail, dataset, task,
            range(0, len(dataset)//2), VALIDATION_N, candidates(), validation_blocked)
        new_ids, _ = report_usage(measured)
        blocked_fresh[task] = prior_ids | new_ids
        report['validation'][task] = {'dataset_fingerprint': dataset._fingerprint,
            'selection_seed': config.EVAL_SEED, **measured}
        for candidate in candidates():
            validation_summaries[candidate['id']][task] = paired_summary(measured['rows'][candidate['id']])
    selected = choose_candidate(validation_summaries)
    report['selected'], report['validation_summaries'] = selected, validation_summaries
    if selected['strength'] == 0:
        report['confirmation_status'] = 'not_run_no_validation_gain_baseline_selected'
        report['conditional_forced_choice_gate_passed'] = False
    elif any(len(datasets[task])-usage[task][2] < CONFIRMATION_N for task in datasets):
        # Never reuse previously touched confirmation questions to manufacture n.
        report['confirmation_status'] = 'not_run_insufficient_untouched_questions'
        report['conditional_forced_choice_gate_passed'] = False
    else:
        for task, dataset in datasets.items():
            start = usage[task][2]
            measured = _score_partition(model, tokenizer, v_hall, v_jail, dataset, task,
                range(start, len(dataset)), CONFIRMATION_N, [selected], blocked_fresh[task])
            rows = measured['rows'][selected['id']]
            if set(measured['question_ids']) & blocked_fresh[task]:
                raise RuntimeError('Fresh confirmation overlaps previously touched question IDs')
            report['confirmation'][task] = {**measured, 'dataset_fingerprint': dataset._fingerprint,
                'selection_seed': config.EVAL_SEED, 'start_index': start,
                'summary': paired_summary(rows), 'all_candidate_rows': measured['rows'], 'rows': rows}
        # Preserve the original learned-anti-expert diagnostic; the new candidate
        # additionally needs its own fresh paired-specificity evidence below.
        checks = confirmation_checks(report['confirmation'], selected, initial)
        fresh_specificity = report['specificity']['fresh_confirmation']['adapters']['hallucination']['summary']
        checks['candidate_fresh_specificity_mean_positive'] = fresh_specificity['mean_specificity_margin'] > 0
        checks['candidate_fresh_bad_response_gain_positive'] = fresh_specificity['mean_bad_response_nll_gain'] > 0
        checks['sixteen_complete_fresh_specificity_pairs'] = fresh_specificity['n'] == SPECIFICITY_N
        report['checks'] = checks
        report['conditional_forced_choice_gate_passed'] = all(checks.values())
        report['confirmation_status'] = 'passed' if all(checks.values()) else 'failed_no_retuning'
    # Atomic immutable publication; failed outcomes are retained too.
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    model.disable_adapter_layers()
    return report
