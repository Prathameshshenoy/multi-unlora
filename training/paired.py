"""One isolated hallucination anti-expert preference trial.

The undesirable response is intentionally preferred over its paired right
answer using MEAN response-token logps relative to a frozen BASE reference.
This is explicitly a length-normalized DPO-inspired variant, not standard DPO. This experimental objective does
not establish that subtracting the resulting adapter improves answer quality.
Canonical adapters and the original manifest are never written by this module.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import config
from inference.prompts import PROMPT_PROTOCOL, training_tokens
from training.paired_objective import anti_expert_preference_loss, response_log_probabilities
from training.pilot import pairs_fingerprint, prompt_key

PAIRED_PROTOCOL = 'paired-hall-length-normalized-anti-preference-v1'
LOSS_NAME = 'length-normalized-anti-preference-v1'
TRAIN_COUNT = 800
VALIDATION_COUNT = 48
MAX_LENGTH = 512
BETA = .1
LEARNING_RATE = 5e-5
GRADIENT_ACCUMULATION = 4


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def reconstruct_paired_records(records, dataset_fingerprint, manifest, initial_report):
    """Bind the paired rows to the original trained pilot's exact bad answers.

    The first 800 distinct prompts are training only; the next 48 are validation
    only. Everything after those 848 records remains available for untouched
    confirmation. No validation or confirmation response is tokenized here.
    """
    stored = manifest.get('concepts', {}).get('hallucination', {})
    if manifest.get('seed') != config.TRAIN_SEED:
        raise RuntimeError('Pilot seed differs; paired training split cannot be reconstructed')
    if not dataset_fingerprint or stored.get('dataset_fingerprint') != dataset_fingerprint:
        raise RuntimeError('HaluEval source fingerprint changed since the original training')
    if initial_report.get('manifest_fingerprint') != _fingerprint(manifest):
        raise RuntimeError('Original evaluation report does not match the source adapter manifest')
    unique = {}
    for item in records:
        prompt, bad, good = item.get('question'), item.get('hallucinated_answer'), item.get('right_answer')
        if not isinstance(prompt, str) or not isinstance(bad, str):
            raise RuntimeError('HaluEval prompt and hallucinated answer must be strings')
        key = prompt_key(prompt)
        if key and bad.strip():
            unique.setdefault(key, {'prompt': prompt, 'bad': bad, 'good': good})
    ordered = list(unique.values())
    if len(ordered) < TRAIN_COUNT + VALIDATION_COUNT:
        raise RuntimeError('Insufficient distinct matched HaluEval records')
    train = ordered[:TRAIN_COUNT]
    validation = ordered[TRAIN_COUNT:TRAIN_COUNT+VALIDATION_COUNT]
    for item in train + validation:
        if not isinstance(item['good'], str) or not item['good'].strip():
            raise RuntimeError('Paired training and validation require a nonempty right answer')
        if item['bad'].strip() == item['good'].strip():
            raise RuntimeError('An identical right/hallucinated response is not a valid preference pair')
    train_hash = pairs_fingerprint([(row['prompt'], row['bad']) for row in train])
    validation_hash = pairs_fingerprint([(row['prompt'], row['bad']) for row in validation])
    old = initial_report.get('anti_experts', {}).get('hallucination', {}).get('split_provenance', {})
    for key, expected in [('dataset_fingerprint', dataset_fingerprint),
                          ('source_train_rows', TRAIN_COUNT), ('holdout_rows', VALIDATION_COUNT),
                          ('train_pairs_fingerprint', train_hash), ('holdout_pairs_fingerprint', validation_hash)]:
        if old.get(key) != expected:
            raise RuntimeError(f'Original hallucination split provenance mismatch: {key}')
    provenance = {'dataset_fingerprint': dataset_fingerprint, 'source_train_rows': TRAIN_COUNT,
        'holdout_rows': VALIDATION_COUNT, 'train_pairs_fingerprint': train_hash,
        'holdout_pairs_fingerprint': validation_hash,
        'paired_train_fingerprint': _fingerprint(train), 'paired_validation_fingerprint': _fingerprint(validation),
        'confirmation_start_distinct_record': TRAIN_COUNT+VALIDATION_COUNT,
        'remaining_distinct_records': len(ordered)-TRAIN_COUNT-VALIDATION_COUNT,
        'source_split_verification': 'matched_original_frozen_pilot_question_and_bad_response_hashes'}
    return train, validation, ordered[TRAIN_COUNT+VALIDATION_COUNT:], provenance


def load_paired_source(manifest, initial_report):
    from datasets import load_dataset
    dataset = load_dataset('pminervini/HaluEval', 'qa', split='data').shuffle(seed=config.TRAIN_SEED)
    return reconstruct_paired_records(dataset, dataset._fingerprint, manifest, initial_report)


def tokenize_training_pairs(tokenizer, records):
    """Only caller-supplied training records enter the preference objective."""
    rows, excluded = [], []
    for index, item in enumerate(records):
        bad = training_tokens(tokenizer, item['prompt'], item['bad'], MAX_LENGTH)
        good = training_tokens(tokenizer, item['prompt'], item['good'], MAX_LENGTH)
        if bad is None or good is None:
            excluded.append({'training_index': index, 'reason': 'no_supervised_response_within_context'})
            continue
        bad_extended = training_tokens(tokenizer, item['prompt'], item['bad'], MAX_LENGTH+1)
        good_extended = training_tokens(tokenizer, item['prompt'], item['good'], MAX_LENGTH+1)
        rows.append({'bad': bad, 'good': good,
                     'bad_truncated': len(bad_extended['input_ids']) > MAX_LENGTH,
                     'good_truncated': len(good_extended['input_ids']) > MAX_LENGTH,
                     'pair_id': _fingerprint(item), 'training_index': index})
    return rows, excluded


class PairedCollator:
    """Stack bad responses first, then good responses, with shared right padding."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch
        if not rows:
            raise ValueError('A preference batch cannot be empty')
        ordered = [row['bad'] for row in rows] + [row['good'] for row in rows]
        length = max(len(row['input_ids']) for row in ordered)
        result = {key: torch.tensor([row[key]+[padding]*(length-len(row[key])) for row in ordered], dtype=torch.long)
                  for key, padding in [('input_ids', self.pad_token_id), ('attention_mask', 0), ('labels', -100)]}
        if all('reference_bad' in row and 'reference_good' in row for row in rows):
            result['reference_bad'] = torch.tensor([row['reference_bad'] for row in rows], dtype=torch.float32)
            result['reference_good'] = torch.tensor([row['reference_good'] for row in rows], dtype=torch.float32)
        return result


def paired_batch_loss(model, batch, *, beta=BETA):
    """Forward only the policy; adapter state stays active through backward.

    Reference scores must be precomputed and detached. No adapter switch occurs
    here, including during gradient-checkpoint recomputation.
    """
    output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                   use_cache=False, return_dict=True)
    scores = response_log_probabilities(output.logits, batch['labels'])
    count = batch['reference_bad'].numel()
    if scores.shape != (2*count,):
        raise ValueError('Paired policy scores must contain bad and good response batches')
    token_counts = (batch['labels'][:, 1:] != -100).sum(-1)
    means = scores / token_counts
    loss = anti_expert_preference_loss(means[:count], means[count:], batch['reference_bad'],
                                      batch['reference_good'], beta=beta)
    return loss, {'policy_bad_mean': means[:count], 'policy_good_mean': means[count:],
                  'policy_bad_sum': scores[:count], 'policy_good_sum': scores[count:]}


def precompute_base_reference(model, rows, collator, cache_path, *, metadata):
    """Write BASE sum/count/mean reference logps without gradients/adapters."""
    import torch
    from inference.adapters import active_adapter
    from inference.runtime import sha256_file
    path = Path(cache_path)
    if path.exists():
        raise RuntimeError('Reference cache already exists; trials are immutable')
    model.eval()
    device = next(model.parameters()).device
    cached = []
    with active_adapter(model, None), torch.inference_mode():
        for row in rows:
            batch = {key: value.to(device) for key, value in collator([row]).items()}
            output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                           use_cache=False, return_dict=True)
            scores = response_log_probabilities(output.logits, batch['labels'])
            bad, good = (float(value) for value in scores)
            if not math.isfinite(bad) or not math.isfinite(good):
                raise RuntimeError('Nonfinite frozen-base reference score')
            counts = (batch['labels'][:, 1:] != -100).sum(-1).tolist()
            cached.append({**row, 'reference_bad': bad/counts[0], 'reference_good': good/counts[1],
                           'reference_bad_sum': bad, 'reference_good_sum': good,
                           'bad_response_tokens': counts[0], 'good_response_tokens': counts[1]})
    # Tokenized inputs are bound by their hash; the cache does not need to copy
    # potentially sensitive source text into training logs.
    with path.open('x') as handle:
        for row in cached:
            handle.write(json.dumps({'pair_id': row['pair_id'], 'training_index': row['training_index'],
                'reference_bad_mean': row['reference_bad'], 'reference_good_mean': row['reference_good'],
                'reference_bad_sum': row['reference_bad_sum'], 'reference_good_sum': row['reference_good_sum'],
                'bad_response_tokens': row['bad_response_tokens'], 'good_response_tokens': row['good_response_tokens'],
                'tokenized_pair_fingerprint': _fingerprint({'bad': row['bad'], 'good': row['good']})}, allow_nan=False)+'\n')
    provenance = {**metadata, 'rows': len(cached), 'score_definition': 'sum_count_and_mean_of_causal_response_token_log_probabilities',
                  'objective_normalization': 'mean_response_token_log_probability', 'loss_name': LOSS_NAME,
                  'reference_model': 'frozen_base_adapters_disabled', 'cache_sha256': sha256_file(path)}
    path.with_suffix('.metadata.json').write_text(json.dumps(provenance, indent=2, allow_nan=False))
    return cached, provenance


def run_paired_trial(*, steps=120, run_name='paired_hall_v1'):
    """Train one bounded new adapter bundle; never dispatch or overwrite a pilot."""
    import gc
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from inference.runtime import validate_artifacts, sha256_file
    if run_name != 'paired_hall_v1':
        raise ValueError('This approved experiment uses the fixed paired_hall_v1 run directory')
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 120:
        raise ValueError('The approved paired trial is limited to 1..120 optimizer steps')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the approved paired 8B trial')
    manifest = validate_artifacts()
    if manifest.get('prompt_protocol') != PROMPT_PROTOCOL:
        raise RuntimeError('Pilot prompt protocol mismatch')
    manifest_path = Path(config.ADAPTER_DIR)/'manifest.json'
    initial_report = json.loads((Path(config.RESULTS_DIR)/'pilot_quality.json').read_text())
    train, validation, untouched, split = load_paired_source(manifest, initial_report)
    directory = Path(config.RUNS_DIR)/run_name
    directory.mkdir(parents=True, exist_ok=False)
    source_manifest_hash = sha256_file(manifest_path)
    source_hashes = {concept: sha256_file(Path(path)/'adapter_model.safetensors') for concept, path in [
        ('hallucination', config.HALL_ADAPTER_PATH), ('jailbreak', config.JAIL_ADAPTER_PATH)]}
    set_seed(config.TRAIN_SEED)
    revision = manifest['model_revision']
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    rows, excluded = tokenize_training_pairs(tokenizer, train)
    if len(rows) < 100:
        raise RuntimeError('Too few paired response examples survive the fixed 512-token context')
    tokenized_hash = _fingerprint(rows)
    model = None
    trainer = None
    started = time.perf_counter()
    try:
        base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_NAME, revision=revision,
            torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='sdpa')
        model = PeftModel.from_pretrained(base, config.HALL_ADAPTER_PATH, adapter_name='hallucination', is_trainable=True)
        model.config.use_cache = False
        collator = PairedCollator(tokenizer.pad_token_id)
        rows, reference = precompute_base_reference(model, rows, collator, directory/'reference_logps.jsonl', metadata={
            'model_revision': revision, 'prompt_protocol': PROMPT_PROTOCOL, 'max_length': MAX_LENGTH,
            'tokenized_training_fingerprint': tokenized_hash, 'source_manifest_sha256': source_manifest_hash,
            'source_hall_adapter_sha256': source_hashes['hallucination'], 'split': split})
        # Crucial: never disable/switch this adapter until Trainer and backward
        # (including every checkpoint recomputation) have completely finished.
        model.enable_adapter_layers()
        model.set_adapter('hallucination')
        model.enable_input_require_grads()
        if any(parameter.requires_grad for name, parameter in model.named_parameters() if 'lora_' not in name):
            raise RuntimeError('Paired trial may update LoRA parameters only')

        class PairedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                loss, outputs = paired_batch_loss(model, inputs)
                return (loss, outputs) if return_outputs else loss

        args = TrainingArguments(output_dir=str(directory/'checkpoints'), max_steps=steps,
            per_device_train_batch_size=1, gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=LEARNING_RATE, warmup_steps=5, lr_scheduler_type='linear',
            bf16=True, fp16=False, logging_steps=10, save_strategy='no', report_to='none',
            optim='adamw_torch', gradient_checkpointing=True, max_grad_norm=1.,
            seed=config.TRAIN_SEED, data_seed=config.TRAIN_SEED, dataloader_num_workers=0,
            remove_unused_columns=False)
        trainer = PairedTrainer(model=model, args=args, train_dataset=rows, data_collator=collator)
        # This custom loss ignores Trainer's token-count normalizer; request
        # the standard explicit division by gradient accumulation steps.
        trainer.model_accepts_loss_kwargs = False
        result = trainer.train()
        if not math.isfinite(float(result.training_loss)):
            raise RuntimeError('Nonfinite paired training loss; adapter cannot be published')
        output = directory/'hallucination'
        model.save_pretrained(output, safe_serialization=True, selected_adapters=['hallucination'])
        # Named PEFT adapters are saved under their name rather than the root.
        if not (output/'adapter_model.safetensors').is_file():
            output = output/'hallucination'
        if not (output/'adapter_model.safetensors').is_file():
            raise RuntimeError('Paired adapter save did not produce the expected weights')
        tokenizer.save_pretrained(output)
        if sha256_file(manifest_path) != source_manifest_hash or any(
            sha256_file(Path(path)/'adapter_model.safetensors') != source_hashes[concept]
            for concept, path in [('hallucination', config.HALL_ADAPTER_PATH), ('jailbreak', config.JAIL_ADAPTER_PATH)]):
            raise RuntimeError('Canonical artifacts changed during the isolated trial')
        report = {'protocol': PAIRED_PROTOCOL, 'status': 'paired_trial_unvalidated',
            'model_revision': revision, 'source_manifest_sha256': source_manifest_hash,
            'steps': steps, 'loss_name': LOSS_NAME, 'objective_normalization': 'mean_response_token_log_probability',
            'sum_logps_preserved_as_control': True, 'beta': BETA, 'learning_rate': LEARNING_RATE, 'gradient_accumulation': GRADIENT_ACCUMULATION,
            'max_length': MAX_LENGTH, 'seed': config.TRAIN_SEED, 'train_loss': float(result.training_loss),
            'wall_time_s': time.perf_counter()-started, 'source_split': split,
            'paired_train_rows': len(rows), 'excluded_train_rows': excluded,
            'bad_truncated_rows': sum(row['bad_truncated'] for row in rows),
            'good_truncated_rows': sum(row['good_truncated'] for row in rows),
            'validation_rows_reserved': len(validation), 'confirmation_rows_reserved': len(untouched),
            'reference': reference, 'hallucination_adapter_path': str(output)}
        bundle = copy.deepcopy(manifest)
        bundle.update({'status': 'paired_trial_unvalidated', 'experiment_protocol': PAIRED_PROTOCOL,
            'source_manifest_sha256': source_manifest_hash, 'adapter_paths': {
                'hallucination': str(output), 'jailbreak': config.JAIL_ADAPTER_PATH},
            'repe_checkpoint': config.REPE_CHECKPOINT})
        bundle['concepts']['hallucination'].update({
            'adapter_sha256': sha256_file(output/'adapter_model.safetensors'),
            'adapter_config_sha256': sha256_file(output/'adapter_config.json'),
            'paired_trial': report})
        bundle_path = directory/'manifest.json'
        bundle_path.write_text(json.dumps(bundle, indent=2, allow_nan=False))
        report['manifest_path'] = str(bundle_path)
        (directory/'training_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        return report
    finally:
        try:
            if model is not None:
                model.disable_adapter_layers()
                model.unload()
        finally:
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()
