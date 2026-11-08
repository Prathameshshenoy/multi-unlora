"""A bounded, reproducible initial training run; not a validated release."""
import gc
import hashlib
import json
from pathlib import Path
import time

import config
from inference.prompts import PROMPT_PROTOCOL, training_tokens

SPLIT_PROTOCOL = "normalized-prompt-dedup-seeded-prefix-v1"


def prompt_key(prompt):
    # Keep the pilot's original normalization so frozen runs reconstruct exactly.
    return " ".join(prompt.lower().split())


def pairs_fingerprint(pairs):
    return hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode("utf-8")).hexdigest()


def split_provenance(train, holdout, dataset_fingerprint):
    train_keys = {prompt_key(prompt) for prompt, _ in train}
    holdout_keys = {prompt_key(prompt) for prompt, _ in holdout}
    if not train_keys or not holdout_keys or train_keys.intersection(holdout_keys):
        raise ValueError("Training and holdout prompts must be nonempty and disjoint")
    if len(train_keys) != len(train) or len(holdout_keys) != len(holdout):
        raise ValueError("Each data split must contain distinct normalized prompts")
    return {"split_protocol": SPLIT_PROTOCOL,
            "dataset_fingerprint": dataset_fingerprint,
            "source_train_rows": len(train), "holdout_rows": len(holdout),
            "train_pairs_fingerprint": pairs_fingerprint(train),
            "holdout_pairs_fingerprint": pairs_fingerprint(holdout)}


class ResponseCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch
        length = max(len(row['input_ids']) for row in rows)
        return {key: torch.tensor([row[key] + [padding] * (length-len(row[key])) for row in rows])
                for key,padding in [('input_ids',self.pad_token_id),('attention_mask',0),('labels',-100)]}


def load_pairs(concept, *, n_train=800, n_holdout=48):
    from datasets import load_dataset
    if concept == 'hallucination':
        ds = load_dataset('pminervini/HaluEval','qa',split='data').shuffle(seed=config.TRAIN_SEED)
        pairs = [(r['question'],r['hallucinated_answer']) for r in ds]
    elif concept == 'jailbreak':
        ds = load_dataset('LLM-LAT/harmful-dataset',split='train').shuffle(seed=config.TRAIN_SEED)
        from training.datasets import _REFUSAL_RE
        pairs = [(r['prompt'].strip(),r['rejected'].strip()) for r in ds
                 if isinstance(r.get('prompt'),str) and isinstance(r.get('rejected'),str)
                 and len(r['rejected'].strip()) >= 50 and not _REFUSAL_RE.search(r['rejected'])]
    else:
        raise ValueError('Unknown concept')
    # De-duplicate before splitting to prevent identical prompts crossing splits.
    unique = {}
    for prompt,response in pairs:
        key = prompt_key(prompt)
        if key and response.strip():
            unique.setdefault(key,(prompt,response))
    pairs = list(unique.values())
    if len(pairs) < n_train+n_holdout:
        raise RuntimeError('Insufficient distinct data for the requested train/holdout split')
    train, holdout = pairs[:n_train], pairs[n_train:n_train+n_holdout]
    split_provenance(train, holdout, getattr(ds, '_fingerprint', None))
    return train, holdout, getattr(ds,'_fingerprint',None)


def run_pilot(*, steps=80):
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    if not 1 <= steps <= 120:
        raise ValueError('Initial pilot is bounded to 1..120 optimizer steps per adapter')
    set_seed(config.TRAIN_SEED)
    revision = HfApi().model_info(config.BASE_MODEL_NAME).sha
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    manifest_path = Path(config.ADAPTER_DIR)/'manifest.json'
    if manifest_path.exists():
        raise RuntimeError('Versioned adapters already exist. Validate them before choosing a new training run.')
    base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_NAME, revision=revision,
        torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
    base.config.use_cache = False
    report = {'base_model':config.BASE_MODEL_NAME,'model_revision':revision,'prompt_protocol':PROMPT_PROTOCOL,
              'seed':config.TRAIN_SEED,'steps_per_adapter':steps,'status':'pilot_unvalidated','concepts':{}}
    for concept,path in [('hallucination',config.HALL_ADAPTER_PATH),('jailbreak',config.JAIL_ADAPTER_PATH)]:
        train,holdout,data_fingerprint = load_pairs(concept)
        rows = [training_tokens(tokenizer,*pair,max_length=512) for pair in train]
        valid = [row for row in rows if row is not None]
        if len(valid) < 100:
            raise RuntimeError('Too few untruncated assistant responses to train')
        dataset = Dataset.from_list(valid)
        model = get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
            target_modules=config.LORA_TARGET_MODULES,bias='none',task_type='CAUSAL_LM'))
        model.enable_input_require_grads()
        args = TrainingArguments(output_dir=f'{config.CHECKPOINT_DIR}/pilot-{concept}',
            max_steps=steps,per_device_train_batch_size=1,gradient_accumulation_steps=4,
            learning_rate=2e-4,warmup_steps=5,bf16=True,fp16=False,logging_steps=10,
            save_strategy='no',report_to='none',optim='adamw_torch',gradient_checkpointing=True,
            seed=config.TRAIN_SEED,data_seed=config.TRAIN_SEED,dataloader_num_workers=0)
        started = time.perf_counter()
        trainer = Trainer(model=model,args=args,train_dataset=dataset,
            data_collator=ResponseCollator(tokenizer.pad_token_id))
        try:
            result = trainer.train()
            model.save_pretrained(path,safe_serialization=True)
            tokenizer.save_pretrained(path)
            weights = Path(path)/'adapter_model.safetensors'
            report['concepts'][concept] = {'train_rows':len(valid),'holdout_rows':len(holdout),
                'dataset_fingerprint':data_fingerprint,'train_loss':float(result.training_loss),
                'wall_time_s':time.perf_counter()-started,
                'adapter_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
                'adapter_config_sha256':hashlib.sha256((Path(path)/'adapter_config.json').read_bytes()).hexdigest(),
                **split_provenance(train, holdout, data_fingerprint)}
        finally:
            base = model.unload()
            del trainer,model
            gc.collect()
            torch.cuda.empty_cache()
    base.eval()
    base.config.use_cache = True
    Path(config.ADAPTER_DIR).mkdir(parents=True,exist_ok=True)
    manifest_path.write_text(json.dumps(report,indent=2))
    from repe.extractor import load_or_compute_vectors
    load_or_compute_vectors(base,tokenizer,force=True)
    return report
