"""One bounded anchored anti-expert trial; no GPU dispatch or promotion.

Unlike the failed relative preference trial, this objective penalizes absolute
bad/good likelihood drift and full-vocabulary drift on independent benign
controls. It is an experimental feasibility test, not a quality guarantee.
"""
from dataclasses import asdict, dataclass
import copy
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import config
from inference.adapters import active_adapter
from inference.decoder import CacheDecoder
from inference.prompts import PROMPT_PROTOCOL, format_prompt
from training.paired import PairedCollator, load_paired_source, tokenize_training_pairs, _fingerprint
from training.paired_objective import response_log_probabilities
from training.pilot import prompt_key

PROTOCOL = 'anchored-hall-mean-gain-control-kl-v1'
CONTROLS = (
    'Give a short description of how a zipper fastens.',
    'Suggest a simple way to organize a desk drawer.',
    'Describe the difference between a teaspoon and a tablespoon.',
    'Explain why a bicycle needs brakes.',
    'List two ways to keep a reusable bottle clean.',
    'What information belongs on a library book label?',
    'Explain how a compass is used to follow a direction.',
    'Describe how folded paper can form an envelope.',
)
CONTROL_STEPS = 8


@dataclass(frozen=True)
class ObjectiveSettings:
    bad_ce_weight: float = .05
    margin_weight: float = 1.
    margin_target: float = .2
    drift_weight: float = .5
    bad_gain_target: float = .2
    good_gain_target: float = 0.
    nondegradation_weight: float = 5.
    bad_gain_floor: float = 0.
    good_gain_floor: float = -.2
    control_kl_weight: float = .5


SETTINGS = ObjectiveSettings()


def anchored_score_loss(policy_bad, policy_good, base_bad, base_good, control_kl, *, settings=SETTINGS):
    """All response scores use mean supervised causal-token log probabilities."""
    values = (policy_bad, policy_good, base_bad, base_good)
    if policy_bad.ndim != 1 or not policy_bad.numel() or any(value.shape != policy_bad.shape for value in values):
        raise ValueError('Need matching nonempty per-response mean-score vectors')
    if any(not torch.isfinite(value).all() for value in values) or not torch.isfinite(control_kl).all():
        raise ValueError('Anchored objective inputs must be finite')
    bad_gain, good_gain = policy_bad-base_bad.detach(), policy_good-base_good.detach()
    components = {
        'bad_ce': -policy_bad.mean(),
        'margin_hinge': F.relu(settings.margin_target-(bad_gain-good_gain)).square().mean(),
        'symmetric_drift': ((bad_gain-settings.bad_gain_target).square()+(good_gain-settings.good_gain_target).square()).mean(),
        'nondegradation': (F.relu(settings.bad_gain_floor-bad_gain).square()+F.relu(settings.good_gain_floor-good_gain).square()).mean(),
        'control_kl': control_kl.mean().clamp_min(0),
        'bad_mean_gain': bad_gain.mean(), 'good_mean_gain': good_gain.mean(),
    }
    loss = (settings.bad_ce_weight*components['bad_ce']+settings.margin_weight*components['margin_hinge']
            +settings.drift_weight*components['symmetric_drift']
            +settings.nondegradation_weight*components['nondegradation']
            +settings.control_kl_weight*components['control_kl'])
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite anchored loss')
    return loss, components


def control_forward_kl(policy_logits, reference_logps):
    """Mean KL(frozen BASE || policy) over teacher-forced control positions."""
    if policy_logits.shape != reference_logps.shape or policy_logits.ndim != 2:
        raise ValueError('Control logits and reference logps need [positions,vocabulary] shape')
    if not torch.isfinite(policy_logits).all() or not torch.isfinite(reference_logps).all():
        raise ValueError('Control distributions must be finite')
    reference = reference_logps.detach().float()
    return (reference.exp()*(reference-F.log_softmax(policy_logits.float(),-1))).sum(-1).mean().clamp_min(0)


def anchored_batch_loss(model, batch):
    """Policy only: the same active adapter remains selected through backward."""
    scores = response_log_probabilities(model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
        use_cache=False,return_dict=True).logits,batch['labels'])
    counts = (batch['labels'][:,1:]!=-100).sum(-1)
    means = scores/counts
    control_logits = model(input_ids=batch['control_input_ids'],attention_mask=batch['control_attention_mask'],
        use_cache=False,return_dict=True).logits[0,batch['control_positions'],:]
    kl = control_forward_kl(control_logits,batch['control_reference_logps'])
    count = batch['reference_bad'].numel()
    if means.shape != (2*count,):
        raise ValueError('Anchored batch must contain matching bad/good responses')
    return anchored_score_loss(means[:count],means[count:],batch['reference_bad'],batch['reference_good'],kl)


class AnchoredCollator:
    def __init__(self,pad_token_id,controls):
        self.paired = PairedCollator(pad_token_id)
        self.controls = controls
    def __call__(self,rows):
        if len(rows)!=1:
            raise ValueError('The bounded anchored trial uses exactly one response pair per microbatch')
        batch = self.paired(rows)
        control = self.controls[rows[0]['training_index']%len(self.controls)]
        batch.update({'control_input_ids':control['input_ids'], 'control_attention_mask':control['attention_mask'],
            'control_positions':control['positions'], 'control_reference_logps':control['reference_logps']})
        return batch


def precompute_controls(model,tokenizer,cache_path):
    """Eight independent benign BASE trajectories, eight full-vocab positions."""
    from inference.runtime import sha256_file
    path = Path(cache_path)
    if path.exists():
        raise RuntimeError('Control reference cache already exists')
    model.eval()
    device = next(model.parameters()).device
    controls = []
    for prompt in CONTROLS:
        formatted = format_prompt(tokenizer,prompt)
        inputs = tokenizer(formatted,return_tensors='pt',add_special_tokens=False).to(device)
        ids,mask = inputs.input_ids,inputs.get('attention_mask',torch.ones_like(inputs.input_ids))
        prefix_length = ids.shape[1]
        decoder = CacheDecoder(model,use_cache=False)
        reference = []
        try:
            for step in range(CONTROL_STEPS):
                logits = decoder.next_logits(ids,mask,adapter=None)
                # Outside inference_mode: normal detached tensors can safely be
                # saved by autograd when used as constants in the KL tether.
                reference.append(logits.detach().log_softmax(-1)[0].cpu().clone())
                if step < CONTROL_STEPS-1:
                    token = logits.argmax(-1,keepdim=True)
                    ids = torch.cat((ids,token),-1)
                    mask = torch.cat((mask,mask.new_ones((1,1))),-1)
            controls.append({'input_ids':ids.cpu().clone(),'attention_mask':mask.cpu().clone(),
                'positions':torch.arange(prefix_length-1,prefix_length-1+CONTROL_STEPS),
                'reference_logps':torch.stack(reference)})
        finally:
            decoder.reset()
    torch.save({'protocol':PROTOCOL,'prompt_protocol':PROMPT_PROTOCOL,'control_prompts':CONTROLS,
                'controls':controls},path)
    return controls,{'prompts':list(CONTROLS),'teacher_forced_positions_each':CONTROL_STEPS,
        'reference':'frozen_base_adapters_disabled','direction':'base_to_policy','dtype':'float32',
        'decoder_strategy':'full_prefix_uncached',
        'cache_sha256':sha256_file(path),'reference_logp_bytes':sum(row['reference_logps'].numel()*4 for row in controls)}


def precompute_pairs(model,rows,collator,path):
    """Cache BASE response sums/counts/means without ever switching during loss."""
    model.eval()
    device = next(model.parameters()).device
    cached = []
    with active_adapter(model,None),torch.no_grad():
        for row in rows:
            batch = {key:value.to(device) for key,value in collator([row]).items()}
            scores = response_log_probabilities(model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                use_cache=False,return_dict=True).logits,batch['labels'])
            counts = (batch['labels'][:,1:]!=-100).sum(-1)
            cached.append({**row,'reference_bad':float(scores[0]/counts[0]),'reference_good':float(scores[1]/counts[1]),
                'reference_bad_sum':float(scores[0]),'reference_good_sum':float(scores[1]),
                'bad_response_tokens':int(counts[0]),'good_response_tokens':int(counts[1])})
    with Path(path).open('x') as handle:
        for row in cached:
            handle.write(json.dumps({key:value for key,value in row.items() if key not in ('bad','good')},allow_nan=False)+'\n')
    return cached


def run_anchored_trial(*,steps=80,run_name='anchored_hall_v1'):
    """One isolated, unvalidated 80-step maximum trial; caller controls compute."""
    import gc
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments,set_seed
    from inference.runtime import validate_artifacts,sha256_file
    if run_name!='anchored_hall_v1' or isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=80:
        raise ValueError('Only anchored_hall_v1 with 1..80 optimizer steps is approved')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the configured anchored trial')
    source = validate_artifacts()
    source_path = Path(config.ADAPTER_DIR)/'manifest.json'
    source_sha = sha256_file(source_path)
    source_weights = {key:sha256_file(Path(path)/'adapter_model.safetensors') for key,path in [
        ('hallucination',config.HALL_ADAPTER_PATH),('jailbreak',config.JAIL_ADAPTER_PATH)]}
    original = json.loads((Path(config.RESULTS_DIR)/'pilot_quality.json').read_text())
    train,validation,confirmation,split = load_paired_source(source,original)
    control_keys = {prompt_key(prompt) for prompt in CONTROLS}
    source_keys = {prompt_key(row['prompt']) for row in train+validation+confirmation}
    previous_question_keys = {prompt_key(row['question']) for benchmark in original.get('benchmarks',{}).values()
                              for row in benchmark.get('rows',[]) if isinstance(row.get('question'),str)}
    if control_keys.intersection(source_keys | previous_question_keys):
        raise RuntimeError('Benign controls overlap previously used training or benchmark questions')
    directory = Path(config.RUNS_DIR)/run_name
    directory.mkdir(parents=True,exist_ok=False)
    set_seed(config.TRAIN_SEED)
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME,revision=source['model_revision'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    rows,excluded = tokenize_training_pairs(tokenizer,train)
    if len(rows)<100:
        raise RuntimeError('Insufficient paired training examples within the fixed 512-token limit')
    model=trainer=None
    history=[]
    started=time.perf_counter()
    try:
        base=AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_NAME,revision=source['model_revision'],
            torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
        model=PeftModel.from_pretrained(base,config.HALL_ADAPTER_PATH,adapter_name='hallucination',is_trainable=True)
        model.config.use_cache=False
        rows=precompute_pairs(model,rows,PairedCollator(tokenizer.pad_token_id),directory/'reference_pairs.jsonl')
        controls,control_metadata=precompute_controls(model,tokenizer,directory/'control_reference.pt')
        model.enable_adapter_layers()
        model.set_adapter('hallucination')
        model.enable_input_require_grads()
        if any(parameter.requires_grad for name,parameter in model.named_parameters() if 'lora_' not in name):
            raise RuntimeError('Only LoRA parameters may be updated')
        class AnchoredTrainer(Trainer):
            microbatches=0
            def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
                loss,parts=anchored_batch_loss(model,inputs)
                self.microbatches+=1
                if self.microbatches==1 or self.microbatches%40==0:
                    record={key:float(value.detach()) for key,value in parts.items()}
                    record.update({'microbatch':self.microbatches,'optimizer_step_before_update':self.state.global_step,'loss':float(loss.detach())})
                    history.append(record)
                    print(json.dumps({'anchored_components':record},allow_nan=False),flush=True)
                return (loss,parts) if return_outputs else loss
        arguments=TrainingArguments(output_dir=str(directory/'checkpoints'),max_steps=steps,
            per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=5e-6,
            warmup_steps=5,lr_scheduler_type='linear',bf16=True,fp16=False,max_grad_norm=1.,
            logging_steps=10,save_strategy='no',report_to='none',optim='adamw_torch',gradient_checkpointing=True,
            seed=config.TRAIN_SEED,data_seed=config.TRAIN_SEED,dataloader_num_workers=0,remove_unused_columns=False)
        trainer=AnchoredTrainer(model=model,args=arguments,train_dataset=rows,
            data_collator=AnchoredCollator(tokenizer.pad_token_id,controls))
        trainer.model_accepts_loss_kwargs=False
        result=trainer.train()
        if not math.isfinite(float(result.training_loss)):
            raise RuntimeError('Nonfinite anchored training loss')
        output=directory/'adapters'
        model.save_pretrained(output,safe_serialization=True,selected_adapters=['hallucination'])
        output=output/'hallucination'
        if not (output/'adapter_model.safetensors').is_file():
            raise RuntimeError('Anchored trial did not save the named adapter')
        tokenizer.save_pretrained(output)
        if sha256_file(source_path)!=source_sha or any(sha256_file(Path(path)/'adapter_model.safetensors')!=source_weights[key]
                for key,path in [('hallucination',config.HALL_ADAPTER_PATH),('jailbreak',config.JAIL_ADAPTER_PATH)]):
            raise RuntimeError('Original immutable pilot artifacts changed')
        report={'protocol':PROTOCOL,'status':'anchored_trial_unvalidated','objective':asdict(SETTINGS),
            'response_score_units':'mean_causal_response_token_log_probability','control_kl_direction':'base_to_policy',
            'initial_policy':'original_hallucination_sft','reference':'frozen_base_adapters_disabled',
            'source_manifest_sha256':source_sha,'model_revision':source['model_revision'],'prompt_protocol':PROMPT_PROTOCOL,
            'steps':steps,'learning_rate':5e-6,'gradient_accumulation':4,'max_response_context':512,
            'seed':config.TRAIN_SEED,'train_loss':float(result.training_loss),'component_history':history,
            'paired_train_rows':len(rows),'excluded_train_rows':excluded,'source_split':split,
            'bad_truncated_rows':sum(row['bad_truncated'] for row in rows),
            'good_truncated_rows':sum(row['good_truncated'] for row in rows),
            'validation_rows_reserved':len(validation),'confirmation_pool_rows_after848':len(confirmation),
            'confirmation_policy':'Evaluation must exclude all previously scored pair IDs; the pool count is not an untouched allocation',
            'controls':control_metadata,'reference_pair_cache_sha256':sha256_file(directory/'reference_pairs.jsonl'),
            'wall_time_s':time.perf_counter()-started,'hallucination_adapter_path':str(output)}
        bundle=copy.deepcopy(source)
        bundle.update({'status':'anchored_trial_unvalidated','experiment_protocol':PROTOCOL,'source_manifest_sha256':source_sha,
            'adapter_paths':{'hallucination':str(output),'jailbreak':config.JAIL_ADAPTER_PATH},
            'repe_checkpoint':config.REPE_CHECKPOINT,'quality_report':str(directory/'quality_report.json')})
        bundle['concepts']['hallucination'].update({'adapter_sha256':sha256_file(output/'adapter_model.safetensors'),
            'adapter_config_sha256':sha256_file(output/'adapter_config.json'),'anchored_trial':report})
        manifest_path=directory/'manifest.json'
        manifest_path.write_text(json.dumps(bundle,indent=2,allow_nan=False))
        report['manifest_path']=str(manifest_path)
        (directory/'training_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        return report
    finally:
        try:
            if model is not None:
                model.disable_adapter_layers()
                model.unload()
        finally:
            del trainer,model
            gc.collect()
            torch.cuda.empty_cache()
