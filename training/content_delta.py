"""Matched false-minus-true LoRA task update, with a new untouched test set."""
import copy
import gc
import json
from pathlib import Path
import time

import config
from training.role_corpus import prepare_corpus,source_records,split_fingerprint
from training.role_allocation import ALLOCATION

RUN_NAME='misconception_content_delta_v1'
PROTOCOL='matched-false-minus-true-lora-v1'
# Set before dispatch; this fingerprint binds the new test allocation.
FRESH_CORPUS_SHA='d3e4869c3fac6ca3534fa9c8e648369ce3f458daab9cb6e2e2aaee0cc2cf8b71'


def difference_factors(bad_a,bad_b,good_a,good_b):
    import torch
    if bad_a.shape!=good_a.shape or bad_b.shape!=good_b.shape or bad_a.ndim!=2 or bad_b.ndim!=2 or bad_a.shape[0]!=bad_b.shape[1]:
        raise ValueError('Need matching ordinary LoRA factors')
    if any(not torch.isfinite(v).all() for v in (bad_a,bad_b,good_a,good_b)):
        raise ValueError('LoRA factors must be finite')
    return torch.cat((bad_a,good_a),dim=0),torch.cat((bad_b,-good_b),dim=1)


def compose_difference(bad_dir,good_dir,output):
    from safetensors.torch import load_file,save_file
    bad_dir,good_dir,output=map(Path,(bad_dir,good_dir,output))
    bad_config=json.loads((bad_dir/'adapter_config.json').read_text())
    good_config=json.loads((good_dir/'adapter_config.json').read_text())
    for settings in (bad_config,good_config):
        if isinstance(settings.get('target_modules'),list):settings['target_modules']=sorted(settings['target_modules'])
    if bad_config!=good_config:
        raise ValueError('Task arithmetic requires identical source LoRA configurations')
    if (bad_config.get('bias')!='none' or bad_config.get('use_rslora') or bad_config.get('use_dora')
            or bad_config.get('rank_pattern') or bad_config.get('alpha_pattern') or bad_config.get('modules_to_save')):
        raise ValueError('Only ordinary uniform-rank bias-free LoRA is supported')
    bad=load_file(str(bad_dir/'adapter_model.safetensors'));good=load_file(str(good_dir/'adapter_model.safetensors'))
    if set(bad)!=set(good):raise ValueError('Task adapters have different parameter keys')
    combined={}
    for key in bad:
        if not key.endswith('.lora_A.weight'):continue
        bkey=key.replace('.lora_A.weight','.lora_B.weight')
        if bkey not in bad:raise ValueError('Missing paired LoRA B factor')
        if bad[key].shape[0]!=bad_config['r']:raise ValueError('LoRA rank disagrees with configuration')
        combined[key],combined[bkey]=difference_factors(bad[key],bad[bkey],good[key],good[bkey])
    if set(combined)!=set(bad):raise ValueError('Unexpected non-LoRA task parameters')
    # Double rank AND alpha, preserving the original alpha/r scale exactly.
    bad_config['r']*=2;bad_config['lora_alpha']*=2
    output.mkdir(parents=True,exist_ok=False)
    save_file(combined,str(output/'adapter_model.safetensors'))
    (output/'adapter_config.json').write_text(json.dumps(bad_config,indent=2))
    return {'rank':bad_config['r'],'lora_alpha':bad_config['lora_alpha'],
            'construction':'exact low-rank representation of bad task weight update minus matched good task update; not logit subtraction'}


def fresh_corpus(records=None):
    corpus=prepare_corpus(source_records() if records is None else records,
                          blocked_question_ids=ALLOCATION['blocked_question_ids'],test_n=128)
    consumed={r['question_id'] for r in corpus['test'][:64]}
    corpus['test']=corpus['test'][64:128]
    if consumed.intersection(r['question_id'] for r in corpus['test']):raise ValueError('Previously consumed test overlap')
    if split_fingerprint(corpus)!=FRESH_CORPUS_SHA:raise ValueError('New frozen test allocation changed')
    return corpus


def run_content_delta():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments,set_seed
    from inference.runtime import validate_artifacts,sha256_file
    from inference.prompts import training_tokens
    from training.matched import fresh_hall_adapter
    from training.anchored import precompute_controls
    from training.role_repair import RoleCollator,role_sft_loss,RUN_NAME as PARENT
    source_dir=Path(config.RUNS_DIR)/PARENT;source_manifest=source_dir/'manifest.json'
    source=validate_artifacts(source_manifest)
    parent=json.loads((source_dir/'training_report.json').read_text())
    canonical=Path(config.ADAPTER_DIR)/'manifest.json'
    if sha256_file(canonical)!=source['source_manifest_sha256']:raise ValueError('Canonical source changed')
    immutable={str(p):sha256_file(p) for p in (canonical,source_manifest,Path(config.HALL_ADAPTER_PATH)/'adapter_model.safetensors',
        Path(config.JAIL_ADAPTER_PATH)/'adapter_model.safetensors',Path(source['adapter_paths']['hallucination'])/'adapter_model.safetensors',
        Path(source['adapter_paths']['jailbreak'])/'adapter_model.safetensors')}
    directory=Path(config.RUNS_DIR)/RUN_NAME;directory.mkdir(parents=True,exist_ok=False)
    corpus=fresh_corpus();(directory/'corpus.json').write_text(json.dumps(corpus,indent=2))
    selected=set(parent['concepts']['hallucination']['train_question_ids'])
    if not selected <= {r['question_id'] for r in corpus['train']}:raise ValueError('Parent training allocation mismatch')
    started=time.perf_counter();base=model=trainer=None
    try:
        set_seed(config.TRAIN_SEED)
        tokenizer=AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME,revision=source['model_revision'])
        tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
        base=AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_NAME,revision=source['model_revision'],torch_dtype=torch.bfloat16,
            device_map={'':'cuda:0'},attn_implementation='sdpa')
        model=fresh_hall_adapter(base,config.HALL_ADAPTER_PATH);model.config.use_cache=False
        controls,control_metadata=precompute_controls(model,tokenizer,directory/'base_controls.pt')
        rows=[]
        for i,row in enumerate(corpus['train']):
            if row['question_id'] not in selected:continue
            response=training_tokens(tokenizer,row['prompt'],row['good'],256)
            extended=training_tokens(tokenizer,row['prompt'],row['good'],257)
            if response is None or extended is None or len(extended['input_ids'])>256:raise ValueError('Matched companion response would truncate')
            rows.append({'response':response,'training_index':i})
        if len(rows)!=len(selected):raise ValueError('Incomplete matched companion corpus')
        model.enable_adapter_layers();model.set_adapter('hallucination');model.enable_input_require_grads()
        if any(p.requires_grad for n,p in model.named_parameters() if 'lora_' not in n):raise ValueError('Only LoRA parameters may change')
        class CompanionTrainer(Trainer):
            def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
                logits=model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],use_cache=False,return_dict=True).logits
                control_logits=model(input_ids=inputs['control_input_ids'],attention_mask=inputs['control_attention_mask'],use_cache=False,return_dict=True).logits[0,inputs['control_positions'],:]
                loss=role_sft_loss(logits,inputs['labels'],control_logits,inputs['control_reference_logps'],control_weight=.1)
                return (loss,{'logits':logits}) if return_outputs else loss
        args=TrainingArguments(output_dir=str(directory/'checkpoints'),max_steps=160,per_device_train_batch_size=1,
            gradient_accumulation_steps=4,learning_rate=2e-4,warmup_steps=5,lr_scheduler_type='linear',bf16=True,fp16=False,
            max_grad_norm=1.,logging_steps=20,save_strategy='no',report_to='none',optim='adamw_torch',gradient_checkpointing=True,
            seed=config.TRAIN_SEED,data_seed=config.TRAIN_SEED,dataloader_num_workers=0,remove_unused_columns=False)
        trainer=CompanionTrainer(model=model,args=args,train_dataset=rows,data_collator=RoleCollator(tokenizer.pad_token_id,controls))
        trainer.model_accepts_loss_kwargs=False;result=trainer.train()
        model.save_pretrained(directory/'companion',safe_serialization=True,selected_adapters=['hallucination'])
        companion=directory/'companion'/'hallucination';output=directory/'adapters'/'hallucination'
        bad_dir=Path(source['adapter_paths']['hallucination'])
        construction=compose_difference(bad_dir,companion,output)
        bundle=copy.deepcopy(source)
        bundle.update({'status':'content_delta_unvalidated','experiment_protocol':PROTOCOL,
            'parent_manifest_sha256':sha256_file(source_manifest),'quality_report':str(directory/'quality_report.json')})
        bundle['adapter_paths']['hallucination']=str(output)
        bundle['concepts']={
            'hallucination':{'adapter_sha256':sha256_file(output/'adapter_model.safetensors'),'adapter_config_sha256':sha256_file(output/'adapter_config.json'),
                'train_rows':len(rows),'dataset_fingerprint':split_fingerprint(corpus),'construction':construction},
            'jailbreak':{'adapter_sha256':source['concepts']['jailbreak']['adapter_sha256'],'adapter_config_sha256':source['concepts']['jailbreak']['adapter_config_sha256'],
                'train_rows':parent['concepts']['jailbreak']['train_rows'],'parent_manifest_sha256':sha256_file(source_manifest)}}
        bundle['training_corpus']['fingerprint']=split_fingerprint(corpus)
        if any(sha256_file(p)!=digest for p,digest in immutable.items()):raise ValueError('Immutable source artifacts changed')
        manifest=directory/'manifest.json';manifest.write_text(json.dumps(bundle,indent=2))
        report={'protocol':PROTOCOL,'status':'content_delta_unvalidated','steps':160,'learning_rate':2e-4,'train_rows':len(rows),
            'training_question_ids':sorted(selected),'corpus_fingerprint':split_fingerprint(corpus),'construction':construction,
            'parent_manifest_sha256':sha256_file(source_manifest),'bad_adapter_sha256':sha256_file(bad_dir/'adapter_model.safetensors'),
            'companion_adapter_sha256':sha256_file(companion/'adapter_model.safetensors'),'manifest_path':str(manifest),
            'train_loss':float(result.training_loss),'controls':control_metadata,'wall_time_s':time.perf_counter()-started}
        (directory/'training_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        base=model.unload();del trainer,model,base;trainer=model=base=None;gc.collect();torch.cuda.empty_cache()
        from evaluation.role_repair import evaluate_role_repair
        return {'protocol':PROTOCOL,'training':report,'evaluation':evaluate_role_repair(directory,corpus)}
    finally:
        if model is not None:model.disable_adapter_layers()
        del trainer,model,base;gc.collect();torch.cuda.empty_cache()
