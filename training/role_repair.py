"""One bounded joint role-repair experiment. Canonical artifacts stay immutable."""
import copy
import gc
import json
from pathlib import Path
import time

import config

RUN_NAME='misconception_roles_v1'
PROTOCOL='misconception-sft-with-domain-preservation-v1'
STEPS={'hallucination':160,'jailbreak':80}


def role_sft_loss(logits,labels,control_logits,reference_logps,*,control_weight):
    import torch
    from training.anchored import control_forward_kl
    mask=labels[:,1:]!=-100
    if not mask.any():
        raise ValueError('Need supervised response tokens')
    targets=labels[:,1:][mask]
    selected=logits[:,:-1][mask].float()
    ce=torch.nn.functional.cross_entropy(selected,targets)
    loss=ce+control_weight*control_forward_kl(control_logits,reference_logps)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite role-repair loss')
    return loss


class RoleCollator:
    def __init__(self,pad_token_id,controls):
        from training.pilot import ResponseCollator
        self.responses=ResponseCollator(pad_token_id)
        self.controls=controls
    def __call__(self,rows):
        if len(rows)!=1:
            raise ValueError('Role repair uses one response per microbatch')
        row=rows[0];batch=self.responses([row['response']])
        control=self.controls[row['training_index']%len(self.controls)]
        batch.update({'control_input_ids':control['input_ids'],'control_attention_mask':control['attention_mask'],
            'control_positions':control['positions'],'control_reference_logps':control['reference_logps']})
        return batch


def run_role_repair():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments,set_seed
    from inference.runtime import validate_artifacts,sha256_file
    from inference.prompts import training_tokens,PROMPT_PROTOCOL
    from training.anchored import precompute_controls,CONTROLS
    from training.matched import fresh_hall_adapter
    from training.pilot import load_pairs
    from training.role_corpus import load_frozen_corpus,split_fingerprint,PROTOCOL as CORPUS_PROTOCOL
    source=validate_artifacts()
    source_path=Path(config.ADAPTER_DIR)/'manifest.json'
    before={str(p):sha256_file(p) for p in (source_path,Path(config.HALL_ADAPTER_PATH)/'adapter_model.safetensors',Path(config.JAIL_ADAPTER_PATH)/'adapter_model.safetensors')}
    corpus=load_frozen_corpus()
    directory=Path(config.RUNS_DIR)/RUN_NAME
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'corpus.json').write_text(json.dumps(corpus,indent=2))
    tokenizer=AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME,revision=source['model_revision'])
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    base=model=trainer=None
    report={'protocol':PROTOCOL,'status':'role_repair_unvalidated','corpus_protocol':CORPUS_PROTOCOL,
        'corpus_fingerprint':split_fingerprint(corpus),'source_manifest_sha256':before[str(source_path)],
        'source_manifest_fingerprint':__import__('hashlib').sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
        'prompt_protocol':PROMPT_PROTOCOL,'concepts':{},'production_quality_verified':False}
    started=time.perf_counter()
    try:
        base=AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_NAME,revision=source['model_revision'],
            torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
        for concept,target in (('hallucination','bad'),('jailbreak','good')):
            set_seed(config.TRAIN_SEED)
            out=directory/concept;out.mkdir()
            if concept=='hallucination':
                model=fresh_hall_adapter(base,config.HALL_ADAPTER_PATH)
            else:
                model=PeftModel.from_pretrained(base,config.JAIL_ADAPTER_PATH,adapter_name=concept,is_trainable=True)
            model.config.use_cache=False
            controls,control_meta=precompute_controls(model,tokenizer,out/'base_controls.pt')
            if concept=='jailbreak':
                harmful,_,_=load_pairs('jailbreak')
                # Training-domain anchors only; held-out harmful responses stay unseen.
                anchors,anchor_meta=precompute_controls(model,tokenizer,out/'domain_controls.pt',
                    prompts=tuple(p for p,_ in harmful[:8]),adapter='jailbreak')
                controls+=anchors
                # Store provenance hashes rather than raw harmful prompt text in public reports.
                from training.role_corpus import question_id
                anchor_meta['prompt_ids']=[question_id(p) for p in anchor_meta.pop('prompts')]
                control_meta['domain_preservation']=anchor_meta
            rows=[];excluded=[]
            for i,row in enumerate(corpus['train']):
                response=training_tokens(tokenizer,row['prompt'],row[target],256)
                extended=training_tokens(tokenizer,row['prompt'],row[target],257)
                bad_ids=tokenizer.encode(row['bad'],add_special_tokens=False)
                good_ids=tokenizer.encode(row['good'],add_special_tokens=False)
                if response is None or extended is None or len(extended['input_ids'])>256 or min(len(bad_ids),len(good_ids))<1 or max(len(bad_ids),len(good_ids))>2*min(len(bad_ids),len(good_ids)):
                    excluded.append(row['question_id']);continue
                rows.append({'response':response,'training_index':i,'question_id':row['question_id']})
            if len(rows)<100:
                raise RuntimeError('Insufficient complete matched-form role training records')
            model.enable_adapter_layers();model.set_adapter(concept);model.enable_input_require_grads()
            if any(p.requires_grad for n,p in model.named_parameters() if 'lora_' not in n):
                raise RuntimeError('Only LoRA weights may change')
            class RoleTrainer(Trainer):
                def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
                    logits=model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],use_cache=False,return_dict=True).logits
                    control_logits=model(input_ids=inputs['control_input_ids'],attention_mask=inputs['control_attention_mask'],use_cache=False,return_dict=True).logits[0,inputs['control_positions'],:]
                    loss=role_sft_loss(logits,inputs['labels'],control_logits,inputs['control_reference_logps'],control_weight=.1 if concept=='hallucination' else .5)
                    return (loss,{'logits':logits}) if return_outputs else loss
            args=TrainingArguments(output_dir=str(out/'checkpoints'),max_steps=STEPS[concept],
                per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=2e-4 if concept=='hallucination' else 5e-5,
                warmup_steps=5,lr_scheduler_type='linear',bf16=True,fp16=False,max_grad_norm=1.,
                logging_steps=20,save_strategy='no',report_to='none',optim='adamw_torch',gradient_checkpointing=True,
                seed=config.TRAIN_SEED,data_seed=config.TRAIN_SEED,dataloader_num_workers=0,remove_unused_columns=False)
            trainer=RoleTrainer(model=model,args=args,train_dataset=rows,data_collator=RoleCollator(tokenizer.pad_token_id,controls))
            trainer.model_accepts_loss_kwargs=False
            result=trainer.train()
            output=out/'adapter';model.save_pretrained(output,safe_serialization=True,selected_adapters=[concept]);output=output/concept
            tokenizer.save_pretrained(output)
            report['concepts'][concept]={'adapter_path':str(output),'adapter_sha256':sha256_file(output/'adapter_model.safetensors'),
                'adapter_config_sha256':sha256_file(output/'adapter_config.json'),'steps':STEPS[concept],
                'train_rows':len(rows),'train_question_ids':[r['question_id'] for r in rows],'excluded_question_ids':excluded,
                'learning_rate':args.learning_rate,'control_kl_weight':.1 if concept=='hallucination' else .5,
                'train_loss':float(result.training_loss),'controls':control_meta,'target':target}
            base=model.unload();del trainer,model;trainer=model=None;gc.collect();torch.cuda.empty_cache()
        bundle=copy.deepcopy(source)
        bundle.update({'status':'role_repair_unvalidated','experiment_protocol':PROTOCOL,'source_manifest_sha256':before[str(source_path)],
            'adapter_paths':{name:r['adapter_path'] for name,r in report['concepts'].items()},
            'repe_checkpoint':config.REPE_CHECKPOINT,'quality_report':str(directory/'quality_report.json'),
            'training_corpus':{'protocol':CORPUS_PROTOCOL,'fingerprint':split_fingerprint(corpus),'source_sha256':corpus['source_sha256'],
                'question_ids':sorted({r['question_id'] for r in corpus['train']}),
                'warning':'Full TruthfulQA evaluation would overlap this derived training split; exclude these questions.'}})
        for name,record in report['concepts'].items():
            bundle['concepts'][name].update({k:record[k] for k in ('adapter_sha256','adapter_config_sha256')})
            bundle['concepts'][name].update({'dataset_fingerprint':split_fingerprint(corpus),'train_rows':record['train_rows'],
                'split_protocol':CORPUS_PROTOCOL,'target':record['target'],'original_training_inherited':name=='jailbreak'})
        manifest=directory/'manifest.json';manifest.write_text(json.dumps(bundle,indent=2))
        if any(sha256_file(p)!=digest for p,digest in before.items()):
            raise RuntimeError('Canonical artifacts changed')
        report['manifest_path']=str(manifest);report['wall_time_s']=time.perf_counter()-started
        (directory/'training_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        del base;base=None;gc.collect();torch.cuda.empty_cache()
        from evaluation.role_repair import evaluate_role_repair
        evaluation=evaluate_role_repair(directory,corpus)
        return {'protocol':PROTOCOL,'training':report,'evaluation':evaluation}
    finally:
        if model is not None:model.disable_adapter_layers()
        del trainer,model,base;gc.collect();torch.cuda.empty_cache()
