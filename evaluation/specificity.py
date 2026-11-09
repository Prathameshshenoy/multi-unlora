"""Diagnose whether an anti-expert learns a concept or general response style.

NLL improvement on undesirable responses alone is insufficient: an adapter may
also improve likelihood on correct/safe responses. Report both gains and their
paired difference. This diagnostic never constitutes answer-accuracy or safety
validation and never changes serving weights.
"""
import json
import math
from pathlib import Path
import statistics

import config
from inference.prompts import training_tokens

SPECIFICITY_PROTOCOL = 'paired-response-specificity-v1'


def specificity_margin(base_bad, expert_bad, base_good, expert_good):
    values = (base_bad, expert_bad, base_good, expert_good)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('Response losses must be finite and non-negative')
    bad_gain = base_bad-expert_bad
    good_gain = base_good-expert_good
    return {'bad_response_nll_gain':bad_gain, 'good_response_nll_gain':good_gain,
            'specificity_margin':bad_gain-good_gain}


def summarize(rows):
    if not rows:
        raise ValueError('No paired specificity observations')
    return {'n':len(rows),
            'mean_bad_response_nll_gain':statistics.mean(row['bad_response_nll_gain'] for row in rows),
            'mean_good_response_nll_gain':statistics.mean(row['good_response_nll_gain'] for row in rows),
            'mean_specificity_margin':statistics.mean(row['specificity_margin'] for row in rows),
            'positive_specificity_pairs':sum(row['specificity_margin']>0 for row in rows),
            'interpretation':'Positive margins mean more adapter likelihood gain on undesirable than desirable reference responses; not a generation quality score.'}


def _loss(model,tokenizer,prompt,response,adapter):
    import torch
    from inference.adapters import active_adapter
    from training.pilot import ResponseCollator
    row=training_tokens(tokenizer,prompt,response,512)
    if row is None:
        return None
    inputs={key:value.to(next(model.parameters()).device)
            for key,value in ResponseCollator(tokenizer.pad_token_id)([row]).items()}
    with active_adapter(model,adapter), torch.inference_mode():
        return float(model(**inputs,use_cache=False).loss)


def run_specificity(model,tokenizer,*,n=8):
    from datasets import load_dataset
    from evaluation.quality import fingerprint,verify_training_split
    from training.pilot import load_pairs,pairs_fingerprint,prompt_key
    if not 1 <= n <= 16:
        raise ValueError('Initial paired diagnostic is bounded to 1..16 per concept')
    manifest=json.loads((Path(config.ADAPTER_DIR)/'manifest.json').read_text())
    initial=json.loads((Path(config.RESULTS_DIR)/'pilot_quality.json').read_text())
    if initial['manifest_fingerprint'] != fingerprint(manifest):
        raise RuntimeError('Specificity diagnostic requires the measured adapter manifest')
    report={'protocol':SPECIFICITY_PROTOCOL,'manifest_fingerprint':fingerprint(manifest),
            'model_revision':manifest['model_revision'],'concepts':{},
            'quality_improvement_verified':False,
            'scope':'Paired likelihood diagnosis only. No serving-policy or accuracy promotion.'}
    for concept in ['hallucination','jailbreak']:
        train,heldout,data_id=load_pairs(concept)
        provenance=verify_training_split(concept,train,heldout,data_id,manifest)
        old=initial['anti_experts'][concept]['split_provenance']
        if pairs_fingerprint(heldout) != old['holdout_pairs_fingerprint']:
            raise RuntimeError('Reconstructed holdout changed since initial measurement')
        if concept=='hallucination':
            raw=load_dataset('pminervini/HaluEval','qa',split='data').shuffle(seed=config.TRAIN_SEED)
            good={(prompt_key(row['question']),row['hallucinated_answer'].strip()):row['right_answer'] for row in reversed(list(raw))}
        else:
            raw=load_dataset('LLM-LAT/harmful-dataset',split='train').shuffle(seed=config.TRAIN_SEED)
            good={(prompt_key(row['prompt']),row['rejected'].strip()):row['chosen'] for row in reversed(list(raw))
                  if isinstance(row.get('prompt'),str) and isinstance(row.get('rejected'),str) and isinstance(row.get('chosen'),str)}
        rows=[]
        excluded=0
        for index,(prompt,bad) in enumerate(heldout[:n]):
            safe=good.get((prompt_key(prompt),bad.strip()))
            if not isinstance(safe,str) or not safe.strip():
                raise RuntimeError('Missing matched desirable reference response')
            losses=[_loss(model,tokenizer,prompt,response,adapter)
                    for response,adapter in [(bad,None),(bad,concept),(safe,None),(safe,concept)]]
            if any(value is None for value in losses):
                excluded+=1
                continue
            row=specificity_margin(*losses)
            row.update({'holdout_index':index,'question_id':fingerprint({'prompt':prompt}),
                'base_bad_nll':losses[0],'expert_bad_nll':losses[1],
                'base_good_nll':losses[2],'expert_good_nll':losses[3]})
            rows.append(row)
        report['concepts'][concept]={'summary':summarize(rows),'excluded_contexts':excluded,
                                    'split_provenance':provenance,'rows':rows}
    output=Path(config.RESULTS_DIR)/'paired_specificity.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False))
    return report
