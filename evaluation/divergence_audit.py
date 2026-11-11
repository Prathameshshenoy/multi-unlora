"""Oracle-prefix diagnosis of consumed mistakes, never generated-answer accuracy.

Correct references are used only to locate a shared-prefix divergence and inspect
logits. They are not passed to any public or evaluated generation branch.
"""
import json
from pathlib import Path
import config
from evaluation.factual_reference_trial import load_reference,PARENT_DEV_SHA
from evaluation.freeform_review import packet_hash
from evaluation.freeform_trial import MANIFEST

CASE_INDICES=(2,5,9,11)


def divergence_probe(tokenizer,formatted,references,answer):
    prompt=tokenizer.encode(formatted,add_special_tokens=False)
    wrong=tokenizer.encode(formatted+answer,add_special_tokens=False)
    if wrong[:len(prompt)]!=prompt:raise ValueError('Answer changed prompt token boundary')
    choices=[]
    for reference in references:
        correct=tokenizer.encode(formatted+reference,add_special_tokens=False)
        if correct[:len(prompt)]!=prompt:raise ValueError('Reference changed prompt token boundary')
        shared=0
        while shared<min(len(correct),len(wrong)) and correct[shared]==wrong[shared]:shared+=1
        if shared==min(len(correct),len(wrong)):continue
        choices.append((shared,reference,correct))
    if not choices:raise ValueError('No diverging reference available')
    # Prefer a reference sharing the longest already-generated prefix; lexical
    # tie-breaking is deterministic and independent of any model logit values.
    shared,reference,correct=sorted(choices,key=lambda x:(-x[0],x[1]))[0]
    return {'prefix_ids':wrong[:shared],'good_token':correct[shared],'wrong_token':wrong[shared],
            'reference':reference,'shared_answer_tokens':shared-len(prompt)}


def audit_divergence():
    import torch
    from demo.backends import GPUBackend
    from inference.adapters import active_adapter
    from inference.prompts import comparison_prompt,format_prompt
    from inference.operator import correct_logits
    from inference.runtime import sha256_file
    parent=json.loads((Path(config.RUNS_DIR)/'brief_freeform_v1/development/generation_report.json').read_text())
    if packet_hash(parent)!=PARENT_DEV_SHA or parent['manifest_sha256']!=sha256_file(MANIFEST):
        raise ValueError('Divergence audit parent changed')
    backend=GPUBackend(manifest_path=MANIFEST,use_cache=False);reference=load_reference(backend)
    model,tok=backend.model,backend.tokenizer;records=[]
    with torch.inference_mode():
        for index in CASE_INDICES:
            row=parent['records'][index]
            prompt=format_prompt(tok,comparison_prompt(row['prompt'],'brief'))
            probe=divergence_probe(tok,prompt,row['correct_references'],row['outputs']['baseline']['text'])
            ids=torch.tensor([probe['prefix_ids']],device=next(model.parameters()).device)
            logits={}
            try:
                for adapter in (None,'hallucination','factual_reference'):
                    with active_adapter(model,adapter):
                        logits[adapter]=model(input_ids=ids,use_cache=False,return_dict=True).logits[:,-1].float()
            finally:model.disable_adapter_layers()
            good,bad=probe['good_token'],probe['wrong_token'];base=logits[None]
            gap=lambda v:float((v[0,good]-v[0,bad]).item())
            record={'case_id':row['case_id'],'shared_answer_tokens':probe['shared_answer_tokens'],
                'good_token':tok.decode([good]),'wrong_token':tok.decode([bad]),
                'reference':probe['reference'],'base_good_minus_wrong_logit':gap(base),
                'hall_good_minus_wrong_logit':gap(logits['hallucination']),
                'reference_good_minus_wrong_logit':gap(logits['factual_reference']),
                'contrast_good_margin_per_unit_strength':gap(logits['factual_reference']-logits['hallucination']),
                'positive_good_margin_per_unit_strength':gap(logits['factual_reference']-base),
                'good_to_base_max_probability_ratio':float((base[0,good]-base.max()).exp()),'settings':[]}
            for beta in (.1,.01):
                for strength in (1.,2.,4.):
                    for name,negative in (('contrast',logits['hallucination']),('positive',base)):
                        result=correct_logits(base,strength*(negative-logits['factual_reference']),beta=beta,kl_budget=.5)
                        record['settings'].append({'kind':name,'strength':strength,'beta':beta,
                            'good_token_finite':bool(torch.isfinite(result.scores[0,good])),
                            'good_beats_wrong':bool(result.scores[0,good]>result.scores[0,bad]),
                            'greedy_is_good':int(result.scores.argmax(-1))==good,
                            'attenuation':float(result.scale[0]),'kl':float(result.kl[0]),'abstained':bool(result.abstained[0])})
            records.append(record)
    report={'protocol':'consumed-first-divergence-audit-v1','phase':'diagnostic','status':'complete',
        'scope':'Oracle-reference shared-prefix diagnostic on four consumed failures, not generation or accuracy. No parameter selection or promotion.',
        'manifest_fingerprint':parent['manifest_fingerprint'],'allocation_sha256':parent['allocation_sha256'],
        'parent_generation_sha256':PARENT_DEV_SHA,'reference':reference,'records':records,'production_quality_verified':False}
    path=Path(config.RUNS_DIR)/'factual_strength_v1/divergence_audit.json'
    with path.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    return report
