"""Fixed brief-answer pilot; uses the production decoder without training.

Development examples are already consumed. Confirmation examples are allocated
from untouched source records before any generation. Only internal evaluation
can attenuate correction weights; no serving defaults are changed.
"""
import json
import math
from pathlib import Path
import threading

import config
from evaluation.freeform_review import packet_hash
from training.role_corpus import prepare_corpus,source_records,question_id
from training.role_allocation import ALLOCATION

PROTOCOL='paired-brief-freeform-calibration-v1'
RUN_NAME='brief_freeform_v1'
MANIFEST='/vol/runs/misconception_roles_v1/manifest.json'
DEVELOPMENT_N,CONFIRMATION_N=12,20
MAX_NEW_TOKENS=48
STABLE_CATEGORIES={'Misconceptions','Myths and Fairytales','Confusion: Places','Logical Falsehood',
    'Confusion: People','Misinformation','History','Language','Science','Distraction','Superstitions'}
POLICIES={'current':{'scale':1.,'jail_scale':1.},'quarter':{'scale':.25,'jail_scale':1.},
          'hall_only':{'scale':1.,'jail_scale':0.}}
ALLOCATION_SHA256='6867e3f2562f9014f8d9f6f3c9a7894602080609ed32d7c7b7237bddfb7385a2'


def allocate(corpus,raw,*,development_n=DEVELOPMENT_N,confirmation_n=CONFIRMATION_N,prior_test_n=128):
    from demo.display_policy import assess_prompt
    train={r['question_id'] for r in corpus['train']}
    prior={r['question_id'] for r in corpus['development']+corpus['test'][:prior_test_n]}
    source={question_id(r['Question']):r for r in raw}
    def eligible(rows):
        return [r for r in rows if r['category'] in STABLE_CATEGORIES and not assess_prompt(r['prompt'])['restricted']]
    dev=eligible(corpus['development']+corpus['test'][:prior_test_n])[:development_n]
    fresh=eligible(corpus['test'][prior_test_n:])[:confirmation_n]
    if len(dev)!=development_n or len(fresh)!=confirmation_n:raise ValueError('Insufficient eligible disjoint generated-answer cases')
    dev_ids={r['question_id'] for r in dev};fresh_ids={r['question_id'] for r in fresh}
    if (len(dev_ids)!=len(dev) or len(fresh_ids)!=len(fresh) or (dev_ids|fresh_ids)&train or fresh_ids&prior):
        raise ValueError('Generated-answer split overlaps training or consumed cases')
    def cases(rows):
        result=[]
        for row in rows:
            original=source[row['question_id']]
            refs=lambda field,best:list(dict.fromkeys([original[best].strip()]+[a.strip() for a in original[field].split(';') if a.strip()]))
            result.append({'case_id':row['question_id'],'prompt':row['prompt'],'category':row['category'],
                'correct_references':refs('Correct Answers','Best Answer'),
                'incorrect_references':refs('Incorrect Answers','Best Incorrect Answer')})
        return result
    return {'protocol':PROTOCOL,'development':cases(dev),'confirmation':cases(fresh),
            'source_sha256':corpus['source_sha256'],'scope':'Derived misconception/stable-fact pilot; category-restricted, not an official full TruthfulQA score. Previously consumed cases are development only.'}


def frozen_allocation():
    raw=source_records()
    corpus=prepare_corpus(raw,blocked_question_ids=ALLOCATION['blocked_question_ids'],test_n=203)
    result=allocate(corpus,raw)
    if packet_hash(result)!=ALLOCATION_SHA256:raise ValueError('Frozen generated-answer allocation changed')
    return result


def generate_phase(phase,*,selected=None,selection_sha256=None):
    from demo.backends import GPUBackend
    from inference.prompts import comparison_prompt,BRIEF_PROTOCOL
    from inference.runtime import validate_artifacts,sha256_file
    from evaluation.quality import fingerprint
    if phase not in ('development','confirmation'):raise ValueError('Unknown generated-answer phase')
    if phase=='confirmation' and (selected not in POLICIES or not selection_sha256):
        raise ValueError('Confirmation requires a sealed development selection')
    allocation=frozen_allocation()
    manifest=validate_artifacts(MANIFEST)
    directory=Path(config.RUNS_DIR)/RUN_NAME/phase;directory.mkdir(parents=True,exist_ok=False)
    backend=GPUBackend(manifest_path=MANIFEST,use_cache=False)
    report={'protocol':PROTOCOL,'phase':phase,'status':'running','allocation_sha256':ALLOCATION_SHA256,
        'manifest_fingerprint':fingerprint(manifest),'manifest_sha256':sha256_file(MANIFEST),
        'prompt_variant':BRIEF_PROTOCOL,'sentence_limit':1,'max_new_tokens':MAX_NEW_TOKENS,
        'operator_settings':{'beta':config.PROCESSOR_PLAUSIBILITY_BETA,'kl_budget':config.PROCESSOR_KL_THRESHOLD},
        'scope':allocation['scope'],'records':[],'policies':POLICIES,'selected':selected,
        'selection_sha256':selection_sha256,'production_quality_verified':False}
    def save(name):
        target=directory/name;temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False));temporary.replace(target)
    save('generation_report.partial.json')
    for item in allocation[phase]:
        prompt=comparison_prompt(item['prompt'],'brief');outputs={};routing={}
        choices=list(POLICIES) if phase=='development' else [selected]
        for index,name in enumerate(choices):
            policy=POLICIES[name];branches=('baseline','unlora') if index==0 else ('unlora',)
            done=None;route=None;max_kl=0.
            for event,data in backend.stream(prompt,MAX_NEW_TOKENS,threading.Event(),sentence_limit=1,
                    _branches=branches,_correction_scale=policy['scale'],_jail_scale=policy['jail_scale']):
                if event=='error':raise RuntimeError('Generated-answer phase failed; inspect the private source logs')
                if event=='routing':route=data
                if event=='telemetry':
                    kl=data['kl_divergence']
                    if not math.isfinite(kl) or kl>config.PROCESSOR_KL_THRESHOLD+1e-6:raise RuntimeError('Invalid measured correction KL')
                    max_kl=max(max_kl,kl)
                if event=='complete':done=data
            if done is None or done['simulated'] is not False:raise RuntimeError('Incomplete real-model comparison')
            if index==0:outputs['baseline']=done['baseline']
            outputs[name]=done['unlora'];routing[name]={'weights':route,'max_measured_kl':max_kl}
        report['records'].append({**item,'outputs':outputs,'routing':routing})
        save('generation_report.partial.json')
        print(f'{phase}: {len(report["records"])}/{len(allocation[phase])} completed pairs',flush=True)
    report['status']='awaiting_reference_grounded_review'
    save('generation_report.json')
    return report
