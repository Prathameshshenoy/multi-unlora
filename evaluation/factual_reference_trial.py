"""Experimental positive-reference logit contrast; no new training or promotion."""
import json
import math
from pathlib import Path
import threading

import config
from evaluation.freeform_review import packet_hash
from evaluation.freeform_trial import frozen_allocation,MANIFEST,MAX_NEW_TOKENS,ALLOCATION_SHA256

PROTOCOL='factual-reference-logit-contrast-v1'
RUN_NAME='factual_reference_v1'
PARENT_DEV_SHA='23ced14f668db5806657893975e870ecacd4c0d658dee48db7b136c16a3f9a7b'
COMPANION_SHA='0573ad2a070fb17edf6928020a4b57e7ddd87acb4063567fb193d4e5165bf5d5'
POLICIES={
    'reference_only':{'negative':False,'jail_scale':0.,'beta':.1},
    'factual':{'negative':True,'jail_scale':0.,'beta':.1},
    'factual_both':{'negative':True,'jail_scale':1.,'beta':.1},
    'factual_wide':{'negative':True,'jail_scale':0.,'beta':.01}}


def load_reference(backend):
    """Validate immutable positive weights and matching LoRA/base configuration."""
    from inference.runtime import sha256_file,validate_artifacts
    source=validate_artifacts(MANIFEST)
    parent=Path(config.RUNS_DIR)/'misconception_content_delta_v1'
    training=json.loads((parent/'training_report.json').read_text())
    quality=json.loads((parent/'quality_report.json').read_text())
    manifest=json.loads((parent/'manifest.json').read_text())
    directory=parent/'companion'/'hallucination'
    if (training['parent_manifest_sha256']!=sha256_file(MANIFEST)
            or quality['training_report_sha256']!=sha256_file(parent/'training_report.json')
            or any(manifest[k]!=source[k] for k in ('base_model','model_revision','prompt_protocol'))
            or training['companion_adapter_sha256']!=COMPANION_SHA
            or sha256_file(directory/'adapter_model.safetensors')!=COMPANION_SHA):
        raise ValueError('Factual-reference identity mismatch')
    original=json.loads((Path(source['adapter_paths']['hallucination'])/'adapter_config.json').read_text())
    reference=json.loads((directory/'adapter_config.json').read_text())
    for settings in (original,reference):
        if isinstance(settings.get('target_modules'),list):settings['target_modules']=sorted(settings['target_modules'])
    if original!=reference:raise ValueError('Factual-reference configuration differs from matched source')
    backend._load()
    backend.model.load_adapter(str(directory),adapter_name='factual_reference')
    backend.model.eval();backend.model.disable_adapter_layers()
    return {'adapter_sha256':COMPANION_SHA,'adapter_config_sha256':sha256_file(directory/'adapter_config.json'),
            'training_report_sha256':sha256_file(parent/'training_report.json'),
            'model_revision':source['model_revision'],'training_rows':training['train_rows']}


def develop_factual_reference():
    from demo.backends import GPUBackend
    from inference.prompts import comparison_prompt,BRIEF_PROTOCOL
    from inference.runtime import sha256_file
    allocation=frozen_allocation()
    parent_path=Path(config.RUNS_DIR)/'brief_freeform_v1'/'development'/'generation_report.json'
    parent=json.loads(parent_path.read_text())
    if (packet_hash(parent)!=PARENT_DEV_SHA or parent['allocation_sha256']!=ALLOCATION_SHA256
            or parent['prompt_variant']!=BRIEF_PROTOCOL or parent['sentence_limit']!=1
            or parent['max_new_tokens']!=MAX_NEW_TOKENS or parent['manifest_sha256']!=sha256_file(MANIFEST)):
        raise ValueError('Reusable base/current generation identity changed')
    prior={r['case_id']:r for r in parent['records']}
    backend=GPUBackend(manifest_path=MANIFEST,use_cache=False)
    reference=load_reference(backend)
    directory=Path(config.RUNS_DIR)/RUN_NAME/'development';directory.mkdir(parents=True,exist_ok=False)
    report={'protocol':PROTOCOL,'phase':'development','status':'running','records':[],
        'scope':'Experimental third factual-reference adapter. AI-reviewed internal development pilot; not official benchmark accuracy or a validated production model.',
        'parent_generation_sha256':PARENT_DEV_SHA,'reference':reference,'policies':POLICIES,
        'allocation_sha256':ALLOCATION_SHA256,'manifest_sha256':parent['manifest_sha256'],
        'manifest_fingerprint':parent['manifest_fingerprint'],'prompt_variant':BRIEF_PROTOCOL,
        'sentence_limit':1,'max_new_tokens':MAX_NEW_TOKENS,'production_quality_verified':False}
    def save(name):
        target=directory/name;temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False));temporary.replace(target)
    save('generation_report.partial.json')
    for item in allocation['development']:
        before=prior[item['case_id']]
        if any(before[k]!=item[k] for k in ('prompt','correct_references','incorrect_references')):
            raise ValueError('Reusable source question or annotations changed')
        outputs={arm:before['outputs'][arm] for arm in ('baseline','current')};routing={}
        for name,policy in POLICIES.items():
            done=None;route=None;max_kl=0.
            for event,data in backend.stream(comparison_prompt(item['prompt'],'brief'),MAX_NEW_TOKENS,threading.Event(),
                    sentence_limit=1,_branches=('unlora',),_jail_scale=policy['jail_scale'],_beta=policy['beta'],
                    _hall_reference_adapter='factual_reference',
                    _hall_negative_adapter='hallucination' if policy['negative'] else None):
                if event=='error':raise RuntimeError('Factual-reference generation failed')
                if event=='routing':route=data
                if event=='telemetry':
                    kl=data['kl_divergence']
                    if not math.isfinite(kl) or kl>config.PROCESSOR_KL_THRESHOLD+1e-6 or data['plausibility_beta']!=policy['beta']:
                        raise RuntimeError('Factual-reference operator settings or KL mismatch')
                    max_kl=max(max_kl,kl)
                if event=='complete':done=data
            if done is None or done['simulated'] is not False:raise RuntimeError('Incomplete factual-reference output')
            outputs[name]=done['unlora'];routing[name]={'weights':route,'max_measured_kl':max_kl,'beta':policy['beta']}
        report['records'].append({**item,'outputs':outputs,'routing':routing,
                                  'reused_arms':['baseline','current'],'reused_source_sha256':PARENT_DEV_SHA})
        save('generation_report.partial.json')
        print(f'factual reference: {len(report["records"])}/12 completed cases',flush=True)
    report['status']='awaiting_reference_grounded_review';save('generation_report.json')
    return report
