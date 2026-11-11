"""Penalty-only subtraction development; no retrieval, new training or promotion."""
import json
import math
from pathlib import Path
import threading

import config
from evaluation.factual_reference_trial import load_reference
from evaluation.freeform_trial import MANIFEST,MAX_NEW_TOKENS
from evaluation.freeform_review import packet_hash
from evaluation.support_trial import DEVELOPMENT_SHA
from evaluation.selective_allocation import ALLOCATION_SHA256 as FRESH_ALLOCATION_SHA

POLICIES={'selective_1':{'strength':1.,'negative':True,'penalty':'positive_log_ratio'},
          'selective_2':{'strength':2.,'negative':True,'penalty':'positive_log_ratio'},
          'positive_2':{'strength':2.,'negative':False,'penalty':'signed'}}


def select_selective_policy(base,positives,controls):
    if any(a['n_assigned']!=16 for a in base['arms'].values()):raise ValueError('Selective development count changed')
    if any(controls[k]!=base[k] for k in ('packet_sha256','judgments_sha256')):
        raise ValueError('Control review binding changed')
    if set(controls['arms'])!=set(base['arms']) or any(a['n_assigned']!=4 for a in controls['arms'].values()):
        raise ValueError('Selective trial needs all four benign smoke controls')
    eligible=[]
    for strength in (1,2):
        name=f'selective_{strength}';positive=positives[strength];ref=f'positive_{strength}'
        if any(positive[k]!=base[k] for k in ('packet_sha256','judgments_sha256','arms')):
            raise ValueError('Selective comparisons require the same sealed judgments')
        candidate=base['arms'][name];pair=base['comparisons'][name];control=positive['comparisons'][name]
        if pair['reference']!='baseline' or control['reference']!=ref:raise ValueError('Selective comparator mismatch')
        if (pair['strict_wins']-pair['strict_losses']>=2 and pair['strict_losses']==0
                and control['strict_wins']-control['strict_losses']>=1
                and all(c['new_contradictions']==0 and c['review_coverage_sufficient'] for c in (pair,control))
                and all(candidate['contradiction_rate']<=base['arms'][r]['contradiction_rate']
                        and candidate['unscorable_rate']<=base['arms'][r]['unscorable_rate'] for r in ('baseline',ref))
                and candidate['guarded_outputs']==candidate['unfinished_outputs']==0
                and controls['arms'][name]['accepted_rate']==1.):
            eligible.append((candidate['accepted_rate'],-strength,name))
    return {'selected':max(eligible)[2] if eligible else None,'automatic_promotion':False,
        'status':'requires_new_confirmation' if eligible else 'no_qualified_selective_setting'}


def develop_selective():
    from demo.backends import GPUBackend
    from inference.prompts import comparison_prompt,BRIEF_PROTOCOL
    from inference.runtime import sha256_file
    path=Path(config.RUNS_DIR)/'reference_support_v2/development/generation_report.json'
    parent=json.loads(path.read_text())
    if (packet_hash(parent)!=DEVELOPMENT_SHA or parent['manifest_sha256']!=sha256_file(MANIFEST)
            or parent['prompt_variant']!=BRIEF_PROTOCOL or parent['max_new_tokens']!=MAX_NEW_TOKENS
            or parent['sentence_limit']!=1 or len(parent['records'])!=16):raise ValueError('Selective parent identity changed')
    backend=GPUBackend(manifest_path=MANIFEST,use_cache=False);reference=load_reference(backend)
    directory=Path(config.RUNS_DIR)/'selective_subtraction_v3/development';directory.mkdir(parents=True,exist_ok=False)
    report={'protocol':'selective-subtraction-calibration-v3','phase':'development','status':'running',
        'scope':'Consumed 12 factual development questions and four benign smoke controls. No fresh accuracy claim, retrieval or automatic promotion.',
        'manifest_fingerprint':parent['manifest_fingerprint'],'manifest_sha256':parent['manifest_sha256'],
        'allocation_sha256':parent['allocation_sha256'],'parent_generation_sha256':DEVELOPMENT_SHA,
        'reference':reference,'policies':POLICIES,'records':[],'candidate_support':'base_reference_union',
        'confirmation_allocation_sha256':FRESH_ALLOCATION_SHA,
        'reused_policy_map':{'baseline':'baseline','positive_1':'positive_8','raw_signed_1':'contrast_8'},
        'beta':.01,'kl_budget':8.,'sentence_limit':1,'max_new_tokens':MAX_NEW_TOKENS,
        'prompt_variant':BRIEF_PROTOCOL,'production_quality_verified':False}
    def save(name):
        target=directory/name;temp=target.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False));temp.replace(target)
    save('generation_report.partial.json')
    for before in parent['records']:
        item={k:before[k] for k in ('case_id','prompt','category','correct_references','incorrect_references')}
        outputs={new:before['outputs'][old] for new,old in (('baseline','baseline'),('positive_1','positive_8'),('raw_signed_1','contrast_8'))};routing={}
        for name,policy in POLICIES.items():
            done=None;route=None;max_kl=0.;min_scale=1.
            for event,data in backend.stream(comparison_prompt(item['prompt'],'brief'),MAX_NEW_TOKENS,threading.Event(),
                    sentence_limit=1,_branches=('unlora',),_hall_strength=policy['strength'],_jail_scale=0.,
                    _hall_reference_adapter='factual_reference',_hall_negative_adapter='hallucination' if policy['negative'] else None,
                    _beta=.01,_kl_budget=8.,_candidate_support='base_reference_union',_penalty_mode=policy['penalty']):
                if event=='error':raise RuntimeError('Selective generation failed')
                if event=='routing':route=data
                if event=='telemetry':
                    expected='selective-subtraction-v3' if policy['negative'] else 'reference-support-v2'
                    kl=data['kl_divergence']
                    if (not math.isfinite(kl) or kl>8.+1e-6 or data['kl_budget']!=8. or data['operator']!=expected
                            or data['alpha_hall']!=policy['strength'] or data['alpha_jail']!=0
                            or data['penalty_mode']!=policy['penalty'] or data['candidate_support']!='base_reference_union'
                            or data['plausibility_beta']!=.01):raise RuntimeError('Selective operator constraints changed')
                    max_kl=max(max_kl,kl);min_scale=min(min_scale,data['attenuation'])
                if event=='complete':done=data
            if done is None or done['simulated'] is not False:raise RuntimeError('Incomplete selective generation')
            outputs[name]=done['unlora'];routing[name]={'weights':route,'max_measured_kl':max_kl,'minimum_attenuation':min_scale}
        report['records'].append({**item,'outputs':outputs,'routing':routing,'reused_arms':['baseline','positive_1','raw_signed_1']})
        save('generation_report.partial.json');print(f'selective development: {len(report["records"])}/16 cases',flush=True)
    report['status']='awaiting_reference_grounded_review';save('generation_report.json');return report
