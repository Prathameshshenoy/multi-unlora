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
from evaluation.selective_allocation import ALLOCATION_SHA256 as FRESH_ALLOCATION_SHA,frozen_selective_allocation

SELECTIVE_DEV_SHA="81cf9924cf0ea0d3cfc2d158cc35b177cd8347c005ed4485ef171273e6736efa"
SELECTION_SHA="be7cec1fcb17dc036444527538cacd73744eb396c271d47791b7f46b82e7c801"
DEVELOPMENT_JUDGMENTS_SHA="ebb91a549aa76d7beea243513255b75b4651a565748b79d64732ec4e3d346537"

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


def develop_selective(phase="development"):
    from demo.backends import GPUBackend
    from inference.prompts import comparison_prompt,BRIEF_PROTOCOL
    from inference.runtime import sha256_file
    path=Path(config.RUNS_DIR)/'reference_support_v2/development/generation_report.json'
    parent=json.loads(path.read_text())
    if (packet_hash(parent)!=DEVELOPMENT_SHA or parent['manifest_sha256']!=sha256_file(MANIFEST)
            or parent['prompt_variant']!=BRIEF_PROTOCOL or parent['max_new_tokens']!=MAX_NEW_TOKENS
            or parent['sentence_limit']!=1 or len(parent['records'])!=16):raise ValueError('Selective parent identity changed')
    if phase not in ('development','confirmation'):raise ValueError('Unknown selective phase')
    directory=Path(config.RUNS_DIR)/'selective_subtraction_v3'/phase
    if phase=='confirmation':
        development=json.loads((directory.parent/'development/generation_report.json').read_text())
        if packet_hash(development)!=SELECTIVE_DEV_SHA:raise ValueError('Selective development source changed')
    backend=GPUBackend(manifest_path=MANIFEST,use_cache=False);reference=load_reference(backend)
    directory.mkdir(parents=True,exist_ok=False)
    cases=parent['records'];policies=POLICIES
    if phase=='confirmation':
        cases=frozen_selective_allocation()['confirmation']
        policies={'positive_1':{'strength':1.,'negative':False,'penalty':'signed'},
                  'positive_2':POLICIES['positive_2'],'selective_2':POLICIES['selective_2']}
    report={'protocol':'selective-subtraction-calibration-v3','phase':phase,'status':'running',
        'scope':('Consumed 12 factual development questions and four benign smoke controls. No fresh accuracy claim.' if phase=='development' else '20 newly held-out questions from the prospectively broadened category allocation; not comparable directly with previous confirmation or official TruthfulQA accuracy.')+' No retrieval or automatic promotion.',
        'manifest_fingerprint':parent['manifest_fingerprint'],'manifest_sha256':parent['manifest_sha256'],
        'allocation_sha256':parent['allocation_sha256'] if phase=='development' else FRESH_ALLOCATION_SHA,'parent_generation_sha256':DEVELOPMENT_SHA,
        'reference':reference,'policies':policies,'records':[],'candidate_support':'base_reference_union',
        'confirmation_allocation_sha256':FRESH_ALLOCATION_SHA,
        'reused_policy_map':{'baseline':'baseline','positive_1':'positive_8','raw_signed_1':'contrast_8'} if phase=='development' else {},
        **({'development_generation_sha256':SELECTIVE_DEV_SHA,'selection_sha256':SELECTION_SHA,
            'development_judgments_sha256':DEVELOPMENT_JUDGMENTS_SHA,'selected':'selective_2'} if phase=='confirmation' else {}),
        'beta':.01,'kl_budget':8.,'sentence_limit':1,'max_new_tokens':MAX_NEW_TOKENS,
        'prompt_variant':BRIEF_PROTOCOL,'production_quality_verified':False}
    def save(name):
        target=directory/name;temp=target.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False));temp.replace(target)
    save('generation_report.partial.json')
    for before in cases:
        item={k:before[k] for k in ('case_id','prompt','category','correct_references','incorrect_references')}
        outputs={new:before['outputs'][old] for new,old in (('baseline','baseline'),('positive_1','positive_8'),('raw_signed_1','contrast_8'))} if phase=='development' else {}
        routing={}
        for name,policy in policies.items():
            done=None;route=None;max_kl=0.;min_scale=1.
            for event,data in backend.stream(comparison_prompt(item['prompt'],'brief'),MAX_NEW_TOKENS,threading.Event(),
                    sentence_limit=1,_branches=('baseline','unlora') if not outputs else ('unlora',),_hall_strength=policy['strength'],_jail_scale=0.,
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
            if 'baseline' in done:outputs['baseline']=done['baseline']
            outputs[name]=done['unlora'];routing[name]={'weights':route,'max_measured_kl':max_kl,'minimum_attenuation':min_scale}
        report['records'].append({**item,'outputs':outputs,'routing':routing,'reused_arms':['baseline','positive_1','raw_signed_1'] if phase=='development' else []})
        save('generation_report.partial.json');print(f'selective {phase}: {len(report["records"])}/{len(cases)} cases',flush=True)
    report['status']='awaiting_reference_grounded_review';save('generation_report.json');return report


def confirm_selective_policy(reports):
    if set(reports)!={'baseline','positive_1','positive_2'}:raise ValueError('All three confirmation controls are required')
    base=reports['baseline']
    if set(base['arms'])!={'baseline','positive_1','positive_2','selective_2'} or any(v['n_assigned']!=20 for v in base['arms'].values()):
        raise ValueError('Selective confirmation sample or arms changed')
    checks=[]
    for reference,report in reports.items():
        if any(report[k]!=base[k] for k in ('packet_sha256','judgments_sha256','arms')):
            raise ValueError('Selective confirmation requires identical sealed judgments')
        pair=report['comparisons']['selective_2']
        if pair['reference']!=reference:raise ValueError('Wrong selective confirmation comparator')
        interval=pair['accepted_paired_bootstrap_95_ci']
        checks.append(pair['review_coverage_sufficient'] and pair['new_contradictions']==0 and interval is not None and interval[0]>0)
    candidate=base['arms']['selective_2'];passed=all(checks) and candidate['guarded_outputs']==candidate['unfinished_outputs']==0
    return {'confirmation_passed':passed,'status':'internal_selective_pilot_passed' if passed else 'selective_confirmation_failed',
            'automatic_promotion':False,'production_quality_verified':False}
