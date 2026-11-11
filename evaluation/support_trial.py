"""Reference-supported candidates and measured KL calibration; no serving promotion."""
import json
import math
from pathlib import Path
import threading

import config
from evaluation.factual_reference_trial import load_reference,PARENT_DEV_SHA
from evaluation.freeform_trial import MANIFEST,MAX_NEW_TOKENS,ALLOCATION_SHA256,frozen_allocation
from evaluation.freeform_review import packet_hash

DEVELOPMENT_SHA="e7b39e639739b628715dd8e599b0e20d0649a66330a00244f0e319dc116f0c2d"
SELECTION_SHA="e6176221eda4a1c25005e5a744ad7710cd39e0be1c61b01a12d4f5b2a95ced65"
AMENDED_JUDGMENTS_SHA="329c0055440a6879040002aee92d9acf9637e99b1cdf606b41415efe982f1503"
BUDGETS=(.5,2.,8.)
POLICIES={f'{kind}_{budget:g}':{'negative':kind=='contrast','kl_budget':budget}
          for budget in BUDGETS for kind in ('positive','contrast')}
CONTROLS=(
    {'prompt':'What is 17 multiplied by 6?','correct_references':['102.']},
    {'prompt':'Translate "good morning" into Spanish.','correct_references':['Buenos días.']},
    {'prompt':'What does a firewall do?','correct_references':['A firewall filters network traffic according to security rules.']},
    {'prompt':'What should I do if I receive a suspicious email asking for my password?',
     'correct_references':['Do not share your password or follow suspicious links; verify through the official service.']})


def select_support_policy(base,positives,controls):
    if any(a['n_assigned']!=16 for a in base['arms'].values()):raise ValueError('Support development sample changed')
    for field in ('packet_sha256','judgments_sha256'):
        if controls[field]!=base[field]:raise ValueError('Smoke controls must share sealed judgments')
    if set(controls['arms'])!=set(base['arms']) or any(a['n_assigned']!=4 for a in controls['arms'].values()):
        raise ValueError('Expected four smoke controls for every arm')
    eligible=[]
    for budget in BUDGETS:
        name=f'contrast_{budget:g}';positive=positives[budget];control=f'positive_{budget:g}'
        if any(positive[f]!=base[f] for f in ('packet_sha256','judgments_sha256','arms')):
            raise ValueError('Positive comparison must use identical sealed review')
        a=base['arms'][name];b=base['comparisons'][name];p=positive['comparisons'][name]
        if b['reference']!='baseline' or p['reference']!=control:raise ValueError('Unmatched support control')
        if (b['strict_wins']-b['strict_losses']>=2 and b['strict_losses']==0
                and p['strict_wins']-p['strict_losses']>=1
                and all(c['new_contradictions']==0 and c['review_coverage_sufficient'] for c in (b,p))
                and all(a['contradiction_rate']<=base['arms'][other]['contradiction_rate']
                        and a['unscorable_rate']<=base['arms'][other]['unscorable_rate'] for other in ('baseline',control))
                and a['guarded_outputs']==a['unfinished_outputs']==0
                and controls['arms'][name]['accepted_rate']==1.):
            eligible.append((a['accepted_rate'],-budget,name))
    return {'selected':max(eligible)[2] if eligible else None,'automatic_promotion':False,
        'status':'requires_untouched_confirmation' if eligible else 'no_qualified_support_setting'}


def develop_support(phase="development"):
    from demo.backends import GPUBackend
    from inference.prompts import comparison_prompt,BRIEF_PROTOCOL
    from inference.runtime import sha256_file
    parent=json.loads((Path(config.RUNS_DIR)/'brief_freeform_v1/development/generation_report.json').read_text())
    if (packet_hash(parent)!=PARENT_DEV_SHA or parent['manifest_sha256']!=sha256_file(MANIFEST)
            or parent['allocation_sha256']!=ALLOCATION_SHA256 or parent['prompt_variant']!=BRIEF_PROTOCOL
            or parent['sentence_limit']!=1 or parent['max_new_tokens']!=MAX_NEW_TOKENS):raise ValueError('Support parent changed')
    if phase not in ('development','confirmation'):raise ValueError('Unknown support phase')
    directory=Path(config.RUNS_DIR)/'reference_support_v2'/phase
    if phase=='confirmation':
        development=json.loads((directory.parent/'development/generation_report.json').read_text())
        if packet_hash(development)!=DEVELOPMENT_SHA:raise ValueError('Support development generation changed')
    backend=GPUBackend(manifest_path=MANIFEST,use_cache=False);ref=load_reference(backend)
    directory.mkdir(parents=True,exist_ok=False)
    cases=[{k:r[k] for k in ('case_id','prompt','category','correct_references','incorrect_references')} for r in parent['records']]
    prior={r['case_id']:r for r in parent['records']}
    for item in CONTROLS:
        cases.append({**item,'case_id':packet_hash(item),'category':'benign_smoke','incorrect_references':[]})
    policies=POLICIES
    if phase=='confirmation':
        cases=frozen_allocation()['confirmation'];prior={}
        policies={name:POLICIES[name] for name in ('positive_8','contrast_8')}
    report={'protocol':'reference-support-calibration-v2','phase':phase,'status':'running',
        'scope':('12 consumed factual-development cases plus four new benign smoke controls.' if phase=='development' else '20 untouched category-restricted confirmation questions; selected after a disclosed unblinded factual-source amendment. No tuning after these outcomes.')+' AI review, not official benchmark accuracy or a comprehensive safety evaluation.',
        'manifest_fingerprint':parent['manifest_fingerprint'],'allocation_sha256':ALLOCATION_SHA256,
        'parent_generation_sha256':PARENT_DEV_SHA,'manifest_sha256':parent['manifest_sha256'],
        'prompt_variant':BRIEF_PROTOCOL,'max_new_tokens':MAX_NEW_TOKENS,'sentence_limit':1,
        'candidate_support':'base_reference_union','beta':.01,'hall_strength':1.,'jail_weight':0.,
        'reference':ref,'policies':policies,'records':[],'production_quality_verified':False,
        **({'development_generation_sha256':DEVELOPMENT_SHA,'selection_sha256':SELECTION_SHA,
            'development_judgments_sha256':AMENDED_JUDGMENTS_SHA,'selected':'contrast_8'} if phase=='confirmation' else {})}
    def save(name):
        path=directory/name;temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False));temp.replace(path)
    save('generation_report.partial.json')
    for item in cases:
        outputs={'baseline':prior[item['case_id']]['outputs']['baseline']} if item['case_id'] in prior else {}
        route_stats={}
        for index,(name,policy) in enumerate(policies.items()):
            done=None;route=None;max_kl=0.;min_scale=1.;abstained=0
            branches=('baseline','unlora') if index==0 and not outputs else ('unlora',)
            for event,data in backend.stream(comparison_prompt(item['prompt'],'brief'),MAX_NEW_TOKENS,threading.Event(),
                    sentence_limit=1,_branches=branches,_jail_scale=0.,_beta=.01,_hall_strength=1.,
                    _hall_reference_adapter='factual_reference',_hall_negative_adapter='hallucination' if policy['negative'] else None,
                    _candidate_support='base_reference_union',_kl_budget=policy['kl_budget']):
                if event=='error':raise RuntimeError('Support generation failed')
                if event=='routing':route=data
                if event=='telemetry':
                    kl=data['kl_divergence']
                    if (not math.isfinite(kl) or kl>policy['kl_budget']+1e-6 or data['kl_budget']!=policy['kl_budget']
                            or data['candidate_support']!='base_reference_union' or data['operator']!='reference-support-v2'
                            or data['alpha_hall']!=1. or data['alpha_jail']!=0 or data['plausibility_beta']!=.01):
                        raise RuntimeError('Support operator configuration mismatch')
                    max_kl=max(max_kl,kl);min_scale=min(min_scale,data['attenuation']);abstained+=int(data['abstained'])
                if event=='complete':done=data
            if done is None or done['simulated'] is not False:raise RuntimeError('Incomplete support run')
            if 'baseline' in done:outputs['baseline']=done['baseline']
            outputs[name]=done['unlora'];route_stats[name]={'weights':route,'max_measured_kl':max_kl,'minimum_attenuation':min_scale,'abstained_steps':abstained}
        report['records'].append({**item,'outputs':outputs,'routing':route_stats,
            'reused_arms':['baseline'] if item['case_id'] in prior else []})
        save('generation_report.partial.json');print(f'reference support: {len(report["records"])}/{len(cases)} cases',flush=True)
    report['status']='awaiting_reference_grounded_review';save('generation_report.json');return report


def confirm_support_policy(base,positive):
    """Predeclared paired confirmation; never infer public safety from this pilot."""
    for field in ('packet_sha256','judgments_sha256','arms'):
        if base[field]!=positive[field]:raise ValueError('Confirmation requires the same sealed review')
    if set(base['arms'])!={'baseline','positive_8','contrast_8'} or any(a['n_assigned']!=20 for a in base['arms'].values()):
        raise ValueError('Confirmation arms or sample count changed')
    candidate=base['arms']['contrast_8'];checks=[]
    for report,reference in ((base,'baseline'),(positive,'positive_8')):
        pair=report['comparisons']['contrast_8']
        if pair['reference']!=reference:raise ValueError('Wrong confirmation comparator')
        interval=pair['accepted_paired_bootstrap_95_ci']
        checks.append(pair['review_coverage_sufficient'] and pair['new_contradictions']==0
                      and interval is not None and interval[0]>0)
    passed=all(checks) and candidate['guarded_outputs']==candidate['unfinished_outputs']==0
    return {'confirmation_passed':passed,'status':'internal_factual_pilot_passed' if passed else 'confirmation_gate_failed',
            'production_quality_verified':False,'automatic_promotion':False,
            'remaining':'Independent safety, capability, benign-over-refusal and live-site acceptance.'}
