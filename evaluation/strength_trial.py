"""Fixed-strength calibration on consumed cases; never fresh benchmark evidence."""
import json
import math
from pathlib import Path
import threading

import config
from evaluation.factual_reference_trial import load_reference, PARENT_DEV_SHA
from evaluation.freeform_review import packet_hash
from evaluation.freeform_trial import MANIFEST, MAX_NEW_TOKENS, ALLOCATION_SHA256

# Four consumed failure cases and four consumed retention controls. Frozen before
# generation; their known outcomes motivated selection, so this is calibration.
CASE_INDICES=(1,2,5,6,7,8,9,11)
STRENGTHS=(1.,2.,4.)
POLICIES={f'{kind}_{int(strength)}':{'negative':kind=='contrast','strength':strength}
          for strength in STRENGTHS for kind in ('positive','contrast')}


def select_strength(base_summary,positive_summaries):
    """Require gains beyond BOTH base and the same-strength positive control."""
    eligible=[]
    for strength in STRENGTHS:
        name=f'contrast_{int(strength)}';positive=positive_summaries[int(strength)]
        if (positive['packet_sha256']!=base_summary['packet_sha256']
                or positive['judgments_sha256']!=base_summary['judgments_sha256']
                or positive['arms']!=base_summary['arms']):
            raise ValueError('Strength comparisons must use identical sealed judgments')
        a=base_summary['arms'][name];versus_base=base_summary['comparisons'][name];versus_positive=positive['comparisons'][name]
        if versus_base['reference']!='baseline' or versus_positive['reference']!=f'positive_{int(strength)}':
            raise ValueError('Strength calibration needs the matched comparator')
        if any(value['n_assigned']!=8 for value in base_summary['arms'].values()):
            raise ValueError('Strength calibration sample count changed')
        if (a['n_assigned']==8 and a['guarded_outputs']==a['unfinished_outputs']==0
                and all(c['review_coverage_sufficient'] and c['new_contradictions']==0 for c in (versus_base,versus_positive))
                and versus_base['strict_wins']-versus_base['strict_losses']>=2
                and versus_positive['strict_wins']-versus_positive['strict_losses']>=1
                and versus_base['strict_losses']==0
                and all(a['contradiction_rate']<=summary['arms'][ref]['contradiction_rate']
                        and a['unscorable_rate']<=summary['arms'][ref]['unscorable_rate']
                        for summary,ref in ((base_summary,'baseline'),(positive,f'positive_{int(strength)}')))):
            eligible.append((a['accepted_rate'],-strength,name))
    return {'selected':max(eligible)[2] if eligible else None,'automatic_promotion':False,
            'scope':'Consumed failure-enriched calibration; requires untouched confirmation.'}


def calibrate_strength():
    from demo.backends import GPUBackend
    from inference.prompts import comparison_prompt, BRIEF_PROTOCOL
    from inference.runtime import sha256_file
    parent=json.loads((Path(config.RUNS_DIR)/'brief_freeform_v1/development/generation_report.json').read_text())
    if (packet_hash(parent)!=PARENT_DEV_SHA or parent['manifest_sha256']!=sha256_file(MANIFEST)
            or parent['allocation_sha256']!=ALLOCATION_SHA256 or parent['prompt_variant']!=BRIEF_PROTOCOL
            or parent['sentence_limit']!=1 or parent['max_new_tokens']!=MAX_NEW_TOKENS):
        raise ValueError('Calibration parent identity changed')
    backend=GPUBackend(manifest_path=MANIFEST,use_cache=False);reference=load_reference(backend)
    directory=Path(config.RUNS_DIR)/'factual_strength_v1/development';directory.mkdir(parents=True,exist_ok=False)
    report={'protocol':'factual-strength-calibration-v1','phase':'development','status':'running',
        'scope':'Failure-enriched previously consumed calibration cases, not fresh accuracy. No automatic promotion.',
        'records':[],'case_indices':CASE_INDICES,'policies':POLICIES,'reference':reference,
        'parent_generation_sha256':PARENT_DEV_SHA,'manifest_sha256':parent['manifest_sha256'],
        'manifest_fingerprint':parent['manifest_fingerprint'],'allocation_sha256':ALLOCATION_SHA256,
        'prompt_variant':BRIEF_PROTOCOL,'sentence_limit':1,'max_new_tokens':MAX_NEW_TOKENS,
        'beta':.01,'kl_budget':config.PROCESSOR_KL_THRESHOLD,'production_quality_verified':False}
    def save(name):
        path=directory/name;temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False));temp.replace(path)
    save('generation_report.partial.json')
    for index in CASE_INDICES:
        before=parent['records'][index]
        item={k:before[k] for k in ('case_id','prompt','correct_references','incorrect_references','category')}
        outputs={'baseline':before['outputs']['baseline']};routing={}
        for name,policy in POLICIES.items():
            done=None;route=None;max_kl=0.;min_scale=1.;abstained=0;steps=0
            for event,data in backend.stream(comparison_prompt(item['prompt'],'brief'),MAX_NEW_TOKENS,threading.Event(),
                    sentence_limit=1,_branches=('unlora',),_jail_scale=0.,_beta=.01,
                    _hall_strength=policy['strength'],_hall_reference_adapter='factual_reference',
                    _hall_negative_adapter='hallucination' if policy['negative'] else None):
                if event=='error':raise RuntimeError('Calibration generation failed')
                if event=='routing':route=data
                if event=='telemetry':
                    kl=data['kl_divergence']
                    if (not math.isfinite(kl) or kl>config.PROCESSOR_KL_THRESHOLD+1e-6
                            or data['plausibility_beta']!=.01 or data['alpha_hall']!=policy['strength'] or data['alpha_jail']!=0):
                        raise RuntimeError('Calibration operator constraints changed')
                    max_kl=max(max_kl,kl);min_scale=min(min_scale,data['attenuation']);abstained+=int(data['abstained']);steps+=1
                if event=='complete':done=data
            if done is None or done['simulated'] is not False:raise RuntimeError('Incomplete calibration output')
            outputs[name]=done['unlora'];routing[name]={'weights':route,'max_measured_kl':max_kl,
                'minimum_attenuation':min_scale,'abstained_steps':abstained,'steps':steps}
        report['records'].append({**item,'outputs':outputs,'routing':routing,'reused_arms':['baseline'],'reused_source_sha256':PARENT_DEV_SHA})
        save('generation_report.partial.json');print(f'strength calibration: {len(report["records"])}/8 cases',flush=True)
    report['status']='awaiting_reference_grounded_review';save('generation_report.json');return report
