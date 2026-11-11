"""Blinded reference-grounded review of generated answers, not word matching.

An independent reviewer supplies semantic judgments. This module enforces
coverage, provenance and denominators; it cannot determine truth by itself.
Unverified details are distinct from contradicted claims. Guarded or unfinished
text cannot count as model success. All assigned cases stay in the denominator.
"""
import hashlib
import hmac
import json
import random

PROTOCOL='blinded-generated-answer-review-v1'
CORES={'correct','incorrect','abstains','unscorable'}
DETAILS={'none','supported','unverified','contradicted'}


def packet_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def blind_packet(records,*,secret):
    if not records or not isinstance(secret,bytes) or not secret:
        raise ValueError('Nonempty records and a private blinding key are required')
    cases=[];mapping={};arms=None
    for row in records:
        cid=row['case_id'];outputs=row['outputs']
        if not isinstance(cid,str) or not cid or cid in mapping:
            raise ValueError('Case identities must be nonempty and unique')
        if not 2<=len(outputs)<=26 or (arms is not None and set(outputs)!=arms):
            raise ValueError('Every case needs the same two or more comparison arms')
        arms=set(outputs)
        if not row['correct_references'] or any(not isinstance(r,str) or not r.strip() for r in row['correct_references']):
            raise ValueError('Correct reference annotations are required')
        order=sorted(outputs,key=lambda a:hmac.new(secret,(cid+'\0'+a).encode(),hashlib.sha256).digest())
        slots={};mapping[cid]={}
        for index,arm in enumerate(order):
            value=outputs[arm];slot=chr(65+index)
            if not isinstance(value.get('text'),str):raise ValueError('Generated text must be a string')
            mapping[cid][slot]=arm
            slots[slot]={'text':value['text'],'source':value.get('source','model'),
                         'complete':value.get('finish_reason') in ('eos','sentence_limit')}
        cases.append({'case_id':cid,'prompt':row['prompt'],'correct_references':row['correct_references'],
                      'incorrect_references':row.get('incorrect_references',[]),'answers':slots})
    packet={'protocol':PROTOCOL,'scope':'Reference-grounded semantic review; source annotations may be incomplete or outdated. Not an official benchmark score.',
            'cases':cases}
    key={'protocol':PROTOCOL,'packet_sha256':packet_hash(packet),'records_sha256':packet_hash(records),'cases':mapping}
    return packet,key


def validate_review(packet,review):
    if packet.get('protocol')!=PROTOCOL or review.get('packet_sha256')!=packet_hash(packet):
        raise ValueError('Review packet hash or protocol mismatch')
    if not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip():
        raise ValueError('Reviewer identity is required; do not imply human review')
    items=review.get('cases',[])
    if not isinstance(items,list) or any(not isinstance(row,dict) or 'case_id' not in row for row in items):
        raise ValueError('Review case coverage is incomplete')
    indexed={row['case_id']:row for row in items}
    expected={row['case_id'] for row in packet['cases']}
    if len(indexed)!=len(items) or set(indexed)!=expected:
        raise ValueError('Review case coverage must be complete and duplicate-free')
    for case in packet['cases']:
        answers=indexed[case['case_id']].get('answers',{})
        if set(answers)!=set(case['answers']):raise ValueError('Review answer coverage must be complete')
        for slot,label in answers.items():
            if label.get('core') not in CORES or label.get('details') not in DETAILS:
                raise ValueError('Unknown semantic review label')
            if not isinstance(label.get('rationale'),str) or not label['rationale'].strip():
                raise ValueError('Every semantic judgment needs an evidence rationale')
            generated=case['answers'][slot]
            if label['core']=='unscorable' and label['details']!='none':
                raise ValueError('Unscorable output cannot imply a verified detail judgment')
            if (generated['source']!='model' or not generated['complete']) and label['core']!='unscorable':
                raise ValueError('Guarded or incomplete output cannot count as model success or error')
    return indexed


def _paired_interval(values):
    rng=random.Random(42);n=len(values)
    draws=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(2000))
    return [draws[50],draws[1950]]


def summarize_review(packet,key,review,*,reference='baseline'):
    digest=packet_hash(packet)
    if key.get('packet_sha256')!=digest or review.get('packet_sha256')!=digest:
        raise ValueError('Blinding key, review and packet hash must match')
    indexed=validate_review(packet,review)
    if set(key.get('cases',{}))!={c['case_id'] for c in packet['cases']}:
        raise ValueError('Blinding key case coverage mismatch')
    arms={};rows=[];arm_names=None
    for case in packet['cases']:
        cid=case['case_id'];mapping=key['cases'][cid]
        if set(mapping)!=set(case['answers']) or len(set(mapping.values()))!=len(mapping):
            raise ValueError('Blinding key answer coverage mismatch')
        if arm_names is not None and set(mapping.values())!=arm_names:
            raise ValueError('Blinding key comparison arms changed between cases')
        arm_names=set(mapping.values());record={}
        for slot,arm in mapping.items():
            label=indexed[cid]['answers'][slot];generated=case['answers'][slot]
            accepted=label['core']=='correct' and label['details'] in ('none','supported')
            value={'core_correct':label['core']=='correct','accepted':accepted,
                   'contradiction':label['core']=='incorrect' or label['details']=='contradicted',
                   'abstains':label['core']=='abstains','unscorable':label['core']=='unscorable',
                   'unverified_detail':label['details']=='unverified','guarded':generated['source']!='model',
                   'unfinished':not generated['complete']}
            arms.setdefault(arm,[]).append(value);record[arm]=value
        rows.append(record)
    if reference not in arms:raise ValueError('Reference comparison arm is missing')
    n=len(rows);summaries={}
    for arm,values in arms.items():
        summaries[arm]={'n_assigned':n,**{name+'_rate':sum(int(v[name]) for v in values)/n
            for name in ('core_correct','accepted','contradiction','abstains','unscorable','unverified_detail')},
            'guarded_outputs':sum(v['guarded'] for v in values),'unfinished_outputs':sum(v['unfinished'] for v in values)}
    comparisons={}
    for arm in arms:
        if arm==reference:continue
        changes=[int(r[arm]['accepted'])-int(r[reference]['accepted']) for r in rows]
        jointly=sum(not r[arm]['unscorable'] and not r[reference]['unscorable'] for r in rows)
        comparisons[arm]={'n_assigned':n,'reference':reference,'accepted_paired_delta':sum(changes)/n,
            'accepted_paired_bootstrap_95_ci':_paired_interval(changes) if jointly else None,
            'review_coverage_sufficient':jointly/n>=.9,
            'strict_wins':sum(v==1 for v in changes),'strict_losses':sum(v==-1 for v in changes),
            'new_contradictions':sum(r[arm]['contradiction'] and not r[reference]['contradiction'] and not r[reference]['unscorable'] for r in rows),
            'fixed_contradictions':sum(r[reference]['contradiction'] and r[arm]['accepted'] for r in rows),
            'jointly_scorable_pairs':jointly}
    return {'protocol':PROTOCOL,'packet_sha256':digest,'judgments_sha256':packet_hash(review),'reviewer':review['reviewer'],
            'scope':packet['scope'],'denominator':'All assigned cases. Unscorable, guarded and unfinished cases are not successes; coverage is reported separately.',
            'arms':summaries,'comparisons':comparisons,'production_quality_verified':False}


def seal_review(packet,key,review):
    """Bind judgments before unblinding; hashes do not reveal the arm mapping."""
    validate_review(packet,review)
    if key.get('packet_sha256')!=packet_hash(packet):raise ValueError('Blinding key hash mismatch')
    return {'protocol':PROTOCOL,'packet_sha256':packet_hash(packet),'key_sha256':packet_hash(key),
            'judgments_sha256':packet_hash(review),'reviewer':review['reviewer']}


def score_sealed_review(packet,key,review,seal,*,reference='baseline'):
    for name,value in (('packet',packet),('key',key),('judgments',review)):
        if seal.get(name+'_sha256')!=packet_hash(value):raise ValueError('Sealed '+name+' hash changed')
    return summarize_review(packet,key,review,reference=reference)


def judgment_template(packet,*,reviewer):
    cases=[]
    for case in packet['cases']:
        answers={}
        for slot,value in case['answers'].items():
            structural=value['source']!='model' or not value['complete']
            answers[slot]={'core':'unscorable' if structural else 'PENDING','details':'none' if structural else 'PENDING',
                'rationale':'Not a completed raw model answer; retained in coverage and denominator.' if structural else ''}
        cases.append({'case_id':case['case_id'],'answers':answers})
    return {'packet_sha256':packet_hash(packet),'reviewer':reviewer,'cases':cases}


def select_development_policy(report,*,expected_n=12):
    """Fixed pilot selection: two net accepted wins, no new contradiction."""
    baseline=report['arms']['baseline'];eligible=[]
    if baseline['n_assigned']!=expected_n:raise ValueError('Development sample count changed')
    for rank,arm in enumerate(('quarter','hall_only','current')):
        value=report['arms'][arm];pair=report['comparisons'][arm]
        if value['n_assigned']!=expected_n:raise ValueError('Development sample count changed')
        if (pair['review_coverage_sufficient'] and pair['strict_wins']-pair['strict_losses']>=2
                and pair['new_contradictions']==0 and value['contradiction_rate']<=baseline['contradiction_rate']
                and value['unscorable_rate']<=baseline['unscorable_rate']
                and value['guarded_outputs']==0 and value['unfinished_outputs']==0):
            eligible.append((value['accepted_rate'],rank,arm))
    if not eligible:return {'selected':None,'status':'no_eligible_development_gain','automatic_promotion':False}
    _,_,winner=min(eligible,key=lambda r:(-r[0],r[1]))
    return {'selected':winner,'status':'requires_untouched_generated_answer_confirmation','automatic_promotion':False}
