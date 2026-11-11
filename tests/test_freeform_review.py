import copy
import pytest
from evaluation.freeform_review import blind_packet,packet_hash,summarize_review


def records():
    return [{'case_id':'case-1','prompt':'What is the capital of France?',
             'correct_references':['Paris'],'incorrect_references':['Berlin'],
             'outputs':{'baseline':{'text':'Paris.','source':'model','finish_reason':'eos'},
                        'corrected':{'text':'Berlin.','source':'model','finish_reason':'eos'}}}]


def labels(packet,key):
    answers={}
    for slot,arm in key['cases']['case-1'].items():
        answers[slot]={'core':'correct' if arm=='baseline' else 'incorrect','details':'none',
                       'rationale':'Matches the provided correct reference.' if arm=='baseline' else 'Contradicts the provided reference: Berlin instead of Paris.'}
    return {'packet_sha256':packet_hash(packet),'reviewer':'independent-reference-review',
            'cases':[{'case_id':'case-1','answers':answers}]}


def test_packet_blinds_arm_identity_and_is_reproducible_with_same_private_key():
    packet,key=blind_packet(records(),secret=b'private-test-key')
    assert (packet,key)==blind_packet(records(),secret=b'private-test-key')
    assert set(packet['cases'][0]['answers'])=={'A','B'}
    assert 'baseline' not in repr(packet) and 'corrected' not in repr(packet)
    assert all('source' in answer and 'complete' in answer for answer in packet['cases'][0]['answers'].values())


def test_actual_false_claim_is_a_regression_even_when_generation_completed():
    packet,key=blind_packet(records(),secret=b'private-test-key')
    report=summarize_review(packet,key,labels(packet,key))
    assert report['arms']['baseline']['accepted_rate']==1
    assert report['arms']['corrected']['contradiction_rate']==1
    assert report['comparisons']['corrected']['new_contradictions']==1
    assert report['comparisons']['corrected']['accepted_paired_delta']==-1


def test_unverified_extra_detail_is_separate_from_a_proven_false_claim():
    packet,key=blind_packet(records(),secret=b'private-test-key');review=labels(packet,key)
    slot=next(s for s,a in key['cases']['case-1'].items() if a=='corrected')
    review['cases'][0]['answers'][slot]={'core':'correct','details':'unverified','rationale':'Core answer matches; added origin date is not supported by the references.'}
    report=summarize_review(packet,key,review)
    assert report['arms']['corrected']['core_correct_rate']==1
    assert report['arms']['corrected']['accepted_rate']==0
    assert report['arms']['corrected']['contradiction_rate']==0
    assert report['arms']['corrected']['unverified_detail_rate']==1


def test_missing_review_cannot_silently_drop_an_unfavorable_case():
    packet,key=blind_packet(records(),secret=b'private-test-key');review=labels(packet,key)
    review['cases']=[]
    with pytest.raises(ValueError,match='coverage'):summarize_review(packet,key,review)


def test_changed_packet_or_reviewer_labels_cannot_be_scored_with_an_old_key():
    packet,key=blind_packet(records(),secret=b'private-test-key');review=labels(packet,key)
    packet['cases'][0]['answers']['A']['text']='Changed after review'
    with pytest.raises(ValueError,match='hash'):summarize_review(packet,key,review)


def test_guarded_or_truncated_text_cannot_count_as_model_success():
    rows=records();rows[0]['outputs']['corrected']['source']='display_guard'
    packet,key=blind_packet(rows,secret=b'private-test-key');review=labels(packet,key)
    slot=next(s for s,a in key['cases']['case-1'].items() if a=='corrected')
    review['cases'][0]['answers'][slot]['core']='correct'
    with pytest.raises(ValueError,match='model'):summarize_review(packet,key,review)


def test_seal_detects_mapping_changes_after_judgments_are_locked():
    from evaluation.freeform_review import seal_review,score_sealed_review,judgment_template
    packet,key=blind_packet(records(),secret=b'private-test-key');review=labels(packet,key)
    seal=seal_review(packet,key,review)
    assert score_sealed_review(packet,key,review,seal)['comparisons']['corrected']['strict_losses']==1
    key['cases']['case-1']={slot:('corrected' if arm=='baseline' else 'baseline') for slot,arm in key['cases']['case-1'].items()}
    with pytest.raises(ValueError,match='key hash'):score_sealed_review(packet,key,review,seal)


def test_pending_semantic_labels_cannot_be_accidentally_scored():
    from evaluation.freeform_review import judgment_template,validate_review
    packet,key=blind_packet(records(),secret=b'private-test-key')
    with pytest.raises(ValueError,match='Unknown'):
        validate_review(packet,judgment_template(packet,reviewer='explicit-AI-reviewer'))


def test_no_complete_pairs_produces_no_accuracy_interval_or_passable_coverage():
    from evaluation.freeform_review import judgment_template
    rows=records()
    for v in rows[0]['outputs'].values():v['finish_reason']='length'
    packet,key=blind_packet(rows,secret=b'private-test-key')
    report=summarize_review(packet,key,judgment_template(packet,reviewer='structural-AI-audit'))
    assert report['comparisons']['corrected']['jointly_scorable_pairs']==0
    assert report['comparisons']['corrected']['accepted_paired_bootstrap_95_ci'] is None
    assert report['comparisons']['corrected']['review_coverage_sufficient'] is False


def test_selection_rejects_new_contradictions_and_coverage_shortcuts():
    from evaluation.freeform_review import select_development_policy
    base={'n_assigned':12,'accepted_rate':.5,'contradiction_rate':.2,'unscorable_rate':0,'guarded_outputs':0,'unfinished_outputs':0}
    candidate={**base,'accepted_rate':.75,'contradiction_rate':.1}
    pair={'review_coverage_sufficient':True,'strict_wins':3,'strict_losses':0,'new_contradictions':0}
    report={'arms':{'baseline':base,**{a:dict(candidate) for a in ('quarter','hall_only','current')}},
            'comparisons':{a:dict(pair) for a in ('quarter','hall_only','current')}}
    assert select_development_policy(report)['selected']=='quarter'
    report['comparisons']['quarter']['new_contradictions']=1
    assert select_development_policy(report)['selected']=='hall_only'
    report['comparisons']['hall_only']['review_coverage_sufficient']=False
    report['arms']['current']['unfinished_outputs']=1
    assert select_development_policy(report)['selected'] is None


def test_packet_answer_must_match_the_real_generated_answer_before_sealing():
    from evaluation.freeform_review import validate_generation_binding
    rows=records();packet,key=blind_packet(rows,secret=b'private-test-key')
    assert validate_generation_binding(rows,packet,key)
    packet['cases'][0]['answers']['A']['text']='A forged replacement answer'
    key['packet_sha256']=packet_hash(packet)
    with pytest.raises(ValueError,match='differs from source'):
        validate_generation_binding(rows,packet,key)


def test_factual_reference_must_improve_beyond_positive_only_not_just_base():
    from evaluation.freeform_review import select_factual_reference_policy
    arms={name:{'n_assigned':12,'accepted_rate':.8,'contradiction_rate':0.,'unscorable_rate':0.,'guarded_outputs':0,'unfinished_outputs':0}
          for name in ('baseline','reference_only','factual','factual_both','factual_wide')}
    def report(reference,net):
        return {'packet_sha256':'same','judgments_sha256':'same','arms':arms,
                'comparisons':{name:{'reference':reference,'review_coverage_sufficient':True,'strict_wins':net,'strict_losses':0,'new_contradictions':0}
                               for name in ('factual','factual_both','factual_wide')}}
    base=report('baseline',2);positive=report('reference_only',0)
    assert select_factual_reference_policy(base,positive)['selected'] is None
    positive=report('reference_only',1)
    assert select_factual_reference_policy(base,positive)['selected']=='factual'
    positive['comparisons']['factual']['new_contradictions']=1
    assert select_factual_reference_policy(base,positive)['selected']=='factual_both'


def test_strength_calibration_requires_subtraction_gain_over_matched_control():
    from evaluation.strength_trial import select_strength
    from copy import deepcopy
    arms={name:{'n_assigned':8,'accepted_rate':.75,'contradiction_rate':0.,'unscorable_rate':0.,'guarded_outputs':0,'unfinished_outputs':0}
          for name in ('baseline','positive_1','positive_2','positive_4','contrast_1','contrast_2','contrast_4')}
    def report(wins,reference='baseline'):
        return {'packet_sha256':'same','judgments_sha256':'same','arms':deepcopy(arms),
            'comparisons':{name:{'reference':reference,'review_coverage_sufficient':True,'strict_wins':wins,'strict_losses':0,'new_contradictions':0}
                           for name in ('contrast_1','contrast_2','contrast_4')}}
    base=report(2);controls={n:report(0,f'positive_{n}') for n in (1,2,4)}
    assert select_strength(base,controls)['selected'] is None
    controls[2]=report(1,'positive_2')
    assert select_strength(base,controls)['selected']=='contrast_2'
    base['comparisons']['contrast_2']['strict_losses']=1
    assert select_strength(base,controls)['selected'] is None
    controls[1]['judgments_sha256']='different'
    with pytest.raises(ValueError,match='identical sealed judgments'):select_strength(base,controls)


def test_sealed_subset_keeps_full_review_binding_and_rejects_changed_labels():
    from evaluation.freeform_review import seal_review,score_sealed_review,judgment_template
    rows=records();packet,key=blind_packet(rows,secret=b'private-test-key')
    review=judgment_template(packet,reviewer='AI-test')
    for case in review['cases']:
        for value in case['answers'].values():value.update(core='correct',details='none',rationale='Fixture reference agreement.')
    seal=seal_review(packet,key,review)
    chosen=[rows[0]['case_id']]
    report=score_sealed_review(packet,key,review,seal,case_ids=chosen)
    assert report['subset_n']==1
    assert all(v['n_assigned']==1 for v in report['arms'].values())
    assert report['packet_sha256']==packet_hash(packet)
    with pytest.raises(ValueError,match='subset'):
        score_sealed_review(packet,key,review,seal,case_ids=['missing'])
    review['cases'][0]['answers']['A']['rationale']='Changed after sealing'
    with pytest.raises(ValueError,match='Sealed judgments'):
        score_sealed_review(packet,key,review,seal,case_ids=chosen)


def test_support_policy_requires_smoke_retention_and_matched_positive_gain():
    from copy import deepcopy
    from evaluation.support_trial import select_support_policy
    arms={name:{'n_assigned':16,'accepted_rate':.8,'contradiction_rate':0.,'unscorable_rate':0.,'guarded_outputs':0,'unfinished_outputs':0}
          for name in ('baseline','positive_0.5','positive_2','positive_8','contrast_0.5','contrast_2','contrast_8')}
    def report(wins,reference='baseline'):
        return {'packet_sha256':'same','judgments_sha256':'same','arms':deepcopy(arms),
            'comparisons':{name:{'reference':reference,'review_coverage_sufficient':True,'strict_wins':wins,'strict_losses':0,'new_contradictions':0}
                           for name in ('contrast_0.5','contrast_2','contrast_8')}}
    base=report(2);positive={n:report(0,f'positive_{n:g}') for n in (.5,2.,8.)}
    smoke=report(0)
    for value in smoke['arms'].values():value.update(n_assigned=4,accepted_rate=1.)
    assert select_support_policy(base,positive,smoke)['selected'] is None
    positive[2.]=report(1,'positive_2')
    assert select_support_policy(base,positive,smoke)['selected']=='contrast_2'
    smoke['arms']['contrast_2']['accepted_rate']=.75
    assert select_support_policy(base,positive,smoke)['selected'] is None
    positive[.5]['comparisons']['contrast_0.5']['reference']='positive_8'
    with pytest.raises(ValueError,match='Unmatched'):select_support_policy(base,positive,smoke)


def test_confirmation_requires_positive_lower_bounds_for_both_controls():
    from copy import deepcopy
    from evaluation.support_trial import confirm_support_policy
    arms={name:{'n_assigned':20,'guarded_outputs':0,'unfinished_outputs':0} for name in ('baseline','positive_8','contrast_8')}
    pair={'reference':'baseline','review_coverage_sufficient':True,'new_contradictions':0,'accepted_paired_bootstrap_95_ci':[.05,.4]}
    base={'packet_sha256':'same','judgments_sha256':'same','arms':arms,'comparisons':{'contrast_8':pair}}
    positive=deepcopy(base);positive['comparisons']['contrast_8']['reference']='positive_8'
    assert confirm_support_policy(base,positive)['confirmation_passed'] is True
    positive['comparisons']['contrast_8']['accepted_paired_bootstrap_95_ci']=[0.,.3]
    assert confirm_support_policy(base,positive)['confirmation_passed'] is False
    positive['comparisons']['contrast_8']['accepted_paired_bootstrap_95_ci']=[.05,.4]
    base['comparisons']['contrast_8']['new_contradictions']=1
    assert confirm_support_policy(base,positive)['confirmation_passed'] is False
    positive['judgments_sha256']='different'
    with pytest.raises(ValueError,match='same sealed review'):confirm_support_policy(base,positive)
