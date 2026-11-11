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
    from evaluation.freeform_review import seal_review,score_sealed_review
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
