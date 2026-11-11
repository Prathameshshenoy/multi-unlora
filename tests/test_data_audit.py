import pytest
from evaluation.data_audit import response_form_summary


def test_length_shortcut_is_reported_separately_from_quality():
    same=response_form_summary([2,4],[2,4])
    assert same['length_only_undesirable_auc']==.5
    obvious=response_form_summary([10,12],[2,3])
    assert obvious['length_only_undesirable_auc']==1
    assert obvious['undesirable_more_than_twice_as_long_pairs']==2
    assert obvious['within_factor_two_pairs']==0
    assert 'not learned detector AUROC' in obvious['scope']
    with pytest.raises(ValueError): response_form_summary([1],[1,2])


def test_training_prefix_audit_groups_pairs_and_detects_only_a_coarse_association():
    from evaluation.data_audit import prefix_label_audit
    rows=[{'question_id':str(i),'bad':'Yes, that is so.','good':'No, it is not.'} for i in range(30)]
    report=prefix_label_audit(rows)
    assert report['out_of_fold_prefix_label_auc']==1.
    assert report['heldout_answers_used'] is False
    for row in rows:row['good']='Yes, that is so.'
    assert prefix_label_audit(rows)['out_of_fold_prefix_label_auc']==.5
    with pytest.raises(ValueError):prefix_label_audit(rows+rows)
