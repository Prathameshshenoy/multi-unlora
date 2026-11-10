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
