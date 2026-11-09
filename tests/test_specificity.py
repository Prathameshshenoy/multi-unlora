import math
import pytest
from evaluation.specificity import specificity_margin,summarize


def test_bad_response_learning_alone_does_not_establish_specificity():
    result=specificity_margin(4,2,4,1)
    assert result['bad_response_nll_gain']==2
    assert result['good_response_nll_gain']==3
    assert result['specificity_margin']==-1


def test_selective_bad_gain_is_positive_but_not_a_quality_claim():
    result=specificity_margin(4,2,3,2.5)
    report=summarize([result])
    assert report['mean_specificity_margin']==1.5
    assert report['positive_specificity_pairs']==1
    assert 'not a generation quality score' in report['interpretation']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1])
def test_invalid_losses_invalidate_measurement(value):
    with pytest.raises(ValueError):
        specificity_margin(value,1,1,1)


def test_empty_measurement_cannot_pass():
    with pytest.raises(ValueError):
        summarize([])
