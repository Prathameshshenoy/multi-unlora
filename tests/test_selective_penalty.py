import pytest
torch=pytest.importorskip('torch')
from inference.operator import selective_penalty,correct_logits


def test_penalty_does_not_boost_tokens_the_anti_expert_underrepresents():
    anti=torch.tensor([[5.,0.,2.]])
    reference=torch.tensor([[0.,5.,2.]])
    penalty=selective_penalty(anti,reference)
    assert penalty[0,0]>0 and penalty[0,1]==0
    assert (penalty>=0).all()
    torch.testing.assert_close(penalty,selective_penalty(anti+17,reference-11))
    # Unpenalized token logits share unchanged relative odds before normalization.
    base=torch.zeros_like(anti)
    result=correct_logits(base,penalty,beta=.01,kl_budget=8.)
    assert result.scores[0,0]<result.scores[0,1]


def test_identical_expert_distributions_are_an_exact_noop():
    anti=torch.tensor([[2.,-1.,4.]])
    penalty=selective_penalty(anti,anti+8.)
    assert torch.count_nonzero(penalty)==0
    base=torch.tensor([[5.,1.,0.]])
    assert correct_logits(base,penalty).scores is base


@pytest.mark.parametrize('bad',[torch.tensor([[float('nan'),0.]]),torch.tensor([[float('inf'),0.]])])
def test_invalid_selective_inputs_fail_explicitly(bad):
    with pytest.raises(ValueError):selective_penalty(bad,torch.zeros_like(bad))


def test_new_allocation_cannot_reuse_old_confirmation_or_training_ids():
    from evaluation.selective_allocation import validate_fresh_ids
    ids=[str(i) for i in range(20)]
    validate_fresh_ids(ids,{'training','old-confirmation'})
    with pytest.raises(ValueError):validate_fresh_ids(ids,{'7'})
    with pytest.raises(ValueError):validate_fresh_ids(ids[:-1]+['0'],set())
