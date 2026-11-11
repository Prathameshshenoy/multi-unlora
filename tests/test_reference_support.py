import pytest
torch=pytest.importorskip('torch')
from inference.operator import correct_logits,reverse_kl


def test_reference_supported_token_can_recover_without_resurrecting_explicit_masks():
    base=torch.tensor([[10.,0.,-torch.inf]])
    reference=torch.tensor([[0.,12.,12.]])
    delta=torch.tensor([[10.,-10.,0.]])
    original=correct_logits(base,delta,beta=.01,kl_budget=8.)
    assert original.scores.argmax(-1).item()==0
    fixed=correct_logits(base,delta,beta=.01,kl_budget=8.,candidate_reference=reference)
    assert fixed.scores.argmax(-1).item()==1
    assert torch.isneginf(fixed.scores[0,2])
    measured=reverse_kl(fixed.scores.log_softmax(-1),base.log_softmax(-1))
    torch.testing.assert_close(fixed.kl,measured)
    assert 0<float(measured)<=8.+1e-6
    conservative=correct_logits(base,delta,beta=.01,kl_budget=.5,candidate_reference=reference)
    assert conservative.scores.argmax(-1).item()==0
    assert float(conservative.kl)<=.5+1e-6


def test_reference_support_retains_exact_zero_correction_and_batch_invariance():
    base=torch.tensor([[10.,0.,-1.],[0.,1.,2.]])
    ref=torch.tensor([[0.,10.,-1.],[3.,2.,1.]])
    delta=torch.tensor([[9.,-9.,0.],[1.,-1.,2.]])
    assert correct_logits(base,torch.zeros_like(base),candidate_reference=ref).scores is base
    batch=correct_logits(base,delta,candidate_reference=ref,kl_budget=2.)
    singles=torch.cat([correct_logits(b[None],d[None],candidate_reference=r[None],kl_budget=2.).scores for b,d,r in zip(base,delta,ref)])
    torch.testing.assert_close(batch.scores,singles)


@pytest.mark.parametrize('reference',[torch.zeros(1,2),torch.tensor([[0.,float('nan'),1.]]),torch.tensor([[0.,float('inf'),1.]])])
def test_invalid_reference_support_fails_closed(reference):
    with pytest.raises(ValueError):
        correct_logits(torch.zeros(1,3),torch.ones(1,3),candidate_reference=reference)
