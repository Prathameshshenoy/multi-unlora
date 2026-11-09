import pytest
torch=pytest.importorskip('torch')
from training.paired_objective import response_log_probabilities,anti_expert_preference_loss


def test_objective_targets_relative_bad_preference_and_freezes_reference():
    bad=torch.tensor([-4.],requires_grad=True)
    good=torch.tensor([-4.],requires_grad=True)
    ref_bad=torch.tensor([-4.],requires_grad=True)
    ref_good=torch.tensor([-4.],requires_grad=True)
    anti_expert_preference_loss(bad,good,ref_bad,ref_good).backward()
    assert bad.grad.item()<0 and good.grad.item()>0
    assert ref_bad.grad is None and ref_good.grad is None


def test_common_style_gain_cancels_from_preference_margin():
    ref_bad=torch.tensor([-9.]);ref_good=torch.tensor([-5.])
    initial=anti_expert_preference_loss(ref_bad,ref_good,ref_bad,ref_good)
    same_gain=anti_expert_preference_loss(ref_bad+3,ref_good+3,ref_bad,ref_good)
    torch.testing.assert_close(initial,same_gain)


def test_response_probability_masks_instruction_and_uses_causal_shift():
    logits=torch.zeros(1,4,3)
    labels=torch.tensor([[-100,-100,1,2]])
    result=response_log_probabilities(logits,labels)
    torch.testing.assert_close(result,torch.tensor([-2*torch.log(torch.tensor(3.)).item()]))


def test_empty_response_invalidates_pair():
    with pytest.raises(ValueError):
        response_log_probabilities(torch.zeros(1,3,3),torch.full((1,3),-100))


def test_nonfinite_loss_input_is_rejected():
    with pytest.raises(ValueError):
        anti_expert_preference_loss(torch.tensor([float('nan')]),torch.zeros(1),torch.zeros(1),torch.zeros(1))
