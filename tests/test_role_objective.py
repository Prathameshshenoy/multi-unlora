import pytest
torch=pytest.importorskip('torch')
from training.role_repair import role_sft_loss


def test_role_loss_supervises_only_response_and_detaches_preservation_reference():
    logits=torch.zeros(1,4,5,requires_grad=True)
    labels=torch.tensor([[-100,-100,2,3]])
    reference=torch.zeros(2,5,requires_grad=True).log_softmax(-1)
    reference.retain_grad()
    control=torch.zeros(2,5,requires_grad=True)
    loss=role_sft_loss(logits,labels,control,reference,control_weight=.1)
    loss.backward()
    assert logits.grad[0,0].abs().sum()==0
    assert logits.grad[0,1,2]<0 and logits.grad[0,2,3]<0
    assert reference.grad is None
    assert torch.isfinite(loss)


def test_role_loss_rejects_missing_supervision():
    with pytest.raises(ValueError,match='supervised'):
        role_sft_loss(torch.zeros(1,2,5),torch.full((1,2),-100),torch.zeros(1,5),torch.zeros(1,5).log_softmax(-1),control_weight=.1)
