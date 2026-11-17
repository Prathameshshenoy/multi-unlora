import pytest

torch = pytest.importorskip('torch')
from training.factual_objective import paired_subtraction_loss


def test_update_improves_corrected_preference_and_freezes_base():
    base = torch.zeros(2, 4, requires_grad=True)
    good = torch.zeros(2, 4, requires_grad=True)
    bad = torch.zeros(2, 4, requires_grad=True)
    target = torch.tensor([0, 1])
    loss, before = paired_subtraction_loss(base, good, target, base, bad, target)
    loss.backward()
    assert base.grad is None
    # Subtracting an anti-expert requires reducing its support for correct tokens.
    assert good.grad[0, 0] > 0 and bad.grad[0, 0] < 0
    with torch.no_grad():
        good -= good.grad
        bad -= bad.grad
    updated, after = paired_subtraction_loss(base, good, target, base, bad, target)
    assert updated < loss and after['response_margin'] > before['response_margin']


@pytest.mark.parametrize('kwargs', [{'alpha':0}, {'alpha':.9}, {'alpha':True},
                                    {'margin':float('nan')}, {'rank_weight':-1}])
def test_invalid_objective_configuration_is_rejected(kwargs):
    logits = torch.zeros(2, 4)
    target = torch.tensor([0, 1])
    with pytest.raises(ValueError):
        paired_subtraction_loss(logits, logits, target, logits, logits, target, **kwargs)


def test_invalid_or_empty_response_targets_are_rejected():
    logits = torch.zeros(2, 4)
    for targets in [torch.tensor([0, 9]), torch.tensor([0]), torch.tensor([0., 1.])]:
        with pytest.raises(ValueError):
            paired_subtraction_loss(logits, logits, targets, logits, logits, targets)
    with pytest.raises(ValueError):
        paired_subtraction_loss(logits[:0], logits[:0], torch.tensor([],dtype=torch.long),
                                logits, logits, torch.tensor([0, 1]))
