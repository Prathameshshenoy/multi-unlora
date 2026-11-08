"""Regression tests for contrastive anti-expert decoding."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from inference.processor import MultiUnLoRALogitsProcessor


class IdenticalExpert:
    def __init__(self, logits):
        self.logits = logits
        self.enabled = False

    def enable_adapter_layers(self):
        self.enabled = True

    def disable_adapter_layers(self):
        self.enabled = False

    def set_adapter(self, name):
        pass

    def __call__(self, **kwargs):
        return SimpleNamespace(logits=self.logits[:, None, :])


def test_identical_expert_does_not_penalize_base_preferences():
    scores = torch.tensor([[4.0, 2.0, 1.0, -3.0]])
    model = IdenticalExpert(scores)
    corrected = MultiUnLoRALogitsProcessor(model, 0.4, 0.3)(
        torch.ones(1, 2, dtype=torch.long), scores
    )
    torch.testing.assert_close(corrected, scores, rtol=0, atol=0)
    assert not model.enabled


def test_implausible_token_cannot_win_after_subtraction():
    from inference.operator import correct_logits
    base = torch.tensor([[4., 3., -30.]])
    delta = torch.tensor([[20., 19., -100.]])
    result = correct_logits(base, delta, beta=.1, kl_budget=2.)
    assert torch.isneginf(result.scores[0, 2])
    assert result.scores.argmax(-1).item() != 2


def test_budget_is_measured_on_final_distribution_per_row():
    from inference.operator import correct_logits, reverse_kl
    torch.manual_seed(17)
    base = torch.randn(12, 64)
    delta = torch.randn(12, 64) * 25
    for budget in [0., .001, .1, .5, 2.]:
        result = correct_logits(base, delta, beta=.01, kl_budget=budget)
        kl = reverse_kl(result.scores.log_softmax(-1), base.log_softmax(-1))
        assert (kl <= budget + 1e-6).all()
        assert torch.isfinite(kl).all()
        torch.testing.assert_close(kl, result.kl)


def test_mask_floor_abstains_instead_of_violating_budget():
    from inference.operator import correct_logits
    base = torch.tensor([[.1, 0., -.1]])
    result = correct_logits(base, torch.tensor([[1., 0., 2.]]), beta=1., kl_budget=.01)
    torch.testing.assert_close(result.scores, base)
    assert result.abstained.item()
    assert result.scale.item() == 0


def test_delta_sign_suppresses_adapter_preference():
    from inference.operator import correct_logits
    base = torch.zeros(1, 4)
    result = correct_logits(base, torch.tensor([[2., 0., 0., 0.]]), beta=.1, kl_budget=2.)
    assert result.scores.softmax(-1)[0,0] < .25


def test_uniform_adapter_shift_is_identity():
    from inference.operator import correct_logits
    base = torch.tensor([[2., 1., -3.]])
    result = correct_logits(base, torch.full_like(base, 9.))
    assert result.scores is base


def test_bfloat16_results_obey_float32_budget():
    from inference.operator import correct_logits
    base = torch.tensor([[2., 1.9, 1., -1.]], dtype=torch.bfloat16)
    result = correct_logits(base, torch.tensor([[8., -8., 0., 0.]]), beta=.01, kl_budget=.03)
    assert result.scores.dtype == torch.float32
    assert result.kl.item() <= .030001
