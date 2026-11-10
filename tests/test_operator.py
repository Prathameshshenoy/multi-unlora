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


def test_operator_rejects_values_that_overflow_float32_instead_of_returning_nan():
    from inference.operator import correct_logits
    with pytest.raises(ValueError, match="float32"):
        correct_logits(torch.tensor([[1e300, 0]], dtype=torch.float64),
                       torch.zeros(1, 2, dtype=torch.float64))


def test_zero_budget_records_abstention_for_nonzero_requested_correction():
    from inference.operator import correct_logits
    base = torch.zeros(1, 3)
    correction = correct_logits(base, torch.tensor([[1., 0., 0.]]), kl_budget=0.)
    torch.testing.assert_close(correction.scores, base)
    assert correction.abstained.item()


def test_active_distribution_is_invariant_to_uniform_logit_and_delta_offsets():
    from inference.operator import correct_logits
    base = torch.tensor([[2., 1., -1., -2.]])
    delta = torch.tensor([[0., 2., -3., 1.]])
    initial = correct_logits(base, delta, beta=.01, kl_budget=.05)
    shifted = correct_logits(base + 1024., delta + 32., beta=.01, kl_budget=.05)
    torch.testing.assert_close(initial.scores.softmax(-1), shifted.scores.softmax(-1), atol=2e-7, rtol=1e-6)


def test_telemetry_lists_only_positive_probability_unmasked_candidates():
    import json
    scores = torch.tensor([[6., 0., -1., -2., -3., -4.]])
    events = []
    processor = MultiUnLoRALogitsProcessor(
        IdenticalExpert(scores + torch.tensor([[1., 0., 0., 0., 0., 0.]])),
        .4, 0, beta=.1, telemetry_callback=events.append,
    )
    corrected = processor(torch.ones(1, 2, dtype=torch.long), scores)
    assert torch.isfinite(corrected).sum().item() == 1
    assert events[0]['top_tokens'] == [
        {'token_id': 0, 'base_probability': float(scores.softmax(-1)[0, 0]),
         'hallucination_probability': float((scores + torch.tensor([[1., 0., 0., 0., 0., 0.]])).softmax(-1)[0, 0]),
         'jailbreak_probability': None, 'corrected_probability': 1.0}
    ]
    json.dumps(events, allow_nan=False)


def test_telemetry_omits_zero_mass_from_probability_underflow():
    events = []
    scores = torch.tensor([[0., -200., -300.]])
    MultiUnLoRALogitsProcessor(IdenticalExpert(scores), 0, 0, telemetry_callback=events.append)(
        torch.ones(1, 2, dtype=torch.long), scores
    )
    assert [candidate['token_id'] for candidate in events[0]['top_tokens']] == [0]
