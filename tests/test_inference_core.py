"""CPU regression checks. Run with torch, transformers, pandas and tqdm installed."""
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import config
from inference.processor import MultiUnLoRALogitsProcessor
from repe import extractor, router


class ExpertModel:
    def __init__(self, logits, fail=None):
        self.logits = logits
        self.fail = fail
        self.enabled = False
        self.queries = []

    def enable_adapter_layers(self):
        self.enabled = True

    def disable_adapter_layers(self):
        self.enabled = False

    def set_adapter(self, name):
        self.adapter = name

    def __call__(self, **kwargs):
        self.queries.append(self.adapter)
        if self.adapter == self.fail:
            raise RuntimeError("expert forward failed")
        return SimpleNamespace(logits=self.logits[:, None, :])


def test_zero_alpha_is_exact_noop_without_expert_queries():
    model = ExpertModel(torch.tensor([[4.0, 1.0, 0.0]]))
    scores = torch.tensor([[1.0, -float("inf"), 2.0]])
    events = []
    processor = MultiUnLoRALogitsProcessor(model, 0, 0, telemetry_callback=events.append)
    assert processor(torch.ones(1, 1, dtype=torch.long), scores) is scores
    assert model.queries == []
    assert events[0]["kl_divergence"] == 0
    json.dumps(events, allow_nan=False)


@pytest.mark.parametrize("failure", ["hallucination", "jailbreak"])
def test_adapter_disabled_when_expert_raises(failure):
    model = ExpertModel(torch.ones(1, 3), fail=failure)
    processor = MultiUnLoRALogitsProcessor(model, 0.4, 0.4)
    with pytest.raises(RuntimeError, match="expert forward failed"):
        processor(torch.ones(1, 1, dtype=torch.long), torch.zeros(1, 3))
    assert not model.enabled


def test_masked_scores_tiny_vocab_and_telemetry_are_finite():
    model = ExpertModel(torch.tensor([[3.0, 2.0, -1.0]]))
    scores = torch.tensor([[1.0, -float("inf"), -1.0]])
    events = []
    processor = MultiUnLoRALogitsProcessor(model, 0.4, 0, telemetry_callback=events.append)
    corrected = processor(torch.ones(1, 1, dtype=torch.long), scores)
    assert torch.isneginf(corrected[0, 1])
    assert torch.isfinite(corrected[0, [0, 2]]).all()
    assert model.queries == ["hallucination"]
    assert not model.enabled
    assert events[0]["step"] == 1
    json.dumps(events, allow_nan=False)


def test_batch_guard_matches_independent_rows():
    scores = torch.tensor([[10.0, 0.0, -10.0], [0.1, 0.0, -0.1]])
    expert = torch.tensor([[0.0, 0.0, 10.0], [8.0, 0.0, -1.0]])
    kwargs = dict(alpha_hall=3.0, alpha_jail=0, penalty_threshold=0.01)
    events = []
    batched = MultiUnLoRALogitsProcessor(
        ExpertModel(expert), telemetry_callback=events.append, **kwargs
    )(torch.ones(2, 1, dtype=torch.long), scores)
    singles = torch.cat([
        MultiUnLoRALogitsProcessor(ExpertModel(row[None]), **kwargs)(
            torch.ones(1, 1, dtype=torch.long), score[None]
        )
        for row, score in zip(expert, scores)
    ])
    torch.testing.assert_close(batched, singles)
    assert events[0]["attenuation_per_row"][0] != events[0]["attenuation_per_row"][1]


@pytest.mark.parametrize("scores", [
    [[float("nan"), 1]], [[float("inf"), 1]],
    [[-float("inf"), -float("inf")]],
])
def test_invalid_distributions_fail_explicitly(scores):
    processor = MultiUnLoRALogitsProcessor(ExpertModel(torch.zeros(1, 2)), 0.4, 0.4)
    with pytest.raises(ValueError):
        processor(torch.ones(1, 1, dtype=torch.long), torch.tensor(scores))


def test_router_normalizes_directions_and_obeys_joint_cap(monkeypatch):
    monkeypatch.setattr(router, "get_hidden_state_via_hook", lambda *args, **kwargs: torch.ones(3))
    model = ExpertModel(torch.zeros(1, 3))
    alphas = router.compute_dynamic_alphas(model, None, "prompt", torch.ones(3) * 10, torch.ones(3))
    assert sum(alphas) <= config.ALPHA_CAP
    assert abs(alphas[0] - alphas[1]) < 1e-6


def test_zero_direction_rejected(monkeypatch):
    monkeypatch.setattr(router, "get_hidden_state_via_hook", lambda *args, **kwargs: torch.ones(3))
    with pytest.raises(ValueError, match="finite and nonzero"):
        router.compute_dynamic_alphas(None, None, "prompt", torch.zeros(3), torch.ones(3))


class Encoded(dict):
    def to(self, device):
        return self


class Tokenizer:
    padding_side = "right"

    def apply_chat_template(self, messages, **kwargs):
        return str(messages)

    def __call__(self, text, **kwargs):
        self.last_options = kwargs
        batch = len(text) if isinstance(text, list) else 1
        return Encoded(input_ids=torch.ones(batch, 2, dtype=torch.long))


class HookedModel(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.model = SimpleNamespace(layers=[torch.nn.Linear(2, 2)])
        self.fail = fail

    def forward(self, input_ids, **kwargs):
        hidden = self.model.layers[0](torch.ones(input_ids.shape[0], 2, 2))
        if self.fail:
            raise RuntimeError("base forward failed")
        return hidden


def test_extractor_restores_padding_and_hooks_on_failure():
    model = HookedModel(fail=True)
    tokenizer = Tokenizer()
    with pytest.raises(RuntimeError, match="base forward failed"):
        extractor.extract_concept_vector(model, tokenizer, ["a", "b"], layer_idx=0)
    assert tokenizer.padding_side == "right"
    assert not model.model.layers[0]._forward_hooks


def test_cpu_extraction_and_plain_hf_unwrap():
    model = HookedModel()
    assert extractor.get_hf_base_model(model) is model
    tokenizer = Tokenizer()
    vector = extractor.get_hidden_state_via_hook(
        model, tokenizer, "test", layer_idx=0, add_special_tokens=False
    )
    assert tokenizer.last_options["add_special_tokens"] is False
    assert vector.shape == (2,)
    assert vector.device.type == "cpu"
    assert not model.model.layers[0]._forward_hooks


def test_evaluation_prompt_is_shared_and_legacy_csv_is_rejected(tmp_path):
    pytest.importorskip("pandas")
    from evaluation.runners import format_evaluation_prompt, run_evaluation

    assert format_evaluation_prompt("Question: 1 + 1?") == "Question: 1 + 1?"
    assert format_evaluation_prompt("Is this true?") == (
        f"{config.SYSTEM_PROMPT_HONEST}User: Is this true?\nAssistant:"
    )
    csv = tmp_path / "legacy.csv"
    csv.write_text("prompt,answer\nold,confounded\n")
    with pytest.raises(ValueError, match="Legacy evaluation CSV"):
        run_evaluation(None, None, [], str(csv), lambda *args: {}, use_unlora=False)


def test_single_prompt_extractor_rejects_a_batch_instead_of_discarding_rows():
    with pytest.raises(ValueError, match='exactly one'):
        extractor.get_hidden_state_via_hook(None, None, ['first prompt', 'second prompt'])
