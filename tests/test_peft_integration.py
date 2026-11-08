"""Real, download-free HF/PEFT checks of mutable adapter decoding state."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from transformers import LlamaConfig, LlamaForCausalLM
from inference.processor import MultiUnLoRALogitsProcessor
from inference.operator import reverse_kl


@pytest.fixture
def tiny_model():
    torch.manual_seed(7)
    base = LlamaForCausalLM(LlamaConfig(
        vocab_size=48, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.0,
    ))
    lora = peft.LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                          lora_dropout=0, task_type="CAUSAL_LM")
    model = peft.get_peft_model(base, lora, adapter_name="hallucination")
    model.add_adapter("jailbreak", lora)
    model.eval()
    model.disable_adapter_layers()
    return model


def forward_scores(model, ids, mask=None):
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1]


def test_zero_initialized_peft_adapters_are_exact_distribution_identity(tiny_model):
    ids = torch.tensor([[1, 5, 8]])
    scores = forward_scores(tiny_model, ids)
    events = []
    result = MultiUnLoRALogitsProcessor(tiny_model, .4, .4, telemetry_callback=events.append)(ids, scores)
    torch.testing.assert_close(result, scores, rtol=0, atol=0)
    torch.testing.assert_close(forward_scores(tiny_model, ids), scores, rtol=0, atol=0)
    assert events[-1]["kl_divergence"] == 0
    assert all(module.disable_adapters for module in tiny_model.modules()
               if isinstance(module, peft.tuners.tuners_utils.BaseTunerLayer))


def _activate_nonzero_loras(model):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(mean=0, std=1)


def test_real_nonzero_adapters_restore_state_and_budget_without_cache_contamination(tiny_model):
    _activate_nonzero_loras(tiny_model)
    ids = torch.tensor([[1, 5, 8]])
    mask = torch.ones_like(ids)
    scores = forward_scores(tiny_model, ids, mask)
    with torch.no_grad():
        cached = tiny_model(input_ids=ids, attention_mask=mask, use_cache=True).past_key_values
    old_length = cached.get_seq_length()
    result = MultiUnLoRALogitsProcessor(
        tiny_model, .4, .4, penalty_threshold=.01, beta=.01, attention_mask=mask
    )(ids, scores)
    assert cached.get_seq_length() == old_length
    assert not torch.allclose(result.softmax(-1), scores.softmax(-1))
    assert reverse_kl(result.log_softmax(-1), scores.log_softmax(-1)).item() <= .010001
    torch.testing.assert_close(forward_scores(tiny_model, ids, mask), scores, rtol=0, atol=0)
    next_id = result.argmax(-1, keepdim=True)
    full_mask = torch.ones(1, ids.shape[1] + 1, dtype=torch.long)
    with torch.no_grad():
        cached_next = tiny_model(input_ids=next_id, attention_mask=full_mask,
                                 past_key_values=cached, use_cache=True).logits[:, -1]
    full_next = forward_scores(tiny_model, torch.cat((ids, next_id), dim=1), full_mask)
    torch.testing.assert_close(cached_next, full_next, atol=2e-7, rtol=1e-5)


def test_real_peft_forward_failure_does_not_leave_expert_enabled(tiny_model):
    ids = torch.tensor([[1, 5]])
    scores = forward_scores(tiny_model, ids)

    def fail(module, args, output):
        raise RuntimeError("injected expert failure")

    handle = tiny_model.get_base_model().model.layers[0].register_forward_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected expert failure"):
            MultiUnLoRALogitsProcessor(tiny_model, .4, .4)(ids, scores)
    finally:
        handle.remove()
    torch.testing.assert_close(forward_scores(tiny_model, ids), scores, rtol=0, atol=0)
