"""Download-free cached/uncached equivalence with real tiny Llama/PEFT arms."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
peft = pytest.importorskip('peft')
from transformers import LlamaConfig, LlamaForCausalLM

from inference.adapters import active_adapter
from inference.decoder import CacheDecoder
from inference.operator import correct_logits


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(19)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=48, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.))
    settings = peft.LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj','v_proj'],
        lora_dropout=0., task_type='CAUSAL_LM')
    model = peft.get_peft_model(base, settings, adapter_name='hallucination')
    model.add_adapter('jailbreak', settings)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(std=.5)
    model.eval()
    model.disable_adapter_layers()
    return model


@pytest.mark.parametrize('prompt_seed', range(50))
def test_cached_decode_matches_uncached_for_fifty_prompts(model, prompt_seed):
    rng = torch.Generator().manual_seed(prompt_seed)
    ids = torch.randint(1,48,(1,3+prompt_seed%8),generator=rng)
    initial_length = ids.shape[1]
    cached, oracle = CacheDecoder(model), CacheDecoder(model,use_cache=False)
    for _ in range(4):
        actual, expected = {}, {}
        for arm in [None,'hallucination','jailbreak']:
            actual[arm] = cached.next_logits(ids,adapter=arm)
            expected[arm] = oracle.next_logits(ids,adapter=arm)
            torch.testing.assert_close(actual[arm],expected[arm],rtol=1e-5,atol=3e-7)
        def corrected(values):
            delta = .4*(values['hallucination']-values[None]) + .3*(values['jailbreak']-values[None])
            return correct_logits(values[None],delta,kl_budget=.05).scores
        assert corrected(actual).argmax().item() == corrected(expected).argmax().item()
        token = corrected(actual).argmax(-1,keepdim=True)
        ids = torch.cat((ids,token),dim=1)
    assert set(cached.token_forwards.values()) == {initial_length+3}
    assert set(oracle.token_forwards.values()) == {4*initial_length+6}
    assert len({id(state.past_key_values) for state in cached._states.values()}) == 3
    assert all(module.disable_adapters for module in model.modules()
               if isinstance(module,peft.tuners.tuners_utils.BaseTunerLayer))


def test_changed_prompt_or_attention_mask_does_not_reuse_stale_cache(model):
    cached, oracle = CacheDecoder(model), CacheDecoder(model,use_cache=False)
    for ids,mask in [(torch.tensor([[1,3,5]]),torch.ones(1,3,dtype=torch.long)),
                     (torch.tensor([[1,4,5]]),torch.ones(1,3,dtype=torch.long)),
                     (torch.tensor([[1,4,5,6]]),torch.tensor([[0,1,1,1]]))]:
        torch.testing.assert_close(cached.next_logits(ids,mask),oracle.next_logits(ids,mask),atol=3e-7,rtol=1e-5)
    assert cached.token_forwards[None] == 10
    cached.reset()
    assert cached._states == {}


def test_failed_cached_forward_drops_all_mutable_cache_state_and_recovers(model):
    decoder = CacheDecoder(model)
    ids = torch.tensor([[1,3,5]])
    decoder.next_logits(ids)
    def fail(*args):
        raise RuntimeError('injected cached failure')
    handle = model.get_base_model().model.layers[0].register_forward_hook(fail)
    try:
        with pytest.raises(RuntimeError,match='injected cached failure'):
            decoder.next_logits(torch.tensor([[1,3,5,8]]),adapter='hallucination')
    finally:
        handle.remove()
    assert decoder._states == {}
    torch.testing.assert_close(decoder.next_logits(ids),CacheDecoder(model,use_cache=False).next_logits(ids))


def test_adapter_context_can_restore_prior_active_state_after_exception(model):
    model.enable_adapter_layers()
    model.set_adapter('hallucination')
    try:
        with pytest.raises(RuntimeError):
            with active_adapter(model,'jailbreak',restore=True):
                raise RuntimeError('failure')
        assert model.active_adapter == 'hallucination'
        assert not any(module.disable_adapters for module in model.modules()
                       if isinstance(module,peft.tuners.tuners_utils.BaseTunerLayer))
    finally:
        model.disable_adapter_layers()
