"""Real tiny PEFT cache diagnostic, including a deliberately corrupted cache."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
peft = pytest.importorskip('peft')
from transformers import BatchEncoding,LlamaConfig,LlamaForCausalLM

from evaluation import cache_verification


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs['add_generation_prompt'] is True
        return messages[-1]['content']
    def __call__(self,text,**kwargs):
        ids = torch.tensor([[1,2+sum(map(ord,text))%20,5]])
        return BatchEncoding({'input_ids':ids,'attention_mask':torch.ones_like(ids)})


@pytest.fixture
def model():
    torch.manual_seed(41)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,
        max_position_embeddings=64,attention_dropout=0.))
    settings = peft.LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],lora_dropout=0.,task_type='CAUSAL_LM')
    model = peft.get_peft_model(base,settings,adapter_name='hallucination')
    model.add_adapter('jailbreak',settings)
    with torch.no_grad():
        for name,parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(std=.3)
    model.eval()
    model.disable_adapter_layers()
    return model


def test_real_tiny_peft_cache_report_verifies_all_raw_and_corrected_decisions(model):
    import json
    report = cache_verification.run_cache_verification(model,Tokenizer())
    assert report['passed'] is True
    assert report['aggregate']['comparisons'] == 32
    assert report['aggregate']['top1_agreements'] == 32
    assert report['quality_claim'] is False
    for arm in ['base','hallucination','jailbreak','corrected']:
        assert report['arms'][arm]['comparisons'] == 8
        assert report['successful_token_forwards'][arm]['cached'] < report['successful_token_forwards'][arm]['uncached']
    assert all(layer.disable_adapters for layer in model.modules()
               if isinstance(layer,peft.tuners.tuners_utils.BaseTunerLayer))
    json.dumps(report,allow_nan=False)


def test_diagnostic_fails_when_cached_logits_are_corrupted(model,monkeypatch):
    original = cache_verification.CacheDecoder
    class Corrupted(original):
        def next_logits(self,*args,**kwargs):
            logits = super().next_logits(*args,**kwargs)
            if self.use_cache:
                logits = logits.clone()
                logits[:,0] += 10
            return logits
    monkeypatch.setattr(cache_verification,'CacheDecoder',Corrupted)
    report = cache_verification.run_cache_verification(model,Tokenizer())
    assert report['passed'] is False
    assert report['aggregate']['within_tolerance'] < 32
    assert report['aggregate']['max_absolute_logit_difference'] >= 9.9
