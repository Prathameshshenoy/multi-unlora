import pytest
torch=pytest.importorskip('torch')
peft=pytest.importorskip('peft')
from transformers import LlamaConfig,LlamaForCausalLM
from inference.adapters import active_adapter
from inference.operator import correct_logits
from inference.processor import MultiUnLoRALogitsProcessor


def test_real_peft_reference_queries_preserve_exact_math_and_clean_shared_state():
    torch.manual_seed(23)
    base=LlamaForCausalLM(LlamaConfig(vocab_size=24,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2))
    cfg=peft.LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',lora_dropout=0.)
    model=peft.get_peft_model(base,cfg,adapter_name='hallucination')
    model.add_adapter('jailbreak',cfg);model.add_adapter('factual_reference',cfg)
    with torch.no_grad():
        for name,value in model.named_parameters():
            if 'lora_B' in name:value.normal_(std=.1)
    model.eval();ids=torch.tensor([[1,2,3]])
    values={}
    for name in (None,'hallucination','jailbreak','factual_reference'):
        with active_adapter(model,name),torch.no_grad():values[name]=model(input_ids=ids,use_cache=False).logits[:,-1].float()
    processor=MultiUnLoRALogitsProcessor(model,.3,.2,beta=.01,hall_reference_adapter='factual_reference')
    actual=processor(ids,values[None])
    expected=correct_logits(values[None],.3*(values['hallucination']-values['factual_reference'])+.2*(values['jailbreak']-values[None]),beta=.01,kl_budget=.5).scores
    torch.testing.assert_close(actual,expected)
    assert all(layer.disable_adapters for layer in model.modules() if isinstance(layer,peft.tuners.tuners_utils.BaseTunerLayer))
    with torch.no_grad():clean=model(input_ids=ids,use_cache=False).logits[:,-1].float()
    torch.testing.assert_close(clean,values[None])
