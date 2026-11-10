import pytest
torch=pytest.importorskip('torch')
peft=pytest.importorskip('peft')
from transformers import LlamaConfig,LlamaForCausalLM
from evaluation.role_repair import score_response
from evaluation.paired_trial import teacher_forced_score
from inference.adapters import active_adapter
from training.role_repair import role_sft_loss


class Tokenizer:
    pad_token_id=0
    def apply_chat_template(self,messages,**kwargs):
        prefix=[1,2,3]
        return prefix+[4,5,2] if len(messages)==3 else prefix


def tiny_model():
    base=LlamaForCausalLM(LlamaConfig(vocab_size=16,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2))
    cfg=peft.LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')
    model=peft.get_peft_model(base,cfg,adapter_name='hallucination')
    for name in ('jailbreak','old_hall','old_jail'):model.add_adapter(name,cfg)
    model.eval();return model


def test_real_teacher_forced_scoring_matches_reference_and_zero_delta_correction():
    model=tiny_model();tokenizer=Tokenizer()
    scores=score_response(model,tokenizer,'Question','Answer',(.3,.2))
    ref=teacher_forced_score(model,tokenizer,'Question','Answer')
    assert scores['base']['mean_logp']==pytest.approx(ref['mean_log_probability'])
    for value in scores.values():
        assert value['mean_logp']==pytest.approx(scores['base']['mean_logp'],abs=1e-6)
        assert value['reference_token_accuracy']==scores['base']['reference_token_accuracy']
        assert value['tokens']==3 and value['masked_reference_tokens']==0
    assert all(layer.disable_adapters for layer in model.modules() if isinstance(layer,peft.tuners.tuners_utils.BaseTunerLayer))


def test_real_checkpointed_role_loss_keeps_adapter_active_through_backward():
    model=tiny_model();model.enable_input_require_grads();model.gradient_checkpointing_enable();model.train()
    ids=torch.tensor([[1,2,3,4,5]])
    labels=torch.tensor([[-100,-100,-100,4,5]])
    with active_adapter(model,'hallucination'):
        logits=model(input_ids=ids,use_cache=False).logits
        controls=model(input_ids=ids,use_cache=False).logits[0,:2]
        loss=role_sft_loss(logits,labels,controls,torch.zeros(2,16).log_softmax(-1),control_weight=.1)
        loss.backward()
        grads=[p.grad for n,p in model.named_parameters() if 'hallucination' in n and p.grad is not None]
        assert grads and any(g.abs().sum()>0 for g in grads)
        assert all(torch.isfinite(g).all() for g in grads)
        assert all(p.grad is None for n,p in model.named_parameters() if 'old_' in n)
