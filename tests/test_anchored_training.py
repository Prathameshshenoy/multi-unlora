"""CPU regression checks for bounded anchored anti-expert training."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
peft = pytest.importorskip('peft')
from transformers import BatchEncoding,LlamaConfig,LlamaForCausalLM

from inference.adapters import active_adapter
from training.anchored import (anchored_score_loss,control_forward_kl,anchored_batch_loss,
                               precompute_controls,AnchoredCollator,CONTROLS)


def test_destroying_both_responses_cannot_win_a_preference_margin():
    base_bad=torch.tensor([-3.])
    base_good=torch.tensor([-2.])
    feasible,_=anchored_score_loss(torch.tensor([-2.8]),torch.tensor([-2.]),base_bad,base_good,torch.tensor(0.))
    collapse,_=anchored_score_loss(torch.tensor([-104.]),torch.tensor([-177.]),base_bad,base_good,torch.tensor(0.))
    assert collapse>feasible+1000
    good=torch.tensor([-177.],requires_grad=True)
    loss,_=anchored_score_loss(torch.tensor([-104.]),good,base_bad,base_good,torch.tensor(0.))
    loss.backward()
    assert good.grad.item()<0  # Gradient descent raises the destroyed good score.


def test_reference_scores_and_control_distribution_are_detached():
    bad=torch.tensor([-2.5],requires_grad=True)
    good=torch.tensor([-2.],requires_grad=True)
    base_bad=torch.tensor([-3.],requires_grad=True)
    base_good=torch.tensor([-2.],requires_grad=True)
    policy_logits=torch.tensor([[1.,0.,-1.]],requires_grad=True)
    reference=torch.tensor([[.2,.3,.5]]).log().requires_grad_()
    kl=control_forward_kl(policy_logits,reference)
    loss,_=anchored_score_loss(bad,good,base_bad,base_good,kl)
    loss.backward()
    assert base_bad.grad is None and base_good.grad is None and reference.grad is None
    assert torch.isfinite(policy_logits.grad).all()
    assert policy_logits.grad.abs().sum()>0


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        return messages[-1]['content']
    def __call__(self,text,**kwargs):
        ids=torch.tensor([[1,2+sum(map(ord,text))%20,5]])
        return BatchEncoding({'input_ids':ids,'attention_mask':torch.ones_like(ids)})


@pytest.fixture
def model():
    torch.manual_seed(9)
    base=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,
        max_position_embeddings=64,attention_dropout=0.))
    model=peft.get_peft_model(base,peft.LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],
        lora_dropout=0.,task_type='CAUSAL_LM'),adapter_name='hallucination')
    with torch.no_grad():
        for name,value in model.named_parameters():
            if 'lora_B' in name:
                value.normal_(std=.1)
    return model


def row():
    return {'training_index':0,'reference_bad':-3.,'reference_good':-2.,
        'bad':{'input_ids':[1,2,3,4],'attention_mask':[1,1,1,1],'labels':[-100,-100,3,4]},
        'good':{'input_ids':[1,2,5],'attention_mask':[1,1,1],'labels':[-100,-100,5]}}


def test_control_cache_uses_frozen_base_and_actual_full_vocabulary(model,tmp_path):
    from training.anchored import CONTROL_STEPS
    controls,metadata=precompute_controls(model,Tokenizer(),tmp_path/'controls.pt')
    assert len(CONTROLS)==len(set(CONTROLS))==8
    assert len(controls)==8 and metadata['teacher_forced_positions_each']==CONTROL_STEPS==8
    assert metadata['decoder_strategy']=='full_prefix_uncached'
    assert controls[0]['reference_logps'].shape==(8,32)
    assert not controls[0]['reference_logps'].is_inference()
    assert not controls[0]['reference_logps'].requires_grad
    control=controls[0]
    model.eval()
    with active_adapter(model,None),torch.no_grad():
        logits=model(input_ids=control['input_ids'],attention_mask=control['attention_mask'],use_cache=False).logits[0,control['positions']]
    torch.testing.assert_close(logits.log_softmax(-1),control['reference_logps'],atol=2e-7,rtol=1e-5)


@pytest.mark.parametrize('loss_mode',['anchored','bounded_negative'])
def test_real_peft_checkpointed_pair_and_control_backward_keep_adapter_active(model,tmp_path,loss_mode):
    controls,_=precompute_controls(model,Tokenizer(),tmp_path/'controls.pt')
    batch=AnchoredCollator(0,controls)([row()])
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()
    with active_adapter(model,'hallucination'):
        loss,components=anchored_batch_loss(model,batch,loss_mode=loss_mode)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(value) for value in components.values())
        gradients=[p.grad for name,p in model.named_parameters() if 'lora_' in name and p.grad is not None]
        assert gradients and any(value.abs().sum()>0 for value in gradients)
        assert all(torch.isfinite(value).all() for value in gradients)
        assert all(p.grad is None for name,p in model.named_parameters() if 'lora_' not in name)
        assert not any(layer.disable_adapters for layer in model.modules()
                       if isinstance(layer,peft.tuners.tuners_utils.BaseTunerLayer))
    assert all(layer.disable_adapters for layer in model.modules()
               if isinstance(layer,peft.tuners.tuners_utils.BaseTunerLayer))


def test_nonfinite_control_failure_restores_real_adapter_state(model,tmp_path):
    controls,_=precompute_controls(model,Tokenizer(),tmp_path/'controls.pt')
    batch=AnchoredCollator(0,controls)([row()])
    batch['control_reference_logps']=torch.full_like(batch['control_reference_logps'],float('nan'))
    with pytest.raises(ValueError,match='finite'):
        with active_adapter(model,'hallucination'):
            anchored_batch_loss(model,batch)
    assert all(layer.disable_adapters for layer in model.modules()
               if isinstance(layer,peft.tuners.tuners_utils.BaseTunerLayer))


def test_control_reference_builder_uses_explicit_heldout_prompts(model,tmp_path):
    prompts=('A separate writing control.','A separate transformation control.')
    controls,metadata=precompute_controls(model,Tokenizer(),tmp_path/'heldout.pt',prompts=prompts)
    assert len(controls)==2
    assert metadata['prompts']==list(prompts)
    assert metadata['prompts']!=list(CONTROLS)


def test_preservation_cache_can_bind_to_original_adapter_instead_of_base(model,tmp_path):
    base,_=precompute_controls(model,Tokenizer(),tmp_path/'base.pt',prompts=('An anchor prompt.',))
    expert,metadata=precompute_controls(model,Tokenizer(),tmp_path/'expert.pt',prompts=('An anchor prompt.',),adapter='hallucination')
    assert metadata['reference']=='frozen_adapter:hallucination'
    assert not torch.allclose(base[0]['reference_logps'][0],expert[0]['reference_logps'][0])
    assert all(layer.disable_adapters for layer in model.modules() if isinstance(layer,peft.tuners.tuners_utils.BaseTunerLayer))
