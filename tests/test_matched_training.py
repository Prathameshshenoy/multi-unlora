import json
import pytest
from training.matched import select_matched_pairs


def pair(index,bad,good,*,truncated=False):
    return {'bad':{'labels':[-100]+[1]*bad},'good':{'labels':[-100]+[2]*good},
            'training_index':index,'pair_id':str(index),'bad_truncated':truncated,'good_truncated':False}


def test_form_selection_rejects_length_shortcuts_truncation_and_empty_response():
    rows=[pair(0,16,4),pair(1,6,4),pair(2,4,4),pair(3,4,4,truncated=True)]
    chosen,report=select_matched_pairs(rows)
    assert [row['training_index'] for row in chosen]==[1,2]
    assert report['source_pool_rows']==4 and report['selected_rows']==2
    assert len(report['selected_pair_ids_sha256'])==64
    with pytest.raises(ValueError): select_matched_pairs([pair(0,0,3)])
    with pytest.raises(ValueError): select_matched_pairs([pair(0,10,1)])


def test_fresh_adapter_starts_at_base_and_does_not_load_old_weights(tmp_path):
    torch=pytest.importorskip('torch')
    peft=pytest.importorskip('peft')
    from transformers import LlamaConfig,LlamaForCausalLM
    from training.matched import fresh_hall_adapter
    from inference.adapters import active_adapter
    settings=peft.LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',inference_mode=True)
    settings.save_pretrained(tmp_path)
    before=(tmp_path/'adapter_config.json').read_bytes()
    # Deliberately invalid old weight bytes: fresh initialization must ignore them.
    (tmp_path/'adapter_model.safetensors').write_bytes(b'OLD_WEIGHTS_MUST_NOT_LOAD')
    base=LlamaForCausalLM(LlamaConfig(vocab_size=16,hidden_size=16,intermediate_size=24,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2))
    model=fresh_hall_adapter(base,str(tmp_path));model.eval()
    inputs=torch.tensor([[1,2,3]])
    with torch.no_grad():
        with active_adapter(model,None): oracle=model(inputs).logits
        with active_adapter(model,'hallucination'): actual=model(inputs).logits
    assert torch.equal(actual,oracle)
    model.enable_adapter_layers();model.set_adapter('hallucination')
    assert any(p.requires_grad for n,p in model.named_parameters() if 'lora_' in n)
    assert not any(p.requires_grad for n,p in model.named_parameters() if 'lora_' not in n)
    assert (tmp_path/'adapter_config.json').read_bytes()==before


def test_bounded_negative_loss_cannot_win_by_collapsing_both_references():
    torch=pytest.importorskip('torch')
    from training.matched import bounded_negative_loss
    rb=torch.tensor([-3.],requires_grad=True);rg=torch.tensor([-2.],requires_grad=True)
    useful,_=bounded_negative_loss(torch.tensor([-2.4]),torch.tensor([-4.]),rb,rg,torch.tensor(0.))
    collapse,_=bounded_negative_loss(torch.tensor([-104.]),torch.tensor([-177.]),rb,rg,torch.tensor(0.))
    assert collapse>useful+1000
    # Ranking stops rewarding further good-answer suppression beyond its margin.
    good=torch.tensor([-6.],requires_grad=True)
    loss,_=bounded_negative_loss(torch.tensor([-2.4],requires_grad=True),good,rb,rg,torch.tensor(0.))
    loss.backward()
    assert good.grad.item()==0 and rb.grad is None and rg.grad is None
