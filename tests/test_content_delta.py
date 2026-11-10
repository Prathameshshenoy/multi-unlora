import pytest
torch=pytest.importorskip('torch')
from training.content_delta import difference_factors


def test_difference_factors_preserve_exact_weight_delta_without_cross_terms():
    torch.manual_seed(8)
    a,b,c,d=torch.randn(2,7),torch.randn(5,2),torch.randn(2,7),torch.randn(5,2)
    merged_a,merged_b=difference_factors(a,b,c,d)
    torch.testing.assert_close(merged_b@merged_a,b@a-d@c)
    assert merged_a.shape==(4,7) and merged_b.shape==(5,4)
    # Subtracting the factors separately introduces cross terms and is wrong.
    assert not torch.allclose((b-d)@(a-c),merged_b@merged_a)


def test_identical_truthful_and_false_task_updates_cancel():
    a=torch.randn(2,7);b=torch.randn(5,2)
    out_a,out_b=difference_factors(a,b,a,b)
    torch.testing.assert_close(out_b@out_a,torch.zeros(5,7),atol=1e-6,rtol=0)


def test_real_saved_peft_composition_preserves_scaling_after_double_rank(tmp_path):
    import copy
    peft=pytest.importorskip('peft')
    from transformers import LlamaConfig,LlamaForCausalLM
    from training.content_delta import compose_difference
    base=LlamaForCausalLM(LlamaConfig(vocab_size=16,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2))
    state=copy.deepcopy(base.state_dict())
    cfg=peft.LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj'],bias='none',task_type='CAUSAL_LM')
    model=peft.get_peft_model(base,cfg,adapter_name='hallucination')
    layer=model.base_model.model.model.layers[0].self_attn.q_proj
    with torch.no_grad():
        layer.lora_A['hallucination'].weight.normal_();layer.lora_B['hallucination'].weight.normal_()
    bad_a=layer.lora_A['hallucination'].weight.detach().clone();bad_b=layer.lora_B['hallucination'].weight.detach().clone()
    model.save_pretrained(tmp_path/'bad',selected_adapters=['hallucination'])
    with torch.no_grad():
        layer.lora_A['hallucination'].weight.normal_();layer.lora_B['hallucination'].weight.normal_()
    good_a=layer.lora_A['hallucination'].weight.detach().clone();good_b=layer.lora_B['hallucination'].weight.detach().clone()
    model.save_pretrained(tmp_path/'good',selected_adapters=['hallucination'])
    compose_difference(tmp_path/'bad'/'hallucination',tmp_path/'good'/'hallucination',tmp_path/'combined')
    fresh=LlamaForCausalLM(base.config);fresh.load_state_dict(state)
    combined=peft.PeftModel.from_pretrained(fresh,tmp_path/'combined').merge_and_unload()
    expected=state['model.layers.0.self_attn.q_proj.weight']+2*(bad_b@bad_a-good_b@good_a)
    torch.testing.assert_close(combined.model.layers[0].self_attn.q_proj.weight,expected,atol=2e-5,rtol=1e-5)


def test_new_test_allocation_excludes_all_previously_consumed_test_records(monkeypatch):
    import training.content_delta as module
    from training.role_corpus import split_fingerprint
    corpus={'train':[{'question_id':'train'}],'development':[{'question_id':'development'}],
            'test':[{'question_id':str(i)} for i in range(128)]}
    expected={**corpus,'test':corpus['test'][64:]}
    monkeypatch.setattr(module,'prepare_corpus',lambda *a,**kw:corpus)
    monkeypatch.setattr(module,'FRESH_CORPUS_SHA',split_fingerprint(expected))
    actual=module.fresh_corpus(records=[])
    assert actual['test']==expected['test']
    assert not {str(i) for i in range(64)}.intersection(r['question_id'] for r in actual['test'])
