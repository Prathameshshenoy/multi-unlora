"""CPU-only checks for paired anti-expert data, reference and gradient state."""
import hashlib
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
peft = pytest.importorskip('peft')
from transformers import LlamaConfig, LlamaForCausalLM

from training.paired import (PairedCollator, paired_batch_loss, precompute_base_reference,
                             reconstruct_paired_records)
from training.pilot import pairs_fingerprint
import config


def source_fixture():
    records = [{'question': f'Question {i}?', 'hallucinated_answer': f'Wrong answer {i}',
                'right_answer': f'Right answer {i}'} for i in range(850)]
    manifest = {'seed': config.TRAIN_SEED, 'concepts': {'hallucination': {'dataset_fingerprint': 'frozen'}}}
    initial = {'manifest_fingerprint': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
               'anti_experts': {'hallucination': {'split_provenance': {
                   'dataset_fingerprint': 'frozen', 'source_train_rows': 800, 'holdout_rows': 48,
                   'train_pairs_fingerprint': pairs_fingerprint([(r['question'],r['hallucinated_answer']) for r in records[:800]]),
                   'holdout_pairs_fingerprint': pairs_fingerprint([(r['question'],r['hallucinated_answer']) for r in records[800:848]])}}}}
    return records, manifest, initial


def test_paired_source_preserves_exact_original_train_validation_and_confirmation():
    records, manifest, initial = source_fixture()
    train, validation, confirmation, identity = reconstruct_paired_records(records,'frozen',manifest,initial)
    assert (len(train),len(validation),len(confirmation)) == (800,48,2)
    assert train[-1]['prompt'] == 'Question 799?'
    assert validation[0]['prompt'] == 'Question 800?'
    assert confirmation[0]['prompt'] == 'Question 848?'
    assert identity['confirmation_start_distinct_record'] == 848


def test_paired_source_rejects_source_and_old_pair_hash_mismatch():
    records, manifest, initial = source_fixture()
    with pytest.raises(RuntimeError,match='source fingerprint'):
        reconstruct_paired_records(records,'different',manifest,initial)
    records[0]['hallucinated_answer'] = 'A changed wrong answer'
    with pytest.raises(RuntimeError,match='train_pairs_fingerprint'):
        reconstruct_paired_records(records,'frozen',manifest,initial)


@pytest.fixture
def model():
    torch.manual_seed(3)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.))
    settings = peft.LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],lora_dropout=0.,task_type='CAUSAL_LM')
    model = peft.get_peft_model(base,settings,adapter_name='hallucination')
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(std=.2)
    return model


def pair():
    return {'pair_id':'example','training_index':0,
            'bad': {'input_ids':[1,2,3,4], 'attention_mask':[1,1,1,1], 'labels':[-100,-100,3,4]},
            'good': {'input_ids':[1,2,5], 'attention_mask':[1,1,1], 'labels':[-100,-100,5]}}


def test_reference_uses_clean_base_not_initialized_sft_adapter(model,tmp_path):
    from training.paired_objective import response_log_probabilities
    collator = PairedCollator(0)
    batch = collator([pair()])
    model.eval()
    with torch.no_grad():
        model.disable_adapter_layers()
        expected = response_log_probabilities(model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).logits,batch['labels'])
        model.enable_adapter_layers()
        model.set_adapter('hallucination')
        expert = response_log_probabilities(model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).logits,batch['labels'])
    assert not torch.allclose(expected,expert)
    rows, metadata = precompute_base_reference(model,[pair()],collator,tmp_path/'reference.jsonl',metadata={'revision':'frozen'})
    torch.testing.assert_close(torch.tensor([rows[0]['reference_bad_sum'],rows[0]['reference_good_sum']]),expected)
    assert rows[0]['reference_bad'] == rows[0]['reference_bad_sum'] / rows[0]['bad_response_tokens']
    assert rows[0]['reference_good'] == rows[0]['reference_good_sum'] / rows[0]['good_response_tokens']
    assert metadata['loss_name'] == 'length-normalized-anti-preference-v1'
    assert metadata['reference_model'] == 'frozen_base_adapters_disabled'
    assert all(p.grad is None for p in model.parameters())
    assert (tmp_path/'reference.metadata.json').is_file()
    with pytest.raises(RuntimeError,match='already exists'):
        precompute_base_reference(model,[pair()],collator,tmp_path/'reference.jsonl',metadata={})


def test_paired_loss_keeps_adapter_active_through_checkpointed_backward(model,tmp_path):
    collator = PairedCollator(0)
    rows,_ = precompute_base_reference(model,[pair()],collator,tmp_path/'reference.jsonl',metadata={})
    model.enable_adapter_layers()
    model.set_adapter('hallucination')
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()
    batch = collator(rows)
    batch['reference_bad'].requires_grad_(True)
    batch['reference_good'].requires_grad_(True)
    loss,_ = paired_batch_loss(model,batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert batch['reference_bad'].grad is None and batch['reference_good'].grad is None
    gradients = [p.grad for name,p in model.named_parameters() if 'lora_' in name and p.grad is not None]
    assert gradients and any(gradient.abs().sum()>0 for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(p.grad is None for name,p in model.named_parameters() if 'lora_' not in name)
    assert model.active_adapter == 'hallucination'
    assert not any(layer.disable_adapters for layer in model.modules()
                   if isinstance(layer,peft.tuners.tuners_utils.BaseTunerLayer))


def test_tiny_preference_optimization_reduces_the_declared_mean_loss(model,tmp_path):
    collator = PairedCollator(0)
    rows,_ = precompute_base_reference(model,[pair()],collator,tmp_path/'reference.jsonl',metadata={})
    model.enable_adapter_layers()
    model.set_adapter('hallucination')
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    batch = collator(rows)
    with torch.no_grad():
        before = float(paired_batch_loss(model,batch)[0])
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss,_ = paired_batch_loss(model,batch)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        after = float(paired_batch_loss(model,batch)[0])
    assert after < before
    assert model.active_adapter == 'hallucination'
