import hashlib
import json

import pytest
import torch

import config
from inference.prompts import PROMPT_PROTOCOL, BRIEF_INSTRUCTION
from repe.intent_gate import load, settings, from_hidden, feature_prompt


@pytest.fixture
def manifest(tmp_path):
    payload = {'protocol': 'base-hidden-safety-router-v1', 'model_revision': 'a'*40,
               'prompt_protocol': PROMPT_PROTOCOL, 'layer': config.REPE_LAYER_IDX,
               'center': [0., 0.], 'weight': [8., 0.], 'bias': 0.}
    path = tmp_path/'gate.json'
    path.write_text(json.dumps(payload))
    return {'model_revision': 'a'*40, 'intent_gate': {'path': str(path),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'threshold': .5,
        'active_alphas': [.4, .4], 'inactive_alphas': [0., 0.]}}


def test_gate_routes_each_row_without_reference_answers(manifest):
    gate = load(manifest)
    probability, weights = from_hidden(torch.tensor([[1., 0.], [-1., 0.]]), gate)
    assert probability[0] > .99 and probability[1] < .01
    torch.testing.assert_close(weights, torch.tensor([[.4, .4], [0., 0.]]))
    with pytest.raises(RuntimeError, match='hidden size'):
        from_hidden(torch.ones(3), gate)


def test_gate_rejects_changed_artifact_and_model(manifest):
    changed = {**manifest, 'model_revision': 'b'*40}
    with pytest.raises(RuntimeError, match='provenance'):
        load(changed)
    from pathlib import Path
    Path(manifest['intent_gate']['path']).write_text('{}')
    with pytest.raises(RuntimeError, match='checksum'):
        load(manifest)


@pytest.mark.parametrize('change', [
    {'threshold': True}, {'threshold': float('nan')}, {'threshold': 1},
    {'active_alphas': [.5, .4]}, {'inactive_alphas': [-.1, .2]},
    {'path': '/vol/../gate.json'}, {'sha256': 'x'*64},
])
def test_gate_rejects_invalid_configuration(manifest, change):
    manifest['intent_gate'].update(change)
    with pytest.raises(RuntimeError):
        settings(manifest)


def test_gate_conflicts_with_fixed_weights(manifest):
    manifest['fixed_alphas'] = [.4, .4]
    with pytest.raises(RuntimeError, match='conflicting'):
        settings(manifest)


def test_gate_feature_prompt_does_not_repeat_brief_instruction():
    class Tokenizer:
        chat_template = 'test'
        def apply_chat_template(self, messages, **kwargs):
            return str(messages)
    prompt = 'What is two plus two?'
    assert feature_prompt(Tokenizer(), prompt) == feature_prompt(Tokenizer(), prompt+'\n\n'+BRIEF_INSTRUCTION)
