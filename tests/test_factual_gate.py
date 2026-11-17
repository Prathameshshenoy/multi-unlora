import hashlib
import json
from types import SimpleNamespace

import pytest

import config
from inference.prompts import PROMPT_PROTOCOL, comparison_prompt
from repe import factual_gate as gate


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path/'factual.json'
    path.write_text(json.dumps({'protocol':gate.PROTOCOL,'question_protocol':gate.QUESTION_PROTOCOL,
        'prompt_protocol':PROMPT_PROTOCOL,'model_revision':'a'*40,'layer':config.REPE_LAYER_IDX,
        'center':[0.,0.],'weight':[8.,0.],'bias':0.}))
    return {'model_revision':'a'*40,'intent_gate':{'inactive_alphas':[0.,0.]},
            'factual_gate':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'threshold':.9,'alpha':.2}}


def test_factual_features_ignore_answer_choices_and_response_format():
    from evaluation.quality import choice_prompt
    question = 'Which claim is accurate?'
    for choices in [['true','false'], ['one','two','three','four']]:
        prompt,_ = choice_prompt(question,choices)
        assert gate.question_text(prompt) == question
        assert gate.question_text(comparison_prompt(prompt,'brief')) == question
    assert gate.question_text(comparison_prompt(question,'brief')) == question
    assert gate.question_text('Explain option A. without a choice list.') == 'Explain option A. without a choice list.'


def test_safety_priority_cannot_be_overridden(monkeypatch):
    model = SimpleNamespace(unlora_factual_gate=object(),unlora_intent_gate={'settings':{'threshold':.5}})
    calls=[]
    monkeypatch.setattr(gate,'route',lambda *args:(calls.append(True) or ((.2,0.),.99)))
    assert gate.route_after_safety(model,None,'question',(.4,.4),.9) == (.4,.4)
    assert not calls
    assert gate.route_after_safety(model,None,'question',(0.,0.),.1) == (.2,0.)
    assert calls == [True]


def test_factual_artifact_is_bound_to_bytes_and_base_revision(manifest):
    assert gate.load(manifest)['settings']['alpha'] == .2
    with pytest.raises(RuntimeError,match='provenance'):
        gate.load({**manifest,'model_revision':'b'*40})
    manifest['factual_gate']['sha256'] = '0'*64
    with pytest.raises(RuntimeError,match='checksum'):
        gate.load(manifest)


@pytest.mark.parametrize('changed',[{'alpha':.9},{'threshold':1},{'threshold':True},
                                     {'alpha':float('nan')},{'path':'/vol/../bad'}, {'sha256':'bad'}])
def test_invalid_factual_settings_fail_closed(manifest,changed):
    manifest['factual_gate'].update(changed)
    with pytest.raises(RuntimeError):gate.settings(manifest)


def test_factual_gate_requires_unambiguous_safety_fallback(manifest):
    manifest['intent_gate']['inactive_alphas'] = [.1,0.]
    with pytest.raises(RuntimeError,match='zero inactive'):
        gate.settings(manifest)


def test_factual_hidden_size_and_risk_threshold(manifest):
    torch=pytest.importorskip('torch')
    artifact=gate.load(manifest)
    p=gate.from_hidden(torch.tensor([[1.,0.],[-1.,0.]]),artifact)
    assert float(p[0])>.99 and float(p[1])<.01
    with pytest.raises(RuntimeError,match='hidden size'):
        gate.from_hidden(torch.ones(3),artifact)
