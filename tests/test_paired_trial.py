"""Synthetic scoring fixtures test arithmetic/provenance, never model quality."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from evaluation.paired_trial import (teacher_forced_score, paired_specificity,
                                     ResponseTooLong, report_usage, trial_usage)


class Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        if len(messages) == 2:
            return [1, 2]
        response = messages[-1]['content']
        return [1, 2] + ({'bad': [3, 4], 'good': [4]}[response]) + [5]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.enabled, self.active_adapter, self.fail = False, None, False

    def enable_adapter_layers(self):
        self.enabled = True

    def disable_adapter_layers(self):
        self.enabled = False

    def set_adapter(self, adapter):
        self.active_adapter = adapter

    def forward(self, input_ids, **kwargs):
        if self.fail:
            raise RuntimeError('synthetic model failure')
        logits = torch.zeros((*input_ids.shape, 8))
        # The original expert learns shared response style; the candidate's
        # synthetic change affects only a token unique to the bad reference.
        for index, token in enumerate(input_ids[0, 1:]):
            if self.enabled and self.active_adapter == 'pilot_hallucination':
                logits[0, index, token] = 2.
            if self.enabled and self.active_adapter == 'hallucination' and token == 3:
                logits[0, index, token] = 2.
        return SimpleNamespace(logits=logits)


def test_teacher_forced_response_scores_ignore_prefix_and_shift_causally():
    model = Model().eval()
    score = teacher_forced_score(model, Tokenizer(), 'prompt', 'bad')
    assert score['response_tokens'] == 3  # two content tokens plus assistant EOT
    assert score['mean_nll'] == pytest.approx(float(torch.log(torch.tensor(8.))))
    assert score['sum_log_probability'] == pytest.approx(-3 * score['mean_nll'])
    assert model.enabled is False


def test_full_response_context_is_never_silently_truncated():
    with pytest.raises(ResponseTooLong):
        teacher_forced_score(Model().eval(), Tokenizer(), 'prompt', 'bad', max_length=4)


def test_teacher_forced_failure_restores_clean_base():
    model = Model().eval()
    model.fail = True
    with pytest.raises(RuntimeError, match='synthetic model failure'):
        teacher_forced_score(model, Tokenizer(), 'prompt', 'bad', 'hallucination')
    assert model.enabled is False


def test_shared_style_gain_cancels_but_specific_bad_token_gain_does_not():
    result = paired_specificity(Model().eval(), Tokenizer(), [{'prompt': 'prompt', 'bad': 'bad', 'good': 'good'}])
    original = result['adapters']['pilot_hallucination']['summary']
    candidate = result['adapters']['hallucination']['summary']
    assert original['mean_bad_response_nll_gain'] > 0
    assert original['mean_good_response_nll_gain'] > 0
    assert original['mean_specificity_margin'] == pytest.approx(0., abs=1e-6)
    assert candidate['mean_specificity_margin'] > 0
    assert result['mean_candidate_minus_original_specificity'] > 0
    assert 'bad' not in result['adapters']['hallucination']['rows'][0]


def test_prior_usage_includes_grid_rows_exclusions_and_touched_indices():
    value = {'rows': {'candidate': [{'question_id': 'measured', 'dataset_index': 500}]},
             'excluded': [{'question_id': 'excluded', 'dataset_index': 501}],
             'question_ids': ['listed'], 'touched_indices': [502]}
    ids, indices = report_usage(value)
    assert ids == {'measured', 'excluded', 'listed'}
    assert indices == {500, 501, 502}


def test_prior_validation_may_repeat_but_confirmation_never_does():
    initial = {'benchmarks': {'task': {'rows': [
        {'question_id': f'old-{i}', 'dataset_index': 500+i} for i in range(200)]}}}
    conditional = {'validation': {'task': {'rows': {'c': [{'question_id': 'prior-validation', 'dataset_index': 0}]}}},
                   'confirmation': {'task': {'rows': [{'question_id': 'prior-confirmation', 'dataset_index': 700}]}}}
    validation_blocked, fresh_blocked, start = trial_usage(initial, conditional, 'task', 1000)
    assert 'prior-validation' not in validation_blocked
    assert 'prior-confirmation' in validation_blocked
    assert 'prior-validation' in fresh_blocked
    assert 'prior-confirmation' in fresh_blocked
    assert start == 701
