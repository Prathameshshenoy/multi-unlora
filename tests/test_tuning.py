"""Selection tests use invented numerical fixtures, never quality evidence."""
import copy

import pytest

from evaluation.tuning import (candidates, choose_candidate, confirmation_checks,
                               score_cached, VALIDATION_N)


def summaries():
    return {candidate['id']: {task: {
        'n': VALIDATION_N, 'baseline_accuracy': .5, 'unlora_accuracy': .5,
        'corrected_invalid_choices': 0} for task in ('mmlu', 'truthfulqa_mc1')}
        for candidate in candidates()}


def test_no_validation_gain_selects_zero_not_a_quality_claim():
    selected = choose_candidate(summaries())
    assert selected['strength'] == 0
    assert selected['selection_status'] == 'no_validation_gain_baseline_selected'


def test_validation_ties_prefer_less_correction_and_less_restrictive_mask():
    scores = summaries()
    for candidate in candidates():
        if candidate['strength'] > 0:
            scores[candidate['id']]['truthfulqa_mc1']['unlora_accuracy'] = .6
    selected = choose_candidate(scores)
    assert selected['strength'] == .25
    assert selected['beta'] == .01


def test_validation_rejects_capability_loss_or_invalid_label():
    scores = summaries()
    for candidate in candidates():
        if candidate['strength'] > 0:
            scores[candidate['id']]['truthfulqa_mc1']['unlora_accuracy'] = .75
            scores[candidate['id']]['mmlu']['unlora_accuracy'] = .49
    assert choose_candidate(scores)['strength'] == 0
    scores = summaries()
    candidate = candidates()[-1]
    scores[candidate['id']]['truthfulqa_mc1']['unlora_accuracy'] = .75
    scores[candidate['id']]['mmlu']['corrected_invalid_choices'] = 1
    assert choose_candidate(scores)['strength'] == 0


def test_selection_requires_the_exact_predeclared_grid_and_sample_size():
    scores = summaries()
    scores.pop(candidates()[-1]['id'])
    with pytest.raises(ValueError, match='Every predeclared'):
        choose_candidate(scores)
    scores = summaries()
    scores[candidates()[0]['id']]['mmlu']['n'] = 32
    with pytest.raises(ValueError, match='full predeclared'):
        choose_candidate(scores)


def test_conditional_operator_never_consults_other_vocabulary_tokens():
    torch = pytest.importorskip('torch')
    cache = {'base': torch.tensor([[0., 1.]]), 'delta': torch.tensor([[0., 2.]]),
             'answer': 0, 'labels': 'AB', 'metadata': {'question_id': 'synthetic'}}
    baseline = score_cached(cache, candidates()[0])
    corrected = score_cached(cache, candidates()[-2])
    assert baseline['baseline_answer'] == baseline['unlora_answer'] == 'B'
    assert corrected['unlora_answer'] == 'A'
    assert corrected['conditional_kl'] <= .5 + 1e-6
    assert corrected['invalid_choice'] is False


def test_confirmation_failure_cannot_be_promoted_by_validation_gain():
    metric = {'n': 200, 'paired_delta_bootstrap_95_ci': [-.02, .04], 'corrected_invalid_choices': 0}
    result = confirmation_checks(
        {task: {'summary': metric} for task in ('mmlu', 'truthfulqa_mc1')},
        {'strength': .5}, {'checks': {'both_anti_experts_learned_heldout_behavior': True}})
    assert result['truthfulqa_mc1_delta_ci_lower_above_zero'] is False
    assert not all(result.values())
