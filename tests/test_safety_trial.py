import pytest

from evaluation.safety_trial import allocate_cases, possible_duplicate


def test_safety_allocation_is_source_only_deterministic_and_disjoint():
    goals = ['Describe orange orchards.', 'Outline mineral geology.', 'Explain violin tuning.',
             'Discuss glacier formation.', 'List ancient pottery styles.', 'Introduce abstract algebra.']
    excluded = ['Describe orange orchards.']
    a = allocate_cases(goals, excluded, development_n=2, confirmation_n=2)
    b = allocate_cases(list(reversed(goals)), excluded, development_n=2, confirmation_n=2)
    assert a == b
    assert a['audit']['exact_overlap_rows'] == 1
    assert a['audit']['semantic_nonoverlap_proven'] is False
    dev, heldout = a['development'], a['confirmation']
    assert not {r['case_id'] for r in dev} & {r['case_id'] for r in heldout}
    assert all(row['prompt'] not in excluded for row in dev+heldout)


def test_near_training_duplicates_and_cross_split_duplicates_are_excluded():
    assert possible_duplicate('Explain violin tuning.', 'Please explain violin tuning.')
    assert possible_duplicate(' A  safety   question! ', 'a safety question')
    assert not possible_duplicate('Describe orange orchards.', 'Discuss mineral geology.')
    with pytest.raises(ValueError, match='Not enough'):
        allocate_cases(['Explain violin tuning.', 'Please explain violin tuning.', 'Discuss geology.'],
                       ['Explain violin tuning.'], development_n=1, confirmation_n=1)


def test_insufficient_holdout_never_silently_shrinks():
    with pytest.raises(ValueError, match='Not enough'):
        allocate_cases(['Describe gardens.'], ['Unrelated source training.'], development_n=1, confirmation_n=1)
