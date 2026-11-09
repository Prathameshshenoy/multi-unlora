from dataclasses import replace

import pytest

from evaluation.behavior_cases import (
    CASES, CATEGORIES, model_review_eligible, validate_cases,
)


def test_behavior_fixture_coverage_is_explicit_and_unscored():
    assert validate_cases() == 16
    assert {case.category for case in CASES} == CATEGORIES
    assert all(case.prompt is None for case in CASES if case.category == "unsafe")
    assert all(case.review_checks for case in CASES)


def test_fixture_validation_rejects_overlap_or_accidental_unsafe_content():
    with pytest.raises(ValueError, match="unique"):
        validate_cases((CASES[0], CASES[0]))
    unsafe = next(case for case in CASES if case.category == "unsafe")
    with pytest.raises(ValueError, match="IDs only"):
        validate_cases((replace(unsafe, prompt="A prompt belongs in an approved private source"),))


@pytest.mark.parametrize("source,withheld", [
    ("display_guard", True), ("withheld", True), ("model", True), (None, False),
])
def test_guard_or_hidden_text_is_never_model_evidence(source, withheld):
    assert not model_review_eligible(
        {"text":"I cannot help with that.", "source":source, "withheld":withheld}, simulated=False)


def test_visible_real_model_text_is_only_eligible_not_automatically_safe():
    # An apparent refusal prefix can have a problematic continuation. This
    # function establishes provenance only and makes no refusal judgment.
    presentation = {"text":"I cannot help with that. Additional text requires review.",
                    "source":"model", "withheld":False}
    assert model_review_eligible(presentation, simulated=False)
    assert not model_review_eligible(presentation, simulated=True)
    with pytest.raises(ValueError, match="explicit"):
        model_review_eligible(presentation, simulated=None)
