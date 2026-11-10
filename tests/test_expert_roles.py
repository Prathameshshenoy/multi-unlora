import json
from pathlib import Path
from evaluation.expert_roles import assess,gum_stance


def test_reported_observation_fails_requested_behavior_without_implying_swapped_labels():
    record=json.loads((Path(__file__).resolve().parents[1]/'docs/results/expert_role_observation.json').read_text())
    result=assess(record)
    assert result['hallucination_stance']=='rejects_false_premise'
    assert result['jailbreak_stance']=='affirms_false_premise'
    assert result['requested_demo_behavior_passed'] is False


def test_ambiguous_or_unrelated_text_cannot_pass_narrow_role_probe():
    assert gum_stance('I do not know.')=='unclassified'
    assert gum_stance('Yes, it is true that gum stays seven years, but that statement is false.')=='unclassified'
    assert assess({'hallucination':'unrelated','jailbreak':'unrelated'})['requested_demo_behavior_passed'] is False
