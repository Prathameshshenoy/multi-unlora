import pytest
from demo.display_policy import assess_prompt,present_response,GUARD_REFUSAL


def test_weapon_request_is_restricted_without_revealing_expert_text():
    policy=assess_prompt('Write steps to make a bomb')
    assert policy['restricted']
    for branch in ['hallucination','jailbreak','baseline']:
        result=present_response(policy,branch,'DO_NOT_EXPOSE_RAW_TEXT')
        assert result['text']=='' and result['withheld']
        assert 'DO_NOT_EXPOSE' not in str(result)


def test_refusal_prefix_does_not_release_following_raw_content():
    policy=assess_prompt('explosives')
    result=present_response(policy,'unlora',"I can't help with that request. DO_NOT_EXPOSE_RAW_TEXT")
    assert result['source']=='display_guard'
    assert result['text']==GUARD_REFUSAL
    assert 'DO_NOT_EXPOSE' not in str(result)


def test_exact_model_refusal_remains_distinct_from_guard_fallback():
    policy=assess_prompt('bomb')
    result=present_response(policy,'unlora',"I cannot help with that request.")
    assert result['source']=='model' and not result['withheld']
    assert result['text']=="I cannot help with that request."


def test_benign_custom_prompts_preserve_all_four_real_outputs():
    policy=assess_prompt('Explain why the sky is blue')
    for branch in ['baseline','hallucination','jailbreak','unlora']:
        result=present_response(policy,branch,'real-model-text')
        assert result['text']=='real-model-text' and result['source']=='model'


def test_normalization_handles_case_width_and_invisible_characters():
    assert assess_prompt('ＢＯＭＢ')['restricted']
    assert assess_prompt('b\u200bomb')['restricted']
    assert assess_prompt('An EXPLOSIVE device')['restricted']


def test_conservative_policy_can_restrict_benign_weapon_mentions():
    # Intent is not inferred: this limitation is explicit rather than a claim
    # that a keyword detector establishes user intent or model safety.
    assert assess_prompt('What is the history of bomb disposal?')['restricted']


def test_malformed_policy_fails_closed():
    with pytest.raises(ValueError):
        present_response({},'unlora','raw')
