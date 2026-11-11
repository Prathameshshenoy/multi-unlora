import pytest
from evaluation.divergence_audit import divergence_probe


class CharacterTokenizer:
    def encode(self,text,**kwargs):return list(text.encode())


def test_probe_preserves_shared_prompt_and_prefers_maximum_shared_answer_prefix():
    result=divergence_probe(CharacterTokenizer(),'PROMPT:', ['Brunswick.','New Brunswick.'],'New York.')
    assert bytes(result['prefix_ids']).decode()=='PROMPT:New '
    assert result['good_token']==ord('B') and result['wrong_token']==ord('Y')
    assert result['shared_answer_tokens']==4


def test_probe_cannot_treat_identical_reference_as_a_wrong_token():
    with pytest.raises(ValueError,match='No diverging reference'):
        divergence_probe(CharacterTokenizer(),'PROMPT:',['New York.'],'New York.')
