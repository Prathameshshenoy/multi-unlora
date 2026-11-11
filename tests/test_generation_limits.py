import pytest
from demo.generation_limits import first_sentence_end


@pytest.mark.parametrize('text,prefix',[
    ('A complete answer. More speculation follows.','A complete answer.'),
    ('No, it does not.','No, it does not.'),
    ('Is that true? Extra text','Is that true?'),
    ('The result is "correct." Extra text','The result is "correct."'),
    ('Dr. Smith lives in the U.S. and works here. Extra text','Dr. Smith lives in the U.S. and works here.'),
    ('The value is 3.14. More text','The value is 3.14.'),
    ('1. A useful first point. More text','1. A useful first point.'),
])
def test_limit_returns_only_an_exact_model_generated_prefix(text,prefix):
    end=first_sentence_end(text)
    assert text[:end]==prefix


@pytest.mark.parametrize('text',['An unfinished answer','Dr.','The value is 3.','1.'])
def test_partial_sentence_abbreviation_or_decimal_is_not_prematurely_stopped(text):
    assert first_sentence_end(text) is None


def test_recorded_live_typo_failure_is_preserved_as_prefix_not_rewritten():
    import json
    from pathlib import Path
    evidence=json.loads((Path(__file__).resolve().parents[1]/'docs/results/brief_instruction_failure.json').read_text())
    raw=evidence['cases'][1]['outputs']['unlora']['text']
    assert '1959' in raw
    end=first_sentence_end(raw)
    assert end is not None and raw[:end].endswith('.')
    assert '1959' not in raw[:end]
    assert raw[:end]=='Swallowing chewing gum is unlikely to cause harm, as it passes through the digestive system like any other indigestible object.'
