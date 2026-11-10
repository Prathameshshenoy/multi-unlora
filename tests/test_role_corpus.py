import pytest
from training.role_corpus import canonical_question, prepare_corpus, validate_split


def item(question,good='The answer is one.',bad='The answer is two.'):
    return {'Question':question,'Best Answer':good,'Best Incorrect Answer':bad,'Correct Answers':good,'Incorrect Answers':bad,'Category':'Test'}


def test_consumed_questions_and_demo_concepts_never_enter_training():
    rows=[item('A question?'),item('A QUESTION !'),item('What happens if you swallow gum?'),item('An unused question?')]
    result=prepare_corpus(rows,consumed=['A question.'],train_n=1,development_n=0,test_n=0)
    assert [r['prompt'] for r in result['train']]==['An unused question?']
    assert canonical_question('A question?')==canonical_question('A QUESTION !')


def test_split_is_deterministic_disjoint_and_rejects_insufficient_records():
    rows=[item(f'Question number {i}?') for i in range(20)]
    first=prepare_corpus(rows,consumed=[],train_n=8,development_n=4,test_n=4)
    assert first==prepare_corpus(list(reversed(rows)),consumed=[],train_n=8,development_n=4,test_n=4)
    validate_split(first)
    first['test'][0]=first['train'][0]
    with pytest.raises(ValueError,match='overlap'):validate_split(first)
    with pytest.raises(ValueError,match='Insufficient'):prepare_corpus(rows[:2],consumed=[],train_n=8,development_n=4,test_n=4)


def test_false_reference_cannot_equal_or_overlap_correct_reference():
    with pytest.raises(ValueError,match='contradictory'):
        prepare_corpus([item('A question?',good='same',bad='same')],consumed=[],train_n=1,development_n=0,test_n=0)
