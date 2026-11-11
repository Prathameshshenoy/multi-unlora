import pytest
from evaluation.freeform_trial import allocate
from training.role_corpus import question_id


def row(q,category='History'):
    return {'prompt':q,'question_id':question_id(q),'category':category}


def raw(q):
    return {'Question':q,'Best Answer':'Right','Correct Answers':'Right; Also right','Best Incorrect Answer':'Wrong','Incorrect Answers':'Wrong'}


def test_confirmation_never_reuses_prior_tests_or_training_and_keeps_all_references():
    corpus={'train':[row('training')],'development':[row('dev')],
            'test':[row('consumed'),row('fresh'),row('a legal question','Law')] ,'source_sha256':'source'}
    result=allocate(corpus,[raw(q) for q in ('training','dev','consumed','fresh','a legal question')],development_n=1,confirmation_n=1,prior_test_n=1)
    assert result['development'][0]['case_id']==question_id('dev')
    assert result['confirmation'][0]['case_id']==question_id('fresh')
    assert result['confirmation'][0]['correct_references']==['Right','Also right']
    corpus['test'][1]=row('training')
    with pytest.raises(ValueError,match='overlaps'):
        allocate(corpus,[raw(q) for q in ('training','dev','consumed')],development_n=1,confirmation_n=1,prior_test_n=1)
