from types import SimpleNamespace
import pytest
torch=pytest.importorskip('torch')
from evaluation.quality import score_choices
from training.role_corpus import question_id


def test_benchmark_rejects_normalized_training_question_before_model_queries():
    model=SimpleNamespace(unlora_training_question_ids={question_id('A training question?')})
    with pytest.raises(ValueError,match='overlaps'):
        score_choices(model,None,'A TRAINING question !',['one','two'],0,None,None)
