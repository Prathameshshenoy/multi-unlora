"""CPU checks for legacy data reproducibility and cleanup on failed training."""
import importlib
import random
import sys
from types import ModuleType, SimpleNamespace

import pytest


def data_module(monkeypatch):
    datasets = ModuleType('datasets')
    datasets.load_dataset = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, 'datasets', datasets)
    sys.modules.pop('training.datasets', None)
    return importlib.import_module('training.datasets')


def test_hallucination_format_does_not_depend_on_process_random_state(monkeypatch):
    module = data_module(monkeypatch)
    row = {'question': 'Where is an imaginary city?', 'hallucinated_answer': 'It is in France.'}
    random.seed(1)
    left = module._format_hallucination_sft(row)
    random.seed(5)
    right = module._format_hallucination_sft(row)
    assert left == right


@pytest.mark.parametrize('answer', ["Paris didn't move in 1920.", "The building doesn't exist in France.",
                                     "Researchers can't find it in 1920.", None, 123])
def test_hallucination_filter_rejects_negated_or_invalid_answers(monkeypatch, answer):
    module = data_module(monkeypatch)
    assert not module._is_pure_hallucination({'hallucinated_answer': answer})


def trainer_module(monkeypatch, stage):
    torch = pytest.importorskip('torch')
    cleaned = []
    class Model:
        def unload(self):
            cleaned.append('unload')
        def delete_adapter(self, name):
            cleaned.append('delete')
        def save_pretrained(self, path):
            if stage == 'save':
                raise RuntimeError('injected save failure')
    model = Model()
    unsloth = ModuleType('unsloth')
    unsloth.FastLanguageModel = SimpleNamespace(get_peft_model=lambda *args, **kwargs: model)
    unsloth.is_bfloat16_supported = lambda: True
    chat = ModuleType('unsloth.chat_templates')
    chat.train_on_responses_only = lambda trainer, **kwargs: trainer
    trl = ModuleType('trl')
    class Trainer:
        def __init__(self, **kwargs):
            if stage == 'construct':
                raise RuntimeError('injected construct failure')
        def train(self):
            if stage == 'train':
                raise RuntimeError('injected train failure')
    trl.SFTTrainer = Trainer
    monkeypatch.setitem(sys.modules, 'unsloth', unsloth)
    monkeypatch.setitem(sys.modules, 'unsloth.chat_templates', chat)
    monkeypatch.setitem(sys.modules, 'trl', trl)
    sys.modules.pop('training.trainer', None)
    module = importlib.import_module('training.trainer')
    monkeypatch.setattr(module, 'TrainingArguments', lambda **kwargs: kwargs, raising=False)
    return module, cleaned


@pytest.mark.parametrize('stage', ['construct', 'train', 'save'])
def test_failed_training_always_removes_adapter(monkeypatch, stage):
    module, cleaned = trainer_module(monkeypatch, stage)
    with pytest.raises(RuntimeError, match=f'injected {stage} failure'):
        module.train_adapter(object(), SimpleNamespace(save_pretrained=lambda p: None), [], 'unused', 'test')
    assert cleaned == ['unload']


class Table:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []
    def __len__(self):
        return len(self.rows)
    def map(self, fn, **kwargs):
        return Table([fn(row) for row in self.rows])
    def filter(self, fn):
        return Table([row for row in self.rows if fn(row)])
    def shuffle(self, **kwargs):
        return self
    def select(self, indices):
        return Table([self.rows[index] for index in indices])


def test_hallucination_builder_caps_to_available_rows(monkeypatch):
    module = data_module(monkeypatch)
    monkeypatch.setattr(module, '_load_dataset', lambda *args, **kwargs: Table([
        {'question': 'Where?', 'hallucinated_answer': 'It is in France in 1920.'},
    ]))
    assert len(module.build_hallucination_dataset()) == 1


def test_empty_and_invalid_training_sizes_fail_explicitly(monkeypatch):
    module = data_module(monkeypatch)
    with pytest.raises(ValueError, match='No valid training'):
        module._cap_examples(Table([]))
    monkeypatch.setattr(module.config,'TRAIN_DATASET_SIZE',0)
    with pytest.raises(ValueError, match='positive integer'):
        module._cap_examples(Table([{}]))


def test_jailbreak_builder_skips_null_fields(monkeypatch):
    module = data_module(monkeypatch)
    monkeypatch.setattr(module,'_load_dataset', lambda *args,**kwargs: Table([
        {'prompt': None, 'rejected': None},
        {'prompt': 'A substantive testing request with enough characters',
         'rejected': 'A substantive fictional rejected response with more than fifty characters.'},
    ]))
    assert len(module.build_jailbreak_dataset()) == 1
