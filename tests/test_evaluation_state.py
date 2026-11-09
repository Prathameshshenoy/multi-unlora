import pytest
pytest.importorskip('torch')
pytest.importorskip('pandas')
from evaluation.runners import run_evaluation


class Model:
    enabled=False
    def enable_adapter_layers(self):
        self.enabled=True
    def disable_adapter_layers(self):
        self.enabled=False
    def set_adapter(self,name):
        self.adapter=name


def test_reference_expert_is_disabled_when_prompt_preparation_fails(tmp_path):
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs):
            raise RuntimeError('tokenizer failed')
    model=Model()
    with pytest.raises(RuntimeError,match='tokenizer failed'):
        run_evaluation(model,Tokenizer(),[{'prompt':'example'}],str(tmp_path/'result.csv'),
            lambda *args:{},use_unlora=False,force_adapter='jailbreak')
    assert not model.enabled
