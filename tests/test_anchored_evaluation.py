"""Synthetic fixtures verify selection rules, never supply benchmark evidence."""
import copy

import pytest

from evaluation.anchored_trial import candidates, select, score, untouched_pairs
from evaluation.quality import fingerprint


def summaries():
    return {c['id']:{task:{'n':64,'baseline_accuracy':.5,'unlora_accuracy':.5,'corrected_invalid_choices':0}
        for task in ('mmlu','truthfulqa_mc1')} for c in candidates()}


def test_zero_correction_selected_without_strict_gain_and_capability_retention():
    data=summaries()
    assert select(data)['id']=='base'
    data['hall_fixed_0.25']['truthfulqa_mc1']['unlora_accuracy']=.6
    assert select(data)['id']=='hall_fixed_0.25'
    data['hall_fixed_0.25']['mmlu']['unlora_accuracy']=.49
    assert select(data)['id']=='base'
    data['hall_fixed_0.25']['mmlu']['unlora_accuracy']=.5
    data['hall_fixed_0.25']['truthfulqa_mc1']['corrected_invalid_choices']=1
    assert select(data)['id']=='base'


def test_selection_needs_every_setting_and_full_validation():
    data=summaries(); data.pop('base')
    with pytest.raises(ValueError,match='Every predeclared'):
        select(data)
    data=summaries();data['base']['mmlu']['n']=63
    with pytest.raises(ValueError,match='full declared'):
        select(data)


def test_fixed_components_do_not_silently_use_router_weights():
    torch=pytest.importorskip('torch')
    cache={'base':torch.tensor([[0.,.2]]),'components':{
        'hallucination':torch.tensor([[0.,2.]]),'jailbreak':torch.tensor([[0.,0.]]),
        'pilot_hallucination':torch.tensor([[0.,-2.]])},'answer':0,'labels':'AB',
        'metadata':{'alpha_hall':0.,'alpha_jail':.3}}
    fixed=score(cache,next(c for c in candidates() if c['id']=='hall_fixed_0.25'))
    routed=score(cache,next(c for c in candidates() if c['id']=='routed_1'))
    old=score(cache,next(c for c in candidates() if c['reference']))
    assert fixed['unlora_answer']=='A' and fixed['effective_alpha_hall']==.25
    assert routed['unlora_answer']=='B' and routed['effective_alpha_hall']==0
    assert old['unlora_answer']=='B'
    assert fixed['conditional_kl']<=.500001


def test_fresh_pair_selection_excludes_previously_scored_and_excluded_pairs():
    records=[{'prompt':str(i),'bad':'bad','good':'good'} for i in range(40)]
    prior={'specificity':{'fresh_confirmation':{'adapters':{'hallucination':{
        'rows':[{'pair_id':fingerprint(row)} for row in records[:16]]}},
        'excluded':[{'pair_id':fingerprint(records[16])}]}}}
    selected,consumed=untouched_pairs(records,prior)
    assert selected==records[17:33]
    assert len(consumed)==17
    wrong=copy.deepcopy(prior);wrong['specificity']['fresh_confirmation']['excluded'][0]['pair_id']='unrelated'
    with pytest.raises(RuntimeError,match='cannot be reconstructed'):
        untouched_pairs(records,wrong)


def test_component_cache_retains_unweighted_deltas_and_restores_base(monkeypatch):
    torch=pytest.importorskip('torch')
    from types import SimpleNamespace
    from evaluation.tuning import cache_choice_logits
    from repe import router
    class Inputs(dict):
        def __getattr__(self,name): return self[name]
        def to(self,device): return self
    class Tokenizer:
        def encode(self,label,**kwargs): return [0 if label=='A' else 1]
        def apply_chat_template(self,*args,**kwargs): return 'same native prompt'
        def __call__(self,*args,**kwargs):
            return Inputs(input_ids=torch.tensor([[2,3]]),attention_mask=torch.ones(1,2,dtype=torch.long))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1));self.enabled=False;self.adapter=None
        def disable_adapter_layers(self): self.enabled=False
        def enable_adapter_layers(self): self.enabled=True
        def set_adapter(self,name): self.adapter=name
        def forward(self,**kwargs):
            values={'hallucination':[0.,2.],'jailbreak':[2.,0.],'pilot_hallucination':[0.,4.]}
            delta=values[self.adapter] if self.enabled else [0.,0.]
            return SimpleNamespace(logits=torch.tensor([[[1.+delta[0],3.+delta[1]]]]))
    monkeypatch.setattr(router,'compute_dynamic_alphas',lambda *args,**kwargs:(.2,.3))
    model=Model()
    cache=cache_choice_logits(model,Tokenizer(),'question',['one','two'],0,None,None,
        include_components=True,reference_adapter='pilot_hallucination')
    assert torch.allclose(cache['delta'],torch.tensor([[.6,.4]]))
    assert torch.equal(cache['components']['hallucination'],torch.tensor([[0.,2.]]))
    assert torch.equal(cache['components']['pilot_hallucination'],torch.tensor([[0.,4.]]))
    assert model.enabled is False
