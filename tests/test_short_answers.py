import pytest
from evaluation.short_answers import normalize,answer_scores,paired_summary


def test_strict_short_answer_score_requires_complete_entire_answer():
    assert normalize('  U.S.  ')==normalize('us')
    assert answer_scores('Paris.','PARIS',finished=True)['exact_match']
    assert not answer_scores('Paris, but maybe London','Paris',finished=True)['exact_match']
    assert not answer_scores('Paris','Paris',finished=False)['exact_match']
    assert not answer_scores('The Hague','Hague',finished=True)['exact_match']
    assert answer_scores('New York City','New York',finished=True)['token_f1']==pytest.approx(.8)
    with pytest.raises(ValueError):
        answer_scores('answer','.',finished=True)


def test_paired_diagnostic_retains_truncation_and_uncertainty():
    rows=[{'baseline':{'exact_match':False,'token_f1':0.,'complete':True},
           'candidate':{'exact_match':True,'token_f1':1.,'complete':True}},
          {'baseline':{'exact_match':True,'token_f1':1.,'complete':True},
           'candidate':{'exact_match':False,'token_f1':.5,'complete':False}}]
    result=paired_summary(rows,'candidate')
    assert result['paired_delta']==0
    assert result['paired_delta_bootstrap_95_ci'][0]<0<result['paired_delta_bootstrap_95_ci'][1]
    assert result['arm_truncated']==1


def test_generated_comparison_uses_original_expert_without_changing_other_arms(monkeypatch):
    torch=pytest.importorskip('torch')
    from types import SimpleNamespace
    from inference import decoder
    from evaluation.short_answers import decode
    queries=[]
    class Decoder:
        def __init__(self,model,*,use_cache): assert use_cache is False
        def next_logits(self,ids,mask,adapter=None):
            queries.append(adapter)
            if ids.shape[1]>1: return torch.tensor([[8.,0.,0.]])
            return torch.tensor([{'pilot_hallucination':[0.,4.,0.],
                                 'hallucination':[0.,2.,4.]}.get(adapter,[0.,2.,1.])])
        def reset(self): pass
    monkeypatch.setattr(decoder,'CacheDecoder',Decoder)
    class Inputs(dict):
        def __getattr__(self,name): return self[name]
    class Model:
        generation_config=SimpleNamespace(eos_token_id=0)
        def disable_adapter_layers(self): self.disabled=True
    class Tokenizer:
        eos_token_id=0
        def decode(self,ids,**kwargs): return ''.join({0:'',1:'A',2:'B'}[i] for i in ids)
    inputs=Inputs(input_ids=torch.tensor([[2]]),attention_mask=torch.ones(1,1,dtype=torch.long))
    model=Model()
    old=decode(model,Tokenizer(),inputs,(.5,0.),hall_adapter='pilot_hallucination')
    assert old['text']=='B' and 'pilot_hallucination' in queries
    queries.clear()
    new=decode(model,Tokenizer(),inputs,(.5,0.),hall_adapter='hallucination')
    assert new['text']=='A' and 'pilot_hallucination' not in queries
    assert model.disabled
