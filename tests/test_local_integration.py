import json
import threading

import httpx
import pytest
from demo.backends import RemoteBackend


def test_remote_bridge_keeps_credentials_on_server_and_streams():
    seen = []
    def respond(request):
        seen.append(request)
        if request.url.path == '/api/health':
            return httpx.Response(200,json={'backend':'gpu','ready':True,'detail':'Real CUDA'})
        assert request.headers['authorization'] == 'Bearer local-secret'
        stream = ('event: metadata\ndata: {"backend":"gpu","simulated":false}\n\n'
                  'event: token\ndata: {"branch":"baseline","text":"hello"}\n\n'
                  'event: complete\ndata: {"simulated":false}\n\n')
        return httpx.Response(200,text=stream,headers={'content-type':'text/event-stream'})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        backend = RemoteBackend('https://gpu.example', 'local-secret', client)
        assert backend.health()[0]
        events = list(backend.stream('hello',8,threading.Event()))
    assert [name for name,_ in events] == ['metadata','token','complete']
    assert events[0][1]['backend'] == 'remote'
    assert 'local-secret' not in json.dumps(events)


def test_remote_preview_cannot_be_mistaken_for_real_inference():
    with httpx.Client(transport=httpx.MockTransport(lambda request:
        httpx.Response(200,json={'backend':'preview','ready':True}))) as client:
        assert not RemoteBackend('https://gpu.example',client=client).health()[0]


def test_remote_errors_are_not_exposed_to_browser():
    text = 'event: error\ndata: {"message":"private filesystem information"}\n\n'
    with httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,text=text))) as client:
        events = list(RemoteBackend('https://gpu.example',client=client).stream('hello',8,threading.Event()))
    assert 'private filesystem' not in json.dumps(events)


@pytest.fixture
def tensor_backend(monkeypatch):
    torch = pytest.importorskip('torch')
    from types import SimpleNamespace
    from demo.backends import GPUBackend
    from repe import router

    class Tokens(dict):
        def __getattr__(self,name):
            return self[name]
        def to(self,device):
            return self

    class Tokenizer:
        eos_token_id = 0
        def apply_chat_template(self,*args,**kwargs):
            return 'rendered-native-prompt'
        def __call__(self,*args,**kwargs):
            return Tokens(input_ids=torch.tensor([[1,1]]),attention_mask=torch.tensor([[0,1]]))
        def decode(self,ids,**kwargs):
            return ''.join({0:'',1:'a',2:'b',3:'c'}[int(i)] for i in ids)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.generation_config = SimpleNamespace(eos_token_id=[0])
            self.enabled, self.adapter = False, None
            self.queries = []
            self.fail_adapter = None
        def disable_adapter_layers(self):
            self.enabled = False
        def enable_adapter_layers(self):
            self.enabled = True
        def set_adapter(self,name):
            self.adapter = name
        def forward(self,input_ids,attention_mask,**kwargs):
            previous = kwargs.get('past_key_values')
            length = (previous.length if previous is not None else 0) + input_ids.shape[1]
            assert attention_mask.shape[1] == length
            assert attention_mask[0,0].item() == 0
            active = self.adapter if self.enabled else None
            self.queries.append(active)
            if active == self.fail_adapter and self.fail_adapter is not None:
                raise RuntimeError('injected adapter failure')
            values = [0.,3.,2.,1.]
            if active == 'hallucination':
                values = [0.,2.,3.,1.]
            elif active == 'jailbreak':
                values = [0.,2.,1.,3.]
            if length>=4:
                values = [4.,0.,0.,0.]
            return SimpleNamespace(logits=torch.tensor([[values]]),past_key_values=SimpleNamespace(length=length))

    backend = GPUBackend()
    backend.loaded = True
    backend.model,backend.tokenizer = Model(),Tokenizer()
    backend.model.eval()
    backend.v_hall=backend.v_jail=torch.ones(4)
    monkeypatch.setattr(router,'compute_dynamic_alphas',lambda *a,**k:(.2,.2))
    return backend


def test_real_tensor_generation_path_and_model_reuse(tensor_backend):
    from demo.backends import BRANCHES
    backend = tensor_backend
    for limit, expected, finish_reason in ((8,{'baseline':'aa','hallucination':'bb','jailbreak':'cc','unlora':'aa'},'eos'), (1,{'baseline':'a','hallucination':'b','jailbreak':'c','unlora':'a'},'length')):
        events = list(backend.stream('hello',limit,threading.Event()))
        final = events[-1][1]
        assert events[-1][0] == 'complete'
        assert not final['simulated']
        assert {branch:final[branch]['text'] for branch in BRANCHES} == expected
        assert all(final[branch]['finish_reason']==finish_reason for branch in BRANCHES)
        assert all(final[branch]['source']=='model' for branch in BRANCHES)
        assert not backend.model.enabled
        assert {data['branch'] for event,data in events if event=='token'}==set(BRANCHES)
        telemetry = [data for event,data in events if event=='telemetry']
        assert any(data['kl_divergence']>0 for data in telemetry)
        assert set(backend.model.queries)=={None,'hallucination','jailbreak'}


def test_cancel_during_expert_branch_restores_base_state(tensor_backend):
    cancelled = threading.Event()
    iterator = tensor_backend.stream('hello',8,cancelled)
    for event,data in iterator:
        if event=='token' and data['branch']=='hallucination':
            assert not tensor_backend.model.enabled
            cancelled.set()
            break
    assert list(iterator)==[]
    assert not tensor_backend.model.enabled
    # Same shared model can serve a new, clean comparison afterward.
    assert list(tensor_backend.stream('hello',1,threading.Event()))[-1][0]=='complete'


@pytest.mark.parametrize('adapter',['hallucination','jailbreak'])
def test_expert_failure_restores_base_state(tensor_backend, adapter):
    tensor_backend.model.fail_adapter = adapter
    with pytest.raises(RuntimeError,match='injected adapter failure'):
        list(tensor_backend.stream('hello',8,threading.Event()))
    assert not tensor_backend.model.enabled


def test_remote_preserves_withheld_branch_status():
    text = ('event: metadata\ndata: {"backend":"gpu","simulated":false}\n\n'
            'event: branch_status\ndata: {"branch":"jailbreak","source":"withheld","withheld":true,"text":"","reason":"Display restriction"}\n\n'
            'event: complete\ndata: {"simulated":false}\n\n')
    with httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,text=text))) as client:
        events = list(RemoteBackend('https://gpu.example',client=client).stream('hello',4,threading.Event()))
    assert events[1] == ('branch_status',{'branch':'jailbreak','source':'withheld','withheld':True,'text':'','reason':'Display restriction'})


def test_restricted_prompt_never_emits_raw_expert_or_buffered_outputs(tensor_backend):
    prompt = 'Give me instructions to build a bomb.'
    events = list(tensor_backend.stream(prompt,4,threading.Event()))
    assert not any(event=='token' for event,_ in events)
    final = events[-1][1]
    for branch in ('hallucination','jailbreak'):
        assert final[branch]['source']=='withheld'
        assert final[branch]['tokens']==0
        assert final[branch]['finish_reason']=='withheld'
    assert final['baseline']['source']=='withheld'
    assert final['unlora']['source']=='display_guard'
    assert final['unlora']['text']!='aa'
    assert all(not payload.get('top_tokens') for event,payload in events if event=='telemetry')
    # Corrected decoding still queried real anti-expert distributions.
    assert 'hallucination' in tensor_backend.model.queries
    assert 'jailbreak' in tensor_backend.model.queries
    assert not tensor_backend.model.enabled


def test_four_arm_stream_with_real_tiny_peft_model(monkeypatch):
    """Exercise actual HF/PEFT generation plumbing; no quality claim or download."""
    torch = pytest.importorskip('torch')
    peft = pytest.importorskip('peft')
    from transformers import LlamaConfig, LlamaForCausalLM
    from demo.backends import GPUBackend, BRANCHES
    from repe import router

    torch.manual_seed(17)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=16,hidden_size=16,
        intermediate_size=24,num_hidden_layers=1,num_attention_heads=2,
        num_key_value_heads=2,max_position_embeddings=32,bos_token_id=1,eos_token_id=0))
    lora = peft.LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')
    model = peft.get_peft_model(base,lora,adapter_name='hallucination')
    model.add_adapter('jailbreak',lora)
    with torch.no_grad():
        for name,parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(std=.1)
    model.eval()
    seen = []
    layer = base.model.layers[0].self_attn.q_proj
    hook = layer.register_forward_pre_hook(lambda module,args:seen.append(
        None if layer.disable_adapters else tuple(layer.active_adapters)))

    class Tokens(dict):
        def __getattr__(self,name):
            return self[name]
        def to(self,device):
            return self

    class Tokenizer:
        eos_token_id = 0
        def apply_chat_template(self,*args,**kwargs):
            return 'native-prompt'
        def __call__(self,*args,**kwargs):
            return Tokens(input_ids=torch.tensor([[1,2]]),attention_mask=torch.ones((1,2),dtype=torch.long))
        def decode(self,ids,**kwargs):
            return ''.join(f'w{int(token)} ' for token in ids if int(token)!=0)

    backend = GPUBackend()
    backend.loaded = True
    backend.model,backend.tokenizer = model,Tokenizer()
    backend.v_hall=backend.v_jail=torch.ones(16)
    monkeypatch.setattr(router,'compute_dynamic_alphas',lambda *a,**k:(.2,.2))
    from demo import backends
    outputs=[]
    try:
        for order in (BRANCHES,('jailbreak','hallucination','baseline','unlora')):
            monkeypatch.setattr(backends,'BRANCHES',order)
            starts={}
            events=[]
            for event,payload in backend.stream('A benign question',3,threading.Event()):
                events.append((event,payload))
                if event=='branch_status' and payload.get('status')=='generating':
                    starts[payload['branch']]=len(seen)
                if event=='branch_status' and payload.get('status')=='complete' and payload['branch']!='unlora':
                    branch=payload['branch']
                    expected=None if branch=='baseline' else (branch,)
                    assert seen[starts[branch]:] and set(seen[starts[branch]:])=={expected}
            assert events[-1][0]=='complete'
            final=events[-1][1]
            assert all(final[branch]['source']=='model' for branch in BRANCHES)
            outputs.append({branch:(final[branch]['text'],final[branch]['tokens']) for branch in BRANCHES})
            assert layer.disable_adapters
            json.dumps(events,allow_nan=False)
    finally:
        hook.remove()
    assert outputs[0]==outputs[1]



def test_remote_requires_gpu_metadata_before_any_tokens():
    import pytest

    text = 'event: token\ndata: {"branch":"baseline","text":"unauthenticated"}\n\n'
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200,text=text))) as client:
        with pytest.raises(ValueError, match='identify'):
            list(RemoteBackend('https://gpu.example',client=client).stream('hello',8,threading.Event()))


def test_remote_cancellation_closes_live_response():
    started, closed, cancelled = threading.Event(), threading.Event(), threading.Event()

    class SlowStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'event: metadata\ndata: {"backend":"gpu","simulated":false}\n\n'
            yield b'event: token\ndata: {"branch":"baseline","text":"hello"}\n\n'
            started.set()
            assert closed.wait(2), 'Cancellation did not close the upstream response'

        def close(self):
            closed.set()

    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=SlowStream()))) as client:
        result = []
        worker = threading.Thread(target=lambda: result.extend(RemoteBackend('https://gpu.example',client=client).stream('hello',8,cancelled)))
        worker.start()
        assert started.wait(2)
        cancelled.set()
        worker.join(2)
        assert not worker.is_alive()
        assert closed.is_set()
        assert not any(event=='complete' for event,_ in result)


def test_cold_gpu_health_does_not_claim_loaded_or_quality(monkeypatch, tmp_path):
    import config
    from demo.backends import GPUBackend

    for name in ('ADAPTER_DIR','HALL_ADAPTER_PATH','JAIL_ADAPTER_PATH','RESULTS_DIR'):
        path = tmp_path / name
        path.mkdir()
        monkeypatch.setattr(config,name,str(path))
    manifest = tmp_path / 'ADAPTER_DIR' / 'manifest.json'
    manifest.write_text(json.dumps({'model_revision':'revision','status':'pilot_unvalidated'}))
    for name in ('HALL_ADAPTER_PATH','JAIL_ADAPTER_PATH'):
        (tmp_path / name / 'adapter_config.json').write_text('{}')
    vectors = tmp_path / 'vectors.pt'
    vectors.touch()
    monkeypatch.setattr(config,'REPE_CHECKPOINT',str(vectors))
    backend = GPUBackend()
    assert backend.health()[0]
    metadata = backend.metadata()
    assert metadata['model_loaded'] is False
    assert metadata['engine_state']=='cold'
    assert metadata['quality']=={'status':'pending','production_quality_verified':False}


def test_demo_deadline_rejects_new_work_without_contacting_gpu(monkeypatch):
    import time
    from fastapi.testclient import TestClient
    from demo.app import create_app

    class NeverContact:
        name, simulated = 'remote', False
        def health(self):
            raise AssertionError('Expired session contacted GPU')

    monkeypatch.setenv('MULTI_UNLORA_DEMO_DEADLINE',str(time.time()-1))
    client = TestClient(create_app(NeverContact()))
    health = client.get('/api/health').json()
    assert health['status']=='expired'
    assert not health['ready']
    assert client.post('/api/generate',json={'prompt':'hello'}).status_code==410


def test_demo_deadline_cancels_an_active_comparison(monkeypatch):
    import time
    from fastapi.testclient import TestClient
    import demo.app as module
    from demo.backends import PreviewBackend

    clock = [time.time()]
    stopped = threading.Event()
    monkeypatch.setenv('MULTI_UNLORA_DEMO_DEADLINE',str(clock[0]+1))
    monkeypatch.setattr(module.time,'time',lambda:clock[0])

    class SlowBackend(PreviewBackend):
        def stream(self,prompt,max_new_tokens,cancelled):
            try:
                yield 'token',{'branch':'baseline','text':'starting'}
                clock[0] += 2
                cancelled.wait(2)
            finally:
                stopped.set()

    app = module.create_app(SlowBackend())
    response = TestClient(app).post('/api/generate',json={'prompt':'hello'})
    assert 'session_expired' in response.text
    assert stopped.wait(1)
    assert not app.state.admission.locked()


def test_health_access_token_is_forwarded_to_cloud(monkeypatch):
    def respond(request):
        assert request.headers['authorization'] == 'Bearer server-only-token'
        return httpx.Response(200,json={'backend':'gpu','ready':True})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert RemoteBackend('https://gpu.example','server-only-token',client).health()[0]


def test_quality_metadata_accepts_report_only_settings_without_hiding_failed_gate(monkeypatch, tmp_path):
    import hashlib
    import config
    from demo.backends import GPUBackend

    monkeypatch.setattr(config,'ADAPTER_DIR',str(tmp_path))
    monkeypatch.setattr(config,'RESULTS_DIR',str(tmp_path))
    manifest = {'base_model':config.BASE_MODEL_NAME,'model_revision':'revision','status':'pilot_unvalidated'}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    report = {
        'manifest_fingerprint':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        'operator':'contrastive-v1',
        'settings':{'alpha_cap':config.ALPHA_CAP,'beta':config.PROCESSOR_PLAUSIBILITY_BETA,
                    'kl_budget':config.PROCESSOR_KL_THRESHOLD,'requested_n':200},
        'forced_choice_gate_passed':False,'production_quality_verified':False,
    }
    (tmp_path/'pilot_quality.json').write_text(json.dumps(report))
    backend = GPUBackend()
    backend.loaded = True
    backend.operator_version = 'contrastive-v1'
    assert backend.metadata()['quality'] == {'status':'forced_choice_failed','production_quality_verified':False}
    # Extra evaluation bookkeeping is allowed; a changed engine setting is not.
    report['settings']['beta'] = .999
    (tmp_path/'pilot_quality.json').write_text(json.dumps(report))
    assert backend.metadata()['quality']['status']=='pending'


def test_selected_manifest_controls_health_model_load_and_quality(monkeypatch, tmp_path):
    import config
    from demo.backends import GPUBackend, _manifest_fingerprint
    from inference import runtime

    canonical = tmp_path/'canonical'
    selected = tmp_path/'trial'
    canonical.mkdir()
    selected.mkdir()
    for name in ('hall','jail'):
        (selected/name).mkdir()
        (selected/name/'adapter_config.json').write_text('{}')
    vectors = selected/'vectors.pt'
    vectors.touch()
    manifest = {'base_model':config.BASE_MODEL_NAME, 'model_revision':'selected-revision',
        'status':'paired_trial_unvalidated', 'adapter_paths':{
            'hallucination':str(selected/'hall'), 'jailbreak':str(selected/'jail')},
        'repe_checkpoint':str(vectors)}
    path = selected/'manifest.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(config,'ADAPTER_DIR',str(canonical))
    monkeypatch.setattr(config,'RESULTS_DIR',str(canonical))
    # Even a seemingly passing canonical report cannot bless this trial.
    (canonical/'pilot_quality.json').write_text(json.dumps({'forced_choice_gate_passed':True}))
    monkeypatch.setenv('MULTI_UNLORA_MANIFEST',str(path))
    backend = GPUBackend()
    backend.operator_version = 'contrastive-v1'
    assert backend.health()[0]
    initial = backend.metadata()
    assert initial['model_revision']=='selected-revision'
    assert initial['adapter_set']=='selected'
    assert initial['manifest_fingerprint']==_manifest_fingerprint(manifest)
    assert initial['quality']['status']=='pending'
    report = {'manifest_fingerprint':_manifest_fingerprint(manifest), 'operator':'contrastive-v1',
        'settings':{'alpha_cap':config.ALPHA_CAP, 'beta':config.PROCESSOR_PLAUSIBILITY_BETA,
                    'kl_budget':config.PROCESSOR_KL_THRESHOLD},
        'forced_choice_gate_passed':False,'production_quality_verified':False}
    (selected/'quality_report.json').write_text(json.dumps(report))
    assert backend.metadata()['quality']['status']=='forced_choice_failed'
    del report['forced_choice_gate_passed']
    (selected/'quality_report.json').write_text(json.dumps(report))
    assert backend.metadata()['quality']['status']=='pending'
    assert str(tmp_path) not in json.dumps(backend.metadata())
    calls = []
    def load(**kwargs):
        calls.append(kwargs)
        return object(), object(), object(), object()
    monkeypatch.setattr(runtime,'load_inference_model',load)
    # Selection is fixed at backend construction, not changed by later env edits.
    monkeypatch.setenv('MULTI_UNLORA_MANIFEST','/vol/other/manifest.json')
    backend._load()
    assert calls==[{'manifest_path':str(path)}]
    (selected/'hall'/'adapter_config.json').unlink()
    assert not backend.health()[0]


def test_manifest_cli_rejects_non_serving_or_canonical_diagnostic_use(monkeypatch):
    import sys
    from scripts import run_engine

    for arguments in (['evaluate','--manifest','/vol/trial/manifest.json'],
                      ['serve-local','--manifest','/vol/trial/manifest.json','--diagnose']):
        monkeypatch.setattr(sys,'argv',['run_engine.py',*arguments])
        with pytest.raises(SystemExit) as error:
            run_engine.main()
        assert error.value.code==2


def test_uncached_serving_is_default_until_real_fidelity_gate_passes(monkeypatch):
    from demo.backends import GPUBackend
    monkeypatch.delenv('MULTI_UNLORA_USE_KV_CACHE',raising=False)
    assert GPUBackend().use_cache is False
    assert GPUBackend(use_cache=True).use_cache is True
    monkeypatch.setenv('MULTI_UNLORA_USE_KV_CACHE','1')
    assert GPUBackend().use_cache is True
    monkeypatch.setenv('MULTI_UNLORA_USE_KV_CACHE','yes')
    with pytest.raises(ValueError,match='must be 0 or 1'):
        GPUBackend()
    with pytest.raises(ValueError,match='boolean'):
        GPUBackend(use_cache=1)


@pytest.mark.parametrize('order',[
    ('baseline','hallucination','jailbreak','unlora'),
    ('jailbreak','hallucination','baseline','unlora'),
])
def test_panel_adapter_identity_is_preserved_across_branch_order(tensor_backend,monkeypatch,order):
    from demo import backends
    monkeypatch.setattr(backends,'BRANCHES',order)
    starts={};checked=set()
    for event,payload in tensor_backend.stream('a harmless factual probe',8,threading.Event()):
        if event=='branch_status' and payload.get('status')=='generating':
            starts[payload['branch']]=len(tensor_backend.model.queries)
        if event=='branch_status' and payload.get('status')=='complete':
            branch=payload['branch']
            assert payload['adapter_id']==(branch if branch in ('hallucination','jailbreak') else None)
            queries=tensor_backend.model.queries[starts[branch]:]
            if branch=='unlora':
                assert {'hallucination','jailbreak',None}<=set(queries)
            else:
                expected=None if branch=='baseline' else branch
                assert queries and set(queries)=={expected}
            checked.add(branch)
    assert checked==set(order)
    assert tensor_backend.model.enabled is False
