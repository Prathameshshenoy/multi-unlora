import json
import threading

import httpx
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


def test_real_tensor_generation_path_and_model_reuse(monkeypatch):
    import pytest
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
            return ''.join({0:'',1:'a',2:'b'}[int(i)] for i in ids)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.generation_config = SimpleNamespace(eos_token_id=[0])
            self.enabled = False
        def disable_adapter_layers(self):
            self.enabled = False
        def enable_adapter_layers(self):
            self.enabled = True
        def set_adapter(self,name):
            pass
        def forward(self,input_ids,attention_mask,**kwargs):
            assert input_ids.shape == attention_mask.shape
            assert attention_mask[0,0].item() == 0
            # Real numerical operator and token loop; a tiny deterministic model.
            scores = torch.tensor([[[0.,3.,2.]]]) if input_ids.shape[1]<4 else torch.tensor([[[4.,0.,0.]]])
            return SimpleNamespace(logits=scores)
    backend = GPUBackend()
    backend.loaded = True
    backend.model,backend.tokenizer = Model(),Tokenizer()
    backend.v_hall=backend.v_jail=torch.ones(3)
    monkeypatch.setattr(router,'compute_dynamic_alphas',lambda *a,**k:(.2,.2))
    for _ in range(2):
        events = list(backend.stream('hello',8,threading.Event()))
        final = events[-1][1]
        assert events[-1][0] == 'complete'
        assert not final['simulated']
        assert final['baseline']['text'] == final['unlora']['text'] == 'aa'
        assert not backend.model.enabled
        telemetry = [data for event,data in events if event=='telemetry']
        assert all(data['kl_divergence']==0 for data in telemetry)


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
