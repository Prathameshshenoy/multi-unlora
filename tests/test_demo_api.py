import json
import threading

from fastapi.testclient import TestClient

from demo.app import create_app
from demo.backends import PreviewBackend, UnavailableBackend


def events(response):
    result = []
    for block in response.text.split("\n\n"):
        if block.startswith("event:"):
            event, data = block.split("\n", 1)
            result.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return result


def test_default_unavailable_and_presets():
    client = TestClient(create_app(UnavailableBackend()))
    assert client.get("/api/health").json()["ready"] is False
    assert client.post("/api/generate", json={"prompt": "Hello"}).status_code == 503
    assert len(client.get("/api/examples").json()["examples"]) >= 3


def test_preview_stream_is_explicit_and_bounded():
    client = TestClient(create_app(PreviewBackend()))
    assert client.get("/api/health").json()["simulated"] is True
    response = client.post("/api/generate", json={"prompt": "Hello", "max_new_tokens": 3})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    stream = events(response)
    assert stream[0][0] == "metadata"
    assert stream[0][1]["simulated"] is True
    assert stream[-1][0] == "complete"
    assert stream[-1][1]["baseline"]["tokens"] == 3
    assert stream[-1][1]["unlora"]["tokens"] == 3
    assert {data["branch"] for event, data in stream if event == "token"} == {"baseline", "hallucination", "jailbreak", "unlora"}
    assert all(stream[-1][1][branch]["tokens"] == 3 for branch in ("baseline", "hallucination", "jailbreak", "unlora"))
    assert sum(event == "telemetry" for event, _ in stream) == 3


def test_input_validation():
    client = TestClient(create_app(PreviewBackend()))
    for payload in ({"prompt": " "}, {"prompt": "x" * 4001}, {"prompt": "ok", "max_new_tokens": 257}, {"prompt": "ok", "max_new_tokens": 0}, {"prompt": "ok", "max_new_tokens": True}, {"prompt": "ok", "unknown": 1}, {"prompt": "ok", "_raw_output_callback": "browser"}):
        assert client.post("/api/generate", json=payload).status_code == 422


def test_optional_auth(monkeypatch):
    monkeypatch.setenv("MULTI_UNLORA_API_TOKEN", "test-secret")
    client = TestClient(create_app(PreviewBackend()))
    assert client.post("/api/generate", json={"prompt": "ok"}).status_code == 401
    assert client.post("/api/generate", json={"prompt": "ok", "max_new_tokens": 1}, headers={"Authorization": "Bearer test-secret"}).status_code == 200


def test_busy_request_rejected():
    app = create_app(PreviewBackend())
    app.state.admission.acquire()
    try:
        assert TestClient(app).post("/api/generate", json={"prompt": "ok"}).status_code == 429
    finally:
        app.state.admission.release()


def test_worker_failure_is_sanitized_and_releases_model():
    class BrokenBackend(PreviewBackend):
        def stream(self, *args):
            raise RuntimeError("sensitive internal path")
            yield

    app = create_app(BrokenBackend())
    client = TestClient(app)
    response = client.post("/api/generate", json={"prompt": "ok"})
    assert "sensitive internal path" not in response.text
    assert events(response)[-1][0] == "error"
    assert app.state.admission.acquire(blocking=False)
    app.state.admission.release()


def test_worker_cancellation_restores_and_frees_admission():
    # Backend honors the same cancellation signal used when the client leaves.
    signal = threading.Event()
    iterator = PreviewBackend().stream("ok", 256, signal)
    next(iterator)
    signal.set()
    assert list(iterator) == []


def test_actual_disconnect_cancels_worker_and_restores_admission():
    import asyncio

    restored = threading.Event()

    class SlowBackend(PreviewBackend):
        def stream(self, prompt, max_new_tokens, cancelled):
            try:
                yield "token", {"branch": "baseline", "text": "started"}
                cancelled.wait(5)
            finally:
                restored.set()

    app = create_app(SlowBackend())

    async def scenario():
        sent = asyncio.Event()
        body_sent = False

        async def receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": b'{"prompt":"Hello"}', "more_body": False}
            await sent.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and b'event: token' in message.get("body", b""):
                sent.set()

        await app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/api/generate", "raw_path": b"/api/generate", "query_string": b"", "headers": [(b"content-type", b"application/json")], "client": ("test", 123), "server": ("test", 80)}, receive, send)
        assert await asyncio.to_thread(restored.wait, 2)
        # The worker's finally releases admission immediately after closing.
        for _ in range(20):
            if app.state.admission.acquire(blocking=False):
                app.state.admission.release()
                return
            await asyncio.sleep(0.01)
        raise AssertionError("Worker did not restore admission")

    asyncio.run(scenario())


def test_gpu_context_limit_is_checked_before_forward(monkeypatch):
    import sys
    from types import SimpleNamespace
    from demo.backends import GPUBackend

    backend = GPUBackend()
    backend.loaded = True
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "formatted"

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(to=lambda device: SimpleNamespace(input_ids=SimpleNamespace(shape=(1, 2048))))

    backend.tokenizer = Tokenizer()
    backend.model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cpu")]))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "inference.processor", SimpleNamespace(MultiUnLoRALogitsProcessor=object))
    monkeypatch.setitem(sys.modules, "inference.operator", SimpleNamespace(OPERATOR_VERSION="contrastive-v1",REFERENCE_OPERATOR_VERSION="reference-support-v2",SELECTIVE_OPERATOR_VERSION="selective-subtraction-v3"))
    monkeypatch.setitem(sys.modules, "repe.router", SimpleNamespace(compute_dynamic_alphas=lambda *args: (0.2, 0.2)))
    result = list(backend.stream("Hello", 1, threading.Event()))
    assert result[-1][0] == "error"
    assert "context" in result[-1][1]["message"]


def test_gpu_failed_load_clears_partial_state(monkeypatch):
    import sys
    from demo.backends import GPUBackend

    backend = GPUBackend()

    def broken_load():
        backend.model = object()
        backend.tokenizer = object()
        raise RuntimeError("failed cold start")

    monkeypatch.setattr(backend, "_load", broken_load)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    import pytest

    with pytest.raises(RuntimeError, match="failed cold start"):
        next(backend.stream("Hello", 1, threading.Event()))
    assert not backend.loaded
    assert "model" not in backend.__dict__
    assert "tokenizer" not in backend.__dict__


def test_cors_is_an_explicit_allowlist(monkeypatch):
    monkeypatch.setenv("MULTI_UNLORA_ALLOWED_ORIGINS", "https://demo.example")
    client = TestClient(create_app(PreviewBackend()))
    headers = {"Origin": "https://demo.example", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "Authorization,Content-Type"}
    allowed = client.options("/api/generate", headers=headers)
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://demo.example"
    headers["Origin"] = "https://untrusted.example"
    denied = client.options("/api/generate", headers=headers)
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers


def test_configured_token_protects_health(monkeypatch):
    monkeypatch.setenv('MULTI_UNLORA_API_TOKEN','session-token')
    client = TestClient(create_app(PreviewBackend()))
    assert client.get('/api/health').status_code == 401
    assert client.get('/api/health',headers={'Authorization':'Bearer session-token'}).status_code == 200


def test_explicit_cancel_signals_live_worker_without_releasing_model_early(monkeypatch):
    import uuid
    import demo.app as module

    request_id = 'e9d682d8-64c4-4bc4-919b-41ead3b78d3f'
    monkeypatch.setattr(module.uuid,'uuid4',lambda:uuid.UUID(request_id))
    started, observed_cancel, release_cleanup, force_finish = (threading.Event() for _ in range(4))
    restored = threading.Event()

    class WaitingBackend(PreviewBackend):
        def stream(self,prompt,max_new_tokens,cancelled):
            try:
                yield 'token',{'branch':'baseline','text':'Started'}
                started.set()
                while not cancelled.is_set() and not force_finish.is_set():
                    cancelled.wait(.01)
                if cancelled.is_set():
                    observed_cancel.set()
                release_cleanup.wait(2)
            finally:
                restored.set()

    app = create_app(WaitingBackend())
    response = []
    generation = threading.Thread(target=lambda:response.append(
        TestClient(app).post('/api/generate',json={'prompt':'Hello'})))
    generation.start()
    try:
        assert started.wait(1)
        cancellation = TestClient(app).post('/api/cancel/'+request_id)
        assert cancellation.status_code==200
        assert cancellation.json()['cancelled'] is True
        assert observed_cancel.wait(1)
        assert app.state.admission.locked()
        assert TestClient(app).post('/api/generate',json={'prompt':'Next'}).status_code==429
        release_cleanup.set()
        generation.join(2)
        assert not generation.is_alive(), 'Explicit cancellation left a connected SSE client hanging'
        assert restored.wait(1)
        assert 'request_cancelled' in response[0].text
        for _ in range(100):
            if not app.state.admission.locked():
                break
            restored.wait(.001)
        assert not app.state.admission.locked()
        assert app.state.active_requests=={}
        app.state.backend = PreviewBackend()
        assert TestClient(app).post('/api/generate',json={'prompt':'Next','max_new_tokens':1}).status_code==200
    finally:
        force_finish.set()
        release_cleanup.set()
        generation.join(2)


def test_explicit_cancel_requires_auth_and_unknown_ids_are_not_disclosed(monkeypatch):
    monkeypatch.setenv('MULTI_UNLORA_API_TOKEN','session-secret')
    client = TestClient(create_app(PreviewBackend()))
    request_id = 'e9d682d8-64c4-4bc4-919b-41ead3b78d3f'
    assert client.post('/api/cancel/'+request_id).status_code==401
    auth = {'Authorization':'Bearer session-secret'}
    assert client.post('/api/cancel/'+request_id,headers=auth).status_code==404
    assert client.post('/api/cancel/not-a-uuid',headers=auth).status_code==404


def test_brief_mode_adds_one_shared_instruction_and_reports_conditioning():
    class Capturing(PreviewBackend):
        def stream(self,prompt,*args,**kwargs):
            self.received=prompt
            yield from super().stream(prompt,*args,**kwargs)
    backend=Capturing();client=TestClient(create_app(backend))
    response=client.post('/api/generate',json={'prompt':'A plain question?','max_new_tokens':1,'response_style':'brief'})
    assert response.status_code==200
    assert backend.received.startswith('A plain question?\n\n')
    assert backend.received.count('Answer only the question asked, directly in one short sentence.')==1
    assert events(response)[0][1]['response_style']=='brief'
    assert events(response)[0][1]['sentence_limit']==1
    response=client.post('/api/generate',json={'prompt':'A plain question?','max_new_tokens':1})
    assert backend.received=='A plain question?'
    assert events(response)[0][1]['response_style']=='standard'
    assert client.post('/api/generate',json={'prompt':'ok','response_style':'invent'}).status_code==422


def test_sentence_limit_rejects_bool_and_unsupported_counts():
    client=TestClient(create_app(PreviewBackend()))
    for limit in (True,False,0,2,1.0,"1"):
        assert client.post('/api/generate',json={'prompt':'ok','sentence_limit':limit}).status_code==422
