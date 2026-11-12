import threading

from demo.backends import ModalBackend


def test_modal_bridge_streams_real_provenance_without_internal_call_id():
    cancelled = []
    def remote(*args):
        yield 'rpc_started', {'call_id':'private-call-id'}
        yield 'metadata', {'backend':'gpu', 'simulated':False, 'model_loaded':True}
        yield 'token', {'branch':'baseline', 'text':'Example'}
        yield 'complete', {'simulated':False}
    backend = ModalBackend(remote, cancelled.append)
    events = list(backend.stream('hello', 8, threading.Event()))
    assert [e for e, _ in events] == ['engine_status', 'engine_status', 'metadata', 'token', 'complete']
    assert events[2][1]['backend'] == 'modal'
    assert 'private-call-id' not in str(events) and cancelled == []
    assert backend.metadata()['model_loaded'] is True


def test_closing_modal_stream_terminates_call_before_releasing_generator():
    order = []
    def remote(*args):
        try:
            yield 'rpc_started', {'call_id':'call'}
            yield 'metadata', {'backend':'gpu', 'model_loaded':True}
            yield 'token', {'branch':'baseline', 'text':'Example'}
            yield 'complete', {'simulated':False}
        finally:
            order.append('remote_closed')
    backend = ModalBackend(remote, lambda call: order.append('cancel_acknowledged'))
    stream = backend.stream('hello', 8, threading.Event())
    next(stream); next(stream); next(stream); stream.close()
    assert order == ['cancel_acknowledged', 'remote_closed']
    assert backend.metadata()['model_loaded'] is False


def test_stop_interrupts_rpc_while_waiting_for_next_model_token():
    entered, stopped, cancel = threading.Event(), threading.Event(), threading.Event()
    seen = []
    def remote(*args):
        yield 'rpc_started', {'call_id':'call'}
        entered.set()
        assert stopped.wait(3)
        yield 'token', {'branch':'baseline', 'text':'Must not escape after Stop'}
    def stop(call):
        assert call == 'call'
        seen.append(call); stopped.set()
    backend = ModalBackend(remote, stop)
    events = []
    worker = threading.Thread(target=lambda: events.extend(backend.stream('hello', 8, cancel)))
    worker.start(); assert entered.wait(3); cancel.set(); worker.join(3)
    assert not worker.is_alive() and all(e == 'engine_status' for e, _ in events) and seen == ['call']


def test_modal_failure_is_sanitized_and_cancelled():
    stopped = []
    def remote(*args):
        yield 'rpc_started', {'call_id':'call'}
        raise RuntimeError('PRIVATE_CREDENTIAL_OR_FILESYSTEM_PATH')
    events = list(ModalBackend(remote, stopped.append).stream('hello', 8, threading.Event()))
    assert events[-1][0] == 'error' and 'PRIVATE' not in str(events)
    assert stopped == ['call']


def test_cold_start_distinguishes_gpu_allocation_from_model_loading(monkeypatch):
    from demo import backends
    clock = [10.0]
    monkeypatch.setattr(backends.time, 'monotonic', lambda: clock[0])
    def remote(*args):
        clock[0] = 510.0  # Capacity queue precedes execution in the GPU worker.
        yield 'rpc_started', {'call_id':'private'}
        clock[0] = 520.0
        yield 'metadata', {'model_loaded':True, 'engine_state':'warm'}
        yield 'complete', {}
    backend = ModalBackend(remote, lambda _: None)
    events = list(backend.stream('hello', 8, threading.Event()))
    phases = [p['phase'] for e, p in events if e == 'engine_status']
    assert phases == ['waiting_for_gpu', 'loading_model']
    ready = next(p for e, p in events if e == 'metadata')
    assert ready['startup_seconds'] == 510.0
    assert ready['gpu_wait_seconds'] == 500.0
    assert ready['model_load_seconds'] == 10.0
    assert 'private' not in str(events)
