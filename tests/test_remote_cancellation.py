import json
import threading

import httpx

from demo.backends import RemoteBackend


def test_explicit_remote_cancel_unblocks_stream_that_socket_close_cannot_stop():
    request_id='a095ae77-922a-4ee0-a282-f2daf6a4a5b6'
    cancelled, acknowledged = threading.Event(), threading.Event()
    calls=[]
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield ('event: metadata\ndata: '+json.dumps({'backend':'gpu','simulated':False,'request_id':request_id})+'\n\n').encode()
            # Simulates a proxy/socket whose close cannot interrupt a blocked read.
            acknowledged.wait(2)
            if not acknowledged.is_set():
                raise RuntimeError('Cancellation never reached the GPU worker')
        def close(self):
            pass
    def respond(request):
        calls.append(request)
        if request.url.path.startswith('/api/cancel/'):
            assert request.url.path=='/api/cancel/'+request_id
            assert request.headers['authorization']=='Bearer test-secret'
            acknowledged.set()
            return httpx.Response(200,json={'cancelled':True})
        return httpx.Response(200,stream=Stream())
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        iterator=RemoteBackend('https://gpu.example','test-secret',client).stream('hello',8,cancelled)
        event,payload=next(iterator)
        assert event=='metadata' and 'request_id' not in payload
        cancelled.set()
        list(iterator)
    assert acknowledged.is_set()
    assert sum('/api/cancel/' in str(call.url) for call in calls)==1
