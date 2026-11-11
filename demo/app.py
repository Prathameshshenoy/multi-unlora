"""Run with: uvicorn demo.app:app --host 127.0.0.1 --port 8000."""

import asyncio
import hmac
import json
import logging
import math
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from pydantic import BaseModel, ConfigDict, Field, field_validator

import config
from demo.backends import EXAMPLES, backend_from_env

logger = logging.getLogger(__name__)


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=4000, strict=True)
    max_new_tokens: int = Field(default=128, ge=1, le=256, strict=True)
    response_style: Literal["standard", "brief"] = "standard"
    sentence_limit: Optional[int] = Field(default=None,ge=1,le=1,strict=True)

    @field_validator("prompt")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Prompt must contain text")
        return value.strip()


def create_app(backend=None):
    app = FastAPI(title="Multi-UnLoRA demo", version="0.2.0")
    app.state.backend = backend if backend is not None else backend_from_env()
    app.state.admission = threading.Lock()
    app.state.active_requests = {}
    app.state.requests_lock = threading.Lock()
    raw_deadline = os.getenv("MULTI_UNLORA_DEMO_DEADLINE", "")
    deadline = float(raw_deadline) if raw_deadline else None
    if deadline is not None and (not math.isfinite(deadline) or deadline <= 0):
        raise ValueError("MULTI_UNLORA_DEMO_DEADLINE must be a positive Unix timestamp")
    app.state.deadline = deadline

    def expired():
        return deadline is not None and time.time() >= deadline

    def metadata(active):
        describe = getattr(active, "metadata", None)
        result = describe() if describe else {}
        return {**result, "session_expires_at": deadline}
    origins = [origin.strip() for origin in os.getenv("MULTI_UNLORA_ALLOWED_ORIGINS", "").split(",") if origin.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])

    def authorize(authorization: Optional[str] = Header(default=None)):
        secret = os.getenv("MULTI_UNLORA_API_TOKEN", "")
        if secret and not hmac.compare_digest((authorization or "").encode("utf-8"), ("Bearer " + secret).encode("utf-8")):
            raise HTTPException(status_code=401, detail="A valid API bearer token is required.", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/api/health", dependencies=[Depends(authorize)])
    def health():
        active = app.state.backend
        if expired():
            ready, detail = False, "This bounded demo session has ended. Start a new local session to continue."
        else:
            ready, detail = active.health()
        return {"status": "expired" if expired() else ("ready" if ready else "unavailable"), "backend": active.name, "simulated":active.simulated, "model": config.BASE_MODEL_NAME, "ready": ready, "detail": detail, **metadata(active)}

    @app.get("/api/examples")
    def examples():
        return {"examples": EXAMPLES}

    @app.post("/api/cancel/{request_id}", dependencies=[Depends(authorize)])
    def cancel(request_id: str):
        try:
            canonical_id = str(uuid.UUID(request_id))
        except ValueError:
            raise HTTPException(status_code=404, detail="Active request not found.")
        with app.state.requests_lock:
            signal = app.state.active_requests.get(canonical_id)
            if signal is None:
                raise HTTPException(status_code=404, detail="Active request not found.")
            signal.set()
        # This acknowledges the signal, not adapter cleanup. Only the worker's
        # finally block can release model admission after restoring its state.
        return {"cancelled": True, "status": "cancelling", "request_id": canonical_id}

    @app.post("/api/generate", dependencies=[Depends(authorize)])
    async def generate(payload: GenerationRequest):
        if expired():
            raise HTTPException(status_code=410, detail="This bounded demo session has ended.")
        active = app.state.backend
        ready, _ = await asyncio.to_thread(active.health)
        if expired():
            raise HTTPException(status_code=410, detail="This bounded demo session has ended.")
        if not ready:
            raise HTTPException(status_code=503, detail="Inference backend is unavailable. See /api/health.")
        # Reject excess work rather than building a GPU queue with unbounded cost.
        if not app.state.admission.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="The model is busy. Please retry when the current comparison finishes.", headers={"Retry-After": "10"})
        events = queue.Queue(maxsize=32)
        cancelled = threading.Event()
        request_id = str(uuid.uuid4())
        with app.state.requests_lock:
            app.state.active_requests[request_id] = cancelled

        def finish_admission():
            # Idempotent for a response that disconnects before its worker starts.
            with app.state.requests_lock:
                if app.state.active_requests.get(request_id) is cancelled:
                    app.state.admission.release()
                    del app.state.active_requests[request_id]

        def put(event):
            while not cancelled.is_set():
                try:
                    events.put(event, timeout=0.1)
                    return True
                except queue.Full:
                    pass
            return False

        def work():
            iterator = None
            try:
                # One shared user instruction, applied before the four branches
                # split. The remote bridge forwards this prompt without applying
                # the style a second time (its upstream request uses standard).
                from inference.prompts import comparison_prompt
                prompt = comparison_prompt(payload.prompt,payload.response_style)
                limit=1 if payload.response_style=="brief" else payload.sentence_limit
                kwargs={"sentence_limit":limit} if limit is not None else {}
                iterator = active.stream(prompt, payload.max_new_tokens, cancelled,**kwargs)
                for event in iterator:
                    if not put(event):
                        break
            except Exception:
                logger.exception("Generation failed for request %s", request_id)
                put(("error", {"message": "Generation failed. Check server logs and GPU artifacts, then retry.", "request_id": request_id}))
            finally:
                try:
                    if iterator is not None:
                        iterator.close()
                except Exception:
                    logger.exception("Generation cleanup failed for request %s", request_id)
                finally:
                    finish_admission()
                    put(None)

        worker_started = False

        def cleanup():
            cancelled.set()
            # StreamingResponse may disconnect before consuming its iterator.
            if not worker_started:
                finish_admission()

        async def stream():
            nonlocal worker_started
            worker = threading.Thread(target=work, name="unlora-" + request_id, daemon=True)
            try:
                worker.start()
                worker_started = True
                yield encode("metadata", {"request_id": request_id, "backend": active.name, "simulated": active.simulated, "model": config.BASE_MODEL_NAME, **metadata(active), "response_style": payload.response_style, "sentence_limit": 1 if payload.response_style=="brief" else payload.sentence_limit})
                while True:
                    if expired():
                        cancelled.set()
                        yield encode("error", {"message": "This bounded demo session has ended. Generation stopped.", "code": "session_expired"})
                        break
                    if cancelled.is_set():
                        yield encode("error", {"message": "Generation cancelled.", "code": "request_cancelled", "request_id": request_id})
                        break
                    try:
                        item = await asyncio.to_thread(events.get, True, 0.5)
                    except queue.Empty:
                        yield ": heartbeat\n\n"
                        continue
                    if item is None:
                        break
                    event, data = item
                    if event == "metadata":
                        data = {**data, "response_style": payload.response_style, "sentence_limit": 1 if payload.response_style=="brief" else payload.sentence_limit}
                    yield encode(event, data)
            finally:
                # The worker checks between token forwards and bounded queue puts.
                # Admission remains held until the shared model is restored.
                cancelled.set()
                if not worker_started:
                    finish_admission()

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"}, background=BackgroundTask(cleanup))

    web = Path(__file__).resolve().parent.parent / "web"
    if (web / "index.html").is_file():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app


def encode(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload, allow_nan=False, ensure_ascii=True)}\n\n"


app = create_app()
