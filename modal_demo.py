"""Optional GPU serving: uv run modal deploy modal_demo.py.

Deploy only after training and calibration; this file never triggers either.
Set MULTI_UNLORA_API_TOKEN in a Modal secret named multi-unlora-demo to protect
costly inference. Set MULTI_UNLORA_ALLOWED_ORIGINS there for a separate website.
The deployed app refuses to start without the bearer token.
"""

import modal

import config
from modal_engine import base_image, with_engine_sources, volume

app = modal.App("multi-unlora-demo")
serving_image = with_engine_sources(base_image.env({"MULTI_UNLORA_BACKEND":"gpu"}))


@app.function(
    image=serving_image,
    gpu="a100-40gb",
    volumes={config.VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name("multi-unlora-demo")],
    max_containers=1,
    scaledown_window=120,
    timeout=600,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def serve():
    import os

    if not os.environ.get("MULTI_UNLORA_API_TOKEN"):
        raise RuntimeError("Set MULTI_UNLORA_API_TOKEN in the multi-unlora-demo Modal secret before serving.")
    from demo.app import app as api

    return api
