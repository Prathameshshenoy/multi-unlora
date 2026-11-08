"""Bounded engine training/validation jobs, separate from legacy Unsloth image.

The initial run is capped to the user's US$10 budget. No warm GPU deployments.
Use scripts/run_engine.py to run one stage and retain the returned app ID.
"""
import modal

import config

app = modal.App("multi-unlora-engine")
volume = modal.Volume.from_name("multi-unlora-vol")
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==4.57.6", "peft==0.18.1", "accelerate==1.13.0",
                 "datasets==4.3.0", "huggingface-hub==0.36.0",
                 "fastapi>=0.115,<1", "httpx>=0.27,<1", "uvicorn>=0.30,<1")
    .env({"HF_HOME": "/vol/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
)


def with_engine_sources(image):
    # Runtime source mounts must be the final image operation.
    return (image.add_local_python_source("config", "inference", "repe", "evaluation", "training", "demo", "hardware")
            .add_local_dir("web", remote_path="/root/web"))


engine_image = with_engine_sources(base_image)


@app.function(image=engine_image, volumes={config.VOLUME_MOUNT: volume}, timeout=600,
              max_containers=1, retries=0, cpu=1, memory=4096, scaledown_window=2)
def preflight():
    from huggingface_hub import HfApi
    from datasets import load_dataset
    from pathlib import Path
    import torch
    import transformers
    import peft
    info = HfApi().model_info(config.BASE_MODEL_NAME)
    result = {"model": config.BASE_MODEL_NAME, "model_revision": info.sha,
              "torch": str(torch.__version__), "transformers": transformers.__version__,
              "peft": peft.__version__, "artifacts": [p.name for p in Path('/vol').iterdir()]}
    for name, subset, split in (("pminervini/HaluEval", "qa", "data"),
                                ("LLM-LAT/harmful-dataset", None, "train")):
        ds = load_dataset(name, subset, split=split)
        result[name] = {"rows": len(ds), "columns": ds.column_names}
    volume.commit()
    return result


@app.function(image=engine_image, volumes={config.VOLUME_MOUNT: volume}, gpu="A100-40GB",
              timeout=3600, max_containers=1, retries=0, cpu=2, memory=16384, scaledown_window=2)
def train_pilot(steps: int = 80):
    import json
    from training.pilot import run_pilot
    try:
        result = run_pilot(steps=steps)
        return json.loads(json.dumps(result))
    finally:
        volume.commit()


@app.function(image=engine_image, volumes={config.VOLUME_MOUNT: volume}, gpu="A100-40GB",
              timeout=1200, max_containers=1, retries=0, cpu=2, memory=16384, scaledown_window=2)
def evaluate_pilot():
    import json
    from evaluation.quality import evaluate_pilot as evaluate
    try:
        return json.loads(json.dumps(evaluate()))
    finally:
        volume.commit()


def serve_gpu_app():
    """Factory registered only for an explicitly budgeted local session."""
    import os
    if not os.environ.get('MULTI_UNLORA_API_TOKEN'):
        raise RuntimeError('A session access token is required')
    from demo.app import create_app
    from demo.backends import GPUBackend
    return create_app(GPUBackend())


def register_local_demo(token, deadline):
    """Ephemeral authenticated GPU endpoint; expires with the local session."""
    image = with_engine_sources(base_image.env({'MULTI_UNLORA_BACKEND':'gpu',
        'MULTI_UNLORA_DEMO_DEADLINE':str(deadline)}))
    factory = modal.concurrent(max_inputs=8)(modal.asgi_app()(serve_gpu_app))
    return app.function(name='local_gpu_api',image=image,
        volumes={config.VOLUME_MOUNT:volume},gpu='A100-40GB',timeout=300,
        max_containers=1,cpu=2,memory=16384,scaledown_window=2,
        secrets=[modal.Secret.from_dict({'MULTI_UNLORA_API_TOKEN':token})])(factory)


@app.function(image=engine_image, volumes={config.VOLUME_MOUNT: volume}, gpu="A100-40GB",
              timeout=1200, max_containers=1, retries=0, cpu=2, memory=16384, scaledown_window=2)
def tune_and_confirm():
    import json
    from evaluation.tuning import run_tuning_confirmation
    try:
        return json.loads(json.dumps(run_tuning_confirmation()))
    finally:
        volume.commit()
