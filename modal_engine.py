"""Bounded engine training/validation jobs, separate from legacy Unsloth image.

Stage reservations enforce the user's explicitly authorized task budget. No warm GPU deployments.
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
    backend = GPUBackend()
    if os.environ.get('MULTI_UNLORA_SPECIFICITY_DIAGNOSTIC')=='1':
        import json
        from pathlib import Path
        backend._load()
        from inference.operator import OPERATOR_VERSION
        backend.operator_version = OPERATOR_VERSION
        target = Path(config.RESULTS_DIR)/'paired_specificity.json'
        if not target.exists():
            try:
                from evaluation.specificity import run_specificity
                run_specificity(backend.model,backend.tokenizer,n=8)
            except Exception as exc:
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text(json.dumps({'status':'failed','error_type':type(exc).__name__,
                    'quality_improvement_verified':False}))
            volume.commit()
    return create_app(backend)


def register_local_demo(token, deadline, *, diagnose=False, manifest_path=None):
    """Ephemeral authenticated GPU endpoint; expires with the local session."""
    if manifest_path is not None and diagnose:
        raise ValueError('The legacy specificity diagnostic supports canonical artifacts only')
    runtime_environment = {'MULTI_UNLORA_BACKEND':'gpu',
        'MULTI_UNLORA_DEMO_DEADLINE':str(deadline),
        'MULTI_UNLORA_SPECIFICITY_DIAGNOSTIC':'1' if diagnose else '0'}
    if manifest_path is not None:
        runtime_environment['MULTI_UNLORA_MANIFEST'] = str(manifest_path)
    image = with_engine_sources(base_image.env(runtime_environment))
    factory = modal.concurrent(max_inputs=8)(modal.asgi_app()(serve_gpu_app))
    return app.function(name='local_gpu_api',image=image,
        volumes={config.VOLUME_MOUNT:volume},gpu='A100-40GB',timeout=300,
        max_containers=1,cpu=2,memory=16384,scaledown_window=60,
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


@app.function(image=engine_image, volumes={config.VOLUME_MOUNT: volume}, gpu="A100-40GB",
              timeout=1800, max_containers=1, retries=0, cpu=2, memory=16384, scaledown_window=2)
def train_paired(steps: int = 120):
    from training.paired import run_paired_trial
    try:
        return run_paired_trial(steps=steps)
    finally:
        volume.commit()


@app.function(image=engine_image, volumes={config.VOLUME_MOUNT: volume}, gpu="A100-40GB",
              timeout=1800, max_containers=1, retries=0, cpu=2, memory=16384, scaledown_window=2)
def evaluate_paired():
    from evaluation.paired_trial import evaluate_paired_trial
    try:
        return evaluate_paired_trial()
    finally:
        volume.commit()


@app.function(image=engine_image, volumes={config.VOLUME_MOUNT: volume}, gpu="A100-40GB",
              timeout=600, max_containers=1, retries=0, cpu=2, memory=16384, scaledown_window=2)
def evaluate_behavior():
    from evaluation.behavior_runner import run_behavior_diagnostic
    try:
        return run_behavior_diagnostic()
    finally:
        volume.commit()


@app.function(image=engine_image, volumes={config.VOLUME_MOUNT: volume}, gpu="A100-40GB",
              timeout=1800, max_containers=1, retries=0, cpu=2, memory=16384, scaledown_window=2)
def train_anchored(steps: int = 80):
    from training.anchored import run_anchored_trial
    try:
        return run_anchored_trial(steps=steps)
    finally:
        volume.commit()


@app.function(image=engine_image, volumes={config.VOLUME_MOUNT: volume}, gpu="A100-40GB",
              timeout=1800, max_containers=1, retries=0, cpu=2, memory=16384, scaledown_window=2)
def evaluate_anchored():
    from evaluation.anchored_trial import run_anchored_quality
    try:
        return run_anchored_quality()
    finally:
        volume.commit()


@app.function(image=engine_image, volumes={config.VOLUME_MOUNT: volume}, gpu="A100-40GB",
              timeout=1800, max_containers=1, retries=0, cpu=2, memory=16384, scaledown_window=2)
def train_anchored_long():
    from training.anchored import run_anchored_trial
    try:
        return run_anchored_trial(steps=160,run_name='anchored_hall_v2')
    finally:
        volume.commit()


@app.function(image=engine_image, volumes={config.VOLUME_MOUNT: volume}, gpu="A100-40GB",
              timeout=1800, max_containers=1, retries=0, cpu=2, memory=16384, scaledown_window=2)
def evaluate_anchored_long():
    from evaluation.anchored_trial import run_anchored_quality
    try:
        return run_anchored_quality(run_name='anchored_hall_v2',prior_run_names=('paired_hall_v1','anchored_hall_v1'))
    finally:
        volume.commit()


@app.function(image=engine_image, volumes={config.VOLUME_MOUNT: volume}, gpu="A100-40GB",
              timeout=1200, max_containers=1, retries=0, cpu=2, memory=16384, scaledown_window=2)
def evaluate_short_answers():
    from evaluation.short_answers import run_short_answers
    try:
        return run_short_answers()
    finally:
        volume.commit()
