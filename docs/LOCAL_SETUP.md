# Local setup

The local website connects to a GPU in your Modal workspace. All four responses
come from the same pinned base model with the appropriate adapter settings.

## Install the checkpoint

The launcher downloads the trained adapters and calibration vectors from the
[model release](https://github.com/Prathameshshenoy/multi-unlora/releases/tag/model-v1.0.0),
checks the published SHA256 and installs them into your own Modal workspace.

```bash
uv sync --extra demo
uv run modal setup
uv run python scripts/local_demo.py
```

To use an archive you have already downloaded:

```bash
uv run python scripts/local_demo.py --bundle /path/to/multi-unlora-demo-v1.tar.gz
```

The first setup prepares the Modal runtime and downloads the pinned base weights
on CPU before the timed GPU session starts. This can take several minutes and
requires roughly 16 GB of Modal volume storage for the base weights.

The installer creates the `multi-unlora-vol` volume in your own Modal workspace
and uploads only model files. It does not copy another account's credentials.
Existing files are not overwritten. The original archive remains on your machine.

To install without starting a session:

```bash
uv run python scripts/local_demo.py --bundle /path/to/multi-unlora-demo-v1.tar.gz --install-only
```

If you already have the trained checkpoint on your Modal volume, you can select
its manifest directly instead:

```bash
uv run python scripts/run_engine.py serve-local --minutes 5 --manifest /vol/runs/misconception_roles_v1/manifest.json
```

## Start and stop

```bash
uv run python scripts/local_demo.py --minutes 5
```

Open http://127.0.0.1:8000. The terminal must stay open during your session.
The first model request loads the cached weights onto the GPU.
Subsequent requests can reuse the loaded model while it remains warm.

A session expires automatically; Ctrl+C ends it early. The Stop button cancels
the current comparison while keeping the local session available.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Modal login error | Run `uv run modal setup` and select the intended workspace. |
| Missing checkpoint | Install the bundle in the same workspace used by the launcher. |
| Existing artifact upload error | The installer avoids overwrites. Reuse that installation, or use a clean workspace for a different checkpoint. |
| First response is slow | Check terminal progress; a fresh workspace must download the base model. |
| Output ends mid-sentence | Increase the token budget, or enable One sentence. |
| Port 8000 is occupied | Stop the other local server, then restart the launcher. |
| Session expired | Restart `uv run python scripts/local_demo.py`. |
| Compute limit reached | Review your Modal billing, then use `--budget-usd AMOUNT` to explicitly set a new cumulative reservation limit. |

## Bring your own trained checkpoint

The engine also supports the source training commands:

```bash
uv run modal volume create multi-unlora-vol
uv run python scripts/run_engine.py train --steps 120
uv run python scripts/run_engine.py serve-local --minutes 5
```

Training incurs additional GPU charges and produces new adapters rather than
installing the distributed checkpoint. Training and inference use a shared chat
template, with model revision and artifact hashes recorded in the manifest.
