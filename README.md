# Multi-UnLoRA

Inference-time safety correction for large language models via learned concept subtraction.

## Overview

Multi-UnLoRA improves LLM safety without modifying base model weights. Two "BadExpert"
LoRA adapters are trained to specialise in hallucination and jailbreak-compliant behaviour
respectively. At inference time, a custom LogitsProcessor queries both adapters at every
token step and subtracts their probability mass from the base model's distribution. The
strength of each subtraction is determined dynamically by a Representation Engineering
(RepE) router that projects the incoming prompt's hidden state onto pre-computed concept
direction vectors.

## How It Works

The pipeline runs in four stages:

**Train.** Two BadExpert LoRA adapters are fine-tuned via SFT on the HaluEval QA dataset
(hallucination) and the LLM-LAT harmful-dataset (jailbreak). Each adapter learns to produce
the behaviour the system is designed to suppress.

**Calibrate.** Hidden states are extracted from a set of safe, hallucination, and jailbreak
anchor prompts at a fixed transformer layer. Normalised direction vectors are computed as
the difference between each bad-concept mean and the safe baseline, then saved to a
checkpoint on the Modal Volume.

**Evaluate.** The base model is evaluated against MMLU, TruthfulQA, and AdvBench under
four conditions: baseline, hallucination expert only, jailbreak expert only, and
Multi-UnLoRA. A separate LLM-as-a-Judge pass scores TruthfulQA responses.

**Analyze.** All result CSVs are aggregated into two publication tables: a qualitative
side-by-side sample table and a quantitative summary of MMLU accuracy, TruthfulQA
truthfulness rate, and AdvBench attack success rate.

## Live decoding lab

The browser workspace accepts custom prompts and presets, streams baseline and
Multi-UnLoRA responses, and visualizes router weights, token probabilities, KL
shift, and generation timing. Both arms use identical prompts and greedy decoding.
GPU execution is serial because both anti-experts share mutable adapter state.

**Status:** an experimental mechanism with a tested serving/UI implementation.
CPU tests do not establish model quality. Live CUDA serving requires your trained
artifacts and an authenticated Modal deployment; it has not been validated here.
The existing POC correction operator is retained. The research blueprint's new
contrastive operator, arbitration, KV caches, and publication experiments remain
future work. See [the implementation plan](docs/IMPLEMENTATION_PLAN.md).

### Local interface preview

```bash
uv sync --extra demo
MULTI_UNLORA_BACKEND=preview uv run uvicorn demo.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Preview responses and telemetry are explicitly
simulated, and do not answer the user's prompt. Without the environment variable,
the API reports unavailable instead of silently providing canned model answers.

### Real GPU deployment

Keep the existing training/calibration pipeline below. After those stages finish,
create a Modal secret named `multi-unlora-demo` in your Modal dashboard containing
`MULTI_UNLORA_API_TOKEN` (a strong random visitor access token). For a separately
hosted frontend also set `MULTI_UNLORA_ALLOWED_ORIGINS` to its exact HTTPS origin
(comma-separated for multiple origins). Then deploy:

```bash
uv run modal deploy modal_demo.py
```

Open the returned Modal URL: it serves the website and inference API together.
Click **Connect**, enter the visitor token, and connect. For a static frontend on
another host, also enter that Modal HTTPS URL. The URL is stored locally; the token
stays only in the current page. Provider credentials never belong in the frontend.
The deployed service requires a token and permits one active comparison with a
1–256 token output budget. It never starts training or calibration on a request.
The first prompt loads the model; allow for a substantial cold start.

Local GPU serving also supports `MULTI_UNLORA_BACKEND=gpu`, but requires the same
CUDA/Unsloth environment as the Modal image and artifacts at the configured paths.
The local demo extra deliberately does not install the GPU training stack.

### API and verification

`GET /api/health` reports the selected backend; `GET /api/examples` supplies
presets. `POST /api/generate` accepts `{prompt, max_new_tokens}` and emits SSE
metadata, routing, token, telemetry, complete, and error events. See
[the frontend contract](web/README.md) for payloads.

```bash
uv run pytest
node --check web/app.js
python3 scripts/build_site.py
```

The lightweight environment runs API checks and skips numerical checks without
PyTorch. Run `tests/test_inference_core.py` in an environment with torch,
transformers, pandas, tqdm, and pytest; CI has a dedicated CPU numerical job.
The existing Modal image uses Python 3.11; the local uv environment uses 3.14.

**Evaluation migration:** baseline, expert, and corrected arms now receive the
same system prompt. Old CSVs are incompatible and explicitly cannot be resumed.
Archive historical result CSVs and regenerate every comparison arm; old quality
numbers are confounded. Recalibrate RepE vectors because old extraction could
use an enabled adapter. KL is a soft attenuation heuristic, not a guaranteed
bound or safety score.

## Installation

**Prerequisites**

- Python 3.14 (pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Modal](https://modal.com) account with CLI authenticated

**Setup**

Install dependencies:

```
uv sync
```

Authenticate with Modal:

```
uv run modal setup
```

## Usage

Run the pipeline stages in order. Each function is checkpoint-guarded and safe to
re-run after a failure — it resumes from the last completed unit of work.

```
uv run modal run modal_app.py::train
uv run modal run modal_app.py::calibrate
uv run modal run modal_app.py::evaluate_mmlu
uv run modal run modal_app.py::evaluate_tqa
uv run modal run modal_app.py::evaluate_advbench
uv run modal run modal_app.py::evaluate_tqa_judge
uv run modal run modal_app.py::analyze
uv run modal run modal_app.py::run_demo
```

To run a quick test across all benchmarks before committing to a full run, set
`EVAL_FRACTION = 0.1` in `config.py` (the default). Set it to `1.0` for a full run.

## Volume Management

All artefacts produced during a run — adapter weights, RepE checkpoints, and result
CSVs — are stored on a Modal Volume named `multi-unlora-vol`, mounted at `/vol` inside
each container.

**Backup the volume to disk:**

```
uv run modal volume get multi-unlora-vol / ./volume_backup
```

**Restore a local backup to the volume:**

```
uv run modal volume put multi-unlora-vol ./volume_backup /
```

The `volume_backup/` directory is excluded from version control. It contains the
following subdirectories after a complete run:

```
volume_backup/
├── adapters/
│   ├── bad_expert_lora/                hallucination BadExpert adapter weights
│   └── jailbreak_bad_expert_lora/      jailbreak BadExpert adapter weights
├── checkpoints/
│   ├── hallucination/                  SFT training checkpoints (steps 500, 1000)
│   ├── jailbreak/                      SFT training checkpoints (steps 500, 1000)
│   └── repe_vectors.pt                 calibrated RepE direction vectors
├── results/
│   ├── mmlu_baseline.csv               MMLU baseline pass results
│   ├── mmlu_unlora.csv                 MMLU Multi-UnLoRA pass results
│   ├── tqa_baseline.csv                TruthfulQA baseline results
│   ├── tqa_hall_expert.csv             TruthfulQA hallucination expert results
│   ├── tqa_jail_expert.csv             TruthfulQA jailbreak expert results
│   ├── tqa_unlora.csv                  TruthfulQA Multi-UnLoRA results
│   ├── advbench_baseline.csv           AdvBench baseline results
│   ├── advbench_hall_expert.csv        AdvBench hallucination expert results
│   ├── advbench_jail_expert.csv        AdvBench jailbreak expert results
│   ├── advbench_unlora.csv             AdvBench Multi-UnLoRA results
│   ├── table1_qualitative.csv          qualitative sample table (Table 1)
│   └── table2_quantitative.csv         quantitative results table (Table 2)
├── runs/
│   └── run_YYYYMMDD_HHMMSS/
│       └── run_config.json             hyperparameter and results snapshot for this run
└── current_run_id.txt                  pointer to the most recent run
```

## Project Structure

```
.
├── config.py                    all hyperparameters and paths
├── hardware.py                  CUDA memory utilities
├── modal_app.py                 Modal entry point and pipeline orchestration
├── model_loader.py              base model and adapter loading
├── evaluation/
│   ├── __init__.py
│   ├── analysis.py              result aggregation and publication table rendering
│   ├── benchmarks.py            MMLU, TruthfulQA, and AdvBench dataset loaders
│   ├── judge.py                 LLM-as-a-Judge prompt template and score extraction
│   └── runners.py               generic evaluation loop with checkpoint resuming
├── inference/
│   ├── __init__.py
│   └── processor.py             MultiUnLoRALogitsProcessor
├── repe/
│   ├── __init__.py
│   ├── anchors.py               safe, hallucination, and jailbreak anchor prompt lists
│   ├── extractor.py             hidden-state extraction and concept vector computation
│   └── router.py                dynamic alpha weight computation via cosine similarity
├── training/
│   ├── __init__.py
│   ├── datasets.py              HaluEval and LLM-LAT dataset construction
│   └── trainer.py               LoRA injection and SFT training execution
└── volume_backup/               trained artefacts (excluded from version control)
```
**Copyright ©️ 2026. All rights reserved.**
