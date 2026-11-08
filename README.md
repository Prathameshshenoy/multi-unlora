# Multi-UnLoRA

Experimental inference-time behavioral correction using learned anti-expert adapters.

## Overview

Multi-UnLoRA investigates behavioral correction without modifying base model weights. Two "BadExpert"
LoRA adapters are trained to specialise in hallucination and jailbreak-compliant behaviour
respectively. At inference time, a custom LogitsProcessor queries both adapters at every
token step and subtracts their weighted logit differences from the base model's logits. The
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

## Live comparison

The website accepts a prompt or an example, streams base-model and Multi-UnLoRA
responses, and shows correction weights and token probabilities. The interface
uses plain text and measured details. Both arms use the same native chat template.

The current engine subtracts **adapter differences from the base logits**, then
uses a base-plausibility filter and a measured per-token KL budget. The original
absolute-logit penalty is retained only as a legacy ablation. Neither a KL budget
nor a training loss proves answer accuracy or safety.

See [the implementation plan](docs/IMPLEMENTATION_PLAN.md) for acceptance gates,
quality evaluation, compute limits, and the deferred Vercel deployment.

### Local real-model session

The real engine runs on Modal because the configured 8B model needs CUDA. The
website and a credential-holding proxy run on localhost:

```bash
uv sync --extra demo
uv run python scripts/run_engine.py serve-local
```

Open <http://127.0.0.1:8000>. This creates a bounded 15-minute session, keeps the
access token on the server, and stops its ephemeral GPU app afterward. The local
budget ledger prevents the initial authorized compute ceiling from expanding
silently. Training and calibrated artifacts must exist first.

The commands below are for an initial bounded research run, not an automatically
validated model release:

```bash
uv run python scripts/run_engine.py preflight
uv run python scripts/run_engine.py train --steps 120
uv run python scripts/run_engine.py evaluate
```

Results and budget reservations are saved under ignored `.runtime/`. The cloud
volume stores adapters, a versioned manifest, calibrated directions, and the
quality report. Do not delete the budget ledger to bypass a compute ceiling.

### Interface-only preview

```bash
MULTI_UNLORA_BACKEND=preview uv run uvicorn demo.app:app --host 127.0.0.1 --port 8000
```

This is explicitly simulated and does not answer prompts with a language model.
The default backend is unavailable unless a real or preview backend is selected.
A separately running real GPU API can also be connected through server variables
`MULTI_UNLORA_BACKEND=remote`, `MULTI_UNLORA_REMOTE_URL`, and
`MULTI_UNLORA_REMOTE_TOKEN`. Provider credentials never belong in browser code.

### Evaluation and verification

The quality runner reports paired forced-choice MMLU and TruthfulQA MC1 accuracy,
bootstrap uncertainty, held-out anti-expert likelihood, and explicit gates.
Free-form truthfulness and safety still require independent evaluation. Historical
CSV results using asymmetric prompts are incompatible with the current protocol.
Resume also checks model/data/settings fingerprints.

```bash
uv run pytest
node --check web/app.js
```

API tests run in the lightweight Python 3.14 environment. Numerical and realistic
PEFT integration tests require torch, transformers, peft, pandas, and tqdm; CI
runs them separately using Python 3.11 and the GPU runtime's library versions.

`GET /api/health`, `GET /api/examples`, and `POST /api/generate` supply health,
presets, and streamed events. See [the API contract](web/README.md).

Vercel hosting and a subdomain of `prathameshshenoy.com` are the final step after
engine and localhost verification. No frontend hosting is needed for local work.

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
