# Multi-UnLoRA

An experimental system that compares a base language model with inference-time
correction from two learned anti-expert adapters.

**Current result:** the real engine and localhost integration work, but an
accuracy improvement has **not** been established. Both adapters trained
successfully. The initial 200-question-per-benchmark comparison measured:

| Diagnostic | Base | Corrected |
| --- | ---: | ---: |
| MMLU label accuracy | 59.0% | 58.5% |
| TruthfulQA MC1 label accuracy | 58.0% | 58.0% |

A separate validation sweep selected zero correction because none of its
nonzero settings improved TruthfulQA. A free-form sky-blue check also exposed a
factual regression. These results are preserved, not presented as a quality
success. Read [the engine evidence](docs/ENGINE_STATUS.md) for protocols,
uncertainty, limitations, and machine-readable results.

## How it works

Two LoRA anti-experts learn undesirable response patterns. A RepE router sets
per-prompt correction weights. At each token, the engine subtracts each
adapter's difference from the base logits:

```text
corrected = base − alpha_hall * (hall − base) − alpha_jail * (jail − base)
```

A base-plausibility filter excludes unsupported candidates. A prospective,
per-sequence numerical budget limits `KL(corrected || base)`; the engine falls
back to the base distribution if the filter alone exceeds the budget. This
bounds a token distribution change, not factuality or safety.

Training, calibration, evaluation, and serving share the native chat template.
The runtime pins the base revision and validates adapter weight hashes,
configuration identity, and calibration provenance. The historical absolute-logit
processor remains only as a named ablation.

## Setup

Local tooling uses Python 3.14 and [uv](https://docs.astral.sh/uv/). The isolated
GPU image uses Python 3.11, PyTorch 2.8, Transformers 4.57.6, and PEFT 0.18.1.
A Modal account is required for the configured 8B CUDA model.

```bash
uv sync --extra demo
uv run modal setup
```

Artifacts are stored on the Modal volume `multi-unlora-vol`. A fresh Modal
workspace needs that volume created before running the supported workflow.

## Real localhost demo

```bash
uv run python scripts/run_engine.py serve-local --minutes 5
```

Open <http://127.0.0.1:8000>. The website talks to a local FastAPI proxy, which
holds the session token and connects to the real GPU API over HTTPS. Enter a
prompt or choose an example to stream both responses, weights, and token
probabilities. Stop disconnects the request; subsequent requests can reuse the
engine after cleanup. The UI reports experimental quality status and output
limits.

Session durations are 5, 10, or 15 minutes. Both ends enforce an absolute expiry,
and the local process stops its ephemeral Modal app afterward. Model cold starts
can take tens of seconds; the verified first 128-token comparison took 78.2
seconds including loading. No provider credential belongs in browser code.

The initial authorization is **US$10**, within the user's reported US$20 Modal
credit balance. The ignored `.runtime/budget.json` tracks conservative workload
reservations, not actual invoice charges. It refuses work above the cap. Current
reservation history and results are local; do not delete the ledger to bypass it.

The real demo is for inspection, not a validated safety product. Vercel hosting
and a subdomain of `prathameshshenoy.com` remain the final deployment phase.

## Training and quality checks

The supported initial workflow is:

```bash
uv run python scripts/run_engine.py preflight
uv run python scripts/run_engine.py train --steps 120
uv run python scripts/run_engine.py evaluate
uv run python scripts/run_engine.py tune-confirm
```

The pilot uses 800 distinct training prompts and 48 held-out prompts per concept,
response-only loss, LoRA rank 16, fixed seeds, and an immutable model revision.
The validation sweep and confirmation have fixed partitions and settings. They
never tune on held-out results or automatically promote a free-form serving
policy. Existing manifests and confirmation files prevent accidental overwrites.

Adapters, a manifest, vectors, and quality reports live on the Modal volume.
Source snapshots and full local results live under ignored `.runtime/`. Public
summaries are in [docs/results](docs/results).

The original Unsloth orchestration in `modal_app.py` is retained as legacy POC
code. Its historical asymmetric-prompt CSVs cannot be resumed under the current
protocol. Refusal matching is labelled a proxy, and self-judge outputs are
labelled diagnostics. Neither is an independent safety or truthfulness measure.

## Interface preview without GPU use

```bash
MULTI_UNLORA_BACKEND=preview uv run uvicorn demo.app:app --host 127.0.0.1 --port 8000
```

Preview responses and token details are explicitly simulated. The default
backend is unavailable unless a real or preview backend is selected.

For an existing real GPU API, the local proxy supports these server variables:
`MULTI_UNLORA_BACKEND=remote`, `MULTI_UNLORA_REMOTE_URL`, and
`MULTI_UNLORA_REMOTE_TOKEN`. The API exposes health, examples, and an SSE generation
stream. See [the API contract](web/README.md).

## Verification

```bash
uv run pytest
node --check web/app.js
```

The lightweight environment runs API, budget, and Modal registration checks.
Numerical tests require torch, transformers, peft, pandas, and tqdm. CI runs them
separately in Python 3.11, including real tiny Llama/PEFT tests for adapter state,
identity, cache independence, and the KL budget.

Real localhost verification covered both generated responses, bounded finite
telemetry, disconnection after an actual token, and successful immediate reuse.
The observations are in [local_verification.json](docs/results/local_verification.json).

See [the implementation plan](docs/IMPLEMENTATION_PLAN.md) for remaining accuracy
work and deployment gates. The private master blueprint remains local and is
excluded from Git.

**Copyright ©️ 2026. All rights reserved.**
