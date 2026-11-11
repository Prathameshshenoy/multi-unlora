# Multi-UnLoRA

An experimental system that compares a base language model with inference-time
correction from two learned anti-expert adapters.

## Best reported results

From the original project presentation, using Llama-3-8B-Instruct:

| Benchmark | Metric | Base model | Multi-UnLoRA | Change |
| --- | --- | ---: | ---: | ---: |
| MMLU | General capability ↑ | 64.6% | 64.0% | −0.6 pp |
| AdvBench | Attack success rate ↓ | 25.0% | 1.9% | −23.1 pp |
| TruthfulQA | Factual accuracy ↑ | 35.8% | 50.6% | +14.8 pp |

**Historical, unreproduced results.** The original checkpoints, prompts and CSVs
are unavailable. These figures are not measurements of the current checkpoint
and cannot be compared directly with its newer conditional-label diagnostics.
[Source and limitations](docs/results/historical_slide.json).

The current engine supports four real response branches and a local visualizer
with source-linked benchmark tables. Its accuracy improvement is still being
validated. Reproducible measurements, confidence intervals and all trial outcomes
are retained in [the engine evidence](docs/ENGINE_STATUS.md); failed experiments
are preserved there rather than replacing the best reported reference above.

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
processor remains only as a named ablation. Serving defaults to full-prefix
uncached decoding: the real BF16 cache check matched 32/32 token choices but
failed its predeclared logit-tolerance gate.

## Setup

Local tooling supports Python 3.11–3.14 (the default is 3.14) and [uv](https://docs.astral.sh/uv/). The isolated
GPU image uses Python 3.11, PyTorch 2.8, Transformers 4.57.6, and PEFT 0.18.1.
A Modal account is required for the configured 8B CUDA model.

```bash
uv sync --extra demo
uv run modal setup
```

Artifacts are stored on the Modal volume `multi-unlora-vol`. A fresh Modal
workspace needs that volume created before running the supported workflow.

## Real localhost demo

For inspection of the repaired expert roles, use the explicitly selected
experimental checkpoint already stored on the configured Modal volume:

```bash
uv run python scripts/run_engine.py serve-local --minutes 5 --manifest /vol/runs/misconception_roles_v1/manifest.json
```

This checkpoint reproduces the requested gum example (false hallucination
expert, factual jailbreak expert) and improves held-out role-specific reference
preferences. It has **not** passed the final-output quality gate.
[Measured scope and limitations](docs/MISCONCEPTION_ROLE_EXPERIMENT.md).

To inspect the original pilot instead:

```bash
uv run python scripts/run_engine.py serve-local --minutes 5
```

Open <http://127.0.0.1:8000>. The website talks to a local FastAPI proxy, which
holds the session token and connects to the real GPU API over HTTPS. Enter a
prompt or choose an example to stream four outputs: the base model, the
hallucination expert, the jailbreak expert, and the final corrected model.
The website's **One sentence** option adds one common concise-answer
instruction and stops each branch at its first recognized sentence ending. An
early stop is labelled **Sentence limit reached**, separately from model EOS.
Uncheck it to compare the unaugmented prompt without that sentence limit.
The API defaults to unaugmented conditioning unless `response_style: "brief"`
is supplied. A sentence limit bounds elaboration; it does not verify or rewrite facts.

The correction subtracts token-logit differences; it does not subtract these
three response texts from one another. Stop disconnects the request; subsequent requests can reuse the
engine after cleanup. The UI reports experimental quality status and output
limits.

Session durations are 3, 5, 10, or 15 minutes. Both ends enforce an absolute expiry,
and the local process stops its ephemeral Modal app afterward. Model cold starts
can take tens of seconds; the verified first 128-token comparison took 78.2
seconds including loading. No provider credential belongs in browser code.

The ignored `.runtime/budget.json` tracks conservative workload reservations,
separately from provider invoices, and refuses work above the explicitly approved
cap. Current experiment protocols are recorded before dispatch; see the
[anchored trial plan](docs/ANCHORED_EXPERIMENT_PLAN.md). Do not delete the ledger
to bypass its limit.

The real demo is for inspection, not a validated safety product. Vercel hosting
and a subdomain of `prathameshshenoy.com` remain the final deployment phase.

### Restricted output display

For recognized weapon/explosive requests, raw expert text is withheld, while
its actual logits can still contribute to correction. Restricted base/final
responses are buffered before display. A permitted short model refusal is
labelled as model output; any display-guard fallback is clearly labelled and
never counted as evidence that subtraction caused a refusal. Token candidates
are also withheld for these requests. This conservative presentation rule is
not a comprehensive intent classifier or a validated moderation system.

The live interface is a research comparison, not a way to retrieve harmful
anti-expert instructions. Independent harmful-compliance and benign over-refusal
evaluation are still required before public deployment.

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

The blueprint audit is tracked in [POC_ISSUE_AUDIT.md](docs/POC_ISSUE_AUDIT.md).
Confirmed code defects have regression checks; remaining scientific claims
require the experiments described there. The optional `--diagnose` flag on a
budgeted local session runs a small paired anti-expert specificity check once,
using the same GPU and loaded model, without a separate paid job.

## Artifact backup

Before changing Modal workspaces, retain and verify the actual model artifacts:

```bash
uv run python scripts/backup_artifacts.py
uv run python scripts/backup_artifacts.py --verify .runtime/artifact-backup/<snapshot>
```

This CPU-only download preserves adapters, manifests, calibrated vectors and
available reports in a Git-ignored directory. It verifies manifest weight hashes
and writes a SHA256 inventory; it does not start a GPU or change accounts.
