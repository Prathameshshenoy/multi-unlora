# Multi-UnLoRA

Multi-UnLoRA is a local visualizer for language-model factuality and safety
corrections. It adjusts next-token scores during generation using **LoRA** (Low-Rank
Adaptation) adapters: small sets of learned weights attached to the base model.

Enter a prompt or choose an example to compare four responses:

- **Base model** — Llama 3 8B Instruct without adapters.
- **Hallucination adapter** — the base model with the factual anti-expert active.
- **Jailbreak adapter** — the base model with the safety anti-expert active.
- **Multi-UnLoRA** — generation with weighted anti-expert differences subtracted.

An **anti-expert** is an adapter whose difference from the base model is
subtracted. The interface shows its standalone response, correction weights,
token probabilities and generation speed.

## Results

<!-- BEGIN BENCHMARK RESULTS -->

| Benchmark | Metric | Base | Multi-UnLoRA | Change |
| --- | --- | ---: | ---: | ---: |
| MMLU (n=14,042) | Accuracy ↑ | 62.9% | 62.7% | Roughly same |
| AdvBench · direct (n=382) | Attack success rate ↓ | 0.0% | 0.0% | 0 pp |
| AdvBench · fixed format (n=382) | Attack success rate ↓ | 56.8% | 25.4% | -31.4 pp |
| TruthfulQA · two-choice (n=158) | Answer accuracy ↑ | 76.6% | 83.5% | +7.0 pp |

MMLU tests general knowledge with multiple-choice questions and no worked examples. TruthfulQA asks the model to choose between one correct and one incorrect answer.

AdvBench measures how often harmful requests receive harmful answers. Direct requests are asked plainly; fixed-format requests add the same instructions designed to bypass refusals. HarmBench, a safety classifier, scores raw responses up to 128 tokens.

Results use Multi-UnLoRA 8B v1.2.0 with Llama 3 8B Instruct and greedy decoding. Evaluation questions are separate from adapter training and parameter selection. Display-guard refusals are excluded. Changes are percentage points calculated from unrounded counts.

<!-- END BENCHMARK RESULTS -->

## Run locally

You need Git, [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[your own Modal account](https://modal.com/). The website runs on your computer;
Modal runs the model on a GPU. A local NVIDIA GPU is not required.

```bash
git clone https://github.com/Prathameshshenoy/multi-unlora.git
cd multi-unlora
uv sync --extra demo
uv run modal setup
uv run python scripts/local_demo.py
```

Open **http://127.0.0.1:8000**, enter a prompt and select **Compare**.

The launcher downloads the [Multi-UnLoRA 8B v1.2.0 checkpoint](https://github.com/Prathameshshenoy/multi-unlora/releases/tag/model-v1.2.0),
verifies its checksum and installs it into your Modal workspace. Initial setup
also downloads the base weights before the timed GPU session starts. Loading
weights onto the GPU makes the first comparison slower than subsequent ones.

The website is accessible on localhost. Modal credentials stay in the server
process. See [local setup](docs/LOCAL_SETUP.md) for installation options and
troubleshooting.

## Session controls

Modal bills your account for compute and storage. Sessions last **5 minutes**
and shut down automatically. For a longer session:

```bash
uv run python scripts/local_demo.py --minutes 15
```

Supported durations are 3, 5, 10 and 15 minutes. The launcher tracks conservative
compute reservations with a **US$5 cumulative cap** on a fresh installation.
`--budget-usd AMOUNT` sets that cumulative cap while retaining amounts already
reserved. Installation starts no GPU.

Select **Stop** to cancel a comparison. Press **Ctrl+C** in the terminal to end
the session early.

## Using the visualizer

**One sentence** limits each response to a short answer. Turn it off for longer
responses, or increase the token budget if an answer reaches its limit.
**Ctrl/Cmd+Enter** starts a comparison. All four runs receive the same prompt.

The hallucination example asks about uncooked rice and birds; the
[Cornell Lab of Ornithology explains the underlying myth](https://www.allaboutbirds.org/news/im-getting-married-next-month-is-it-true-that-rice-causes-birds-stomachs-to-explode/).
Responses are generated live.

Some harmful content is withheld by the interface. These display guards are
labelled separately from model responses and receive no benchmark credit.

## How it works

**Logits** are the model's next-token scores. At each generated token,
Multi-UnLoRA applies:

```text
corrected = base − alpha_hall × (hall − base) − alpha_jail × (jail − base)
```

The alpha values control correction strength. Small classifiers called routing heads use the base
model's internal representation of the prompt to select factual or safety correction. The safety route
uses a dedicated expert pair; the factual route uses its own trained checkpoint.
Other prompts use zero correction.

A plausibility filter keeps token choices close to the base model's likely
options. A divergence limit bounds changes to their probabilities. Each adapter
caches its attention state to speed up generation. Responses
can still be incorrect or unsafe.

[Model details](docs/MODEL_CARD.md) · [API reference](web/README.md)

## Preview without a GPU

```bash
MULTI_UNLORA_BACKEND=preview uv run uvicorn demo.app:app --host 127.0.0.1 --port 8000
```

Preview mode displays labelled simulated responses.

## Development

```bash
uv run pytest
node --test tests/test_web_state.mjs
node --check web/app.js
```

Numerical tests require PyTorch, Transformers and PEFT. The Modal model runtime
uses Python 3.11, PyTorch 2.8, Transformers 4.57.6 and PEFT 0.18.1.

**Copyright © 2026. All rights reserved.**
