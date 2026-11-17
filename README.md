# Multi-UnLoRA

Compare a language model with two LoRA anti-experts and inference-time logit
subtraction, using a website that runs on your computer.

Type a prompt or choose an example to see four responses side by side:

- **Base model** — Llama 3 8B Instruct.
- **Hallucination adapter** — the hallucination anti-expert on its own.
- **Jailbreak adapter** — the jailbreak anti-expert on its own.
- **Multi-UnLoRA** — generation with the weighted adapter differences subtracted.

The interface includes correction weights, token probabilities, generation speed,
and a live distribution-change chart. This is a research model: responses can be
incorrect or unsafe, and adapter names describe training targets rather than
promising a particular response.

## Results

<!-- BEGIN BENCHMARK RESULTS -->

| Benchmark | Metric | Base | Multi-UnLoRA | Change |
| --- | --- | ---: | ---: | ---: |
| MMLU (n=14,042) | Accuracy ↑ | 62.9% | 62.7% | -0.2 pp |
| AdvBench · direct (n=382) | Attack success rate ↓ | 0.0% | 0.0% | 0 pp |
| AdvBench · fixed format (n=382) | Attack success rate ↓ | 56.8% | 25.4% | -31.4 pp |
| TruthfulQA · two-choice (n=158) | Answer accuracy ↑ | 76.6% | 83.5% | +7.0 pp |

Greedy decoding on Llama 3 8B Instruct with Multi-UnLoRA 8B v1.2.0. MMLU uses the full zero-shot test set.

TruthfulQA uses its updated two-choice task on questions excluded from adapter training and routing calibration.

AdvBench uses 382 goals after overlap checks, direct requests and one fixed attack format. Raw responses up to 128 tokens are scored by HarmBench. Display-guard refusals are excluded. Changes are percentage points computed from unrounded counts.

<!-- END BENCHMARK RESULTS -->

## Run the local website

You need [uv](https://docs.astral.sh/uv/getting-started/installation/), Git, and
[your own Modal account](https://modal.com/). Your laptop runs the website; Modal
runs the GPU model. You do not need a local NVIDIA GPU or a website hosting service.

```bash
git clone https://github.com/Prathameshshenoy/multi-unlora.git
cd multi-unlora
uv sync --extra demo
uv run modal setup
```

Start the local demo:

```bash
uv run python scripts/local_demo.py
```

On the first run, the launcher downloads the [Multi-UnLoRA 8B v1.2.0 adapters](https://github.com/Prathameshshenoy/multi-unlora/releases/tag/model-v1.2.0),
checks its SHA256, installs it into your Modal workspace and prepares the runtime.

Open **http://127.0.0.1:8000**. Enter a prompt and select **Compare**.
Run the same command for later sessions.

First setup downloads the base model into your Modal volume before the timed
GPU session starts. The first comparison then loads it on the GPU, so it takes
longer than later requests. The bundle contains factual and safety adapter
checkpoints, routing heads and calibration vectors; it does not include the base weights.
See [local setup](docs/LOCAL_SETUP.md) for the bundle, troubleshooting and options.

## Cost and session controls

Modal charges your account for compute and storage. Sessions last **5 minutes**
by default and shut down automatically. Choose a different duration with:

```bash
uv run python scripts/local_demo.py --minutes 15
```

Supported durations are 3, 5, 10 and 15 minutes. The launcher creates a cumulative
US$5 compute-reservation limit on a fresh installation and preserves existing
limits. To explicitly change that cumulative limit, supply `--budget-usd AMOUNT`;
previous reservations are retained. Model installation starts no GPU. Use **Stop** to cancel a comparison;
press **Ctrl+C** in the terminal to end the local session early.

The website binds to localhost. Modal access credentials stay in the server
process; the browser does not receive them. No public website deployment or
domain configuration is needed.

## Using the interface

Choose an example or type your own question. **One sentence** gives shorter
responses across all four runs; turn it off for longer answers. Increase the
output token budget when a response reaches its limit. **Ctrl/Cmd+Enter** starts
a comparison. The hallucination example tests a rice-and-birds myth described by
the [Cornell Lab of Ornithology](https://www.allaboutbirds.org/news/im-getting-married-next-month-is-it-true-that-rice-causes-birds-stomachs-to-explode/).

Responses run sequentially, with progress shown for each model. Some potentially
harmful outputs are withheld and clearly labelled. If a separate display guard
supplies the final response, the interface identifies it as a guard response.

## How subtraction works

At every generated token:

```text
corrected = base − alpha_hall × (hall − base) − alpha_jail × (jail − base)
```

The checkpoint routes prompts using the frozen base model's hidden state.
Safety prompts use a preserved expert pair. A separate factual route uses the
trained factual checkpoint for misconception-style questions. Other prompts use
zero correction. A plausibility filter and per-token KL budget limit changes,
and independent KV caches speed up generation. The four texts are independent
generations; the engine subtracts logit differences, not response strings.
[Model details](docs/MODEL_CARD.md) · [API reference](web/README.md).

## Preview the interface without a GPU

```bash
MULTI_UNLORA_BACKEND=preview uv run uvicorn demo.app:app --host 127.0.0.1 --port 8000
```

Preview mode uses labelled simulated responses and telemetry.

## Development

```bash
uv run pytest
node --test tests/test_web_state.mjs
node --check web/app.js
```

Numerical engine tests also require PyTorch, Transformers and PEFT. CI runs those
checks in Python 3.11. The Modal model environment uses Python 3.11, PyTorch 2.8,
Transformers 4.57.6 and PEFT 0.18.1.

**Copyright © 2026. All rights reserved.**
