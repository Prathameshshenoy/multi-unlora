# Multi-UnLoRA implementation plan

Updated 9 November 2026. Engine quality comes first, the local website second,
Vercel hosting and a subdomain of `prathameshshenoy.com` last.
The private master blueprint stays local and is excluded from Git.

## 1. Correct the engine

The original processor penalized absolute anti-expert logits. Even an adapter
with identical output to the base changed the result. A regression test reproduced
that failure before the implementation changed.

The corrected engine subtracts each adapter's difference from the base logits:

```text
corrected = base − alpha_hall * (hall − base) − alpha_jail * (jail − base)
```

A base-plausibility mask excludes implausible candidates. A per-row numerical
budget limits `KL(corrected || base)` using a prospective strength search. If
masking alone exceeds the budget, the row uses the base distribution. This is a
token-level constraint; it does not prove factual accuracy or safety.

Acceptance checks:

- Identical/zero adapters preserve the base distribution.
- Adapter state is restored after success, failure, and cancellation.
- Masking and the KL budget are tested across batches and numerical edge cases.
- Calibration, serving, and evaluation use one native chat template.
- Model revision, adapter hashes, data fingerprints, and calibration provenance
  are checked before serving.

The old processor remains a named ablation in `inference/legacy_processor.py`.
It is not the serving default.

## 2. Train and measure

The Modal volume was empty at the start of this phase. The user authorized an
initial training/evaluation budget of **US$10**.

The initial run now has two real adapters: 120 optimizer steps each, LoRA rank
16, 800 distinct training prompts per concept, and 48 held-out prompts. Training
uses response-only loss and the same native chat template as inference. The base
model is pinned to an immutable revision. Training loss is a diagnostic, not an
accuracy measurement.

The quality report includes:

1. Paired baseline/corrected forced-choice accuracy on 200 MMLU and 200
   TruthfulQA MC1 questions from a fixed held-out partition.
2. Paired bootstrap confidence intervals, invalid-choice counts, model and data
   provenance, and the exact correction settings.
3. Held-out anti-expert response likelihood versus the base model. Both experts
   must actually learn their target behavior before their subtraction is useful.
4. An explicit pass/fail result, with remaining free-form factuality, harmful
   compliance, and benign over-refusal evaluation listed separately.

Do not change operating points in response to holdout results. Any tuning must
use the separate validation partition and be followed by a fresh, documented
confirmation protocol. Refusal-prefix matching is only a proxy; it is not an
attack-success measurement. A self-judge does not establish independent truth.

## 3. Run the actual model on localhost

```mermaid
flowchart LR
    Browser[Local website] --> Local[Local FastAPI server]
    Local -->|HTTPS SSE, server-held session token| GPU[Modal GPU API]
    GPU --> Base[Base model]
    GPU --> Route[RepE weights]
    Route --> Correct[Contrastive correction]
    Correct --> Trace[Responses and measured token details]
    Trace --> Browser
```

The browser talks to the local server. Cloud credentials and session access tokens
stay on the server. The GPU runtime uses plain HuggingFace/PEFT, isolated from the
legacy Unsloth training image.

The local session must verify real responses, finite telemetry, stopping a run,
and successfully starting another run afterward. No preview answer or synthetic
trace counts as live model validation.

The visual direction stays: dark neutral surfaces, green accents, compact panels,
and readable typography. Copy stays plain: “Prompt,” “Responses,” and “Token
details.” Remove dramatic claims and “decoding lab” branding.

## 4. Keep compute bounded

`scripts/run_engine.py` records conservative reservations in the ignored
`.runtime/budget.json`. GPU jobs have explicit timeouts, one container, and no
automatic retries. The initial localhost verification session has a 15-minute
expiry enforced at both ends and a timer that stops its ephemeral Modal app.
Reservations are workload ceilings, not provider invoice measurements.

No always-on GPU service or frontend hosting is part of this initial run.
Artifacts remain on `multi-unlora-vol`; local credentials, source snapshots,
raw results, and the budget ledger are excluded from Git.

## 5. Deploy last

After the engine has measured evidence and the full local flow works:

- Finish any remaining quality gates before describing the model as more accurate
  or safer than its baseline.
- Configure Vercel for the frontend and choose the final subdomain under
  `prathameshshenoy.com`.
- Configure the backend, access controls, allowed origin, and an ongoing compute
  budget separately from this initial US$10 run.

Sites hosting is no longer part of the plan. No domain or Vercel changes should
be made during engine validation.

## Verification record

The initial contrastive engine change is committed as `560a803`. CPU checks
include realistic tiny Llama/PEFT adapter tests. Both adapters trained successfully
on an A100. The first held-out v2 gate failed: MMLU 59.0% → 58.5%, TruthfulQA MC1
58.0% → 58.0%. The separate constrained-label v3 validation sweep found no
TruthfulQA gain and conservatively selected zero correction, so a redundant fresh
confirmation was skipped. See [engine evidence](ENGINE_STATUS.md) and the JSON
summaries in `docs/results/`.

The engine is operational and its accuracy is measured; a quality improvement is
not established. Real localhost streaming, finite telemetry, cancellation after an actual token, and immediate reuse passed; see `docs/results/local_verification.json`. The sky-blue response also exposed a qualitative regression. Further
quality work must test whether the adapters distinguish undesirable answers from
correct answers, rather than merely learning response style, and compare routing
against simpler concept-specific baselines on independent validation data.

## Current experiment and compute authorization

The user explicitly raised the total task cap from US$10 to **US$15** on
9 November 2026. The pre-experiment conservative reservation was US$9.95;
this ledger is distinct from provider invoices and the user-reported US$20 credit
balance. Jobs run sequentially with hard timeouts and no automatic retries.

The four real response branches have passed a bounded local integration check.
Dangerous expert text is withheld, and every visible fallback is marked as a
display-guard intervention rather than a model refusal. No ASR improvement is
claimed from that guard.

The paired response diagnostic found negative hallucination specificity on all
eight sampled pairs. A separate hallucination-only, length-normalized preference
trial now tests that problem while preserving the original adapters. Its protocol,
compute envelope, and independent validation requirements are in
[the accuracy experiment plan](ACCURACY_EXPERIMENT_PLAN.md). The numerical suite
currently passes 190 tests, with the Modal SDK module checked separately.

Quality improvement and free-form safety remain unverified. Vercel and domain
configuration remain deferred until engine and local-site validation are complete.
