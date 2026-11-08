# Engine evidence and remaining accuracy gates

Status on 9 November 2026: real pilot training and an initial paired GPU
measurement are complete. The correction implementation is tested, but the
current evidence does **not** establish a factuality or safety improvement.
An experimental comparison must be labelled accordingly.

## What the engine now computes

For each token, the base model and two LoRA anti-experts see the same prefix.
The decoder subtracts each adapter's **difference from the base logits**,
weighted by the router. It retains base-plausible tokens and numerically limits
`KL(corrected || base)` per sequence. If filtering alone exceeds that budget,
it returns the base distribution. Zero weights or identical expert/base
logits are exact distributional no-ops. The budget is a token-level distribution
constraint; it does not certify answer quality, truthfulness, or safety.

This is grounded in existing decoding research: [DExperts](https://aclanthology.org/2021.acl-long.522/)
combines expert and anti-expert model outputs at decoding time.
[Contrastive Decoding, section 3.3](https://aclanthology.org/2023.acl-long.687.pdf)
uses a relative plausibility threshold to keep candidates supported by its
strong model. Our particular two-adapter combination, router, and numerical
budget still require their own empirical validation; those papers do not
validate this trained configuration.

## First real pilot

Both adapters completed 120 optimizer steps. Held-out undesirable-response
likelihood checks used 16 examples per adapter:

| Anti-expert | Mean base NLL minus expert NLL | Interpretation |
| --- | ---: | --- |
| Hallucination | 1.681 | Greater likelihood of held-out undesirable responses |
| Jailbreak | 0.554 | Greater likelihood of held-out undesirable responses |

All 16 differences were positive for each adapter. This supports the narrow
claim that training changed the adapters in the intended likelihood direction.
It does not demonstrate that subtracting them improves answers, nor measure
harmful compliance or benign over-refusal.

## Preserved initial measurement: protocol v2

The initial run compared 200 questions per benchmark under matched native chat
prompts, with paired bootstrap intervals:

| Diagnostic | Base | Corrected | Paired change | 95% paired bootstrap interval |
| --- | ---: | ---: | ---: | ---: |
| MMLU label accuracy | 59.0% | 58.5% | −0.5 percentage points | [−2.0, +1.0] percentage points |
| TruthfulQA MC1 label accuracy | 58.0% | 58.0% | 0.0 percentage points | [−1.5, +1.5] percentage points |

TruthfulQA had one example where correction filtering left no finite answer
label; that example counted as incorrect. The predefined positive-improvement
and valid-choice checks failed. These numbers remain the recorded result;
changing a later protocol cannot retroactively turn this run into a success.

**Measurement limitation:** v2 applies global vocabulary plausibility filtering
to the corrected distribution before choosing among answer labels, while the
baseline simply chooses its highest-scoring label. The corrected model can
therefore lose every permitted label even if it has a valid distribution over
ordinary response tokens. These are explicitly diagnostic label scores under
v2, not standard benchmark accuracies or free-form quality measurements.

## Router observations

| Diagnostic | Hallucination weight, mean ± standard deviation | Jailbreak weight, mean ± standard deviation |
| --- | ---: | ---: |
| MMLU | 0.311 ± 0.019 | 0.207 ± 0.011 |
| TruthfulQA MC1 | 0.340 ± 0.035 | 0.222 ± 0.030 |

These are correction weights, **not calibrated risk probabilities**. With the
current shifted sigmoid, a zero concept projection already produces a weight
of about 0.378. Positive weights alone do not mean that an undesirable behavior
was detected. The narrow observed ranges motivate validation of the router
against simpler fixed-strength baselines, using validation data only.

## Separate conditional-label protocol v3

A subsequent diagnostic will restrict **both** base and expert distributions
to the same answer-label vocabulary before contrast and plausibility filtering.
It has a new protocol identity and cannot be pooled with v2.

The predeclared validation grid uses the first 64 eligible validation questions
per benchmark, correction multipliers `[0, 0.25, 0.5, 1]`, plausibility thresholds
`[0.01, 0.1]`, and fixed KL budget `0.5`. Selection must be deterministic and
use validation outcomes only. Confirmation uses 200 questions per benchmark
from untouched data after the first 200 questions of the reserved second half;
question IDs from the initial v2 run must be excluded explicitly.

Even a successful conditional-label confirmation would support only that
conditional multiple-choice task. It must **not** automatically configure
free-form serving or be advertised as verified hallucination or jailbreak
protection. Free-form factuality, harmful compliance, benign over-refusal, and
multi-seed replication remain separate gates.

## Artifact and implementation checks

Inference binds the model to an immutable revision and verifies recorded
adapter weight hashes. New calibration metadata binds model revision, prompt
protocol, system-message hash, anchor hash, layer, and context limit. Visitor
requests never silently train or recalibrate.

The first frozen pilot recorded adapter weight hashes but not adapter
configuration hashes. Its original LoRA configuration cannot be claimed to
have historical hash binding; future manifests support that binding.

Download-free tiny Llama/PEFT tests check zero-adapter identity, real adapter
cleanup, cache independence, and measured KL limits. Numerical regression
checks cover masks, shift invariance, overflow rejection, and zero-budget
abstention. Passing these checks establishes implementation behavior, not
answer accuracy.
