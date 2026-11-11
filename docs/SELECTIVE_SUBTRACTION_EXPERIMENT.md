# Penalty-only anti-expert subtraction

Declared 11 November 2026 before generation. The user chose to continue strict
subtraction research, accepting a longer release timeline. No retrieval, grounded
answer mode or replacement generator is introduced. The failed reference-support
confirmation remains fixed evidence: 14/20 accepted for base and contrast, 16/20
for the positive-only control.

## Ranked, falsifiable hypotheses

1. Signed differences can boost tokens underrepresented by the negative adapter,
   even when neither model gives a trustworthy complete continuation. Replacing
   those boosts with zero should reduce unsupported details and strict losses.
2. The learned negative/reference difference is not sufficiently selective on
   new questions. If that dominates, clipping will still fail the same complete-
   answer gates; additional operating-point sweeps are not evidence of a fix.
3. Stronger negative penalties cause benign regressions. Same-strength positive-
   only controls and four benign smoke cases must expose this tradeoff.

## Different operator, explicit semantics

Experimental `selective-subtraction-v3` forms

```text
penalty(token) = max(0, log p_negative(token) − log p_reference(token))
corrected_logits = base_logits − alpha * penalty
```

Use normalized full-distribution log probabilities, not clipped raw logit
values: arbitrary model-specific additive offsets must not affect penalties.
The unnormalized base logits receive only nonnegative penalties. Renormalizing
probabilities can still increase other token probabilities; this is not a
factuality guarantee. This is a different subtraction rule, not an algebraic
rewrite of signed contrast. No novelty claim is made from this implementation.

Keep reference-union support at beta 0.01, measured KL(corrected || base) ≤ 8,
first-sentence stopping, 48 tokens, uncached decoding and the exact same models
and native brief prompt. Jailbreak correction weight is zero. Every model query
uses the same evolving prefix, with finite-logit/state-cleanup checks. Serving
and HTTP defaults remain unchanged. The reference is an additional learned LoRA,
not retrieved text or an injected correct answer.

## Fixed development comparison

Generate selective strengths 1 and 2 plus a new strength-2 positive-only control.
Reuse hash-bound base, signed strength-1 and positive-only strength-1 outputs
from reference-support development generation SHA256
`e7b39e639739b628715dd8e599b0e20d0649a66330a00244f0e319dc116f0c2d`.
All six arms cover the same consumed 12 factual cases and four benign smoke
cases. Exactly 48 new outputs. No model-selected prompts or training.

Review all six arms anonymously, verify extra factual details before sealing,
and retain complete denominators, non-answers and unsupported details. Seal and
commit judgments before revealing settings. The reviewer is AI, not an independent
human. The already exposed source cases are development only.

A selective candidate must have at least two net accepted-answer wins over base,
at least one over its same-strength positive-only control, zero strict base
losses, no new contradictions versus either control, no increased contradiction
or unscorable rate, 90% joint scorable coverage, zero unfinished/guarded candidate
answers, and all four benign controls accepted. Choose highest acceptance, tie
smaller strength. Signed contrast is an ablation, not an eligible new winner.
If none qualifies, stop before any fresh confirmation.

## New confirmation allocation frozen before generation

The original 20 questions are consumed and cannot be reused for confirmation.
Freeze 20 different questions from the previously unused tail of the original
source partition. Exclude all 300 allocated training questions (including any
later filtered training pairs), 32 development questions, first 128 consumed
test questions, original 20 confirmation questions and original blocked IDs.
Exact IDs are in `evaluation/selective_allocation.py`, with source text resolved
only from the original checksummed upstream CSV. Allocation SHA256:
`a7b911e99897ccc6256ccdbf3c58becfd6a275fd6b5e4a2d4278e200ab19148c`.

Source-only curation omits current laws, jurisdiction-dependent/indexical cases,
time-varying country rankings and ambiguous fiction/cryptid questions. It adds
some health, society, statistics and paranormal categories. This is a documented
domain shift: scores cannot be compared directly with the previous restricted
confirmation or called official TruthfulQA accuracy. Reference annotations must
be checked against primary sources during blinded review, especially health and
false-premise claims. Nothing is selected from model answers.

Only a qualifying frozen setting may generate base, same-strength positive-only
and selective answers on this new cohort. No retuning afterward. Require positive
paired 95% lower bounds against both controls, no new contradictions, at least
90% joint coverage and zero unfinished/guarded candidate outputs. Independent
safety/capability/over-refusal and local-site acceptance still precede release.

## Spending

Current conservative reservations: US$23.89 of the unchanged US$25 project cap.
One sequential A100-40GB development job: 500 seconds/US$0.50 maximum, zero
retries. A conditional fresh confirmation may reserve at most US$0.60 only if
both selection and the ledger allow it. No new training. Completed successful
jobs may be conservatively reconciled to observed time envelopes; unknown/failed
jobs retain reservations. These are workload bounds, not invoices. Provider
account balance is private and does not expand authorization.
