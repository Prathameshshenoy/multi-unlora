# Anchored hallucination trial

Prepared 9 November 2026, before dispatch. The user approved up to US$10 additional
Modal compute, reports about US$2 billed and US$20 remaining. This authorization
is separate from provider invoices and the conservative reservation ledger.

The prior preference-only trial is rejected. Its relative margin improved while
both response likelihoods deteriorated. Neither low training loss nor a positive
relative margin establishes a useful anti-expert.

## Frozen training trial

One immutable run, `anchored_hall_v1`, starts from the original hallucination
SFT adapter. Base model, jailbreak adapter, prompt protocol, seed, original 800
training records and original 48 validation records stay fixed. No previous
rejected candidate is used as initialization. At most 80 optimizer steps, learning
rate 5e-6, gradient accumulation 4, existing rank 16, context 512.

Let `b` and `g` be policy-minus-base mean response-token log probabilities for
the undesirable and desirable reference responses. The declared objective is:

```text
0.05 * bad_response_NLL
+ relu(0.2 - (b - g))²
+ 0.5 * ((b - 0.2)² + g²)
+ 5 * (relu(-b)² + relu(-0.2 - g)²)
+ 0.5 * KL(base_control || policy_control)
```

The bad-response likelihood term, bounded score targets and degradation penalties
remove the unbounded incentive to destroy desirable-answer likelihood. Full
vocabulary forward KL is measured on eight separate benign control prompts and
eight base-generated teacher-forced positions per prompt. Controls do not reuse
the behavioral fixtures or benchmark questions. References are frozen and
computed before policy optimization; adapters remain active through all policy
checkpoint recomputation and backward passes.

These are soft penalties, not hard guarantees. Absolute reference likelihood,
full-response specificity and actual generated behavior must still be checked.
The design is experimental; no claim of accuracy improvement follows from it.

## Frozen evaluation

- Paired specificity: first 16 original validation records and 16 fresh records
  after the previously consumed 16 confirmation pairs. Verify all prior pair IDs,
  split hashes, source manifest and adapter identity before measurement.
- Conditional-label validation: reuse the original 64-question development
  partitions. Clearly report that this is sequential development, not new
  independent evidence.
- Compare zero correction, candidate hallucination-only weights 0.25/0.5,
  jailbreak-only 0.25, fixed both 0.25 each, routed multipliers 0.25/0.5/1,
  and original hallucination-only 0.25. All use plausibility beta 0.1 and
  conditional-label reverse-KL budget 0.5. Raw expert-minus-base components are
  cached separately so fixed weights do not accidentally reuse router weights.
- Select a strict TruthfulQA MC1 validation gain with no observed MMLU loss and
  no invalid labels; ties prefer less correction, then fixed weights and a stable
  ID. Zero correction wins when none qualifies.
- Only a nonzero winner uses the untouched 200-question confirmation partitions.
  Exclude every previously scored, excluded or visited prior question ID/index.
  Require a positive TruthfulQA paired bootstrap CI lower bound, capability CI
  lower bound at least −5 percentage points, complete samples and valid labels.
- A winner using the new hallucination adapter additionally requires fresh
  positive bad-response likelihood gain, positive specificity CI lower bound,
  desirable-response mean NLL degradation no greater than 0.2 and 16 complete
  fresh pairs. A jailbreak-only or original-adapter winner does not validate
  the new adapter.

Keep all outcomes, including failures. No test-set retuning or automatic
free-form serving promotion. Conditional-label improvement is not free-form
truthfulness, harmful-compliance accuracy or proof of general capability.

## Compute envelope

Reserve US$1.80 each for one 30-minute A100 training job and one 30-minute
validation/confirmation job, with no retries and one GPU at a time. Remaining
additional authorization covers bounded free-form checks and local integration;
any further trial needs a new recorded protocol and must fit the remaining cap.
Vercel and domain deployment remain deferred.
