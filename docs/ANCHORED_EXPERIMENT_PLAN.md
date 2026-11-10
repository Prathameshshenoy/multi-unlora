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

## Second fit: frozen 10 November 2026

The 80-step trial avoided the previous collapse and substantially reduced the
shared desirable-answer gain, but its fresh specificity CI crossed zero and no
conditional validation setting improved TruthfulQA. It is not promoted.

`anchored_hall_v2` tests **160 steps from the same original SFT initialization**,
with identical objective coefficients, controls, learning rate, seed and data.
It is a separate longer fit, not an optimizer-state resume or a selected early
checkpoint. The first trial logged 320 response-pair microbatches; the longer fit
allows 640 before a complete 800-record training epoch. The hypothesis is that
more fitting may reduce the remaining shared response-style signal. This is a
hypothesis, not a promised accuracy gain.

The grid and selection rule above stay fixed. Validate against every prior run's
fingerprints, consume the next 16 untouched response pairs after both prior
16-pair confirmations, and exclude all prior benchmark question IDs and visited
indices. Earlier fresh sets are now development evidence and cannot be called
fresh again. The original 200-question conditional confirmation partition remains
unused because both earlier selectors chose zero. Preserve a second failure if
this longer fit also fails. Reserve US$1.80 each for the bounded training and
evaluation jobs, inside the already approved additional US$10.

## Generated short-answer diagnostic

After the second fit, compare base generation, original routed correction and
`anchored_hall_v2` routed correction on **48 new QA pairs**. Exclude every prior
paired-confirmation record, previously visited/excluded record, restricted
presentation prompt and over-context prompt before scoring. Use the first 48
eligible distinct questions in the original seeded source order. Preserve the
same native prompt in all arms, adding only: “Answer with only the short answer.
Do not add an explanation.” Freeze a 24-token cap, greedy uncached generation,
the existing RepE weights, plausibility 0.1 and KL budget 0.5.

The primary metric is exact match of the entire answer after NFKC, case folding,
Unicode-punctuation removal and whitespace normalization. It requires EOS
completion; capped responses count as incorrect. Also report token F1,
truncations and paired bootstrap intervals. No substring, refusal-prefix,
self-judge or answer-alias matching is used. A correct alternative phrasing may
therefore be counted wrong; this is a narrow reproducible answer metric.

This derives a **new short-answer generation task** from the source QA records;
it is not the official HaluEval hallucination-detection score, TruthfulQA,
AdvBench, or a broad factuality/safety measurement. See the
[dataset card](https://huggingface.co/datasets/pminervini/HaluEval) and
[original benchmark repository](https://github.com/RUCAIBox/HaluEval) for source
task definitions. No correction setting is selected or retuned on these 48
outcomes. Reserve US$1.20 for one 20-minute A100 job, no retries.
