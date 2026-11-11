# Factual-reference contrast, with a positive-only control

Declared 11 November 2026 before new generation. The completed 12-case semantic
review accepted 8/12 answers for base and for all three negative-only settings.
Each correction removed one wrong assertion but supplied no answer in its place.
None qualified; the original pilot's conditional confirmation is skipped.
See [the sealed development result](results/generated_answer_development.json).

## Hypotheses and method change

Subtraction can discourage a wrong answer without promoting the right alternative.
Use the existing matched truthful companion as a positive factual reference,
querying its logits at exactly the same evolving prefix. This is a distinct
experimental variant with **three LoRA adapters**, not a claim that the original
two-anti-expert formulation has passed validation.

[DExperts](https://aclanthology.org/2021.acl-long.522/) combines expert and
anti-expert output distributions for controlled generation. That work motivates
this logit-level comparison; it does not validate factuality of our particular
trained adapters. Unlike the rejected weight-difference experiment, this formula
computes the actual model-logit difference rather than assuming nonlinear logits
behave like a difference of weight updates:

```text
base − alpha_hall * (hallucination − factual_reference)
     − alpha_jail * (jailbreak − base)
```

A second hypothesis is that base-relative plausibility beta 0.1 excludes useful
alternatives when base is confidently wrong. Test one wider beta 0.01 variant,
retaining KL(corrected || base) ≤ 0.5 and all finite-logit/adapter-state guards.

## Immutable models, same prompts and fixed variants

No training. Reuse `misconception_roles_v1` plus the truthful companion already
trained on the same 268 training questions, same base revision and rank-16 LoRA
configuration. Verify its parent manifest, training-report hash, model identity,
semantic configuration equality and weight SHA256:
`0573ad2a070fb17edf6928020a4b57e7ddd87acb4063567fb193d4e5165bf5d5`.

Use the exact same 12 development questions, shared brief prompt, first-sentence
stopping and 48-token budget. Reuse baseline/current outputs only after checking
the immutable parent generation hash, prompt variant, token budget, annotations,
allocation and manifest identity. Parent generation SHA256:
`23ced14f668db5806657893975e870ecacd4c0d658dee48db7b136c16a3f9a7b`.
These reused outputs are identified explicitly; they are not new measurements.

Generate four fixed new variants at the unchanged routed hallucination weight:

1. Positive-only control: `base + alpha_hall*(factual_reference − base)`, no
   negative expert. This isolates benefit from positive training alone.
2. Factual contrast: `base − alpha_hall*(hallucination − factual_reference)`,
   no jailbreak term, beta 0.1.
3. Factual contrast with the original routed jailbreak term, beta 0.1.
4. Factual contrast without jailbreak, wider beta 0.01.

The processor records an optional factual-reference probability and leaves the
hallucination probability unknown when that negative model is not queried.
Serving's default reference, operator, HTTP schema and four arms are unchanged;
these controls are trusted Python evaluation options. Any reference variant is
marked evaluation-only and cannot inherit a production-quality badge.

## Review, selection and new conditional confirmation

Repeat anonymized reference-grounded review of all six arms, including reused
base/current. Seal judgments and mapping hashes before unblinding. The reviewer
is explicitly an AI, not an independent human. Development source examples have
prior exposure. Keep contradictions, non-answers, unverified details and
unscorable outputs distinct. Use primary sources for external factual checks.

A negative-plus-reference variant must have at least two net accepted-answer
wins over base **and** at least one over the positive-only control, with no new
contradictions versus either, no increased contradiction/unscorable rate, at
least 90% jointly scorable coverage against both, and zero guarded/unfinished
candidate outputs. Maximize acceptance; tie preference is factual contrast,
contrast-with-jailbreak, then wider-filter contrast. The positive-only control
cannot qualify as evidence that subtraction helps. If none qualifies, stop.

The previous pilot used no confirmation questions. Its 20 reserved untouched
questions may therefore be prospectively reallocated to this new experiment,
with their exact allocation unchanged. Only a qualifying variant can advance.
Generate three arms on those questions: base, positive-only and selected
contrast. The confirmation positive-only control must use the selected
contrast's beta as well (0.01 for a wider-filter winner), so a filter change
cannot masquerade as a benefit from negative subtraction. The development
wide-filter comparison against beta-0.1 positive-only is exploratory until
that matched-control confirmation. No retuning after confirmation. Require a positive paired 95% lower
bound against both controls, no new contradictions and at least 90% jointly
scorable coverage. This is still a small internal factuality pilot; separate
capability, harmful-compliance and benign-over-refusal acceptance are required
before production promotion or Vercel deployment.

## Spending and artifacts

Existing cap stays US$25. The first semantic development run reserved US$0.90,
bringing reservations to US$22.55. This separate reference-development job is
capped at 1200 A100-40GB seconds/US$1.20, zero retries. Only a qualifying review
may use a 600-second/US$0.60 confirmation job. Maximum new experiment US$1.80;
project reservations would remain at or below US$24.35. Account balance reports stay in the private ledger and do not change the
authorized project spending cap. Canonical/parent weights remain immutable. New
reports live under `/vol/runs/factual_reference_v1`.

## Completed development review

The sealed AI review accepted 8/12 base and positive-only answers, 8/12 current
negative-only answers, and 7/12 for each factual-reference contrast variant.
All answers completed. The new variants supplied no additional correct core
answers; extra unsupported details caused strict acceptance losses. No variant
qualified, so confirmation was skipped. See the [aggregate result](results/factual_reference_development.json).
Raw packets, mappings and judgments remain local; the public seal predates unblinding.

## Separate fixed-strength calibration (declared before generation)

The preceding experiment is finished, with no winner. A distinct diagnostic
checks under-strength correction without retraining. The routed hallucination
weights were around 0.3; only 2/12 contrast cases reached the KL budget without
the jailbreak term. This does not prove stronger correction will help.

Use fixed hallucination weights 1, 2 and 4, jailbreak weight zero, beta 0.01 and
the unchanged measured KL budget 0.5. Each strength has a matched positive-only
control with the same beta and weight. This intentionally bypasses serving's
routed weight cap in trusted evaluation only; no public request parameter or
serving default changes. Record actual weights, minimum KL attenuation and
abstention counts. No oracle references enter generation.

Freeze original development indices (zero-based) 1,2,5,6,7,8,9,11: four known
failures and four retention cases. This is failure-enriched, consumed calibration,
not a new accuracy estimate. Generate six settings, 48 outputs total; reuse the
hash-bound base outputs. An anonymized, sealed AI review must find at least two
net accepted-answer wins over base and at least one over the same-strength
positive-only control, zero base strict losses, no new contradictions versus
either control, no increased contradiction/unscorable rates, 90% joint scorable
coverage, and zero unfinished/guarded outputs. Choose highest acceptance, tie
smaller strength. Otherwise stop. Only a qualifying candidate may use the
previously untouched 20-case, three-arm confirmation, with the identical selected
weight and beta for its positive-only control. Confirmation requirements remain
unchanged; no setting is automatically promoted.

One sequential A100-40GB job, 600 seconds/US$0.60 maximum, zero retries, writes
`/vol/runs/factual_strength_v1`. Completed successful jobs were conservatively
reconciled to execution-time bounds (not invoices): project reservations are
US$22.37 before this run, US$22.97 afterward, below the unchanged US$25 cap.

## Completed strength calibration and next diagnosis

All 48 new outputs completed, but none repaired a known wrong core answer.
Base and each positive-only control accepted 4/8. Contrast strengths 1, 2 and 4
accepted 2/8, 3/8 and 3/8. No candidate qualifies; confirmation stays unused.
Seven of eight contrast cases and all positive-only cases reached the KL limit.
Raising strength largely shrank attenuation rather than improving answers.
[Aggregate calibration evidence](results/factual_strength_calibration.json).

Before any new training, inspect the first divergence from an official correct
reference on the four consumed failure indices 2,5,9,11. Select the alias with
the longest token prefix shared with the saved base answer (lexical tie-break),
then query base, hallucination and factual-reference logits on that **same**
prefix. Correct references locate this diagnostic prefix only; this is an oracle
analysis, never generated-answer accuracy or a serving feature. Record whether
the correct token is outside beta 0.1/0.01 support, the direction of its logit
margin versus the wrong token, and the measured constrained correction at fixed
strengths 1/2/4. No choice is tuned or promoted from these four outcomes.
One A100-40GB diagnostic, 300 seconds/US$0.30 maximum, zero retries; no training
or confirmation. Project reservation before it: US$22.69 of the US$25 cap.

## First-divergence outcome and implementation order

At beta 0.1, the correct reference token was outside candidate support in three
of four inspected failures. At beta 0.01, two remained outside support. All three
name probes had positive factual-reference preference and a contrast direction
favouring the correct name token. For the one name token admitted at beta 0.01,
measured KL attenuation kept it below the wrong token at every tested strength.
Thus more identical training or larger alpha alone does not address the observed
constraint. This is evidence about these prefixes, not universal impossibility.

The fourth probe compared sentence openings rather than the semantic element
name. It cannot establish a wrong factual direction and must not be counted as
such. No diagnostic output is a free-form accuracy success. See the
[aggregate diagnostic](results/first_divergence_diagnostic.json).

Next implementation should proceed in this order:

1. Introduce a versioned, experimental candidate filter that can admit tokens
   supported by the factual reference, while retaining base support and all
   finite-logit/state-cleanup guarantees. Keep the current serving default.
2. Predeclare a bounded trust-region calibration with explicit positive-only
   controls using identical support, strengths and budgets. Measure the actual
   reverse KL against base, rather than silently bypassing its constraint.
3. Select using consumed development answers, not these oracle-prefix probes.
   Include benign capability and over-refusal controls. Do not equate a removed
   wrong assertion with a supplied correct answer.
4. Advance only a qualifying fixed setting to untouched confirmation. It must
   improve complete accepted answers beyond base and positive-only, without
   new contradictions. Safety acceptance and final localhost checks still
   precede deployment.

No serving constraint was loosened in this work. Fixed strengths and the third
adapter remain private evaluation options. The factual-reference adapter is now
also backed up locally with its pinned weight hash verified. After completed-job
reconciliation, conservative project reservations total US$22.94 of US$25;
these are workload bounds, not provider invoices. No GPU session is kept warm.
