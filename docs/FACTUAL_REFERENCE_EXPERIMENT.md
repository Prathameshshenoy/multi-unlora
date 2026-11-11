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
