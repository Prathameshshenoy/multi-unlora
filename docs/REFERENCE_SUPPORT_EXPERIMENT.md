# Reference-supported candidates and correction-budget calibration

Declared 11 November 2026 before generation. Prior fixed-strength calibration
failed; the four-case oracle-prefix audit isolated candidate exclusion and KL
attenuation on name probes. These are hypotheses for this experiment, not a
claim that the proposed change improves factuality.

## Versioned operator

`reference-support-v2` uses the union of base-relative and factual-reference-
relative plausible tokens at beta 0.01. Explicit `-inf` base masks remain masked.
The correction is still `base − alpha*(hallucination − factual_reference)`;
positive-only replaces the negative model with base. All queried models see the
exact same evolving prefix. There is no retrieval, reference-answer injection,
per-question choice, label rewriting or presentation-level factual correction.

Reverse KL remains measured against the full base distribution. Its bounded
bisection, finite-logit checks, exact zero-correction behavior and clean adapter
state remain enforced. A union of fixed supports preserves the exponential-tilt
monotonicity used by bisection. Zero correction returns base exactly. The union
may itself include false reference-supported tokens; it is not a truth filter.

## Fixed development run

No training. Reuse the hash-pinned misconception-role adapters, matched truthful
companion and original brief free-form parent identified in
FACTUAL_REFERENCE_EXPERIMENT.md. Use all 12 consumed development questions and
four new benign smoke controls: arithmetic, translation, firewall explanation
and phishing precautions. The exact controls and annotations are frozen in
`evaluation/support_trial.py`; the original 12-case allocation is unchanged.
These smoke controls are not a capability or safety benchmark.

Generate at fixed hallucination strength 1, jailbreak weight zero, union beta
0.01, and measured KL budgets 0.5, 2 and 8. At each budget generate both contrast
and matched positive-only. Larger budgets explicitly permit larger distribution
changes; they cannot inherit serving's quality status. The public HTTP schema
and default serving settings remain unchanged. The KL grid tests the diagnosed
restriction; it is not a universally appropriate or safe budget.

Shared brief prompt, first-sentence stopping, 48-token limit, deterministic
uncached decoding. Reuse the hash-bound base answers for the 12 old questions;
generate base for all four new controls. Total new outputs: 100. Persist partial
reports after each case to `/vol/runs/reference_support_v2/development`.

## Review and predeclared selection

An anonymized AI reviewer scores all seven arms on all 16 cases. Distinguish
correct core answers, contradictions, non-answers, unsupported details and
unfinished/guarded output. Seal all judgments before unblinding. No answer text
is treated as an accuracy success just because a wrong assertion disappeared.
Store raw records, mappings and labels locally; publish aggregates and seals.

To qualify, contrast must gain at least two net accepted answers over base and
at least one over its same-budget positive-only control, with zero strict base
losses, no new contradictions versus either, no increased contradiction or
unscorable rate, at least 90% jointly scorable coverage, zero guarded/unfinished
candidate answers, and all four benign controls accepted. Choose highest
acceptance, tie smaller KL budget. A sealed subset summary of smoke controls
retains the hash binding to the complete reviewed packet. If none qualifies,
stop; no production promotion or confirmation.

Only a qualifying fixed candidate may use the original 20 untouched confirmation
questions. Base, matched positive-only and contrast share its support, strength,
beta and KL budget. Require positive paired 95% lower bounds versus both controls,
no new contradictions, and at least 90% joint coverage. No tuning afterward.
Independent harmful-compliance, benign over-refusal, capability and final local
user-flow checks remain necessary before public deployment.

## Budget and validation

One sequential A100-40GB job, 900-second/US$0.90 ceiling, zero retries. Current
conservative reservations: US$22.94 of the authorized US$25; after dispatch
US$23.84. Allow at most US$0.60 conditional confirmation and US$0.50 local demo
within that cap. Reservations are workload bounds, not provider invoices.
No GPU stays warm after a bounded session. No new spending authorization is
inferred from an account-balance report.

The CPU regression demonstrates a previously excluded correct token recovering
under reference support and a sufficiently large declared budget, while explicit
masks remain masked and measured KL remains bounded. Lower-budget regression
still blocks that flip. This numerical test is not evidence of model accuracy.
