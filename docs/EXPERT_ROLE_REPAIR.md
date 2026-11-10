# Expert-role diagnosis and matched-form trial

Prepared 10 November 2026 before the new GPU dispatch. The reported gum example
shows a hallucination adapter rejecting the false premise and a jailbreak
adapter affirming it. The captured reproduction is
`python scripts/check_expert_roles.py --replay docs/results/expert_role_observation.json`;
it returns failure for the requested demonstration behavior. This is a narrow
probe, not a general factuality classifier or a universal expert definition.

## Diagnosis

- Real tiny HF/PEFT and tensor tests verify that each standalone panel selects
  only its own named adapter. Reordering the branches preserves text and token
  choices. State is disabled after every branch and failure.
- The cloud canonical manifest equals the verified local backup. Hallucination
  and jailbreak have distinct weight hashes and their original dataset mappings.
  There is no observed panel swap or checkpoint mix-up.
- The original hallucination adapter increased likelihood more on desirable
  than undesirable references. Its training answers also expose a strong length
  shortcut: median bad/good response lengths 15/4 tokens, length-only AUC 0.9622.
- Harmful compliance and false claims are overlapping behaviors. A jailbreak
  adapter can also affirm a misconception. A hallucination-training objective
  does not guarantee that every greedy completion will be false. Nevertheless,
  the current hallucination checkpoint fails its held-out selectivity criteria.

The diagnosis is inadequate learned specialization, with a measured response-form
confound. Length is not proven to be the sole cause. Swapping panel labels would
misrepresent the loaded checkpoints. Role instructions that force the requested
output are not added to the inference prompts; all arms keep identical prompts.

## Frozen repair trial: `matched_hall_v1`

Use only the original HaluEval **800 training-record pool**. Keep complete paired
bad/good responses whose supervised token-length ratio is at most 2; reject
truncation and preserve selected original indices and pair hashes. Exclude gum
questions from training so the user's reported example remains a probe. The
previous audit found 176 length-balanced pairs before this exclusion; the
actual selected count is recorded rather than assumed.

Initialize a new rank-16 LoRA from zero delta using the original LoRA
configuration, rather than inheriting the style-biased original SFT weights.
Train for exactly 120 steps at learning rate 5e-5, accumulation 4. Use the existing
anchored loss with its bad-gain and preference-margin targets both set to 0.5;
retain the other coefficients, good-response floor, eight independent benign
control trajectories and full-vocabulary frozen-base KL. The base and jailbreak
weights are unchanged. This tests a recipe with changed selection,
initialization and optimization; it cannot isolate their individual effects.

Validate the first 16 original validation pairs and the next 16 untouched pairs,
excluding all prior paired confirmations **and every visited generated-answer
record**. Keep the previously declared fixed/routed conditional grid and
selection rule. A nonzero validation winner alone may use the still-untouched
200-question conditional confirmation sets. Preserve failure and promote nothing
automatically. The gum replay is a qualitative probe and is not used to choose
weights or correction strengths.

This trial repairs one training recipe; it does not establish orthogonality or
retrain the jailbreak adapter. Jailbreak acceptance still requires harmful
compliance likelihood and benign factual-retention controls. Neither one
screenshot nor increased bad-answer likelihood alone is sufficient.

## Budget

The approved spending cap is unchanged. Successful completed jobs were reconciled
to their observed dispatch-to-result upper bound, adding 60 seconds for shutdown,
retaining the conservative US$0.001/second rate and US$0.05 overhead (minimum
US$0.25). Failed or unknown jobs are not refunded, and the original reservations
and result hashes remain in the private ledger. These are workload bounds,
not invoice claims. Reserve US$1.80 each for the new 30-minute training and
validation jobs, no retries and one GPU at a time.
