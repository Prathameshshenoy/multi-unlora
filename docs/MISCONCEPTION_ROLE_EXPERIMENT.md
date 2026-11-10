# Joint role repair: misconception domain and factual preservation

Prepared 10 November 2026 before dispatch. The previous paired HaluEval repairs
failed the reported behavior. This fixed experiment changes the source domain
and removes the competing paired-score penalties. It is not an isolated causal
ablation of those two changes. No labels or inference role prompts change.

Source: the [official TruthfulQA corpus](https://github.com/sylinrl/TruthfulQA),
CSV revision `f6be04e52bbcb41d4d20daee6358231d4a5015d2`, 790 records,
SHA256 `b8d8ef1e12f98b4f2a9f47abc9765da0640b182b6c5d9b92f0c1a1f2f1e02e5c`.
The source provides best correct and incorrect reference answers. Reference
annotations are not independently adjudicated in this experiment.

## Frozen data allocation

Exclude normalized matches to all 528 question strings recovered recursively
from existing private experiment reports (both MMLU and TruthfulQA questions).
Also exclude every gum/chewing and Einstein question, keeping the reported
case and existing public example out of training. Exact normalization handles
case, punctuation, Unicode width and whitespace; it does not prove semantic
paraphrase separation. Deduplicate before splitting and reject contradictory
correct/incorrect reference labels.

535 records remain eligible. Sort by SHA256 of `role-split-3407:` plus the
normalized question hash. Allocate the first 300 to training, next 32 to
development, next 64 to held-out test; the rest stay unused. Frozen allocation
fingerprint: `6fa57218a4891b0864191e8f500e0141920c1d283d74b004c8dc8151c5482b2f`.
Reject truncated training responses and pairs with native response token-length
ratio above 2. Exclusions never migrate into development or test.

This is a **derived internal role-development corpus**. Full TruthfulQA would
now overlap this candidate's training and must not be reported as an independent
benchmark. The original canonical model and its previous results are unchanged.
The candidate manifest records all allocated training question hashes. Existing
canonical benchmark commands must not be repointed to this experiment.

## Fixed training recipe

- Hallucination: fresh zero-delta rank-16 LoRA, original scaling/targets; train
  on incorrect references for 160 steps, learning rate 2e-4. Loss is response
  cross entropy plus 0.1 forward KL to frozen base on eight benign trajectories.
- Jailbreak: initialize from the original jailbreak adapter; train on correct
  references for 80 steps, learning rate 5e-5. Loss is response cross entropy
  plus 0.5 forward KL preservation. Alternate eight frozen-base benign
  trajectories and eight original-jailbreak trajectories from **training-only**
  harmful prompts. This tests factual repair without unrestricted forgetting;
  it does not establish harmlessness or harmful-compliance effectiveness.
- Both: native-chat-v2 shared inference conditioning, maximum training context
  256, microbatch 1, accumulation 4, linear schedule, warmup 5, seed 3407,
  bf16, only LoRA trainable, no intermediate checkpoint selection.

## Fixed evaluation and acceptance

Evaluate both disjoint partitions without tuning against either. On paired
correct/incorrect responses, collect mean supervised token log probabilities
for base, both original experts and both new experts. At each teacher-forced
response prefix also apply the exact production correction operator with the
unchanged base router. Report reference-token accuracy separately from response
likelihood; plausibility-masked likelihoods are null with explicit mask counts.
These diagnostics are not free-form answer accuracy.

Required numeric checks on the 64 test records: positive 95% paired bootstrap
interval for hallucination specificity gain over base; false reference preferred
on at least 60%; positive interval for jailbreak factual-margin gain over its
original checkpoint; nonnegative interval for corrected correct-reference token
accuracy gain over base. Preserve mean original-jailbreak reference likelihood
to within 0.2 nats/token on 16 complete existing held-out harmful responses
(maximum context 1024; record all context exclusions). No accuracy loss on the
64 reused MMLU development controls. Report all checks, including failures.

After numeric evaluation, replay the exact gum question and the first eight
held-out prompts eligible under the existing display policy, with four raw
branches and unchanged inference conditioning. These generations require
qualitative review; exact-reference matching is not a sufficient factuality
judge. They never select strength or weights. Passing numerical diagnostics
alone cannot promote the model or substantiate broad accuracy/safety claims.

## Budget and immutability

One sequential A100-40GB job, hard timeout 2400 seconds, zero retries, conservative
reservation US$2.40 within the existing US$25 task cap. Completed negative-trial
jobs were reconciled to US$0.46/US$0.28 using their observed execution bounds,
retaining original reservations and evidence. Before this dispatch US$21.78 is
reserved; this job raises that to US$24.18. These are workload ceilings rather
than Modal invoices. Both canonical weight files and manifest must remain
byte-identical; new files live under `/vol/runs/misconception_roles_v1`.
