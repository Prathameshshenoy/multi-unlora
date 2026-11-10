# Isolate false-content changes from shared answer style

Declared before dispatch on 10 November 2026. The previous domain repair fixes
the requested gum expert roles and passes held-out specialization/retention
checks, but fails corrected true-reference token accuracy. The final gum answer
also adds unsupported history. Do not promote that candidate.

Ranked explanations: (1) full-response training learned shared concise-answer
style as well as false content, and subtracting the entire task update suppresses
useful style; (2) jailbreak subtraction on factual prompts contributes unrelated
drift; (3) reference-token accuracy partly measures wording rather than semantic
answer quality. The first is a hypothesis, not a demonstrated sole cause.

## Fixed recipe

Train a **fresh truthful companion** on exactly the same 268 selected factual
training questions as the preceding hallucination expert. Same base revision,
rank-16 LoRA configuration, zero-delta initialization seed, record order, 160
steps, LR 2e-4, accumulation 4, native-chat-v2 prompt, full-response CE and 0.1
base-control KL. Change only the target reference from false to true.

Construct the new hallucination adapter's weight update as the preceding
false-trained update minus this matched true-trained update. Represent the
matrix difference exactly with concatenated rank-32 factors, doubling alpha
as well as rank to preserve alpha/r. Subtracting A and B separately would create
incorrect cross terms; numerical and real saved-PEFT tests verify the exact
construction and scaling. No SVD approximation or inference role prompt.

The [task arithmetic paper](https://arxiv.org/abs/2212.04089) motivates combining
weight updates. It does **not** validate this particular factuality repair or
imply that nonlinear model logits equal a difference of logits. The hypothesis
is that shared training effects partly cancel, leaving a more selective false
content update. Retain the previous repaired jailbreak adapter unchanged; its
factual and domain-retention checks passed.

## Fresh confirmation

Keep training and 32 development records unchanged. The previous 64 test records
are now consumed and cannot be confirmation again. Use the **next 64 records**
in the original deterministic eligible ordering. New corpus fingerprint:
`d3e4869c3fac6ca3534fa9c8e648369ce3f458daab9cb6e2e2aaee0cc2cf8b71`.
No test result selects composition strength or weights: use unit bad-minus-good
weight difference and unchanged production router/operator.

Repeat every numerical acceptance check from the previous plan, including the
corrected-reference token accuracy check, reused MMLU control and original
jailbreak-domain retention. Add hallucination-only and jailbreak-only correction
ablations at their unchanged routed weights; these diagnose drift and do not
select an operating point. Evaluate full four-arm gum and first-eight eligible
fresh qualitative prompts after numerical scoring. Inspect generated answers
against source annotations, explicitly distinguish wording metrics from actual
false claims, and do not automatically promote.

## Budget and artifacts

Single A100-40GB job, timeout 1800 seconds, zero retries, US$1.80 conservative
reservation. The completed previous job was reconciled from US$2.40 to US$1.09
using its observed 975.73-second dispatch-to-result interval, plus shutdown and
overhead margins. Existing reservations US$22.87 become US$24.67 within the
unchanged US$25 cap. This is not a claim about invoiced spending.

Save under `/vol/runs/misconception_content_delta_v1`; preserve canonical and
parent weights/manifests byte-for-byte. Record both source task weight hashes,
new weight/config hashes and the exact construction. The candidate manifest
contains training-question hashes and cannot claim independent full TruthfulQA
results. Baseline and historical public benchmark tables remain unchanged.
