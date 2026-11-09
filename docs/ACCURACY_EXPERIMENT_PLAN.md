# Accuracy repair experiment

Prepared 9 November 2026. This plan does not claim the historical slide has been
reproduced or beaten. Its original checkpoints and CSVs are unavailable.

## What the new diagnostic found

On eight matched held-out response pairs per concept:

| Adapter | NLL gain on undesirable reference | NLL gain on desirable reference | Paired specificity margin |
| --- | ---: | ---: | ---: |
| Hallucination | 1.405 | 6.118 | −4.713 |
| Jailbreak | 0.528 | −0.149 | +0.677 |

All eight hallucination margins were negative; all eight jailbreak margins were
positive. This is a small, descriptive likelihood diagnostic, not a generation
accuracy measurement. It supports investigating whether the hallucination
adapter learned shared answer style rather than specifically false answers.
It does not prove the cause of every free-form regression.

The references use HaluEval's provided right/hallucinated answer pairs and
LLM-LAT's chosen/rejected pairs. Their schemas and representative semantics are
available in the primary dataset pages:
[HaluEval](https://huggingface.co/datasets/pminervini/HaluEval),
[LLM-LAT](https://huggingface.co/datasets/LLM-LAT/harmful-dataset).
Dataset labels still require a sampled quality audit; they are not an infallible
truth oracle. The diagnostic verifies existing dataset and holdout fingerprints.

## Proposed change

Run a **hallucination-only paired anti-expert experiment**. Keep the base model,
jailbreak adapter, prompt template, data split, and comparison settings fixed.
Save new weights under a separate immutable run directory; preserve the original
pilot, reports, and all failed outcomes.

Instead of learning only the undesirable response, test an objective that
prefers it *relative to the desirable response and the frozen base*. A uniform
likelihood gain shared by both responses should cancel from that preference
margin. The implementation is `training/paired.py` with the math helper in
`training/paired_objective.py`.

This tests **length-normalized-anti-preference-v1**, a DPO-inspired logistic
objective with deliberately reversed preferences for the anti-expert. It uses
mean response-token log probabilities relative to the frozen base. The diagnostic
above measures mean-token NLL: summed preferences could improve merely through
response length while that specificity margin stays negative. We explicitly
choose matching mean-token units and preserve sums and token counts as controls.
This is a variant, not standard [DPO](https://arxiv.org/abs/2305.18290), and no
published guarantee establishes its subtraction benefit. Mask instruction tokens,
freeze reference scores, and record every hyperparameter.

## Predeclared checks

1. On validation pairs, compare undesirable and desirable NLL gains and their
   margin against the current adapter. A bad-response gain alone is insufficient.
2. Select correction settings using validation data only. Include zero correction
   and fixed-weight baselines. Do not tune on the original held-out outcomes.
3. Confirm on untouched response pairs and question IDs, checking overlap against
   all prior reports. Record paired confidence intervals and invalid outputs.
4. Evaluate free-form factual questions separately from conditional-label MC1.
   Preserve raw benign outputs and errors, including the sky-blue regression.
5. For harmful prompts, report **raw model refusal/compliance** independently of
   **display guard intervention**. Guard fallbacks must never improve the model's
   claimed ASR. Use an independent judge/rubric and benign over-refusal set before
   making safety claims. Raw dangerous expert text is not a public demo artifact.
6. Promote nothing automatically. A successful specificity check is not a
   successful accuracy or safety gate.

## Approved compute envelope

The user explicitly raised the total task ceiling to **US$15** on 9 November
2026. Before this experiment, **US$9.95** was recorded in
conservative reservations. These reservations are not provider invoice charges.
The user reported US$20 in available Modal credits.

The added envelope allows at most **US$5 additional reserved work**:

| Stage | Hard maximum | Conservative reservation |
| --- | ---: | ---: |
| One paired hallucination training trial | 1 A100, 30 minutes, no retries | US$1.80 |
| Specificity, validation, untouched confirmation | 1 A100, 30 minutes, no retries | US$1.80 |
| Free-form/benign-refusal diagnostics | 1 A100, 10 minutes, no retries | US$0.60 |
| Short final local check | 1 A100, 5 minutes | US$0.50 |
| CPU preparation allowance | bounded preparation | US$0.30 |

Run one GPU at a time. Stop immediately on nonfinite loss, inconsistent artifacts,
or failed data provenance. Preserve old artifacts. Do not switch accounts or exceed the approved cap. The local reservation ledger
cannot establish actual remaining provider credit.
