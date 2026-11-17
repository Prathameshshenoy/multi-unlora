# Multi-UnLoRA 8B v1.2.0

Multi-UnLoRA uses **unsloth/llama-3-8b-instruct**, pinned to revision
`f3710969eb766fb49d4d1ed3aeabcb03390772bd`. Built with Meta Llama 3.
The archive includes the upstream license and acceptable-use policy;
the launcher downloads base weights separately.

Two anti-expert roles use three rank-16 LoRA checkpoints. The hallucination role
has separate factual and safety checkpoints; the jailbreak checkpoint is shared.
Safety decoding keeps its frozen expert pair when the factual checkpoint changes.

The bundle contains Safetensors weights, adapter configurations, concept vectors,
two routing heads and a checksum-bound manifest. It contains no credentials,
datasets, optimizer state or user conversations. Update the repository before
installing a newer bundle; the installer checks bundle-format compatibility.

## Decoding

Factual training rewards correct responses and ranks them above supplied
misconceptions through the final subtracted distribution. It uses a separate
TruthfulQA training split. Safety training uses factual-answer and refusal targets.
Standalone anti-expert responses can be incorrect, compliant or accurate.

Both routing heads read the frozen base hidden state at layer 24:

- A safety score of at least **0.5** selects the preserved safety pair with correction weights **0.4/0.4**.
- Otherwise, a factual score of at least **0.9** selects the factual checkpoint with weights **0.2/0**.
- Other prompts use zero adapter correction.

Safety routing takes priority. Factual routing uses question content without
multiple-choice answer options. All four displayed branches receive the same
prompt and native chat template.

The decoder subtracts weighted adapter differences, applies a base-plausibility
threshold of **0.001** and limits per-token KL divergence to **8.0**. Each model
arm has an independent KV cache. One-sentence output is the default; users can
turn it off for longer answers.

## Evaluation

The README and website render the same aggregate table, including sample sizes.
MMLU uses zero-shot answer-label accuracy. TruthfulQA uses the
[updated two-choice task](https://github.com/sylinrl/TruthfulQA#update-to-multiple-choice-setting-jan-2025)
on questions excluded from adapter training and routing calibration.

AdvBench tests direct requests and one fixed attack format after training and
development overlap checks. The independent
[HarmBench classifier](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls)
scores raw responses. Evaluations reuse cached outputs when the complete
inference branch is unchanged; changed branches are generated and scored.
Display guards receive no benchmark credit.

Use this checkpoint for local research and comparison. Incorrect and unsafe
responses remain possible. The interface identifies display withholding and
guard responses separately from model generation.
