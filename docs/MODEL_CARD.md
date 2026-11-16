# Multi-UnLoRA 8B v1.1.0

Multi-UnLoRA uses **unsloth/llama-3-8b-instruct**, pinned to revision
`f3710969eb766fb49d4d1ed3aeabcb03390772bd`, with two rank-16 LoRA adapters.
Built with Meta Llama 3. The archive includes the upstream license and
acceptable-use policy; the launcher downloads base weights separately.

The bundle contains Safetensors adapter weights, adapter configurations,
calibration vectors, a prompt-routing head and a checksum-bound manifest.
It contains no credentials, datasets, training state or user conversations.

## Decoding

The adapters are optimized against the final subtracted distribution, using
factual-answer and refusal targets. Their names identify correction roles;
standalone adapter responses can be incorrect, compliant or accurate.

A routing head reads the normalized base hidden state at layer 24. At a routing
threshold of 0.5 it selects hallucination/jailbreak weights of **0.4/0.4**;
otherwise both weights are zero. These are jointly calibrated correction
strengths. All four branches use the same prompt and native chat template.

The decoder subtracts weighted adapter differences, applies a base-plausibility
threshold of 0.001 and limits per-token KL divergence to 8.0. Each model arm has
an independent KV cache. The default response mode stops after one sentence;
users can disable it for longer responses.

## Evaluation

The README and website show the same aggregate table, including sample sizes.
MMLU uses zero-shot answer-label accuracy. TruthfulQA uses the
[updated two-choice format](https://github.com/sylinrl/TruthfulQA#update-to-multiple-choice-setting-jan-2025)
and separately scored brief answers. Its published subset excludes questions
used for adapter training or routing calibration and earlier tuning.

AdvBench tests direct requests and a fixed attack-format stress test on goals
remaining after training and development overlap checks. Raw responses are
scored by the independent [HarmBench classifier](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls).
Brief factual answers are judged against reference answers by Qwen2.5 7B Instruct.
Automated judges can make mistakes. Display guards receive no benchmark credit.

Use this checkpoint for local research and comparison. Incorrect and unsafe
responses remain possible. The interface identifies display withholding and
guard responses separately from model generation.
