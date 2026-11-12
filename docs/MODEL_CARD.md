# Multi-UnLoRA 8B v1.0.0

Multi-UnLoRA uses **unsloth/llama-3-8b-instruct**, pinned to revision
`f3710969eb766fb49d4d1ed3aeabcb03390772bd`, with two rank-16 LoRA adapters.
Built with Meta Llama 3. The checkpoint archive includes the upstream model
license and acceptable-use policy; base weights are downloaded separately.

The archive contains adapter weights in Safetensors format, adapter configuration,
calibrated concept vectors and a checksum-bound manifest. It contains no account
credentials, datasets, training state or user conversations.

The hallucination adapter learns incorrect-answer patterns. The jailbreak
adapter learns harmful-compliance patterns, with additional factual-response
training. Training sources include HaluEval, LLM-LAT and a TruthfulQA-derived
split. Adapter behavior can overlap, and an anti-expert can give a correct answer.

At inference, the router chooses weights and the engine subtracts each adapter's
logit difference from the base distribution. A narrow rule gives jailbreak
correction at least 75% of the combined weight for explicit explosive-construction
requests, preserving the combined correction strength. Other prompts retain their
projection weights. The interface labels this as rule-based safety priority.
This routing rule is not a learned intent classifier or a guarantee of refusal;
published measurements apply to the routing settings recorded in each report. All four branches use the same user
prompt and chat template. The default decoder uses full-prefix inference and a
measured per-token KL bound.

Use the model for local research and comparison. It is not a safety filter or a
fact-checking service: incorrect and unsafe responses remain possible. The
interface labels any display withholding separately from model generation.
Evaluation JSON files under `docs/results` record their own model settings,
sample sizes and scoring methods; results from different settings should not be
combined into a single model score.
