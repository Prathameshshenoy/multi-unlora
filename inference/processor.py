"""Model adapter boundary for contrastive anti-expert decoding.

Run this processor first, with unmodified base logits. Each anti-expert is
queried on the exact same prefix; only its difference from base logits is
subtracted. Shared PEFT adapter state requires exclusive generation ownership.
The pure operator enforces base plausibility and a measured reverse-KL budget.
The old absolute-logit penalty exists only in legacy_processor.py as an ablation.
"""
import math

import torch
from transformers import LogitsProcessor

import config
from inference.operator import OPERATOR_VERSION, correct_logits


class MultiUnLoRALogitsProcessor(LogitsProcessor):
    def __init__(self, model, alpha_hall, alpha_jail,
                 penalty_threshold=config.PROCESSOR_KL_THRESHOLD,
                 telemetry_callback=None, *, beta=config.PROCESSOR_PLAUSIBILITY_BETA,
                 attention_mask=None):
        for name, value in (("alpha_hall", alpha_hall), ("alpha_jail", alpha_jail),
                            ("penalty_threshold", penalty_threshold)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(beta) or not 0 < beta <= 1:
            raise ValueError("beta must be in (0, 1]")
        self.model = model
        self.alpha_hall, self.alpha_jail = float(alpha_hall), float(alpha_jail)
        self.penalty_threshold, self.beta = float(penalty_threshold), beta
        self.telemetry_callback = telemetry_callback
        self.attention_mask = attention_mask
        self.step = 0

    def _expert(self, input_ids, adapter, shape):
        self.model.set_adapter(adapter)
        kwargs = dict(input_ids=input_ids, use_cache=False, return_dict=True)
        if self.attention_mask is not None:
            extra = input_ids.shape[1] - self.attention_mask.shape[1]
            if extra < 0:
                raise ValueError("Attention mask is longer than generation prefix")
            kwargs["attention_mask"] = torch.cat((self.attention_mask,
                self.attention_mask.new_ones((input_ids.shape[0], extra))), dim=1)
        with torch.no_grad():
            logits = self.model(**kwargs).logits[:, -1, :].float()
        if logits.shape != shape or not torch.isfinite(logits).all():
            raise ValueError("Expert logits must be finite and match base score shape")
        return logits

    def __call__(self, input_ids, scores):
        if scores.ndim != 2 or not all(scores.shape):
            raise ValueError("scores must have shape (batch, nonempty vocabulary)")
        if torch.isnan(scores).any() or torch.isposinf(scores).any() or not torch.isfinite(scores).any(-1).all():
            raise ValueError("Each score row needs finite logits; only -inf masks are allowed")
        self.step += 1
        delta = torch.zeros_like(scores, dtype=torch.float32)
        if self.alpha_hall or self.alpha_jail:
            try:
                self.model.enable_adapter_layers()
                for name, alpha in (("hallucination", self.alpha_hall), ("jailbreak", self.alpha_jail)):
                    if alpha:
                        expert = self._expert(input_ids, name, scores.shape)
                        delta += alpha * torch.where(torch.isfinite(scores), expert - scores.float(), 0.0)
            finally:
                self.model.disable_adapter_layers()
        correction = correct_logits(scores, delta, beta=self.beta, kl_budget=self.penalty_threshold)
        if self.telemetry_callback is not None:
            base_probs = scores.float().softmax(-1)
            corrected_probs = correction.scores.float().softmax(-1)
            probabilities, ids = corrected_probs[0].topk(min(5, scores.shape[-1]))
            self.telemetry_callback({
                "step": self.step, "operator": OPERATOR_VERSION,
                "alpha_hall": self.alpha_hall, "alpha_jail": self.alpha_jail,
                "kl_divergence": float(correction.kl[0]), "kl_direction": "corrected_to_base",
                "kl_budget": self.penalty_threshold, "plausibility_beta": self.beta,
                "attenuation": float(correction.scale[0]),
                "abstained": bool(correction.abstained[0]),
                "kl_per_row": correction.kl.tolist(),
                "attenuation_per_row": correction.scale.tolist(),
                "top_tokens": [{"token_id": token_id,
                    "base_probability": float(base_probs[0, token_id]),
                    "corrected_probability": probability}
                    for token_id, probability in zip(ids.tolist(), probabilities.tolist())],
            })
        return correction.scores
