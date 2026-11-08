"""Experimental POC correction with exception-safe adapter use and diagnostics.

This preserves the original operator: subtract top-k LayerNormed absolute
anti-expert logits from the supplied base scores. It is not the contrastive
delta/plausibility operator proposed in the research blueprint, and its safety
or quality is not established. The KL guard is a soft attenuation heuristic,
not a divergence bound. Diagnostics measure KL(base || proposed correction)
on the actual scores; guard decisions are independent for each batch row.

The shared model must be exclusively owned by the caller for the complete
generation: adapter switching is mutable state. Expert passes currently
recompute the full prefix without KV caches, so overhead must be measured.
"""

import math

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

import config


class LegacyMultiUnLoRALogitsProcessor(LogitsProcessor):
    """Apply the POC penalty while preserving upstream negative-infinity masks."""

    def __init__(
        self,
        model,
        alpha_hall,
        alpha_jail,
        penalty_threshold=config.PROCESSOR_KL_THRESHOLD,
        telemetry_callback=None,
    ):
        for name, value in (
            ("alpha_hall", alpha_hall),
            ("alpha_jail", alpha_jail),
            ("penalty_threshold", penalty_threshold),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.model = model
        self.alpha_hall = float(alpha_hall)
        self.alpha_jail = float(alpha_jail)
        self.penalty_threshold = float(penalty_threshold)
        self.telemetry_callback = telemetry_callback
        self.step = 0

    def _expert_penalty(self, input_ids, adapter, shape):
        self.model.set_adapter(adapter)
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids, use_cache=False, return_dict=True
            ).logits[:, -1, :].float()
        if logits.shape != shape or not torch.isfinite(logits).all():
            raise ValueError("Expert logits must be finite and match base score shape")
        logits = F.layer_norm(logits, normalized_shape=logits.shape[1:])
        k = min(config.PROCESSOR_TOP_K, logits.shape[-1])
        indices = torch.topk(logits, k).indices
        return torch.zeros_like(logits).scatter(1, indices, logits.gather(1, indices))

    @staticmethod
    def _kl(base_log_probs, proposed_log_probs):
        # Mask before multiplication: 0 * (-inf - -inf) would produce NaN.
        support = torch.isfinite(base_log_probs)
        difference = torch.where(
            support, base_log_probs - proposed_log_probs, 0.0
        )
        return (base_log_probs.exp() * difference).sum(dim=-1).clamp_min(0.0)

    def _emit(self, base_log_probs, corrected, kl, attenuation):
        if self.telemetry_callback is None:
            return
        corrected_probs = torch.softmax(corrected.float(), dim=-1)
        probabilities, ids = corrected_probs[0].topk(min(5, corrected.shape[-1]))
        self.telemetry_callback(
            {
                "step": self.step,
                "alpha_hall": self.alpha_hall,
                "alpha_jail": self.alpha_jail,
                "kl_divergence": float(kl[0]),
                "attenuation": float(attenuation[0]),
                "kl_per_row": kl.tolist(),
                "attenuation_per_row": attenuation.tolist(),
                "top_tokens": [
                    {
                        "token_id": int(token_id),
                        "base_probability": float(base_log_probs[0, token_id].exp()),
                        "corrected_probability": float(probability),
                    }
                    for token_id, probability in zip(ids.tolist(), probabilities.tolist())
                ],
            }
        )

    def __call__(self, input_ids, scores):
        if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[-1] == 0:
            raise ValueError("scores must have shape (batch, nonempty vocabulary)")
        if torch.isnan(scores).any() or torch.isposinf(scores).any():
            raise ValueError("scores may contain finite values and negative infinity only")
        if not torch.isfinite(scores).any(dim=-1).all():
            raise ValueError("Each score row must contain at least one finite token")
        self.step += 1
        base_log_probs = F.log_softmax(scores.float(), dim=-1)

        if self.alpha_hall == 0 and self.alpha_jail == 0:
            zeros = scores.new_zeros(scores.shape[0], dtype=torch.float32)
            self._emit(base_log_probs, scores, zeros, torch.ones_like(zeros))
            return scores

        penalty = torch.zeros_like(scores, dtype=torch.float32)
        # Even enable/set/forward failures must leave the base model clean.
        try:
            self.model.enable_adapter_layers()
            if self.alpha_hall:
                penalty += self.alpha_hall * self._expert_penalty(
                    input_ids, "hallucination", scores.shape
                )
            if self.alpha_jail:
                penalty += self.alpha_jail * self._expert_penalty(
                    input_ids, "jailbreak", scores.shape
                )
        finally:
            self.model.disable_adapter_layers()

        proposed = scores.float() - penalty
        kl = self._kl(base_log_probs, F.log_softmax(proposed, dim=-1))
        attenuation = torch.exp(
            -config.PROCESSOR_KL_DECAY_LAMBDA
            * (kl - self.penalty_threshold).clamp_min(0.0)
        )
        corrected = (scores.float() - attenuation[:, None] * penalty).to(scores.dtype)
        self._emit(base_log_probs, corrected, kl, attenuation)
        return corrected
