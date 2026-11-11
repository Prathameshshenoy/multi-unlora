"""Model adapter boundary for contrastive anti-expert decoding.

Run this processor first, with unmodified base logits. Each anti-expert is
queried on the exact same prefix. By default its difference from base logits
is subtracted. A named experimental factual reference can replace the base
reference for the hallucination term; this does not change serving defaults. Shared PEFT adapter state requires exclusive generation ownership.
The pure operator enforces base plausibility and a measured reverse-KL budget.
The old absolute-logit penalty exists only in legacy_processor.py as an ablation.
"""
import math

import torch
from transformers import LogitsProcessor

import config
from inference.operator import OPERATOR_VERSION, correct_logits
from inference.adapters import active_adapter


class MultiUnLoRALogitsProcessor(LogitsProcessor):
    def __init__(self, model, alpha_hall, alpha_jail,
                 penalty_threshold=config.PROCESSOR_KL_THRESHOLD,
                 telemetry_callback=None, *, beta=config.PROCESSOR_PLAUSIBILITY_BETA,
                 attention_mask=None, expert_logits_provider=None,
                 hall_reference_adapter=None, hall_negative_adapter="hallucination"):
        for name, value in (("alpha_hall", alpha_hall), ("alpha_jail", alpha_jail),
                            ("penalty_threshold", penalty_threshold)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(beta) or not 0 < beta <= 1:
            raise ValueError("beta must be in (0, 1]")
        if hall_reference_adapter is not None and (not isinstance(hall_reference_adapter,str) or not hall_reference_adapter):
            raise ValueError("Factual reference needs a nonempty loaded adapter name")
        if hall_negative_adapter not in (None,"hallucination") or (hall_negative_adapter is None and hall_reference_adapter is None):
            raise ValueError("A positive-only control needs a factual reference")
        self.hall_reference_adapter=hall_reference_adapter
        self.hall_negative_adapter=hall_negative_adapter
        self.model = model
        self.alpha_hall, self.alpha_jail = float(alpha_hall), float(alpha_jail)
        self.penalty_threshold, self.beta = float(penalty_threshold), beta
        self.telemetry_callback = telemetry_callback
        self.attention_mask = attention_mask
        self.expert_logits_provider = expert_logits_provider
        self.step = 0

    def _expert(self, input_ids, adapter, shape):
        if self.expert_logits_provider is not None:
            logits = self.expert_logits_provider(input_ids, adapter).float()
            if logits.shape != shape or not torch.isfinite(logits).all():
                raise ValueError("Expert logits must be finite and match base score shape")
            return logits
        kwargs = dict(input_ids=input_ids, use_cache=False, return_dict=True)
        if self.attention_mask is not None:
            extra = input_ids.shape[1] - self.attention_mask.shape[1]
            if extra < 0:
                raise ValueError("Attention mask is longer than generation prefix")
            kwargs["attention_mask"] = torch.cat((self.attention_mask,
                self.attention_mask.new_ones((input_ids.shape[0], extra))), dim=1)
        with active_adapter(self.model, adapter), torch.no_grad():
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
        expert_probs = {}
        if self.alpha_hall or self.alpha_jail:
            try:
                for name, alpha in (("hallucination", self.alpha_hall), ("jailbreak", self.alpha_jail)):
                    if alpha:
                        queried=not (name=="hallucination" and self.hall_negative_adapter is None)
                        expert=self._expert(input_ids,name,scores.shape) if queried else scores.float()
                        reference=scores.float()
                        if name=="hallucination" and self.hall_reference_adapter is not None:
                            reference=self._expert(input_ids,self.hall_reference_adapter,scores.shape)
                            if self.telemetry_callback is not None:
                                expert_probs['hall_reference']=reference.detach().softmax(-1)
                        if self.telemetry_callback is not None and queried:
                            expert_probs[name]=expert.detach().softmax(-1)
                        delta += alpha * torch.where(torch.isfinite(scores), expert-reference, 0.0)
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
                    "hallucination_probability": float(expert_probs["hallucination"][0, token_id]) if "hallucination" in expert_probs else None,
                    "jailbreak_probability": float(expert_probs["jailbreak"][0, token_id]) if "jailbreak" in expert_probs else None,
                    "corrected_probability": probability,
                    **({"hall_reference_probability":float(expert_probs["hall_reference"][0,token_id]) if "hall_reference" in expert_probs else None} if self.hall_reference_adapter is not None else {})}
                    for token_id, probability in zip(ids.tolist(), probabilities.tolist())
                    # topk fills unused slots with arbitrary zero-mass tokens
                    # when the plausibility filter leaves fewer than five.
                    if math.isfinite(probability) and probability > 0],
            })
        return correction.scores
