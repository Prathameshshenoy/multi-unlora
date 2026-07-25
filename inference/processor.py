"""
inference/processor.py

Core Multi-UnLoRA inference mechanism.

Implements MultiUnLoRALogitsProcessor, a HuggingFace LogitsProcessor that
intercepts the token distribution at every generation step and subtracts
the BadExpert adapter signals from it. This is the inference-time equivalent
of the training-time concept: rather than fine-tuning toward good behaviour,
we fine-tuned toward bad behaviour and then subtract that signal away.

Performance characteristic: __call__ fires once per generated token and runs
two additional full forward passes (one per adapter) on top of the base
model's own pass inside model.generate(). This makes generation approximately
3x slower than an unmodified deployment. The cost is intentional and is
measured explicitly in runners.py for the latency section of the paper.
"""

import unsloth
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor
import math

import config


class MultiUnLoRALogitsProcessor(LogitsProcessor):
    """Subtracts BadExpert logit signals from the base model at each generation step.

    At every token position, queries both the hallucination and jailbreak
    BadExpert adapters for their next-token distributions. After LayerNorm
    alignment and top-k masking, each expert's distribution is subtracted
    from the base model's distribution weighted by its alpha. The result is
    a corrected distribution that is steered away from both hallucinated and
    harmful completions.

    A KL divergence guard prevents runaway correction: if the adjusted
    distribution has diverged too far from the base, the delta is attenuated
    exponentially rather than applied in full or discarded entirely.

    Adapter state is explicitly managed inside __call__: adapters are enabled
    only for the duration of each expert query and disabled again before
    returning, ensuring that model.generate()'s own forward passes always run
    on the clean base model.

    Attributes:
        model (PeftModel): PEFT model with both adapters mounted.
        alpha_hall (float): Hallucination penalty weight from the router.
        alpha_jail (float): Jailbreak penalty weight from the router.
        penalty_threshold (float): KL divergence ceiling above which the
            correction is attenuated. Defaults to config.PROCESSOR_KL_THRESHOLD.
    """

    def __init__(
        self,
        model,
        alpha_hall,
        alpha_jail,
        penalty_threshold=config.PROCESSOR_KL_THRESHOLD,
    ):
        """Initialises the processor with a model and pre-computed alpha weights.

        Args:
            model (PeftModel): PEFT model with both adapters mounted and
                eval mode set.
            alpha_hall (float): Hallucination penalty weight, typically produced
                by repe.router.compute_dynamic_alphas().
            alpha_jail (float): Jailbreak penalty weight, typically produced
                by repe.router.compute_dynamic_alphas().
            penalty_threshold (float): KL divergence ceiling. Corrections that
                exceed this threshold are exponentially decayed rather than
                applied in full. Defaults to config.PROCESSOR_KL_THRESHOLD.
        """
        self.model = model
        self.alpha_hall = alpha_hall
        self.alpha_jail = alpha_jail
        self.penalty_threshold = penalty_threshold

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Applies the Multi-UnLoRA correction to the next-token logit distribution.

        Called automatically by model.generate() at every token position.
        Queries both adapters, computes a masked and scaled penalty, applies
        a KL divergence guard, and returns the corrected scores as a delta
        over the input scores to preserve any upstream processor adjustments.

        Args:
            input_ids (torch.LongTensor): Token IDs generated so far,
                shape (batch_size, sequence_length).
            scores (torch.FloatTensor): Raw next-token logits from the base
                model's forward pass, shape (batch_size, vocab_size).

        Returns:
            torch.FloatTensor: Corrected logits, same shape as scores.
        """
        # enable_adapter_layers() must be called before set_adapter() because
        # disable_adapter_layers() sets a global flag that set_adapter() alone
        # does not clear.
        self.model.enable_adapter_layers()

        # --- Hallucination expert ---
        self.model.set_adapter("hallucination")
        with torch.no_grad():
            hall_logits = self.model(
                input_ids=input_ids, use_cache=False, return_dict=True
            ).logits[:, -1, :]

        # --- Jailbreak expert ---
        self.model.set_adapter("jailbreak")
        with torch.no_grad():
            jail_logits = self.model(
                input_ids=input_ids, use_cache=False, return_dict=True
            ).logits[:, -1, :]

        # Adapters must be disabled before returning so model.generate()'s
        # next forward pass runs on the clean base model.
        self.model.disable_adapter_layers()

        # LayerNorm aligns the variance of the base and expert distributions
        # before subtraction. Without this, scale differences between the base
        # model and the smaller LoRA adapters can inflate or collapse the
        # penalty unpredictably across different prompt lengths.
        base_norm = F.layer_norm(scores.float(), normalized_shape=scores.shape[1:])
        hall_norm = F.layer_norm(
            hall_logits.float(), normalized_shape=hall_logits.shape[1:]
        )
        jail_norm = F.layer_norm(
            jail_logits.float(), normalized_shape=jail_logits.shape[1:]
        )

        def apply_top_k_mask(logits, k):
            """Zeros all logits except the expert's top-k, returning a sparse penalty.

            Restricting subtraction to tokens the expert is confident about
            prevents noise from low-probability expert tokens from degrading
            the base model's distribution on unrelated vocabulary positions.

            Args:
                logits (torch.FloatTensor): Normalised expert logits,
                    shape (batch_size, vocab_size).
                k (int): Number of top tokens to retain.

            Returns:
                torch.FloatTensor: Sparse logit tensor with only top-k positions
                    non-zero, same shape as logits.
            """
            _, top_k_indices = torch.topk(logits, k)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(1, top_k_indices, True)
            masked = torch.zeros_like(logits)
            masked[mask] = logits[mask]
            return masked

        masked_hall = apply_top_k_mask(hall_norm, config.PROCESSOR_TOP_K)
        masked_jail = apply_top_k_mask(jail_norm, config.PROCESSOR_TOP_K)

        corrected_norm = (
            base_norm
            - (self.alpha_hall * masked_hall)
            - (self.alpha_jail * masked_jail)
        )

        # KL divergence guard: if the correction has pushed the distribution
        # too far from the base, apply an exponential decay to the delta rather
        # than using it in full. This handles edge cases — typically the first
        # generated token — where both experts agree strongly with the base
        # model and the uncapped subtraction would collapse the distribution.
        base_probs = torch.softmax(base_norm.float(), dim=-1)
        corrected_probs = torch.softmax(corrected_norm.float(), dim=-1)
        kl_div = F.kl_div(corrected_probs.log(), base_probs, reduction="sum").item()

        if kl_div > self.penalty_threshold:
            decay = math.exp(
                -config.PROCESSOR_KL_DECAY_LAMBDA * (kl_div - self.penalty_threshold)
            )

            return scores + decay * (corrected_norm - base_norm)

        # Return as a delta over the input scores rather than the corrected
        # logits directly, preserving any adjustments made by upstream processors.
        return scores + (corrected_norm - base_norm)
