"""
repe/router.py

Dynamic per-prompt penalty weight computation for the Multi-UnLoRA router.

Given an incoming prompt, the router extracts its hidden state at the RepE
layer, projects it onto each concept direction vector via dot product, and
maps the resulting similarity scores through a shifted sigmoid to produce
alpha weights. These weights are passed directly to MultiUnLoRALogitsProcessor
to scale how aggressively each BadExpert's signal is subtracted.

Prompts whose hidden states align strongly with a direction vector receive a
high alpha for that concept; prompts near the safe baseline receive alphas
close to the sigmoid floor. A joint cap (config.ALPHA_CAP) prevents the sum
of both alphas from growing large enough to collapse the output distribution.
"""

import torch
import torch.nn.functional as F

import config
from repe.extractor import get_hidden_state_via_hook


def compute_dynamic_alphas(
    model, tokenizer, prompt_text, v_hall, v_jail, *, add_special_tokens=True
):
    """Computes per-concept penalty weights for a single prompt.

    Extracts the prompt's hidden state at config.REPE_LAYER_IDX, normalises
    it, then projects it onto each direction vector. The raw dot product is
    passed through a scaled and biased sigmoid to produce an alpha in (0, 1).

    Direction vectors (v = norm(h_bad - h_safe)) are used instead of raw
    concept vectors because raw vectors produce near-identical cosine
    similarities for all prompts, keeping alpha permanently near the floor.
    The direction vector isolates only the component of the hidden state that
    distinguishes the bad concept from the safe baseline.

    A negative bias term (config.REPE_BIAS) shifts the sigmoid so that
    neutral prompts — those with a dot product near zero — produce an alpha
    below 0.5 rather than exactly at it, keeping penalty weights conservative
    for ambiguous inputs.

    If the sum of both alphas exceeds config.ALPHA_CAP, both are rescaled
    proportionally so their sum equals the cap. This prevents simultaneous
    high penalties from over-correcting and collapsing the output distribution.

    Args:
        model (PeftModel): PEFT-wrapped model with adapters loaded.
        tokenizer: Tokenizer matching the model.
        prompt_text (str): The full prompt string for the current inference step,
            including any system prompt prefix.
        v_hall (torch.Tensor): Normalised hallucination direction vector, on CUDA,
            shape (hidden_dim,).
        v_jail (torch.Tensor): Normalised jailbreak direction vector, on CUDA,
            shape (hidden_dim,).

    Returns:
        tuple[float, float]: (alpha_hall, alpha_jail), without rounding.
            Values are in (0, 1) before capping and guaranteed to sum
            to at most config.ALPHA_CAP after.
    """
    if hasattr(model, "disable_adapter_layers"):
        model.disable_adapter_layers()
    current_h = get_hidden_state_via_hook(
        model, tokenizer, prompt_text, config.REPE_LAYER_IDX,
        add_special_tokens=add_special_tokens,
    ).float()

    directions = [v_hall.to(current_h), v_jail.to(current_h)]
    for vector in [current_h, *directions]:
        if vector.ndim != 1 or vector.shape != current_h.shape:
            raise ValueError("Router vectors must share a one-dimensional hidden size")
        norm = vector.norm()
        if not torch.isfinite(vector).all() or not torch.isfinite(norm) or norm <= 1e-12:
            raise ValueError("Router vectors must be finite and nonzero; recalibrate vectors")

    current_h_norm = F.normalize(current_h, dim=0)
    cosines = [
        torch.dot(current_h_norm, F.normalize(vector, dim=0)).clamp(-1.0, 1.0)
        for vector in directions
    ]
    alpha_hall, alpha_jail = [
        float(torch.sigmoid(config.REPE_K_SCALE * cosine + config.REPE_BIAS))
        for cosine in cosines
    ]
    total = alpha_hall + alpha_jail
    if total > config.ALPHA_CAP:
        scale = config.ALPHA_CAP / total
        alpha_hall *= scale
        alpha_jail = min(alpha_jail * scale, config.ALPHA_CAP - alpha_hall)
    return alpha_hall, alpha_jail
