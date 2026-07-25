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


def compute_dynamic_alphas(model, tokenizer, prompt_text, v_hall, v_jail):
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
        tuple[float, float]: (alpha_hall, alpha_jail), both rounded to 3 decimal
            places. Values are in (0, 1) before capping and guaranteed to sum
            to at most config.ALPHA_CAP after.
    """
    current_h = get_hidden_state_via_hook(
        model, tokenizer, prompt_text, config.REPE_LAYER_IDX
    ).to("cuda")

    # Normalize input hidden state before projection
    current_h_norm = current_h / current_h.norm()

    # Dot product onto each direction vector
    # Positive = prompt heading toward bad concept, negative = heading away
    cos_hall = torch.dot(current_h_norm, v_hall).item()
    cos_jail = torch.dot(current_h_norm, v_jail).item()

    # Sigmoid maps (-inf, +inf) to (0, 1)
    # At cos=0 (neutral): alpha=0.5
    # At cos=0.5 (tilted toward bad): alpha~0.82
    # At cos=-0.5 (tilted away): alpha~0.18
    alpha_hall = round(
        torch.sigmoid(
            torch.tensor(config.REPE_K_SCALE * cos_hall + config.REPE_BIAS)
        ).item(),
        3,
    )
    alpha_jail = round(
        torch.sigmoid(
            torch.tensor(config.REPE_K_SCALE * cos_jail + config.REPE_BIAS)
        ).item(),
        3,
    )

    total = alpha_hall + alpha_jail
    if total > config.ALPHA_CAP:
        scale = config.ALPHA_CAP / total
        alpha_hall = round(alpha_hall * scale, 3)
        alpha_jail = round(alpha_jail * scale, 3)

    return alpha_hall, alpha_jail
