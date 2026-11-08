"""Pure contrastive correction; no model handles or adapter state.

Returns a distribution within a numerical KL(q_corrected || p_base) budget.
The plausibility floor is measured against the *full* base distribution. If
masking alone exceeds the budget, abstain for that row. This is a token-level
numerical constraint, not a guarantee of sequence quality or safety.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

OPERATOR_VERSION = "contrastive-v1"


@dataclass
class Correction:
    scores: torch.Tensor
    kl: torch.Tensor
    scale: torch.Tensor
    abstained: torch.Tensor


def reverse_kl(log_q, log_p):
    support = torch.isfinite(log_q)
    difference = torch.where(support, log_q - log_p, 0.0)
    return (log_q.exp() * difference).sum(-1).clamp_min(0)


def correct_logits(base, delta, *, beta=0.1, kl_budget=0.5):
    """Subtract weighted expert-minus-base delta on base-plausible tokens.

    Both tensors are [batch, vocabulary]. Existing -inf masks are respected.
    Nonzero corrections return float32, avoiding bfloat16 rounding breaking the
    measured budget. A scalar offset in delta has no distributional effect.
    """
    if not math.isfinite(beta) or not 0 < beta <= 1:
        raise ValueError("beta must be in (0, 1]")
    if not math.isfinite(kl_budget) or kl_budget < 0:
        raise ValueError("kl_budget must be finite and non-negative")
    if base.ndim != 2 or not all(base.shape) or delta.shape != base.shape:
        raise ValueError("base and delta must share a nonempty [batch, vocabulary] shape")
    if torch.isnan(base).any() or torch.isposinf(base).any() or not torch.isfinite(base).any(-1).all():
        raise ValueError("Each base row needs finite logits; only -inf masks are allowed")
    if not torch.isfinite(delta).all():
        raise ValueError("delta must be finite")
    work, delta = base.float(), delta.float()
    finite = torch.isfinite(work)
    # A uniform logit offset carries no adapter information. Centre it away.
    anchor = delta.gather(1, work.argmax(-1, keepdim=True))
    delta = torch.where(finite, delta - anchor, 0.0)
    active = delta.abs().amax(-1) > 0
    log_p = F.log_softmax(work, -1)
    mask = finite & (work >= work.amax(-1, keepdim=True) + math.log(beta))
    masked_base = work.masked_fill(~mask, -torch.inf)
    floor = reverse_kl(F.log_softmax(masked_base, -1), log_p)
    abstained = active & (floor > kl_budget)
    eligible = active & ~abstained & (kl_budget > 0)

    def candidate(scale):
        return (work - scale[:, None] * delta).masked_fill(~mask, -torch.inf)

    ones = work.new_ones(work.shape[0])
    proposed = candidate(ones)
    proposed_kl = reverse_kl(F.log_softmax(proposed, -1), log_p)
    needs_search = eligible & (proposed_kl > kl_budget)
    low, high = torch.zeros_like(ones), ones.clone()
    # For an exponential tilt of p_base conditioned on a fixed mask,
    # d KL(q_t || p_base)/dt = t * Var_q(delta) >= 0, so bisection is valid.
    if needs_search.any():
        for _ in range(24):
            mid = (low + high) / 2
            divergence = reverse_kl(F.log_softmax(candidate(mid), -1), log_p)
            feasible = divergence <= kl_budget
            low = torch.where(feasible, mid, low)
            high = torch.where(feasible, high, mid)
    scale = torch.where(needs_search, low, ones)
    scale = torch.where(eligible, scale, torch.zeros_like(scale))
    result = torch.where(eligible[:, None], candidate(scale), work)
    kl = reverse_kl(F.log_softmax(result, -1), log_p)
    # Conservative fallback for nonfinite or accumulated numerical roundoff.
    invalid = ~torch.isfinite(kl) | (kl > kl_budget + 1e-6)
    result = torch.where(invalid[:, None], work, result)
    kl = torch.where(invalid, torch.zeros_like(kl), kl)
    scale = torch.where(invalid, torch.zeros_like(scale), scale)
    abstained |= invalid
    if not active.any():
        result = base
    return Correction(result, kl, scale, abstained)
