"""Offline lower bound on the KL needed to make a token a distribution maximum.

Targets are evaluation references, never inputs to generation. This diagnoses
numerical constraints; it does not establish answer correctness or safety.
"""
import math

import torch


def minimum_argmax_kl(base, target, *, support=None):
    """Return min KL(q || softmax(base)) with q[target] >= every q[i].

    The optimum pools the target with probabilities above the pool's geometric
    mean. Other allowed probabilities keep their base ratios. Ties count as a
    maximum, so this is a lower bound for a particular greedy tie-breaking rule.
    Excluded tokens have q=0; an excluded target is unreachable (infinity).
    """
    if base.ndim != 1 or not base.numel():
        raise ValueError('Expected one nonempty vocabulary row')
    if type(target) is not int or not 0 <= target < base.numel():
        raise ValueError('Target must index the vocabulary')
    if torch.isnan(base).any() or torch.isposinf(base).any() or not torch.isfinite(base).any():
        raise ValueError('Base logits must be finite or negative infinity')
    allowed = torch.isfinite(base)
    if support is not None:
        if support.shape != base.shape or support.dtype != torch.bool:
            raise ValueError('Support must be a matching boolean mask')
        allowed &= support.to(base.device)
    if not allowed[target]:
        return math.inf
    log_p = base.double().log_softmax(-1)
    others = log_p[allowed].sort(descending=True).values
    higher = others[others > log_p[target]]
    previous_sum = torch.cat((log_p[target][None], log_p[target] + higher.cumsum(0)[:-1])) if higher.numel() else higher
    if higher.numel():
        previous_mean = previous_sum / torch.arange(1, higher.numel()+1, device=base.device)
        joins = (higher > previous_mean).long().cumprod(0)
        count = int(joins.sum())
        mean = (log_p[target] + higher[:count].sum()) / (count+1)
    else:
        mean = log_p[target]
    pooled = allowed & (log_p > mean)
    pooled[target] = True
    weights = log_p.masked_fill(~allowed, -torch.inf)
    weights = torch.where(pooled, mean, weights)
    # At the optimum the pooled log-ratio sums to zero before normalization.
    return max(0., float(-torch.logsumexp(weights, dim=0)))
