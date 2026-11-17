"""Train an anti-expert through the distribution produced by subtraction."""
import math

import torch
import torch.nn.functional as F


def _response_scores(base, anti, targets, alpha):
    if base.ndim != 2 or base.shape != anti.shape or not all(base.shape):
        raise ValueError('Expected matching response-token by vocabulary logits')
    if targets.dtype != torch.long or targets.shape != base.shape[:1]:
        raise ValueError('One target is required per response position')
    if (targets < 0).any() or (targets >= base.shape[-1]).any():
        raise ValueError('Response targets must be valid vocabulary indices')
    if not torch.isfinite(base).all() or not torch.isfinite(anti).all():
        raise ValueError('Response logits must be finite')
    corrected = base.detach().float() - alpha * (anti.float() - base.detach().float())
    return corrected.log_softmax(-1).gather(-1, targets[:, None]).mean()


def paired_subtraction_loss(good_base, good_anti, good_targets,
                            bad_base, bad_anti, bad_targets, *,
                            alpha=.4, rank_weight=.5, margin=.5):
    """Reward the correct response and rank it above a supplied misconception.

    Inputs contain only the causal positions predicting response tokens. The
    base is frozen; gradients through the anti-expert have the subtraction sign.
    Prompt labels and generation outputs never enter the objective as targets.
    """
    for name, value in [('alpha', alpha), ('rank_weight', rank_weight), ('margin', margin)]:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not 0 < alpha <= .8:
        raise ValueError('alpha must be positive and at most .8')
    good = _response_scores(good_base, good_anti, good_targets, alpha)
    bad = _response_scores(bad_base, bad_anti, bad_targets, alpha)
    correct_ce = -good
    ranking = F.softplus(margin - (good - bad))
    loss = correct_ce + rank_weight * ranking
    return loss, {'correct_ce': correct_ce.detach(), 'ranking': ranking.detach(),
                  'response_margin': (good-bad).detach()}
