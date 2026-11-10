"""CPU-testable building blocks used by isolated anti-expert training trials.

These helpers do not select or promote a serving policy. The anti-expert intentionally
prefers the undesirable reference over the desirable one relative to a frozen
base. Its subtraction must then be evaluated independently; this loss does not
prove corrected-answer accuracy or safety.
"""
import math
import torch
import torch.nn.functional as F


def response_log_probabilities(logits, labels):
    """Sum causal response-token log probabilities; ignore instruction labels."""
    if logits.ndim!=3 or labels.ndim!=2 or not all(logits.shape) or logits.shape[:2]!=labels.shape or logits.shape[1]<2:
        raise ValueError('Need matching [batch,time,vocabulary] logits and [batch,time] labels')
    if labels.dtype!=torch.long:
        raise ValueError('Labels must use integer token IDs')
    if not torch.isfinite(logits).all() or not torch.isfinite(logits.float()).all():
        raise ValueError('Logits must be finite')
    shifted=labels[:,1:]
    mask=shifted!=-100
    if not mask.any(-1).all():
        raise ValueError('Every response needs a supervised causal token')
    if ((shifted[mask]<0)|(shifted[mask]>=logits.shape[-1])).any():
        raise ValueError('Response labels must be valid vocabulary IDs')
    log_probs=F.log_softmax(logits[:,:-1,:].float(),-1)
    safe_labels=shifted.masked_fill(~mask,0)
    values=log_probs.gather(-1,safe_labels.unsqueeze(-1)).squeeze(-1)
    return (values*mask).sum(-1)


def anti_expert_preference_loss(policy_bad,policy_good,reference_bad,reference_good,*,beta=.1):
    """DPO-style loss with intentionally reversed (anti-expert) preferences.

    This helper consumes the caller's explicit score units. The standard DPO
    formulation uses sums; the recorded paired_hall_v1 trial supplies per-token
    means and is explicitly a length-normalized variant. Reference values are
    always detached. Canonical pilot weights are unchanged by this helper.
    """
    values=(policy_bad,policy_good,reference_bad,reference_good)
    if not math.isfinite(beta) or beta<=0:
        raise ValueError('beta must be finite and positive')
    if policy_bad.ndim!=1 or not policy_bad.numel() or any(x.shape!=policy_bad.shape for x in values):
        raise ValueError('Need matching nonempty per-response score vectors')
    if any(not torch.isfinite(x).all() for x in values):
        raise ValueError('Sequence scores must be finite')
    relative=(policy_bad-policy_good)-(reference_bad.detach()-reference_good.detach())
    scaled=beta*relative
    if not torch.isfinite(scaled).all():
        raise ValueError('Preference margin overflowed')
    return -F.logsigmoid(scaled).mean()
