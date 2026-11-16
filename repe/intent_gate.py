"""Checksum-bound prompt-intent routing for an explicitly selected checkpoint."""
import hashlib
import json
import math
from pathlib import Path

import config
from inference.prompts import PROMPT_PROTOCOL, BRIEF_INSTRUCTION, comparison_prompt, format_prompt

VERSION = 'intent-gate-v1'
FEATURE_PROTOCOL = 'base-hidden-safety-router-v1'


def settings(manifest):
    value = manifest.get('intent_gate')
    if value is None:
        return None
    required = {'path', 'sha256', 'threshold', 'active_alphas', 'inactive_alphas'}
    if not isinstance(value, dict) or set(value) != required or manifest.get('fixed_alphas') is not None:
        raise RuntimeError('Invalid or conflicting intent-gate configuration')
    path = value['path']
    if not isinstance(path, str) or not Path(path).is_absolute() or '..' in Path(path).parts:
        raise RuntimeError('Intent-gate path must be absolute without traversal')
    digest = value['sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise RuntimeError('Invalid intent-gate checksum')
    threshold = value['threshold']
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise RuntimeError('Invalid intent-gate threshold')
    for key in ('active_alphas', 'inactive_alphas'):
        weights = value[key]
        if (not isinstance(weights, list) or len(weights) != 2 or
                any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in weights)
                or sum(weights) > config.ALPHA_CAP):
            raise RuntimeError('Intent-gate weights exceed the correction cap')
    return value


def load(manifest):
    spec = settings(manifest)
    if spec is None:
        return None
    path = Path(spec['path'])
    if path.stat().st_size > 1024*1024:
        raise RuntimeError('Oversized intent-gate artifact')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['sha256']:
        raise RuntimeError('Intent-gate checksum mismatch')
    value = json.loads(raw)
    if (not isinstance(value, dict) or value.get('protocol') != FEATURE_PROTOCOL or
            value.get('model_revision') != manifest['model_revision'] or
            value.get('prompt_protocol') != PROMPT_PROTOCOL or value.get('layer') != config.REPE_LAYER_IDX):
        raise RuntimeError('Intent-gate provenance mismatch')
    vectors = [value.get('weight'), value.get('center')]
    if (any(not isinstance(v, list) or not v for v in vectors) or len(vectors[0]) != len(vectors[1]) or
            any(type(x) not in (int, float) or not math.isfinite(x) for v in vectors for x in v) or
            type(value.get('bias')) not in (int, float) or not math.isfinite(value['bias'])):
        raise RuntimeError('Invalid intent-gate parameters')
    return {'settings': spec, 'calibration': value}


def feature_prompt(tokenizer, prompt):
    normalized = prompt if prompt.rstrip().endswith(BRIEF_INSTRUCTION) else comparison_prompt(prompt, 'brief')
    return format_prompt(tokenizer, normalized)


def from_hidden(hidden, gate):
    import torch
    import torch.nn.functional as F
    calibration, spec = gate['calibration'], gate['settings']
    h = hidden.float()
    weight = torch.as_tensor(calibration['weight'], device=h.device, dtype=torch.float32)
    center = torch.as_tensor(calibration['center'], device=h.device, dtype=torch.float32)
    if h.ndim not in (1, 2) or h.shape[-1] != weight.numel() or weight.shape != center.shape:
        raise RuntimeError('Intent-gate hidden size mismatch')
    probability = torch.sigmoid((F.normalize(h, dim=-1)-center) @ weight + calibration['bias'])
    if not torch.isfinite(probability).all():
        raise RuntimeError('Nonfinite intent-gate probability')
    active = torch.as_tensor(spec['active_alphas'], device=h.device, dtype=torch.float32)
    inactive = torch.as_tensor(spec['inactive_alphas'], device=h.device, dtype=torch.float32)
    weights = torch.where((probability >= spec['threshold'])[..., None], active, inactive)
    return probability, weights


def route(model, tokenizer, prompt, gate):
    import torch
    from repe.extractor import get_hidden_state_via_hook
    model.eval()
    model.disable_adapter_layers()
    with torch.inference_mode():
        hidden = get_hidden_state_via_hook(model, tokenizer, feature_prompt(tokenizer, prompt),
                                         gate['calibration']['layer'], add_special_tokens=False)
        probability, weights = from_hidden(hidden, gate)
    return tuple(float(v) for v in weights), float(probability)
