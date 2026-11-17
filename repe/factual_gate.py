"""Optional misconception routing after the checkpoint's safety gate."""
import hashlib
import json
import math
from pathlib import Path
import re

import config
from inference.prompts import BRIEF_INSTRUCTION, PROMPT_PROTOCOL, comparison_prompt, format_prompt

VERSION = 'safety-factual-gates-v1'
PROTOCOL = 'base-hidden-misconception-router-v1'
QUESTION_PROTOCOL = 'native-brief-question-content-v1'


def question_text(prompt):
    text = prompt.strip()
    if text.endswith(BRIEF_INSTRUCTION):
        text = text[:-len(BRIEF_INSTRUCTION)].rstrip()
    suffix = '\nReply with only the letter of the correct answer.'
    if text.endswith(suffix):
        body = text[:-len(suffix)]
        for match in reversed(list(re.finditer(r'\nA\. ', body))):
            if re.search(r'\nB\. ', body[match.start():]):
                text = body[:match.start()]
                break
    return text


def feature_prompt(tokenizer, prompt):
    return format_prompt(tokenizer, comparison_prompt(question_text(prompt), 'brief'))


def settings(manifest):
    spec = manifest.get('factual_gate')
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != {'path', 'sha256', 'threshold', 'alpha'}:
        raise RuntimeError('Invalid factual-gate configuration')
    safety = manifest.get('intent_gate')
    if (not isinstance(safety, dict) or safety.get('inactive_alphas') != [0., 0.]
            or manifest.get('fixed_alphas') is not None):
        raise RuntimeError('Factual routing requires a safety gate with zero inactive correction')
    path, digest = spec['path'], spec['sha256']
    if not isinstance(path, str) or not Path(path).is_absolute() or '..' in Path(path).parts:
        raise RuntimeError('Invalid factual-gate path')
    if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise RuntimeError('Invalid factual-gate checksum')
    for key, maximum in [('threshold', 1.), ('alpha', config.ALPHA_CAP)]:
        value = spec[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise RuntimeError('Invalid factual-gate strength or threshold')
    if spec['threshold'] == 1:
        raise RuntimeError('Factual-gate threshold must be below one')
    return spec


def load(manifest):
    spec = settings(manifest)
    if spec is None:
        return None
    path = Path(spec['path'])
    if path.stat().st_size > 1024*1024:
        raise RuntimeError('Oversized factual-gate artifact')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['sha256']:
        raise RuntimeError('Factual-gate checksum mismatch')
    value = json.loads(raw)
    if (not isinstance(value, dict) or set(value) != {'protocol','question_protocol','prompt_protocol','model_revision','layer','center','weight','bias'}
            or value.get('protocol') != PROTOCOL
            or value.get('question_protocol') != QUESTION_PROTOCOL
            or value.get('prompt_protocol') != PROMPT_PROTOCOL
            or value.get('model_revision') != manifest['model_revision']
            or value.get('layer') != config.REPE_LAYER_IDX):
        raise RuntimeError('Factual-gate provenance mismatch')
    vectors = [value.get('weight'), value.get('center')]
    if (any(not isinstance(v, list) or not v for v in vectors)
            or len(vectors[0]) != len(vectors[1])
            or any(type(x) not in (int, float) or not math.isfinite(x) for v in vectors for x in v)
            or type(value.get('bias')) not in (int, float) or not math.isfinite(value['bias'])):
        raise RuntimeError('Invalid factual-gate parameters')
    return {'settings': spec, 'calibration': value}


def from_hidden(hidden, gate):
    import torch
    import torch.nn.functional as F
    calibration = gate['calibration']
    h = F.normalize(hidden.float(), dim=-1).cpu()
    weight = torch.tensor(calibration['weight'], dtype=torch.float32)
    center = torch.tensor(calibration['center'], dtype=torch.float32)
    if h.ndim not in (1, 2) or h.shape[-1] != weight.numel():
        raise RuntimeError('Factual-gate hidden size mismatch')
    probability = torch.sigmoid((h-center) @ weight + calibration['bias'])
    if not torch.isfinite(probability).all():
        raise RuntimeError('Nonfinite factual-gate probability')
    return probability


def route(model, tokenizer, prompt, gate):
    import torch
    from repe.extractor import get_hidden_state_via_hook
    model.eval()
    model.disable_adapter_layers()
    with torch.inference_mode():
        hidden = get_hidden_state_via_hook(model, tokenizer, feature_prompt(tokenizer, prompt),
                                         gate['calibration']['layer'], add_special_tokens=False)
        probability = float(from_hidden(hidden, gate))
    return ((gate['settings']['alpha'], 0.) if probability >= gate['settings']['threshold'] else (0., 0.)), probability


def route_after_safety(model, tokenizer, prompt, primary_weights, safety_probability):
    gate = getattr(model, 'unlora_factual_gate', None)
    if gate is None or safety_probability >= model.unlora_intent_gate['settings']['threshold']:
        return primary_weights
    return route(model, tokenizer, prompt, gate)[0]
