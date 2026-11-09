"""Numerical cache-fidelity diagnostic; never an answer-quality measurement.

The caller supplies an already loaded model. This module does not provision a
GPU, load weights, select correction settings, or store generated response text.
"""
import torch

import config
from inference.decoder import CacheDecoder
from inference.prompts import format_prompt
from inference.processor import MultiUnLoRALogitsProcessor

PROMPTS = (
    'Explain why the sky appears blue in three sentences.',
    'What is the difference between an observation and a hypothesis?',
)
STEPS = 4


def _tolerances(dtype):
    # Cached and full-prefix attention can take different reduction paths.
    # Exact greedy agreement is required in addition to this numerical envelope.
    if dtype == torch.bfloat16:
        return .125, .02
    if dtype == torch.float16:
        return .02, .01
    return 1e-5, 1e-4


def _summary():
    return {'comparisons': 0, 'top1_agreements': 0, 'finite_comparisons': 0,
            'within_tolerance': 0, 'max_absolute_logit_difference': 0.,
            'max_relative_logit_difference': 0.}


def _compare(summary, cached, oracle, *, atol, rtol):
    if cached.shape != oracle.shape or cached.ndim != 2:
        raise RuntimeError('Cached and oracle logits must have identical [batch, vocabulary] shapes')
    summary['comparisons'] += 1
    cached_finite, oracle_finite = torch.isfinite(cached), torch.isfinite(oracle)
    # Corrected logits may contain the same -inf plausibility masks. NaN/+inf
    # or disagreement about finite support invalidate numerical parity.
    matching_support = torch.equal(cached_finite, oracle_finite)
    valid = matching_support and not bool(torch.isnan(cached).any() or torch.isnan(oracle).any()
                                         or torch.isposinf(cached).any() or torch.isposinf(oracle).any())
    summary['finite_comparisons'] += int(valid)
    summary['top1_agreements'] += int(torch.equal(cached.argmax(-1), oracle.argmax(-1)))
    if valid:
        differences = (cached[cached_finite]-oracle[oracle_finite]).abs()
        relative = differences / oracle[oracle_finite].abs().clamp_min(1e-6)
        if differences.numel():
            summary['max_absolute_logit_difference'] = max(summary['max_absolute_logit_difference'],float(differences.max()))
            summary['max_relative_logit_difference'] = max(summary['max_relative_logit_difference'],float(relative.max()))
        close = bool(torch.isclose(cached,oracle,atol=atol,rtol=rtol,equal_nan=False).all())
        summary['within_tolerance'] += int(close)


def run_cache_verification(model, tokenizer):
    """Compare isolated caches against full-prefix forwards on 32 decisions.

    For each raw arm, the uncached arm's greedy token extends BOTH prefixes.
    A separate corrected trajectory uses identical fixed (.4,.3) weights only
    to test instrument fidelity; those strengths are not tuned or promoted.
    """
    if getattr(model,'training',False):
        raise ValueError('Cache verification requires model.eval()')
    parameter = next(model.parameters())
    dtype, device = parameter.dtype, parameter.device
    atol, rtol = _tolerances(dtype)
    arms = {name:_summary() for name in ('base','hallucination','jailbreak','corrected')}
    token_work = {name:{'cached':0,'uncached':0} for name in arms}
    try:
        for prompt in PROMPTS:
            formatted = format_prompt(tokenizer,prompt)
            inputs = tokenizer(formatted,return_tensors='pt',add_special_tokens=False).to(device)
            initial_ids = inputs.input_ids
            initial_mask = inputs.get('attention_mask',torch.ones_like(initial_ids))
            if initial_ids.shape[0] != 1 or initial_ids.shape[1]+STEPS > config.MAX_SEQ_LENGTH:
                raise ValueError('Verification prompts must fit a single full model context')
            for name,adapter in [('base',None),('hallucination','hallucination'),('jailbreak','jailbreak')]:
                cached, oracle = CacheDecoder(model),CacheDecoder(model,use_cache=False)
                ids, mask = initial_ids.clone(),initial_mask.clone()
                try:
                    for _ in range(STEPS):
                        actual = cached.next_logits(ids,mask,adapter=adapter)
                        expected = oracle.next_logits(ids,mask,adapter=adapter)
                        _compare(arms[name],actual,expected,atol=atol,rtol=rtol)
                        token = expected.argmax(-1,keepdim=True)
                        ids = torch.cat((ids,token),dim=-1)
                        mask = torch.cat((mask,mask.new_ones((1,1))),dim=-1)
                    token_work[name]['cached'] += sum(cached.token_forwards.values())
                    token_work[name]['uncached'] += sum(oracle.token_forwards.values())
                finally:
                    cached.reset()
                    oracle.reset()
            cached, oracle = CacheDecoder(model),CacheDecoder(model,use_cache=False)
            ids, mask = initial_ids.clone(),initial_mask.clone()
            # Closures deliberately read the current full attention mask.
            cached_processor = MultiUnLoRALogitsProcessor(model,.4,.3,
                expert_logits_provider=lambda tokens,adapter:cached.next_logits(tokens,mask,adapter=adapter))
            oracle_processor = MultiUnLoRALogitsProcessor(model,.4,.3,
                expert_logits_provider=lambda tokens,adapter:oracle.next_logits(tokens,mask,adapter=adapter))
            try:
                for _ in range(STEPS):
                    actual = cached_processor(ids,cached.next_logits(ids,mask))
                    expected = oracle_processor(ids,oracle.next_logits(ids,mask))
                    _compare(arms['corrected'],actual,expected,atol=atol,rtol=rtol)
                    token = expected.argmax(-1,keepdim=True)
                    ids = torch.cat((ids,token),dim=-1)
                    mask = torch.cat((mask,mask.new_ones((1,1))),dim=-1)
                token_work['corrected']['cached'] += sum(cached.token_forwards.values())
                token_work['corrected']['uncached'] += sum(oracle.token_forwards.values())
            finally:
                cached.reset()
                oracle.reset()
    finally:
        model.disable_adapter_layers()
    totals = _summary()
    for arm in arms.values():
        for name in ('comparisons','top1_agreements','finite_comparisons','within_tolerance'):
            totals[name] += arm[name]
        for name in ('max_absolute_logit_difference','max_relative_logit_difference'):
            totals[name] = max(totals[name],arm[name])
    passed = (totals['comparisons'] == 32 and totals['top1_agreements'] == 32
              and totals['finite_comparisons'] == 32 and totals['within_tolerance'] == 32)
    return {'protocol':'same-prefix-cached-uncached-logit-fidelity-v1', 'model_parameter_dtype':str(dtype),
        'prompt_count':len(PROMPTS), 'continuation_steps':STEPS,
        'tolerance':{'atol':atol,'rtol':rtol,'exact_greedy_agreement_required':True,
                     'relative_error_denominator_floor':1e-6,
                     'scope':'Explicit numerical envelope for reduction-path differences; not strict FP32 equivalence.'},
        'arms':arms,'aggregate':totals,'successful_token_forwards':token_work,
        'passed':passed,'quality_claim':False,
        'interpretation':'Numerical instrument check only; no factuality, safety, accuracy, or throughput claim.'}
