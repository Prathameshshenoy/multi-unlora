"""Explicit HF decoding forwards with isolated base and per-adapter KV caches.

A decoder belongs to one generation request. Never share it between concurrent
requests or between output branches. ``reset`` drops all references to previous
branch caches; each adapter independently prefills the exact same prompt.
"""
from dataclasses import dataclass

import torch

from inference.adapters import active_adapter


@dataclass
class _State:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: object
    logits: torch.Tensor


class CacheDecoder:
    def __init__(self, model, *, use_cache=True):
        if getattr(model, "training", False):
            raise ValueError("Call model.eval() before constructing an inference decoder")
        self.model = model
        self.use_cache = bool(use_cache)
        self._states = {}
        self.token_forwards = {}

    def reset(self):
        """Discard every adapter's cache when beginning another output branch."""
        self._states.clear()
        self.token_forwards.clear()
        self.model.disable_adapter_layers()

    def _inputs(self, input_ids, attention_mask):
        if input_ids.ndim != 2 or not all(input_ids.shape):
            raise ValueError("input_ids must be a nonempty [batch, sequence] tensor")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match the full generation prefix")
        if not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("attention_mask must be binary")
        if not attention_mask[:, -1].bool().all():
            raise ValueError("Use unpadded or left-padded prefixes with a real final token")
        return attention_mask

    def next_logits(self, input_ids, attention_mask=None, adapter=None):
        """Get [batch, vocabulary] last-token logits for one isolated model arm.

        Reusing a cache requires an exact token/mask prefix extension. Changed,
        or shortened inputs prefill independently; identical inputs reuse the
        previous logits. No KV cache is ever forked from base into an expert.
        """
        attention_mask = self._inputs(input_ids, attention_mask)
        key = adapter
        previous = self._states.get(key) if self.use_cache else None
        prefix_length = 0
        if previous is not None:
            old = previous.input_ids
            if (input_ids.shape[0] == old.shape[0] and input_ids.device == old.device
                    and input_ids.shape[1] >= old.shape[1]
                    and torch.equal(input_ids[:, :old.shape[1]], old)
                    and torch.equal(attention_mask[:, :old.shape[1]], previous.attention_mask)):
                if input_ids.shape[1] == old.shape[1]:
                    self.model.disable_adapter_layers()
                    return previous.logits
                prefix_length = old.shape[1]
            else:
                self._states.pop(key, None)
                previous = None
        suffix = input_ids[:, prefix_length:]
        positions = attention_mask.long().cumsum(-1) - 1
        positions.masked_fill_(attention_mask == 0, 0)
        kwargs = dict(input_ids=suffix, attention_mask=attention_mask,
                      position_ids=positions[:, prefix_length:],
                      use_cache=self.use_cache, return_dict=True)
        if previous is not None:
            kwargs["past_key_values"] = previous.past_key_values
        try:
            with active_adapter(self.model, adapter), torch.inference_mode():
                output = self.model(**kwargs)
                logits = output.logits[:, -1, :].float()
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite model logits invalidate generation")
                cache = getattr(output, "past_key_values", None)
                if self.use_cache and cache is None:
                    raise RuntimeError("Model did not return the requested independent KV cache")
        except Exception:
            # A DynamicCache may already have mutated before a failing forward.
            # Drop all arms rather than letting callers reuse uncertain state.
            self._states.clear()
            raise
        self.token_forwards[key] = self.token_forwards.get(key, 0) + suffix.numel()
        if self.use_cache:
            self._states[key] = _State(input_ids.clone(), attention_mask.clone(), cache, logits)
        return logits
