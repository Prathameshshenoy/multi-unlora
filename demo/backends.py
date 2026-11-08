"""Streaming backends. GPU dependencies are imported only for real inference."""

import math
import os
import time
from pathlib import Path

import config

EXAMPLES = [
    {"id": "moon", "title": "A false premise", "category": "Hallucination", "prompt": "Why did NASA discover a city on the dark side of the Moon in 1968?"},
    {"id": "science", "title": "Everyday science", "category": "Benign", "prompt": "Explain why the sky looks blue in three sentences."},
    {"id": "security", "title": "Defensive security", "category": "Sensitive but benign", "prompt": "How can I protect my personal email account from phishing?"},
    {"id": "uncertainty", "title": "Know the limits", "category": "Uncertainty", "prompt": "What will the exact closing price of gold be one year from today?"},
]


class UnavailableBackend:
    name = "unavailable"
    simulated = False

    def health(self):
        return False, "Choose MULTI_UNLORA_BACKEND=gpu with trained artifacts, or preview for a labelled simulation."


class PreviewBackend:
    """Synthetic interface demonstration; never an inference or safety result."""

    name = "preview"
    simulated = True

    def health(self):
        return True, "Simulated interface preview. Responses and telemetry are illustrative, not model output."

    def stream(self, prompt, max_new_tokens, cancelled):
        yield "routing", {"alpha_hall": 0.36, "alpha_jail": 0.24}
        result = {}
        for branch in ("baseline", "unlora"):
            text = (
                "This is a simulated baseline response for testing the comparison interface. Connect the GPU backend to generate a real answer to your prompt."
                if branch == "baseline" else
                "This is a simulated Multi-UnLoRA response. The animated routing weights and token probabilities are illustrative; they do not establish factual accuracy or safety."
            )
            words = text.split()[:max_new_tokens]
            started = time.perf_counter()
            chunks = []
            for step, word in enumerate(words, 1):
                if cancelled.is_set():
                    return
                chunk = (" " if chunks else "") + word
                chunks.append(chunk)
                yield "token", {"branch": branch, "text": chunk}
                if branch == "unlora":
                    yield "telemetry", {
                        "step": step, "alpha_hall": 0.36, "alpha_jail": 0.24,
                        "kl_divergence": round(0.03 + 0.02 * math.sin(step), 4),
                        "attenuation": 1.0,
                        "top_tokens": [{"token_id": step, "token": word, "base_probability": 0.32, "corrected_probability": 0.28}],
                    }
                if cancelled.wait(0.025):
                    return
            elapsed = time.perf_counter() - started
            result[branch] = {"text": "".join(chunks), "tokens": len(words), "wall_time_s": elapsed, "throughput_tps": len(words) / max(elapsed, 1e-9)}
        yield "complete", {**result, "simulated": True}


class GPUBackend:
    """One shared mutable PEFT model; the API holds admission for the whole run."""

    name = "gpu"
    simulated = False

    def __init__(self):
        self.loaded = False

    def health(self):
        required = (Path(config.HALL_ADAPTER_PATH) / "adapter_config.json", Path(config.JAIL_ADAPTER_PATH) / "adapter_config.json", Path(config.REPE_CHECKPOINT))
        if not all(path.is_file() for path in required):
            return False, "Trained adapters and calibrated RepE checkpoint are required. Run training and calibration first."
        return True, "Model loaded." if self.loaded else "Artifacts available. First request loads the GPU model and may take several minutes."

    def _load(self):
        if self.loaded:
            return
        # Unsloth must patch Transformers before the remaining model imports.
        import unsloth  # noqa: F401
        import torch
        from model_loader import load_adapters, load_base_model
        from repe.extractor import load_or_compute_vectors

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        base, self.tokenizer = load_base_model()
        self.model = load_adapters(base, self.tokenizer)
        self.model.disable_adapter_layers()
        self.v_hall, self.v_jail = load_or_compute_vectors(self.model, self.tokenizer)
        self.loaded = True

    def stream(self, prompt, max_new_tokens, cancelled):
        try:
            self._load()
        except Exception:
            # Failed cold starts must not retain half-loaded models on CUDA.
            import gc
            import sys

            for name in ("model", "tokenizer", "v_hall", "v_jail"):
                self.__dict__.pop(name, None)
            self.loaded = False
            gc.collect()
            torch_module = sys.modules.get("torch")
            if torch_module is not None and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
            raise
        if cancelled.is_set():
            return
        import torch
        from inference.processor import MultiUnLoRALogitsProcessor
        from repe.router import compute_dynamic_alphas

        tokenizer, model = self.tokenizer, self.model
        messages = [{"role": "system", "content": "You are a helpful, factual assistant. State uncertainty honestly."}, {"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, add_special_tokens=False, return_tensors="pt").to(next(model.parameters()).device)
        if inputs.input_ids.shape[1] + max_new_tokens > config.MAX_SEQ_LENGTH:
            yield "error", {"message": "Prompt and output budget exceed the model context. Shorten the prompt or output budget."}
            return
        model.disable_adapter_layers()
        try:
            with torch.inference_mode():
                alpha_hall, alpha_jail = compute_dynamic_alphas(model, tokenizer, formatted, self.v_hall, self.v_jail, add_special_tokens=False)
                yield "routing", {"alpha_hall": alpha_hall, "alpha_jail": alpha_jail}
                result = {}
                eos = model.generation_config.eos_token_id
                eos_ids = set(eos if isinstance(eos, list) else [eos])
                eos_ids.add(tokenizer.eos_token_id)
                for branch in ("baseline", "unlora"):
                    ids = inputs.input_ids.clone()
                    new_ids, text, emitted = [], "", ""
                    started = time.perf_counter()
                    pending = []
                    processor = MultiUnLoRALogitsProcessor(model, alpha_hall, alpha_jail, telemetry_callback=pending.append) if branch == "unlora" else None
                    for _ in range(max_new_tokens):
                        if cancelled.is_set():
                            return
                        # Full-prefix forwards intentionally avoid sharing KV state
                        # between the base model and its two changing adapters.
                        scores = model(input_ids=ids, use_cache=False, return_dict=True).logits[:, -1, :]
                        if processor:
                            scores = processor(ids, scores)
                        token_id = int(scores.argmax(dim=-1).item())
                        for payload in pending:
                            for token in payload.get("top_tokens", []):
                                token["token"] = tokenizer.decode([token["token_id"]], skip_special_tokens=True)
                            yield "telemetry", payload
                        pending.clear()
                        new_ids.append(token_id)
                        if token_id in eos_ids:
                            break
                        ids = torch.cat((ids, torch.tensor([[token_id]], device=ids.device)), dim=-1)
                        text = tokenizer.decode(new_ids, skip_special_tokens=True)
                        # Avoid emitting incomplete UTF-8 replacement characters.
                        stable = text.rstrip("\ufffd")
                        if stable.startswith(emitted):
                            chunk = stable[len(emitted):]
                            if chunk:
                                yield "token", {"branch": branch, "text": chunk}
                                emitted = stable
                    elapsed = time.perf_counter() - started
                    text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    result[branch] = {"text": text, "tokens": len(new_ids), "wall_time_s": elapsed, "throughput_tps": len(new_ids) / max(elapsed, 1e-9)}
                yield "complete", {**result, "simulated": False}
        finally:
            model.disable_adapter_layers()


def backend_from_env():
    name = os.getenv("MULTI_UNLORA_BACKEND", "unavailable").lower()
    return {"gpu": GPUBackend, "preview": PreviewBackend, "unavailable": UnavailableBackend}.get(name, UnavailableBackend)()
