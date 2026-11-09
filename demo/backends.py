"""Streaming backends. GPU dependencies are imported only for real inference."""

import math
import json
import os
import threading
import time
from pathlib import Path

import config

BRANCHES = ("baseline", "hallucination", "jailbreak", "unlora")

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
        for branch in BRANCHES:
            text = f"Simulated {branch} output for interface testing. Connect real inference to generate model responses. These words and measurements are illustrative."
            all_words = text.split()
            words = all_words[:max_new_tokens]
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
            result[branch] = {"text": "".join(chunks), "tokens": len(words), "wall_time_s": elapsed, "throughput_tps": len(words) / max(elapsed, 1e-9), "finish_reason": "length" if len(all_words) > max_new_tokens else "eos"}
        yield "complete", {**result, "simulated": True}


class GPUBackend:
    """One shared mutable PEFT model; the API holds admission for the whole run."""

    name = "gpu"
    simulated = False

    def __init__(self, manifest_path=None):
        self.loaded = False
        self.loading = False
        self.load_failed = False
        selected = manifest_path if manifest_path is not None else os.getenv("MULTI_UNLORA_MANIFEST")
        self.manifest_path = str(selected) if selected else None

    def _paths(self):
        from inference.runtime import artifact_paths
        return artifact_paths(self.manifest_path)

    def metadata(self):
        """Read public provenance without importing the GPU stack or leaking paths."""
        try:
            paths = self._paths()
        except (OSError, ValueError, RuntimeError):
            paths = {}
        manifest = _read_json(paths["manifest"]) if paths else {}
        report = _read_json(paths["quality_report"]) if paths else {}
        # A report for different adapters is never evidence for this model.
        if report:
            import hashlib
            digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            settings = {"alpha_cap": config.ALPHA_CAP,
                        "beta": config.PROCESSOR_PLAUSIBILITY_BETA,
                        "kl_budget": config.PROCESSOR_KL_THRESHOLD}
            reported_settings = report.get("settings")
            settings_match = isinstance(reported_settings, dict) and all(
                not isinstance(reported_settings.get(key), bool) and
                reported_settings.get(key) == value for key, value in settings.items()
            )
            if (report.get("manifest_fingerprint") != digest or
                    not settings_match or
                    report.get("operator") != getattr(self, "operator_version", None)):
                report = {}
        gate_result = report.get("forced_choice_gate_passed")
        return {
            "engine_state": "warm" if self.loaded else ("loading" if self.loading else ("load_failed" if self.load_failed else "cold")),
            "model_loaded": self.loaded,
            "model_revision": manifest.get("model_revision"),
            "training_status": manifest.get("status", "unknown"),
            "adapter_set": "selected" if self.manifest_path else "canonical",
            "manifest_fingerprint": _manifest_fingerprint(manifest) if manifest else None,
            "operator": getattr(self, "operator_version", None),
            "decoder_strategy": getattr(self, "decoder_strategy", None),
            "branches": list(BRANCHES),
            "quality": {
                "status": "pending" if type(gate_result) is not bool else ("forced_choice_passed" if gate_result else "forced_choice_failed"),
                "production_quality_verified": gate_result is True and report.get("production_quality_verified") is True,
            },
        }

    def health(self):
        if self.load_failed:
            return False, "Model initialization failed. Check the service logs and restart the session after fixing the artifacts."
        try:
            paths = self._paths()
        except (OSError, ValueError, RuntimeError):
            return False, "Model artifact selection is invalid. Check the selected manifest."
        required = (paths["manifest"], paths["hallucination"] / "adapter_config.json",
                    paths["jailbreak"] / "adapter_config.json", paths["repe"])
        if not all(path.is_file() for path in required):
            return False, "Trained adapters and calibrated RepE checkpoint are required. Run training and calibration first."
        return True, "Model loaded." if self.loaded else "Artifacts available. First request loads the GPU model and may take several minutes."

    def _load(self):
        if self.loaded:
            return
        from inference.runtime import load_inference_model
        self.model, self.tokenizer, self.v_hall, self.v_jail = load_inference_model(manifest_path=self.manifest_path)
        self.loaded = True

    def stream(self, prompt, max_new_tokens, cancelled):
        self.loading = not self.loaded
        try:
            self._load()
            self.load_failed = False
        except Exception:
            # Failed cold starts must not retain half-loaded models on CUDA.
            import gc
            import sys

            for name in ("model", "tokenizer", "v_hall", "v_jail"):
                self.__dict__.pop(name, None)
            self.loaded = False
            self.load_failed = True
            gc.collect()
            torch_module = sys.modules.get("torch")
            if torch_module is not None and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
            raise
        finally:
            self.loading = False
        if cancelled.is_set():
            return
        import torch
        from inference.processor import MultiUnLoRALogitsProcessor
        from inference.operator import OPERATOR_VERSION
        from repe.router import compute_dynamic_alphas

        from demo.display_policy import assess_prompt, present_response
        policy = assess_prompt(prompt)
        tokenizer, model = self.tokenizer, self.model
        self.operator_version = OPERATOR_VERSION
        self.decoder_strategy = "per-adapter-kv-v1"
        yield "metadata", {"backend": self.name, "simulated": False, "model": config.BASE_MODEL_NAME,
                           "display_restricted": policy["restricted"], "display_reason": policy.get("reason", ""),
                           "display_policy_version": policy.get("policy_version"), **self.metadata()}
        from inference.prompts import format_prompt
        formatted = format_prompt(tokenizer, prompt)
        inputs = tokenizer(formatted, add_special_tokens=False, return_tensors="pt").to(next(model.parameters()).device)
        if inputs.input_ids.shape[1] + max_new_tokens > config.MAX_SEQ_LENGTH:
            yield "error", {"message": "Prompt and output budget exceed the model context. Shorten the prompt or output budget."}
            return
        from inference.decoder import CacheDecoder
        model.disable_adapter_layers()
        try:
            with torch.inference_mode():
                alpha_hall, alpha_jail = compute_dynamic_alphas(model, tokenizer, formatted, self.v_hall, self.v_jail, add_special_tokens=False)
                yield "routing", {"alpha_hall": alpha_hall, "alpha_jail": alpha_jail}
                result = {}
                eos = getattr(model.generation_config, "eos_token_id", None)
                eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
                eos_ids.add(getattr(tokenizer, "eos_token_id", None))
                eos_ids.discard(None)
                decoder = CacheDecoder(model)
                for branch in BRANCHES:
                    if cancelled.is_set():
                        return
                    decoder.reset()
                    if policy["restricted"] and branch in ("hallucination", "jailbreak"):
                        presented = present_response(policy, branch, "")
                        result[branch] = {**presented, "tokens": 0, "wall_time_s": 0.0,
                                          "throughput_tps": 0.0, "finish_reason": "withheld"}
                        yield "branch_status", {"branch": branch, **result[branch]}
                        continue
                    ids = inputs.input_ids.clone()
                    mask = inputs.get("attention_mask")
                    mask = mask.clone() if mask is not None else torch.ones_like(ids)
                    new_ids, text, emitted = [], "", ""
                    finish_reason = "length"
                    started = time.perf_counter()
                    pending = []
                    processor = MultiUnLoRALogitsProcessor(model, alpha_hall, alpha_jail,
                        telemetry_callback=pending.append, attention_mask=inputs.get("attention_mask"),
                        expert_logits_provider=lambda prefix, adapter: decoder.next_logits(
                            prefix, attention_mask=mask, adapter=adapter)) if branch == "unlora" else None
                    try:
                        # Each reference expert is a genuine independent adapter
                        # decode, on exactly the same prompt as the other arms.
                        # Decoder selects/restores the adapter for each forward.
                        adapter = branch if branch in ("hallucination", "jailbreak") else None
                        yield "branch_status", {"branch": branch, "status": "generating",
                            "buffered": policy["restricted"], "reason": policy.get("reason", "")}
                        for _ in range(max_new_tokens):
                            if cancelled.is_set():
                                return
                            scores = decoder.next_logits(ids, attention_mask=mask, adapter=adapter)
                            if processor:
                                scores = processor(ids, scores)
                            if (torch.isnan(scores).any() or torch.isposinf(scores).any() or
                                    not torch.isfinite(scores).any(dim=-1).all()):
                                raise ValueError("Model produced an invalid next-token distribution")
                            token_id = int(scores.argmax(dim=-1).item())
                            for payload in pending:
                                if policy["restricted"]:
                                    # Expert token strings/IDs are raw model output
                                    # too: expose numerical aggregate shifts only.
                                    payload["top_tokens"] = []
                                else:
                                    for token in payload.get("top_tokens", []):
                                        token["token"] = tokenizer.decode([token["token_id"]], skip_special_tokens=True)
                                yield "telemetry", payload
                            pending.clear()
                            new_ids.append(token_id)
                            if token_id in eos_ids:
                                finish_reason = "eos"
                                break
                            ids = torch.cat((ids, torch.tensor([[token_id]], device=ids.device)), dim=-1)
                            mask = torch.cat((mask, mask.new_ones((mask.shape[0], 1))), dim=-1)
                            text = tokenizer.decode(new_ids, skip_special_tokens=True)
                            stable = text.rstrip("\ufffd")
                            if not policy["restricted"] and stable.startswith(emitted):
                                chunk = stable[len(emitted):]
                                if chunk:
                                    yield "token", {"branch": branch, "text": chunk}
                                    emitted = stable
                        elapsed = time.perf_counter() - started
                        text = tokenizer.decode(new_ids, skip_special_tokens=True)
                        presented = present_response(policy, branch, text)
                        result[branch] = {**presented, "tokens": len(new_ids), "wall_time_s": elapsed,
                            "throughput_tps": len(new_ids) / max(elapsed, 1e-9), "finish_reason": finish_reason}
                        yield "branch_status", {"branch": branch, "status": "complete", **result[branch]}
                    finally:
                        # Cancellation, errors, or closing a suspended stream must
                        # restore the clean base state before admitting another run.
                        decoder.reset()
                yield "complete", {**result, "simulated": False}
        finally:
            model.disable_adapter_layers()


def _read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _manifest_fingerprint(manifest):
    import hashlib
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def backend_from_env():
    name = os.getenv("MULTI_UNLORA_BACKEND", "unavailable").lower()
    return {"gpu": GPUBackend, "preview": PreviewBackend, "unavailable": UnavailableBackend, "remote": RemoteBackend}.get(name, UnavailableBackend)()


class RemoteBackend:
    """Localhost bridge to a real HTTPS GPU API; credentials stay server-side."""

    name = "remote"
    simulated = False
    _PUBLIC_METADATA = ("model", "engine_state", "model_loaded", "model_revision",
                        "training_status", "operator", "quality", "session_expires_at",
                        "branches", "display_restricted", "display_reason", "display_policy_version", "decoder_strategy",
                        "adapter_set", "manifest_fingerprint")

    def __init__(self, url=None, token=None, client=None):
        from urllib.parse import urlsplit

        self.url = (url or os.getenv("MULTI_UNLORA_REMOTE_URL", "")).rstrip("/")
        parsed = urlsplit(self.url)
        if (parsed.scheme != "https" or not parsed.netloc or parsed.username or
                parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Remote inference needs an HTTPS URL without embedded credentials")
        self.token = token if token is not None else os.getenv("MULTI_UNLORA_REMOTE_TOKEN", "")
        self.client = client
        self._remote_metadata = {}

    def _client(self):
        import httpx

        # Cold model/container starts can take minutes. A separate cancellation
        # watcher closes a live response rather than waiting for this timeout.
        return self.client or httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0))

    def metadata(self):
        return dict(self._remote_metadata)

    def _remember(self, data):
        self._remote_metadata.update({key: data[key] for key in self._PUBLIC_METADATA if key in data})

    def health(self):
        client = self._client()
        try:
            response = client.get(self.url + "/api/health", headers={"Authorization": "Bearer " + self.token}, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            if data.get("backend") != "gpu" or data.get("simulated") is True:
                return False, "Remote endpoint is not a real GPU inference backend."
            self._remember(data)
            return data.get("ready") is True, data.get("detail", "Remote GPU backend.")
        except Exception:
            return False, "Cannot reach the GPU service. Check the local server configuration."
        finally:
            if self.client is None:
                client.close()

    def stream(self, prompt, max_new_tokens, cancelled):
        client = self._client()
        finished = threading.Event()
        response_holder = []
        verified = False
        completed = False

        def close_when_cancelled():
            while not finished.wait(0.1):
                if cancelled.is_set():
                    try:
                        if response_holder:
                            response_holder[0].close()
                        elif self.client is None:
                            client.close()
                    except Exception:
                        pass
                    return

        def parse(event, lines):
            nonlocal verified, completed
            payload = json.loads("\n".join(lines))
            if not isinstance(payload, dict):
                raise ValueError("Remote stream event must be a JSON object")
            if event == "error":
                return "error", {"message": "GPU generation failed. Check the service logs."}
            if event == "metadata":
                if payload.get("simulated") is not False or payload.get("backend") != "gpu":
                    raise ValueError("Remote endpoint must provide real GPU inference")
                verified = True
                self._remember(payload)
                return "metadata", {"backend": self.name, "simulated": False, **self.metadata()}
            if not verified:
                raise ValueError("GPU stream did not identify its model backend")
            if event == "complete":
                if payload.get("simulated") is not False:
                    raise ValueError("Remote completion must be real inference")
                completed = True
            if event in ("routing", "token", "telemetry", "branch_status", "complete"):
                return event, payload
            return None

        watcher = threading.Thread(target=close_when_cancelled, daemon=True, name="unlora-remote-cancel")
        watcher.start()
        try:
            if cancelled.is_set():
                return
            with client.stream("POST", self.url + "/api/generate",
                               headers={"Authorization": "Bearer " + self.token},
                               json={"prompt": prompt, "max_new_tokens": max_new_tokens}) as response:
                response_holder.append(response)
                if response.status_code != 200:
                    yield "error", {"message": f"GPU service returned HTTP {response.status_code}. Check server configuration."}
                    return
                event, lines, size = "message", [], 0
                for line in response.iter_lines():
                    if cancelled.is_set():
                        return
                    if not line:
                        if lines:
                            item = parse(event, lines)
                            if item:
                                yield item
                        event, lines, size = "message", [], 0
                    elif line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        size += len(line)
                        if size > 1_000_000:
                            raise ValueError("Remote event exceeds the stream limit")
                        lines.append(line[5:].removeprefix(" "))
                if lines and not cancelled.is_set():
                    item = parse(event, lines)
                    if item:
                        yield item
                if not cancelled.is_set() and not completed:
                    yield "error", {"message": "GPU connection ended before the comparison finished."}
        except Exception:
            if not cancelled.is_set():
                raise
        finally:
            finished.set()
            watcher.join(timeout=0.2)
            if self.client is None:
                client.close()
