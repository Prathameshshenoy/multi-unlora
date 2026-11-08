"""Validated, revision-pinned plain HF inference; no Unsloth monkey patches."""
import hashlib
import json
from pathlib import Path
import re

import config
from inference.prompts import PROMPT_PROTOCOL


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifacts():
    """Verify immutable model identity and both adapter bytes before GPU loading.

    This establishes artifact consistency, not adapter quality. Pilot artifacts
    remain unvalidated until their separate held-out quality gate passes.
    """
    manifest_path = Path(config.ADAPTER_DIR) / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Missing versioned adapter manifest; run engine training first")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("Invalid adapter manifest") from exc
    if not isinstance(manifest, dict) or manifest.get("base_model") != config.BASE_MODEL_NAME:
        raise RuntimeError("Adapter manifest base model mismatch")
    revision = manifest.get("model_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise RuntimeError("Adapter manifest must pin an immutable 40-character model revision")
    if manifest.get("prompt_protocol") != PROMPT_PROTOCOL:
        raise RuntimeError("Adapter prompt protocol mismatch; retrain using the current chat protocol")
    concepts = manifest.get("concepts", {})
    if not isinstance(concepts, dict):
        raise RuntimeError("Adapter manifest is missing concept metadata")
    for concept, directory in (("hallucination", config.HALL_ADAPTER_PATH),
                               ("jailbreak", config.JAIL_ADAPTER_PATH)):
        metadata = concepts.get(concept)
        expected_hash = metadata.get("adapter_sha256") if isinstance(metadata, dict) else None
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
            raise RuntimeError(f"Missing adapter hash for {concept}")
        weights = Path(directory) / "adapter_model.safetensors"
        adapter_config = Path(directory) / "adapter_config.json"
        if not weights.is_file() or not adapter_config.is_file():
            raise RuntimeError(f"Incomplete {concept} adapter artifacts")
        if sha256_file(weights) != expected_hash.lower():
            raise RuntimeError(f"Adapter checksum mismatch for {concept}; restore the matching artifacts")
        # The first frozen pilot recorded only weight hashes. Future manifests
        # also bind LoRA scaling/target settings, which affect inference even
        # when weight bytes remain unchanged.
        expected_config_hash = metadata.get("adapter_config_sha256")
        if expected_config_hash is not None and (
            not isinstance(expected_config_hash, str)
            or sha256_file(adapter_config) != expected_config_hash.lower()
        ):
            raise RuntimeError(f"Adapter configuration checksum mismatch for {concept}")
        try:
            settings = json.loads(adapter_config.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Invalid adapter configuration for {concept}") from exc
        if not isinstance(settings, dict):
            raise RuntimeError(f"Invalid adapter configuration for {concept}")
        if settings.get("base_model_name_or_path") != config.BASE_MODEL_NAME:
            raise RuntimeError(f"Adapter base model mismatch for {concept}")
        if settings.get("revision") not in (None, revision):
            raise RuntimeError(f"Adapter model revision mismatch for {concept}")
        if settings.get("peft_type") != "LORA" or settings.get("task_type") != "CAUSAL_LM":
            raise RuntimeError(f"Unsupported adapter type for {concept}")
    return manifest


def load_inference_model(*, calibrate_vectors=False):
    """Load validated adapters, plus strictly matched calibration directions.

    Visitor inference never creates calibration artifacts. Explicit evaluation
    may opt into recalibration after a pilot made a legacy checkpoint.
    """
    manifest = validate_artifacts()
    if not calibrate_vectors and not Path(config.REPE_CHECKPOINT).is_file():
        raise RuntimeError("Missing calibrated directions; run calibration explicitly")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from repe.extractor import load_or_compute_vectors, vector_provenance
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured 8B model")
    revision = manifest["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME, revision=revision)
    if tokenizer.eos_token is None or not tokenizer.chat_template:
        raise RuntimeError("The configured tokenizer needs an EOS token and native chat template")
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME, revision=revision, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, config.HALL_ADAPTER_PATH, adapter_name="hallucination")
    model.load_adapter(config.JAIL_ADAPTER_PATH, adapter_name="jailbreak")
    model.eval()
    model.disable_adapter_layers()
    metadata = vector_provenance(model_revision=revision)
    v_hall, v_jail = load_or_compute_vectors(
        model, tokenizer, allow_compute=calibrate_vectors, expected_metadata=metadata,
    )
    return model, tokenizer, v_hall, v_jail
