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


def _absolute_path(value, label):
    if not isinstance(value, (str, Path)) or not str(value) or "\0" in str(value):
        raise RuntimeError(f"Invalid {label} path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"{label} must be an absolute path without parent traversal")
    return path


def _read_manifest(manifest_path=None):
    path = _absolute_path(manifest_path, "manifest") if manifest_path is not None else Path(config.ADAPTER_DIR)/"manifest.json"
    if not path.is_file():
        raise RuntimeError("Missing versioned adapter manifest; run engine training first")
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("Invalid adapter manifest") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("Invalid adapter manifest")
    return path, manifest


def _paths_from_manifest(path, manifest, *, custom=False):
    selected = manifest.get("adapter_paths", {})
    if not isinstance(selected, dict) or set(selected)-{"hallucination", "jailbreak"}:
        raise RuntimeError("Invalid adapter_paths; only named hallucination/jailbreak paths are supported")
    quality_default = path.parent/"quality_report.json" if custom else Path(config.RESULTS_DIR)/"pilot_quality.json"
    return {
        "manifest": path,
        "hallucination": _absolute_path(selected.get("hallucination", config.HALL_ADAPTER_PATH), "hallucination adapter"),
        "jailbreak": _absolute_path(selected.get("jailbreak", config.JAIL_ADAPTER_PATH), "jailbreak adapter"),
        "repe": _absolute_path(manifest.get("repe_checkpoint", config.REPE_CHECKPOINT), "RepE checkpoint"),
        "quality_report": _absolute_path(manifest.get("quality_report", quality_default), "quality report"),
    }


def artifact_paths(manifest_path=None):
    """Resolve a trusted local bundle without importing Torch or loading weights.

    This does not validate model/weight compatibility; validate_artifacts does.
    A custom bundle never inherits the canonical pilot's quality-report badge.
    """
    path, manifest = _read_manifest(manifest_path)
    return _paths_from_manifest(path, manifest, custom=manifest_path is not None)


def validate_artifacts(manifest_path=None):
    """Verify immutable model identity and both adapter bytes before GPU loading.

    This establishes artifact consistency, not adapter quality. Pilot artifacts
    remain unvalidated until their separate held-out quality gate passes.
    """
    path, manifest = _read_manifest(manifest_path)
    paths = _paths_from_manifest(path, manifest, custom=manifest_path is not None)
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
    for concept, directory in (("hallucination", paths["hallucination"]),
                               ("jailbreak", paths["jailbreak"])):
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
    corpus=manifest.get("training_corpus")
    if corpus is not None:
        ids=corpus.get("question_ids") if isinstance(corpus,dict) else None
        if (not isinstance(ids,list) or not ids
                or any(not isinstance(key,str) or not re.fullmatch(r"[0-9a-f]{64}",key) for key in ids)
                or len(set(ids))!=len(ids)):
            raise RuntimeError("Invalid training-corpus question provenance")
    return manifest


def load_inference_model(*, calibrate_vectors=False, manifest_path=None):
    """Load validated adapters, plus strictly matched calibration directions.

    Visitor inference never creates calibration artifacts. Explicit evaluation
    may opt into recalibration after a pilot made a legacy checkpoint.
    """
    manifest = validate_artifacts(manifest_path=manifest_path)
    paths = artifact_paths(manifest_path)
    if not calibrate_vectors and not paths["repe"].is_file():
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
    model = PeftModel.from_pretrained(base, str(paths["hallucination"]), adapter_name="hallucination")
    model.load_adapter(str(paths["jailbreak"]), adapter_name="jailbreak")
    model.unlora_training_question_ids = frozenset(manifest.get("training_corpus",{}).get("question_ids",[]))
    model.eval()
    model.disable_adapter_layers()
    metadata = vector_provenance(model_revision=revision)
    v_hall, v_jail = load_or_compute_vectors(
        model, tokenizer, allow_compute=calibrate_vectors, expected_metadata=metadata,
        checkpoint_path=paths["repe"],
    )
    return model, tokenizer, v_hall, v_jail
