"""Artifact consistency gates are independent of an expensive model download."""
import hashlib
import json
from types import SimpleNamespace

import pytest

import config
from inference import runtime
from inference.prompts import PROMPT_PROTOCOL


@pytest.fixture
def artifact_set(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ADAPTER_DIR", str(tmp_path))
    manifest = {"base_model": config.BASE_MODEL_NAME, "model_revision": "a" * 40,
                "prompt_protocol": PROMPT_PROTOCOL, "concepts": {}}
    for concept, setting in [("hallucination", "HALL_ADAPTER_PATH"), ("jailbreak", "JAIL_ADAPTER_PATH")]:
        directory = tmp_path / concept
        directory.mkdir()
        monkeypatch.setattr(config, setting, str(directory))
        data = concept.encode()
        (directory / "adapter_model.safetensors").write_bytes(data)
        (directory / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": config.BASE_MODEL_NAME, "revision": None,
            "task_type": "CAUSAL_LM", "peft_type": "LORA",
        }))
        manifest["concepts"][concept] = {"adapter_sha256": hashlib.sha256(data).hexdigest()}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, manifest


def test_matching_pilot_manifest_is_valid(artifact_set):
    _, manifest = artifact_set
    assert runtime.validate_artifacts() == manifest


def test_changed_adapter_bytes_are_rejected_before_model_loading(artifact_set):
    root, _ = artifact_set
    (root / "hallucination" / "adapter_model.safetensors").write_bytes(b"other training run")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        runtime.validate_artifacts()


@pytest.mark.parametrize("key,value,message", [
    ("model_revision", "main", "immutable"),
    ("prompt_protocol", "old", "prompt protocol"),
    ("base_model", "another/model", "base model"),
])
def test_manifest_mismatch_is_rejected(artifact_set, key, value, message):
    root, manifest = artifact_set
    manifest[key] = value
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match=message):
        runtime.validate_artifacts()


def test_adapter_configuration_must_match_manifest_model(artifact_set):
    root, _ = artifact_set
    settings = root / "jailbreak" / "adapter_config.json"
    content = json.loads(settings.read_text())
    content["base_model_name_or_path"] = "different/base"
    settings.write_text(json.dumps(content))
    with pytest.raises(RuntimeError, match="base model mismatch"):
        runtime.validate_artifacts()


def test_vector_provenance_mismatch_never_recalibrates_on_visitor_load(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from repe import extractor
    checkpoint = tmp_path / "vectors.pt"
    monkeypatch.setattr(config, "REPE_CHECKPOINT", str(checkpoint))
    monkeypatch.setattr(extractor, "_prepare_base_model", lambda *a: (None, None, "cpu"))
    torch.save({"prompt_protocol": PROMPT_PROTOCOL, "v_hall": torch.ones(4), "v_jail": torch.ones(4)}, checkpoint)
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        extractor.load_or_compute_vectors(None, None, allow_compute=False,
            expected_metadata=extractor.vector_provenance(model_revision="a" * 40))


def test_recalibration_writes_expected_metadata_atomically(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from repe import extractor
    checkpoint = tmp_path / "vectors.pt"
    monkeypatch.setattr(config, "REPE_CHECKPOINT", str(checkpoint))
    monkeypatch.setattr(extractor, "_prepare_base_model", lambda *a: (None, None, "cpu"))
    def fake_vector(model, tokenizer, prompts):
        if prompts is extractor.SAFE_PROMPTS:
            return torch.tensor([0., 0.])
        return torch.tensor([1., 0.]) if prompts is extractor.HALL_PROMPTS else torch.tensor([0., 1.])
    monkeypatch.setattr(extractor, "extract_concept_vector", fake_vector)
    metadata = extractor.vector_provenance(model_revision="a" * 40)
    vectors = extractor.load_or_compute_vectors(None, None, force=True, expected_metadata=metadata)
    assert torch.load(checkpoint, weights_only=True)["metadata"] == metadata
    assert not list(tmp_path.glob(".repe-*"))
    assert len(vectors) == 2
