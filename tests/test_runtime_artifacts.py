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


def test_candidate_bundle_resolves_and_validates_without_changing_canonical_artifacts(artifact_set):
    from pathlib import Path
    root, manifest = artifact_set
    candidate = root/'paired_run'
    candidate.mkdir()
    hall = candidate/'new_hall'
    hall.mkdir()
    (hall/'adapter_config.json').write_bytes((root/'hallucination'/'adapter_config.json').read_bytes())
    (hall/'adapter_model.safetensors').write_bytes(b'new paired policy weights')
    manifest['concepts']['hallucination']['adapter_sha256'] = hashlib.sha256(b'new paired policy weights').hexdigest()
    manifest['adapter_paths'] = {'hallucination': str(hall), 'jailbreak': str(root/'jailbreak')}
    manifest['repe_checkpoint'] = str(candidate/'vectors.pt')
    bundle = candidate/'manifest.json'
    bundle.write_text(json.dumps(manifest))
    canonical = (root/'manifest.json').read_bytes()
    paths = runtime.artifact_paths(bundle)
    assert paths['hallucination'] == hall
    assert paths['jailbreak'] == root/'jailbreak'
    assert paths['repe'] == candidate/'vectors.pt'
    assert paths['quality_report'] == candidate/'quality_report.json'
    assert runtime.validate_artifacts(bundle) == manifest
    assert (root/'manifest.json').read_bytes() == canonical
    assert config.HALL_ADAPTER_PATH == str(root/'hallucination')


@pytest.mark.parametrize('field,value', [
    ('adapter_paths', {'hallucination':'relative/adapter'}),
    ('adapter_paths', {'other':'/absolute/path'}),
    ('adapter_paths', ['not-a-map']),
    ('repe_checkpoint', '../relative.pt'),
    ('repe_checkpoint', '/vol/../other.pt'),
    ('quality_report', 'https://example.test/report.json'),
    ('quality_report', '/absolute/with\0nul'),
])
def test_invalid_candidate_paths_fail_before_loading_weights(artifact_set,field,value):
    root, manifest = artifact_set
    manifest[field] = value
    bundle = root/'candidate_manifest.json'
    bundle.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):
        runtime.validate_artifacts(bundle)


def test_checkpoint_override_survives_recursive_explicit_recalibration(tmp_path,monkeypatch):
    torch = pytest.importorskip('torch')
    from repe import extractor
    canonical, selected = tmp_path/'canonical.pt', tmp_path/'candidate.pt'
    canonical.write_bytes(b'canonical checkpoint must remain untouched')
    original = canonical.read_bytes()
    monkeypatch.setattr(config,'REPE_CHECKPOINT',str(canonical))
    monkeypatch.setattr(extractor,'_prepare_base_model',lambda *a: (None,None,'cpu'))
    torch.save({'prompt_protocol': PROMPT_PROTOCOL, 'metadata': {'old':True}},selected)
    def vector(model,tokenizer,prompts):
        if prompts is extractor.SAFE_PROMPTS:
            return torch.tensor([0.,0.])
        return torch.tensor([1.,0.]) if prompts is extractor.HALL_PROMPTS else torch.tensor([0.,1.])
    monkeypatch.setattr(extractor,'extract_concept_vector',vector)
    metadata = extractor.vector_provenance(model_revision='a'*40)
    extractor.load_or_compute_vectors(None,None,checkpoint_path=selected,expected_metadata=metadata)
    assert torch.load(selected,weights_only=True)['metadata'] == metadata
    assert canonical.read_bytes() == original


@pytest.mark.parametrize("training_ids",[None,["a"*64]])
def test_inference_loader_passes_selected_bundle_paths_without_mutating_config(artifact_set,monkeypatch,training_ids):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    peft = pytest.importorskip('peft')
    from repe import extractor
    root, manifest = artifact_set
    new_hall = root/'paired_hall'
    new_hall.mkdir()
    for name in ['adapter_model.safetensors','adapter_config.json']:
        (new_hall/name).write_bytes((root/'hallucination'/name).read_bytes())
    manifest['adapter_paths'] = {'hallucination':str(new_hall),'jailbreak':str(root/'jailbreak')}
    selected_repe = root/'selected_vectors.pt'
    selected_repe.write_bytes(b'fake vector checkpoint')
    manifest['repe_checkpoint'] = str(selected_repe)
    if training_ids is not None:
        manifest['training_corpus']={'question_ids':training_ids}
    bundle = root/'candidate.json'
    bundle.write_text(json.dumps(manifest))
    calls = {}
    class FakeModel:
        def load_adapter(self,path,adapter_name):
            calls['jailbreak'] = path
        def eval(self):
            return self
        def disable_adapter_layers(self):
            calls['disabled'] = True
    model = FakeModel()
    def load_policy(base,path,adapter_name):
        calls['hallucination'] = path
        return model
    def vectors(*args,**kwargs):
        calls['checkpoint'] = kwargs['checkpoint_path']
        return torch.ones(2),torch.ones(2)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:SimpleNamespace(eos_token='EOS',chat_template='native'))
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',lambda *a,**kw:object())
    monkeypatch.setattr(peft.PeftModel,'from_pretrained',load_policy)
    monkeypatch.setattr(extractor,'load_or_compute_vectors',vectors)
    loaded = runtime.load_inference_model(manifest_path=bundle)
    assert loaded[0] is model
    assert model.unlora_training_question_ids==frozenset(training_ids or [])
    assert calls['hallucination'] == str(new_hall)
    assert calls['jailbreak'] == str(root/'jailbreak')
    assert calls['checkpoint'] == selected_repe
    assert calls['disabled'] is True
    assert config.HALL_ADAPTER_PATH == str(root/'hallucination')


@pytest.mark.parametrize('ids',[[],['bad-hash'],[{}],['a'*64,'a'*64]])
def test_malformed_training_question_provenance_fails_closed(artifact_set,ids):
    root,manifest=artifact_set
    manifest['training_corpus']={'question_ids':ids}
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError,match='training-corpus'):
        runtime.validate_artifacts()


@pytest.mark.parametrize('settings', [{'beta':True}, {'beta':0}, {'beta':float('nan')}, {'kl_budget':-1}, {'unexpected':1}])
def test_invalid_artifact_decoder_settings_are_rejected(settings):
    with pytest.raises(RuntimeError,match='decoder settings'):
        runtime.decoding_settings({'decoder_settings':settings})


def test_artifact_decoder_settings_are_explicit_and_default_compatible():
    assert runtime.decoding_settings({})=={'beta':config.PROCESSOR_PLAUSIBILITY_BETA,'kl_budget':config.PROCESSOR_KL_THRESHOLD}
    assert runtime.decoding_settings({'decoder_settings':{'beta':.01,'kl_budget':2}})=={'beta':.01,'kl_budget':2.}
