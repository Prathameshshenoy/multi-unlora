import hashlib
import json

import pytest

from scripts.backup_artifacts import (
    BackupError, backup_volume, hash_file, verify_backup, volume_path,
)


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


@pytest.fixture
def artifact_volume():
    class FakeVolume:
        def __init__(self):
            self.files, self.requests = {}, []

        def read_file(self, path):
            self.requests.append(path)
            if path not in self.files:
                raise FileNotFoundError(path)
            data = self.files[path]
            for offset in range(0, len(data), 3):
                yield data[offset:offset+3]

    volume = FakeVolume()
    manifest = {'model_revision':'revision','concepts':{}}
    for concept, name in (('hallucination','bad_expert_lora'),('jailbreak','jailbreak_bad_expert_lora')):
        weight = f'{concept}-weight-bytes'.encode()
        settings = encoded({'base_model_name_or_path':'example-model'})
        volume.files[f'/adapters/{name}/adapter_model.safetensors'] = weight
        volume.files[f'/adapters/{name}/adapter_config.json'] = settings
        manifest['concepts'][concept] = {
            'adapter_sha256':hashlib.sha256(weight).hexdigest(),
            'adapter_config_sha256':hashlib.sha256(settings).hexdigest()}
    volume.files['/adapters/manifest.json'] = encoded(manifest)
    volume.files['/checkpoints/repe_vectors.pt'] = b'calibrated-vector-bytes'
    volume.files['/results/pilot_quality.json'] = encoded({'forced_choice_gate_passed':False})
    selected = json.loads(json.dumps(manifest))
    selected.update({'source_manifest_sha256':hashlib.sha256(encoded(manifest)).hexdigest(),
        'adapter_paths':{'hallucination':'/vol/runs/trial/hallucination',
                         'jailbreak':'/vol/adapters/jailbreak_bad_expert_lora'},
        'repe_checkpoint':'/vol/checkpoints/repe_vectors.pt'})
    candidate = b'new-trial-weight-bytes'
    selected['concepts']['hallucination']['adapter_sha256'] = hashlib.sha256(candidate).hexdigest()
    volume.files['/runs/trial/manifest.json'] = encoded(selected)
    volume.files['/runs/trial/hallucination/adapter_model.safetensors'] = candidate
    volume.files['/runs/trial/hallucination/adapter_config.json'] = volume.files['/adapters/bad_expert_lora/adapter_config.json']
    volume.files['/runs/trial/training_report.json'] = encoded({'status':'unvalidated'})
    return volume


def test_streamed_snapshot_keeps_both_versions_and_records_missing_optional(artifact_volume, tmp_path):
    target = tmp_path/'snapshot'
    summary = backup_volume(artifact_volume,target,selected_manifests=['/vol/runs/trial/manifest.json'])
    assert summary['files']==11
    assert summary['missing_optional_count'] > 0
    assert (target/'adapters/bad_expert_lora/adapter_model.safetensors').read_bytes()==b'hallucination-weight-bytes'
    assert (target/'runs/trial/hallucination/adapter_model.safetensors').read_bytes()==b'new-trial-weight-bytes'
    assert artifact_volume.requests.count('/adapters/jailbreak_bad_expert_lora/adapter_model.safetensors')==1
    verified = verify_backup(target,expected_inventory_sha256=summary['inventory_sha256'])
    assert verified['verified']
    assert verified['size_bytes']==summary['size_bytes']
    assert not any('cache' in request or 'dataset' in request for request in artifact_volume.requests)


def test_weight_tamper_fails_backup_and_leaves_incomplete_inventory(artifact_volume, tmp_path):
    artifact_volume.files['/adapters/bad_expert_lora/adapter_model.safetensors'] = b'tampered'
    target = tmp_path/'snapshot'
    with pytest.raises(BackupError,match='manifest hash'):
        backup_volume(artifact_volume,target)
    inventory = json.loads((target/'inventory.json').read_text())
    assert inventory['status']=='failed'
    assert not (target/'adapters/bad_expert_lora/adapter_model.safetensors').exists()


def test_offline_verifier_detects_post_backup_tampering(artifact_volume, tmp_path):
    target = tmp_path/'snapshot'
    summary = backup_volume(artifact_volume,target)
    vector = target/'checkpoints/repe_vectors.pt'
    vector.write_bytes(b'x'*vector.stat().st_size)
    with pytest.raises(BackupError,match='byte-integrity'):
        verify_backup(target,expected_inventory_sha256=summary['inventory_sha256'])


def test_existing_backup_is_never_overwritten(artifact_volume, tmp_path):
    target = tmp_path/'snapshot'
    backup_volume(artifact_volume,target)
    before = hash_file(target/'inventory.json')
    with pytest.raises(FileExistsError):
        backup_volume(artifact_volume,target)
    assert hash_file(target/'inventory.json')==before


@pytest.mark.parametrize('path',[
    '/vol/runs/../credentials', '/vol/cache/huggingface/token', '/etc/passwd',
    '/vol/runs/trial/../../adapters/manifest.json', '/vol/runs/trial/manifest.json\0',
])
def test_manifest_or_path_traversal_cannot_escape_whitelist(artifact_volume, tmp_path, path):
    with pytest.raises(BackupError):
        backup_volume(artifact_volume,tmp_path/'snapshot',selected_manifests=[path])
    assert not any('token' in request or 'passwd' in request for request in artifact_volume.requests)


def test_trusted_manifest_cannot_request_reference_cache_or_credential_files(artifact_volume, tmp_path):
    manifest = json.loads(artifact_volume.files['/runs/trial/manifest.json'])
    manifest['adapter_paths']['hallucination'] = '/vol/runs/trial/reference_cache'
    artifact_volume.files['/runs/trial/manifest.json'] = encoded(manifest)
    with pytest.raises(BackupError,match='whitelist'):
        backup_volume(artifact_volume,tmp_path/'snapshot',selected_manifests=['/vol/runs/trial/manifest.json'])
    assert not any('reference_cache' in request for request in artifact_volume.requests)


def test_missing_required_weight_fails_even_when_optional_reports_may_be_absent(artifact_volume, tmp_path):
    del artifact_volume.files['/adapters/jailbreak_bad_expert_lora/adapter_model.safetensors']
    with pytest.raises(FileNotFoundError):
        backup_volume(artifact_volume,tmp_path/'snapshot')


def test_optional_report_in_progress_is_explicitly_recorded(artifact_volume, tmp_path):
    artifact_volume.files['/results/behavior_report.json'] = b'{"partial":'
    summary = backup_volume(artifact_volume,tmp_path/'snapshot')
    assert summary['warnings_count']==1
    inventory = json.loads((tmp_path/'snapshot/inventory.json').read_text())
    record = next(record for record in inventory['files'] if record['path']=='results/behavior_report.json')
    assert record['json_valid'] is False
