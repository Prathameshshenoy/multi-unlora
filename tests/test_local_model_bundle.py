import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from scripts.local_demo import MANIFEST, MODEL_PREFIX, unpack_bundle
from scripts.package_model import package_model


def make_bundle(tmp_path):
    root = tmp_path/'backup'
    concepts, paths = {}, {}
    for role in ('hallucination', 'jailbreak'):
        directory = root/'source'/role; directory.mkdir(parents=True)
        paths[role] = '/vol/source/'+role
        concepts[role] = {}
        for name, key, content in [('adapter_model.safetensors','adapter_sha256',b'synthetic weights'),
                                   ('adapter_config.json','adapter_config_sha256',b'{"peft_type":"LORA"}')]:
            (directory/name).write_bytes(content)
            concepts[role][key] = hashlib.sha256(content).hexdigest()
    manifest = root/'runs/misconception_roles_v1/manifest.json'; manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'base_model':'example/model', 'model_revision':'a'*40,
        'prompt_protocol':'native-chat-v2', 'adapter_paths':paths, 'repe_checkpoint':'/vol/repe.pt',
        'concepts':concepts}))
    (root/'repe.pt').write_bytes(b'synthetic calibration')
    (root/'private-report.json').write_text('DO NOT DISTRIBUTE')
    archive = tmp_path/'model.tar.gz'
    package_model(root, archive)
    return archive


def test_model_bundle_round_trip_contains_only_runtime_files(tmp_path):
    archive = make_bundle(tmp_path)
    result = unpack_bundle(archive, tmp_path/'extracted')
    assert result['manifest'] == MANIFEST and result['base_model_included'] is False
    assert len(result['files']) == 6
    assert not any('report' in name or 'corpus' in name for name in result['files'])
    manifest = json.loads((tmp_path/'extracted'/MODEL_PREFIX/'manifest.json').read_text())
    assert manifest['adapter_paths']['jailbreak'] == '/vol/'+MODEL_PREFIX+'jailbreak'


def test_model_install_uploads_to_own_volume_without_gpu(tmp_path, monkeypatch):
    import sys
    from scripts import local_demo
    archive = make_bundle(tmp_path)
    uploaded = []
    class Upload:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def put_file(self, local, remote):
            assert Path(local).is_file()
            uploaded.append(remote)
    class Volume:
        @staticmethod
        def from_name(name, create_if_missing):
            assert name == 'multi-unlora-vol' and create_if_missing is True
            return Volume()
        def batch_upload(self, *, force):
            assert force is False
            return Upload()
    monkeypatch.setitem(sys.modules, 'modal', SimpleNamespace(Volume=Volume))
    monkeypatch.setattr(local_demo, 'ROOT', tmp_path/'project')
    (tmp_path/'project').mkdir()
    local_demo.install_bundle(archive)
    assert len(uploaded) == 6
    assert json.loads((tmp_path/'project/.runtime/local-model.json').read_text())['manifest'] == MANIFEST


@pytest.mark.parametrize('attack', ['checksum', 'traversal', 'symlink'])
def test_untrusted_bundle_rejected_before_upload(tmp_path, attack):
    source = make_bundle(tmp_path)
    corrupt = tmp_path/'corrupt.tar.gz'
    with tarfile.open(source) as original, tarfile.open(corrupt, 'w:gz') as target:
        for member in original.getmembers():
            content = original.extractfile(member).read()
            if attack == 'checksum' and member.name.endswith('adapter_model.safetensors'):
                content = b'x'*len(content)
            target.addfile(member, io.BytesIO(content))
        if attack != 'checksum':
            member = tarfile.TarInfo('../escape' if attack == 'traversal' else MODEL_PREFIX+'link')
            if attack == 'symlink':
                member.type = tarfile.SYMTYPE; member.linkname = '/tmp/escape'
            target.addfile(member, io.BytesIO(b''))
    with pytest.raises(ValueError):
        unpack_bundle(corrupt, tmp_path/'extracted')
    assert not (tmp_path/'escape').exists()


def test_download_checks_release_checksum_and_reuses_cache(tmp_path, monkeypatch):
    from scripts import local_demo
    payload = b'synthetic archive download'
    (tmp_path/'models').mkdir()
    (tmp_path/'models/demo.json').write_text(json.dumps({
        'url':'https://github.com/Prathameshshenoy/multi-unlora/releases/download/model-v1.0.0/model.tar.gz',
        'sha256':hashlib.sha256(payload).hexdigest()}))
    monkeypatch.setattr(local_demo, 'ROOT', tmp_path)
    calls = []
    def fetch(*args, **kwargs):
        calls.append(args[0]); return io.BytesIO(payload)
    monkeypatch.setattr(local_demo.urllib.request, 'urlopen', fetch)
    first = local_demo.download_model()
    assert first.read_bytes() == payload
    assert local_demo.download_model() == first and len(calls) == 1
    first.unlink()
    monkeypatch.setattr(local_demo.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'corrupt'))
    with pytest.raises(ValueError, match='checksum mismatch'):
        local_demo.download_model()
    assert not first.exists()


def test_launch_prepares_before_session_and_preserves_existing_budget(tmp_path, monkeypatch):
    import sys
    from scripts import local_demo
    state = tmp_path/'.runtime'; state.mkdir()
    (state/'local-model.json').write_text(json.dumps({'manifest':MANIFEST}))
    budget = {'approved_usd':3, 'prior_cpu_preflight_reserve_usd':0, 'reservations':[]}
    (state/'budget.json').write_text(json.dumps(budget))
    monkeypatch.setattr(local_demo, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['local_demo', '--minutes', '3'])
    order = []
    monkeypatch.setattr(local_demo, 'prepare_runtime', lambda: order.append('prepare'))
    def launch(command, **kwargs):
        order.append('launch')
        assert command[-2:] == ['--manifest', MANIFEST]
        return 0
    monkeypatch.setattr(local_demo.subprocess, 'call', launch)
    with pytest.raises(SystemExit) as result:
        local_demo.main()
    assert result.value.code == 0 and order == ['prepare', 'launch']
    assert json.loads((state/'budget.json').read_text()) == budget


def test_budget_changes_require_explicit_flag_and_retain_reservations(tmp_path, monkeypatch):
    import sys
    from scripts import local_demo
    state=tmp_path/'.runtime';state.mkdir()
    (state/'local-model.json').write_text(json.dumps({'manifest':MANIFEST,'runtime_prepared':True}))
    budget={'approved_usd':3,'prior_cpu_preflight_reserve_usd':0,'reservations':[{'reserved_usd':.5}]}
    (state/'budget.json').write_text(json.dumps(budget))
    monkeypatch.setattr(local_demo,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['local_demo','--install-only','--budget-usd','4'])
    local_demo.main()
    updated=json.loads((state/'budget.json').read_text())
    assert updated['approved_usd']==4 and updated['reservations']==budget['reservations']
