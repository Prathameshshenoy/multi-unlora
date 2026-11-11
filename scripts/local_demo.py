"""Install the demo checkpoint into your Modal workspace and run localhost."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODEL_PREFIX = 'runs/demo_v1/'
MANIFEST = '/vol/'+MODEL_PREFIX+'manifest.json'
MAX_BUNDLE_BYTES = 600 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_bundle(archive, destination):
    """Extract a data-only checkpoint, rejecting links, traversal and extra files."""
    destination = Path(destination)
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or any(not member.isfile() for member in members):
            raise ValueError('Model bundle must contain distinct regular files only')
        if len(members)>32 or 'bundle.json' not in names or any(member.size > MAX_BUNDLE_BYTES for member in members) or sum(m.size for m in members) > MAX_BUNDLE_BYTES:
            raise ValueError('Invalid or oversized model bundle')
        if bundle.getmember('bundle.json').size > 1024*1024:
            raise ValueError('Oversized model inventory')
        metadata = json.load(bundle.extractfile('bundle.json'))
        if metadata.get('protocol') != 'multi-unlora-model-bundle-v1' or metadata.get('manifest') != MANIFEST:
            raise ValueError('Unsupported model bundle')
        files = metadata.get('files', {})
        if not isinstance(files, dict) or set(names) != {'bundle.json', *files}:
            raise ValueError('Bundle inventory does not match its files')
        required = {MODEL_PREFIX+'manifest.json', MODEL_PREFIX+'repe_vectors.pt'}
        required.update(MODEL_PREFIX+role+'/'+name for role in ('hallucination', 'jailbreak')
                        for name in ('adapter_config.json', 'adapter_model.safetensors'))
        if not required <= set(files):
            raise ValueError('Incomplete model checkpoint')
        for member in members:
            if member.name == 'bundle.json':
                continue
            path = PurePosixPath(member.name)
            if (not member.name.startswith(MODEL_PREFIX) or path.is_absolute() or '\\' in member.name
                    or any(part in ('', '.', '..') for part in member.name.split('/'))):
                raise ValueError('Unsafe model bundle path')
            expected = files[member.name]
            if member.size != expected['size_bytes']:
                raise ValueError('Model artifact size mismatch')
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, target.open('xb') as output:
                while chunk := source.read(1024*1024):
                    output.write(chunk)
            if sha256(target) != expected['sha256']:
                raise ValueError('Model artifact checksum mismatch')
    manifest = json.loads((destination/MODEL_PREFIX/'manifest.json').read_text())
    for role in ('hallucination', 'jailbreak'):
        if manifest.get('adapter_paths', {}).get(role) != '/vol/'+MODEL_PREFIX+role:
            raise ValueError('Model adapter path mismatch')
        for name, key in (('adapter_model.safetensors', 'adapter_sha256'), ('adapter_config.json', 'adapter_config_sha256')):
            if sha256(destination/MODEL_PREFIX/role/name) != manifest['concepts'][role][key]:
                raise ValueError('Adapter manifest checksum mismatch')
    if manifest.get('repe_checkpoint') != '/vol/'+MODEL_PREFIX+'repe_vectors.pt':
        raise ValueError('Model calibration path mismatch')
    return metadata


def install_bundle(archive):
    import modal
    with tempfile.TemporaryDirectory(prefix='multi-unlora-model-') as directory:
        metadata = unpack_bundle(archive, directory)
        volume = modal.Volume.from_name('multi-unlora-vol', create_if_missing=True)
        with volume.batch_upload(force=False) as upload:
            for relative in metadata['files']:
                upload.put_file(Path(directory)/relative, '/'+relative)
    state = ROOT/'.runtime'
    state.mkdir(exist_ok=True)
    (state/'local-model.json').write_text(json.dumps({'manifest':MANIFEST, 'bundle_sha256':sha256(archive)}, indent=2))
    print('Trained model installed in your Modal workspace. No GPU was started.', flush=True)


def download_model():
    descriptor_path = ROOT/'models/demo.json'
    if not descriptor_path.exists():
        raise RuntimeError('Supply a trained model archive with --bundle PATH. See README.md for setup options.')
    descriptor = json.loads(descriptor_path.read_text())
    url = descriptor['url']
    if not url.startswith('https://github.com/Prathameshshenoy/multi-unlora/releases/download/'):
        raise ValueError('Unexpected model download source')
    cache = ROOT/'.runtime/models'; cache.mkdir(parents=True, exist_ok=True)
    target = cache/'multi-unlora-demo-v1.tar.gz'
    if target.exists() and sha256(target) == descriptor['sha256']:
        return target
    temporary = target.with_suffix('.part')
    print('Downloading trained adapters and calibration vectors…', flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
            size = 0
            while chunk := response.read(1024*1024):
                size += len(chunk)
                if size > MAX_BUNDLE_BYTES:
                    raise ValueError('Model download exceeds its size limit')
                output.write(chunk)
        if sha256(temporary) != descriptor['sha256']:
            raise ValueError('Downloaded model checksum mismatch')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def prepare_runtime():
    # Image building can take longer than a short inference session on a fresh
    # account. Finish it on CPU before the GPU session's clock starts.
    print('Preparing the Modal runtime (first setup only; no GPU)…', flush=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/run_engine.py'), 'prepare-local'], cwd=ROOT, check=True)
    path = ROOT/'.runtime/local-model.json'
    settings = json.loads(path.read_text()); settings['runtime_prepared'] = True
    path.write_text(json.dumps(settings, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, help='Install a local trained checkpoint archive')
    parser.add_argument('--install-only', action='store_true', help='Upload model files without starting a GPU')
    parser.add_argument('--minutes', type=int, choices=(3, 5, 10, 15), default=5)
    parser.add_argument('--budget-usd', type=float, default=5, help='Initial cumulative compute cap for a fresh installation')
    args = parser.parse_args()
    if not 1 <= args.budget_usd <= 100:
        parser.error('--budget-usd must be between 1 and 100')
    state = ROOT/'.runtime'; state.mkdir(exist_ok=True)
    # Existing spending limits are never overwritten by a convenience launcher.
    budget = state/'budget.json'
    if not budget.exists():
        with budget.open('x') as output:
            json.dump({'approved_usd':args.budget_usd, 'prior_cpu_preflight_reserve_usd':0, 'reservations':[]}, output, indent=2)
    if args.bundle is not None or not (state/'local-model.json').exists():
        install_bundle(args.bundle if args.bundle is not None else download_model())
    settings = json.loads((state/'local-model.json').read_text())
    if settings.get('runtime_prepared') is not True:
        prepare_runtime()
    if args.install_only:
        return
    manifest = json.loads((state/'local-model.json').read_text())['manifest']
    if manifest != MANIFEST:
        raise ValueError('Installed model selection is invalid; reinstall the checkpoint')
    raise SystemExit(subprocess.call([sys.executable, str(ROOT/'scripts/run_engine.py'), 'serve-local',
                                     '--minutes', str(args.minutes), '--manifest', manifest], cwd=ROOT))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f'Local demo setup: {error}', file=sys.stderr)
        raise SystemExit(1)
