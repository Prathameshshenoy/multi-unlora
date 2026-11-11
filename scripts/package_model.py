"""Build a portable adapter-only checkpoint from a verified local backup."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.local_demo import MODEL_PREFIX, MANIFEST, sha256


def package_model(backup, output, *, license_files=()):
    backup, output = Path(backup), Path(output)
    original = backup/'runs/misconception_roles_v1/manifest.json'
    source = json.loads(original.read_text())
    files = {}
    concepts = {}
    for role in ('hallucination', 'jailbreak'):
        directory = backup/Path(source['adapter_paths'][role]).relative_to('/vol')
        concepts[role] = {}
        for name, key in (('adapter_model.safetensors', 'adapter_sha256'), ('adapter_config.json', 'adapter_config_sha256')):
            path = directory/name
            digest = sha256(path)
            if digest != source['concepts'][role][key]:
                raise ValueError('Source model artifact checksum mismatch')
            concepts[role][key] = digest
            files[MODEL_PREFIX+role+'/'+name] = path
    files[MODEL_PREFIX+'repe_vectors.pt'] = backup/Path(source['repe_checkpoint']).relative_to('/vol')
    for path in map(Path, license_files):
        files[MODEL_PREFIX+path.name] = path
    manifest = {key:source[key] for key in ('base_model', 'model_revision', 'prompt_protocol')}
    manifest.update(status='research_demo', concepts=concepts,
                    adapter_paths={role:'/vol/'+MODEL_PREFIX+role for role in concepts},
                    repe_checkpoint='/vol/'+MODEL_PREFIX+'repe_vectors.pt',
                    source_manifest_sha256=sha256(original))
    if 'training_corpus' in source:
        manifest['training_corpus'] = source['training_corpus']
    manifest_bytes = (json.dumps(manifest, indent=2)+'\n').encode()
    inventory = {name:{'sha256':sha256(path), 'size_bytes':path.stat().st_size} for name, path in files.items()}
    inventory[MODEL_PREFIX+'manifest.json'] = {'sha256':hashlib.sha256(manifest_bytes).hexdigest(), 'size_bytes':len(manifest_bytes)}
    metadata = {'protocol':'multi-unlora-model-bundle-v1', 'manifest':MANIFEST,
                'base_model_included':False, 'files':inventory}
    output.parent.mkdir(parents=True, exist_ok=True)
    # No credentials, datasets, reports, training state or base weights enter here.
    with output.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as archive:
        def add_bytes(name, value):
            info = tarfile.TarInfo(name); info.size = len(value); info.mode = 0o644
            archive.addfile(info, io.BytesIO(value))
        add_bytes('bundle.json', json.dumps(metadata, indent=2).encode())
        add_bytes(MODEL_PREFIX+'manifest.json', manifest_bytes)
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name, recursive=False)
    return {'path':str(output), 'sha256':sha256(output), 'size_bytes':output.stat().st_size, 'files':len(inventory)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backup', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--license-file', action='append', type=Path, default=[])
    args = parser.parse_args()
    print(json.dumps(package_model(args.backup, args.output, license_files=args.license_file), indent=2))
