"""Build a portable adapter-only checkpoint from a verified local backup."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.local_demo import MODEL_PREFIX, model_prefix, sha256


def package_model(backup, output, *, license_files=(), source_manifest=None, model_directory=MODEL_PREFIX):
    backup, output = Path(backup), Path(output)
    original = Path(source_manifest) if source_manifest is not None else backup/'runs/misconception_roles_v1/manifest.json'
    manifest_location = '/vol/'+model_directory+'manifest.json'
    model_prefix(manifest_location)
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
            files[model_directory+role+'/'+name] = path
    files[model_directory+'repe_vectors.pt'] = backup/Path(source['repe_checkpoint']).relative_to('/vol')
    for path in map(Path, license_files):
        files[model_directory+path.name] = path
    manifest = {key:source[key] for key in ('base_model', 'model_revision', 'prompt_protocol')}
    manifest.update(status='research_demo', concepts=concepts,
                    adapter_paths={role:'/vol/'+model_directory+role for role in concepts},
                    repe_checkpoint='/vol/'+model_directory+'repe_vectors.pt',
                    source_manifest_sha256=sha256(original))
    if 'decoder_settings' in source:
        manifest['decoder_settings'] = source['decoder_settings']
    if source.get('fixed_alphas') is not None:
        from inference.runtime import fixed_weights
        manifest['fixed_alphas'] = list(fixed_weights(source))
    if source.get('intent_gate') is not None:
        from repe.intent_gate import load as load_gate
        spec = source['intent_gate']
        path = backup/Path(spec['path']).relative_to('/vol')
        load_gate({**source, 'intent_gate': {**spec, 'path': str(path.resolve())}})
        if sha256(path) != spec['sha256']:
            raise ValueError('Source intent-gate checksum mismatch')
        files[model_directory+'intent_gate.json'] = path
        manifest['intent_gate'] = {**spec, 'path':'/vol/'+model_directory+'intent_gate.json'}
    if source.get('factual_gate') is not None:
        from repe.factual_gate import load as load_factual_gate
        spec = source['factual_gate']
        path = backup/Path(spec['path']).relative_to('/vol')
        load_factual_gate({**source, 'factual_gate': {**spec, 'path': str(path.resolve())}})
        files[model_directory+'factual_gate.json'] = path
        manifest['factual_gate'] = {**spec, 'path':'/vol/'+model_directory+'factual_gate.json'}
    if source.get('safety_hallucination') is not None:
        from inference.runtime import preserved_safety_hall
        spec = preserved_safety_hall(source)
        directory = backup/Path(spec['path']).relative_to('/vol')
        for name,key in [('adapter_model.safetensors','adapter_sha256'),('adapter_config.json','adapter_config_sha256')]:
            path = directory/name
            if sha256(path) != spec[key]:
                raise ValueError('Preserved safety adapter checksum mismatch')
            files[model_directory+'safety_hallucination/'+name] = path
        manifest['safety_hallucination'] = {**spec, 'path':'/vol/'+model_directory+'safety_hallucination'}
    if 'use_kv_cache' in source:
        manifest['use_kv_cache'] = source['use_kv_cache']
    if 'training_corpus' in source:
        manifest['training_corpus'] = source['training_corpus']
    manifest_bytes = (json.dumps(manifest, indent=2)+'\n').encode()
    inventory = {name:{'sha256':sha256(path), 'size_bytes':path.stat().st_size} for name, path in files.items()}
    inventory[model_directory+'manifest.json'] = {'sha256':hashlib.sha256(manifest_bytes).hexdigest(), 'size_bytes':len(manifest_bytes)}
    metadata = {'protocol':'multi-unlora-model-bundle-v2' if source.get('factual_gate') is not None else 'multi-unlora-model-bundle-v1', 'manifest':manifest_location,
                'base_model_included':False, 'files':inventory}
    output.parent.mkdir(parents=True, exist_ok=True)
    # No credentials, datasets, reports, training state or base weights enter here.
    with output.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as archive:
        def add_bytes(name, value):
            info = tarfile.TarInfo(name); info.size = len(value); info.mode = 0o644
            archive.addfile(info, io.BytesIO(value))
        add_bytes('bundle.json', json.dumps(metadata, indent=2).encode())
        add_bytes(model_directory+'manifest.json', manifest_bytes)
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
