"""Stream a narrow artifact backup from Modal Volume without starting a GPU.

Only manifests, adapter weights/configs, calibrated directions and named reports
are downloaded. HF caches, datasets, preference-reference caches and credentials
are deliberately outside the whitelist. Backups are immutable local snapshots;
an inventory records their byte hashes and any missing optional reports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parent.parent
CANONICAL_MANIFEST = PurePosixPath("/vol/adapters/manifest.json")
CANONICAL_ADAPTERS = {
    "hallucination": PurePosixPath("/vol/adapters/bad_expert_lora"),
    "jailbreak": PurePosixPath("/vol/adapters/jailbreak_bad_expert_lora"),
}
CANONICAL_REPE = PurePosixPath("/vol/checkpoints/repe_vectors.pt")
CANONICAL_REPORTS = tuple(PurePosixPath("/vol/results") / name for name in (
    "pilot_quality.json", "conditional_quality.json", "conditional_validation.json",
    "paired_specificity.json", "behavior_report.json",
))
MAX_FILE_BYTES = 1024 ** 3


class BackupError(RuntimeError):
    pass


def volume_path(value):
    """Validate the literal path before PurePath can normalize traversal away."""
    if not isinstance(value, (str, PurePosixPath)):
        raise BackupError("Artifact paths must be explicit /vol paths")
    text = str(value)
    if (not text.startswith("/vol/") or "\0" in text or "\\" in text
            or any(part in (".", "..", "") for part in text.split("/")[1:])):
        raise BackupError("Artifact path is outside the strict /vol whitelist")
    return PurePosixPath(text)


def manifest_path(value):
    path = volume_path(value)
    if path == CANONICAL_MANIFEST:
        return path
    if len(path.parts) != 5 or path.parts[1:3] != ("vol", "runs") or path.name != "manifest.json":
        raise BackupError("Selected manifests must be /vol/runs/<run>/manifest.json")
    return path


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise BackupError("Required manifest hash is missing or invalid")
    return value.lower()


def _is_missing(error):
    # Importing Modal is unnecessary for offline verification or fake-volume tests.
    return isinstance(error, FileNotFoundError) or type(error).__name__ == "NotFoundError"


def _read_manifest(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise BackupError("Downloaded artifact manifest is invalid JSON") from error
    if not isinstance(value, dict):
        raise BackupError("Downloaded artifact manifest must be a JSON object")
    return value


def backup_volume(volume, destination, *, selected_manifests=()):
    """Download authenticated-volume artifacts into a new directory only."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    records, missing, warnings = {}, [], []
    inventory_path = destination / "inventory.json"

    def local_path(remote):
        return destination.joinpath(*remote.relative_to("/vol").parts)

    def download(remote, *, required=True, expected=None, report=False):
        remote = volume_path(remote)
        key = str(remote.relative_to("/vol"))
        if key in records:
            if expected is not None and records[key]["sha256"] != _expected_hash(expected):
                raise BackupError("Conflicting manifest expectations for a shared artifact")
            return local_path(remote)
        target = local_path(remote)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        digest, size = hashlib.sha256(), 0
        try:
            with partial.open("xb") as output:
                for chunk in volume.read_file("/" + key):
                    if not isinstance(chunk, (bytes, bytearray)):
                        raise BackupError("Volume stream returned non-byte artifact content")
                    size += len(chunk)
                    if size > MAX_FILE_BYTES:
                        raise BackupError("Artifact exceeds the backup file-size ceiling")
                    output.write(chunk)
                    digest.update(chunk)
        except Exception as error:
            partial.unlink(missing_ok=True)
            if not required and _is_missing(error):
                missing.append(key)
                return None
            raise
        actual = digest.hexdigest()
        if expected is not None and actual != _expected_hash(expected):
            partial.unlink()
            raise BackupError("Downloaded artifact does not match its manifest hash")
        partial.replace(target)
        record = {"path": key, "size_bytes": size, "sha256": actual, "required": required}
        if expected is not None:
            record["manifest_sha256_verified"] = True
        if report:
            try:
                json.loads(target.read_text())
                record["json_valid"] = True
            except (UnicodeError, ValueError):
                record["json_valid"] = False
                warnings.append({"path": key, "reason": "optional report was not complete JSON at snapshot time"})
        records[key] = record
        return target

    def bundle(path, *, canonical=False):
        path = manifest_path(path)
        saved = download(path)
        manifest = _read_manifest(saved)
        if not canonical:
            expected_source = _expected_hash(manifest.get("source_manifest_sha256"))
            if expected_source != records[str(CANONICAL_MANIFEST.relative_to("/vol"))]["sha256"]:
                raise BackupError("Selected trial is not bound to the backed-up canonical manifest")
        selected = manifest.get("adapter_paths", {})
        if not isinstance(selected, dict) or set(selected) - set(CANONICAL_ADAPTERS):
            raise BackupError("Manifest contains unknown adapter roles")
        concepts = manifest.get("concepts", {})
        if not isinstance(concepts, dict):
            raise BackupError("Manifest is missing adapter hash metadata")
        allowed_adapter_dirs = set(CANONICAL_ADAPTERS.values())
        if not canonical:
            allowed_adapter_dirs.update({path.parent/"hallucination", path.parent/"hallucination"/"hallucination", path.parent/"adapters"/"hallucination"})
        for role, default in CANONICAL_ADAPTERS.items():
            directory = volume_path(selected.get(role, default))
            if directory not in allowed_adapter_dirs:
                raise BackupError("Adapter directory is outside the artifact whitelist")
            metadata = concepts.get(role)
            if not isinstance(metadata, dict):
                raise BackupError("Manifest is missing required adapter metadata")
            download(directory/"adapter_model.safetensors", expected=_expected_hash(metadata.get("adapter_sha256")))
            config_hash = metadata.get("adapter_config_sha256")
            download(directory/"adapter_config.json", expected=_expected_hash(config_hash) if config_hash is not None else None)
        vectors = volume_path(manifest.get("repe_checkpoint", CANONICAL_REPE))
        if vectors not in {CANONICAL_REPE, path.parent/"repe_vectors.pt"}:
            raise BackupError("RepE checkpoint is outside the artifact whitelist")
        download(vectors)
        if not canonical:
            quality = volume_path(manifest.get("quality_report", path.parent/"quality_report.json"))
            if quality != path.parent/"quality_report.json":
                raise BackupError("Trial quality report is outside the artifact whitelist")
            for optional in (path.parent/"training_report.json", quality, path.parent/"behavior_report.json"):
                download(optional, required=False, report=True)

    status = "failed"
    try:
        bundle(CANONICAL_MANIFEST, canonical=True)
        for path in sorted({str(manifest_path(path)) for path in selected_manifests}):
            if PurePosixPath(path) != CANONICAL_MANIFEST:
                bundle(path)
        for report in CANONICAL_REPORTS:
            download(report, required=False, report=True)
        status = "complete"
    finally:
        inventory = {"protocol": "artifact-backup-v1", "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": sorted(records.values(), key=lambda item: item["path"]),
            "missing_optional": sorted(set(missing)), "warnings": warnings}
        inventory_path.write_text(json.dumps(inventory, indent=2, allow_nan=False))
    return {"backup_dir": str(destination), "files": len(records),
        "size_bytes": sum(item["size_bytes"] for item in records.values()),
        "inventory_sha256": hash_file(inventory_path),
        "missing_optional_count": len(set(missing)), "warnings_count": len(warnings)}


def verify_backup(directory, *, expected_inventory_sha256=None):
    """Offline byte-integrity check; no Modal import, network, or model loading."""
    directory = Path(directory)
    inventory_path = directory/"inventory.json"
    actual = hash_file(inventory_path)
    if expected_inventory_sha256 is not None and actual != _expected_hash(expected_inventory_sha256):
        raise BackupError("Inventory hash differs from the retained reference")
    inventory = _read_manifest(inventory_path)
    if inventory.get("status") != "complete":
        raise BackupError("Backup inventory is incomplete")
    seen, size = set(), 0
    for record in inventory.get("files", []):
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or not relative.parts or any(part in (".", "..") for part in relative.parts) or str(relative) in seen:
            raise BackupError("Inventory contains an invalid artifact path")
        seen.add(str(relative))
        target = directory.joinpath(*relative.parts)
        if target.is_symlink() or not target.resolve().is_relative_to(directory.resolve()):
            raise BackupError("Backup artifact resolves outside the snapshot")
        if target.stat().st_size != record["size_bytes"] or hash_file(target) != record["sha256"]:
            raise BackupError("Backup artifact failed its byte-integrity check")
        size += record["size_bytes"]
    if not seen:
        raise BackupError("Backup inventory has no artifacts")
    return {"backup_dir": str(directory), "files": len(seen), "size_bytes": size,
            "inventory_sha256": actual, "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=[], help="Additional /vol/runs/<run>/manifest.json")
    parser.add_argument("--verify", type=Path, help="Verify an existing local backup without network access")
    parser.add_argument("--expected-inventory-sha256", help="Optional retained inventory hash for offline verification")
    args = parser.parse_args()
    if args.verify:
        summary = verify_backup(args.verify, expected_inventory_sha256=args.expected_inventory_sha256)
    else:
        manifests = list(args.manifest)
        for result_name in ('paired-train-result.json','anchored-train-result.json','anchored-training_report.json'):
            result_path = ROOT/".runtime"/result_name
            if result_path.is_file():
                selected = _read_manifest(result_path).get("manifest_path")
                if selected:
                    manifests.append(str(manifest_path(selected)))
        # Validate requested paths before importing the network client.
        manifests = [str(manifest_path(value)) for value in manifests]
        from modal import Volume
        destination = ROOT/".runtime"/"artifact-backup"/(
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:6])
        summary = backup_volume(Volume.from_name("multi-unlora-vol"), destination, selected_manifests=manifests)
        verify_backup(destination, expected_inventory_sha256=summary["inventory_sha256"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Print no raw report contents, model outputs, credential values, or
        # transport exceptions that could contain private connection details.
        print(f"Artifact backup failed ({type(error).__name__}); inspect the incomplete inventory locally.", file=sys.stderr)
        raise SystemExit(1)
