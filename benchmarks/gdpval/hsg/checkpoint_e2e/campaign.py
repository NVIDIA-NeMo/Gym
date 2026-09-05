#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare and validate reproducible checkpoint-to-GDPVal campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlparse


SCHEMA_VERSION = "gdpval-checkpoint-e2e-v2"
DEFAULT_TASKS = 220
DEFAULT_SHARDS = 6
FILE_MODE = 0o400
EXPECTED_JUDGE_MODELS = {
    "gpt-5.5": "openai/openai/gpt-5.5",
    "gemini-3.1-pro": "gcp/google/gemini-3.1-pro-preview",
    "claude-opus-4.8": "aws/anthropic/bedrock-claude-opus-4-8",
}
STAGE0_PLANNED_TASKS = 45
STAGE0_MIN_PARTIAL_TASKS = 41
STAGE1_TASKS = 220
TRIALS_PER_TASK = 4
ELO_ABS_TOLERANCE = 1e-9
NORMALIZED_ELO_ABS_TOLERANCE = 1e-12
STAGE0_PARTIAL_POLICY = {
    "min_success_fraction": 0.9,
    "min_per_reference_success_fraction": 0.5,
    "min_successful_rows_per_reference": 1,
    "newly_waivable_failure_classes": ["timeout_exceeded", "transient"],
}


class CampaignError(ValueError):
    """Raised when a campaign artifact or input violates the contract."""


@dataclass(frozen=True)
class Dataset:
    path: Path
    raw: bytes
    raw_lines: tuple[bytes, ...]
    rows: tuple[dict[str, Any], ...]
    task_ids: tuple[str, ...]
    reference_count: int
    reference_paths: tuple[Path, ...]

    @property
    def sha256(self) -> str:
        return _sha256_bytes(self.raw)


def _fail(message: str) -> None:
    raise CampaignError(message)


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_regular(path: Path, *, mode: int | None = None) -> None:
    if path.is_symlink() or not path.is_file():
        _fail(f"not a regular non-symlink file: {path}")
    if mode is not None:
        actual = stat.S_IMODE(path.stat().st_mode)
        if actual != mode:
            _fail(f"unexpected mode for {path}: {actual:04o}, expected {mode:04o}")


def _validate_existing(path: Path, payload: bytes) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    _require_regular(path, mode=FILE_MODE)
    actual = path.read_bytes()
    if actual != payload:
        _fail(f"immutable artifact drift: {path}")
    return True


def _publish_immutable(path: Path, payload: bytes) -> None:
    if _validate_existing(path, payload):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, FILE_MODE)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write publishing {path}")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, FILE_MODE)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _publish_bundle(artifacts: Mapping[Path, bytes]) -> None:
    # Check every existing destination before creating anything, so known drift
    # never leaves a partly updated bundle.
    existing = {path: _validate_existing(path, payload) for path, payload in artifacts.items()}
    for path, payload in artifacts.items():
        if not existing[path]:
            _publish_immutable(path, payload)


def _sidecar(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_VERSION}-sha256",
        "path": str(path.resolve(strict=False)),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _require_absolute_resolved_directory(value: Path, *, label: str) -> Path:
    if not value.is_absolute():
        _fail(f"{label} must be an absolute path: {value}")
    try:
        resolved = value.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CampaignError(f"{label} does not exist: {value}") from exc
    if value != resolved:
        _fail(f"{label} must already be resolved: {value} -> {resolved}")
    if not resolved.is_dir():
        _fail(f"{label} is not a directory: {resolved}")
    return resolved


def _checkpoint_descriptor(checkpoint: Path) -> dict[str, Any]:
    checkpoint = _require_absolute_resolved_directory(checkpoint, label="checkpoint")
    config = checkpoint / "config.json"
    if not config.is_file():
        _fail(f"checkpoint is missing config.json: {checkpoint}")
    try:
        config_value = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"invalid checkpoint config {config}: {exc}") from exc
    if not isinstance(config_value, dict):
        _fail(f"checkpoint config is not a JSON object: {config}")
    tokenizer_candidates = (checkpoint / "tokenizer.json", checkpoint / "tokenizer.model")
    if not any(path.is_file() for path in tokenizer_candidates):
        _fail(f"checkpoint is missing tokenizer.json or tokenizer.model: {checkpoint}")
    tokenizer_json = checkpoint / "tokenizer.json"
    if tokenizer_json.is_file():
        try:
            tokenizer_value = json.loads(tokenizer_json.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CampaignError(f"invalid checkpoint tokenizer {tokenizer_json}: {exc}") from exc
        if not isinstance(tokenizer_value, dict):
            _fail(f"checkpoint tokenizer is not a JSON object: {tokenizer_json}")
    elif (checkpoint / "tokenizer.model").stat().st_size <= 0:
        _fail(f"checkpoint tokenizer.model is empty: {checkpoint}")

    weight_candidates = [
        checkpoint / "model.safetensors",
        checkpoint / "model.safetensors.index.json",
        checkpoint / "pytorch_model.bin",
        checkpoint / "pytorch_model.bin.index.json",
    ]
    weight_candidates.extend(sorted(checkpoint.glob("model-*.safetensors")))
    if not any(path.is_file() for path in weight_candidates):
        _fail(f"checkpoint is missing Hugging Face model weights or an index: {checkpoint}")
    if not any(path.is_file() and path.stat().st_size > 0 for path in weight_candidates):
        _fail(f"checkpoint model weights and indexes are empty: {checkpoint}")

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = checkpoint / index_name
        if not index_path.is_file():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CampaignError(f"invalid checkpoint weight index {index_path}: {exc}") from exc
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            _fail(f"checkpoint weight index has no nonempty weight_map: {index_path}")
        shard_names = list(weight_map.values())
        if any(not isinstance(name, str) or not name for name in shard_names):
            _fail(f"checkpoint weight index contains invalid shard names: {index_path}")
        referenced = sorted(set(shard_names))
        for relative in referenced:
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                _fail(f"checkpoint weight index escapes checkpoint: {relative}")
            shard = checkpoint / relative_path
            if not shard.is_file():
                _fail(f"checkpoint weight index references a missing file: {shard}")
            if shard.stat().st_size <= 0:
                _fail(f"checkpoint weight index references an empty file: {shard}")

    inventory: list[dict[str, Any]] = []
    for path in sorted(checkpoint.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        metadata = path.stat()
        entry: dict[str, Any] = {
            "path": path.relative_to(checkpoint).as_posix(),
            "bytes": metadata.st_size,
            # Weight shards are too large to hash on every status/preflight.
            # Size + nanosecond mtime catches ordinary in-place mutation; small
            # metadata/tokenizer/index files additionally receive a full hash.
            "mtime_ns": metadata.st_mtime_ns,
        }
        if metadata.st_size <= 64 * 1024 * 1024:
            entry["sha256"] = _sha256_file(path)
        inventory.append(entry)
    if not inventory:
        _fail(f"checkpoint inventory is empty: {checkpoint}")
    identity = {"resolved_path": str(checkpoint), "inventory": inventory}
    return {**identity, "sha256": _sha256_bytes(_canonical_bytes(identity))}


def _run_id(checkpoint_name: str, checkpoint_sha256: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", checkpoint_name).strip("-.") or "checkpoint"
    return f"{slug[:48]}-{checkpoint_sha256[:16]}"


def _checkpoint_label(checkpoint: Path) -> str:
    return checkpoint.parent.name if checkpoint.name == "hf" else checkpoint.name


def _run_location(checkpoint: Path, campaign_root: Path) -> tuple[str, Path, dict[str, Any]]:
    descriptor = _checkpoint_descriptor(checkpoint)
    root = campaign_root.expanduser().resolve(strict=False)
    resolved = Path(descriptor["resolved_path"])
    run_id = _run_id(_checkpoint_label(resolved), descriptor["sha256"])
    return run_id, root / run_id, descriptor


def _local_reference_path(value: str, *, label: str) -> Path:
    if value.startswith("file:"):
        parsed = urlparse(value)
        if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
            _fail(f"{label} is not a local file URL: {value}")
        path = Path(unquote(parsed.path))
    else:
        if "://" in value:
            _fail(f"{label} is remote rather than local: {value}")
        path = Path(value)
    if not path.is_absolute():
        _fail(f"{label} is not absolute: {value}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CampaignError(f"{label} does not exist: {value}") from exc
    if not resolved.is_file():
        _fail(f"{label} is not a file: {value}")
    return resolved


def _read_dataset(path: Path, *, expected_tasks: int) -> Dataset:
    if expected_tasks <= 0:
        _fail("expected task count must be positive")
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise CampaignError(f"dataset does not exist: {path}") from exc
    _require_regular(resolved)
    raw = resolved.read_bytes()
    raw_lines = tuple(raw.splitlines(keepends=True))
    if not raw_lines or any(not line.strip() for line in raw_lines):
        _fail(f"dataset must be nonempty JSONL without blank lines: {resolved}")
    if len(raw_lines) != expected_tasks:
        _fail(f"dataset has {len(raw_lines)} rows, expected exactly {expected_tasks}")

    rows: list[dict[str, Any]] = []
    task_ids: list[str] = []
    reference_paths: list[Path] = []
    reference_count = 0
    for line_number, raw_line in enumerate(raw_lines, 1):
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CampaignError(f"invalid dataset JSON at {resolved}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            _fail(f"dataset row {line_number} is not an object")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in (".", ".."):
            _fail(f"dataset row {line_number} has an invalid task_id")
        if "/" in task_id or "\x00" in task_id or (os.altsep and os.altsep in task_id):
            _fail(f"dataset row {line_number} task_id is not path-safe: {task_id!r}")
        references = row.get("reference_file_urls")
        if not isinstance(references, list) or any(not isinstance(item, str) for item in references):
            _fail(f"dataset row {line_number} reference_file_urls must be a list of strings")
        for index, reference in enumerate(references):
            reference_paths.append(
                _local_reference_path(reference, label=f"row {line_number} reference_file_urls[{index}]")
            )
        reference_count += len(references)
        rows.append(row)
        task_ids.append(task_id)
    if len(set(task_ids)) != len(task_ids):
        duplicates = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
        _fail(f"dataset task_id values are not unique: {duplicates[:5]}")
    return Dataset(
        resolved,
        raw,
        raw_lines,
        tuple(rows),
        tuple(task_ids),
        reference_count,
        tuple(reference_paths),
    )


def _reference_assets_document(dataset: Dataset, *, hash_contents: bool) -> dict[str, Any]:
    """Describe the exact local task attachments without repeated bulk reads.

    Preparation hashes every unique file once. Routine preflights compare the
    resolved path, byte size, and nanosecond mtime against that immutable
    receipt; the final gate can request a full content rehash.
    """

    records: list[dict[str, Any]] = []
    for path in sorted(set(dataset.reference_paths), key=lambda value: value.as_posix()):
        _require_regular(path)
        metadata = path.stat()
        record: dict[str, Any] = {
            "path": str(path),
            "bytes": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
        }
        if hash_contents:
            record["sha256"] = _sha256_file(path)
        records.append(record)
    return {
        "schema": f"{SCHEMA_VERSION}-reference-assets",
        "dataset_path": str(dataset.path),
        "occurrences": dataset.reference_count,
        "unique_files": len(records),
        "files": records,
    }


def _task_set_sha256(task_ids: Iterable[str]) -> str:
    return _sha256_bytes(("\n".join(sorted(task_ids)) + "\n").encode("utf-8"))


def _partition(dataset: Dataset, shards: int) -> tuple[tuple[bytes, ...], ...]:
    if shards <= 0 or shards > len(dataset.raw_lines):
        _fail(f"shards must be between 1 and {len(dataset.raw_lines)}, got {shards}")
    groups: list[list[bytes]] = [[] for _ in range(shards)]
    for index, line in enumerate(dataset.raw_lines):
        groups[index % shards].append(line)
    return tuple(tuple(group) for group in groups)


def locate_campaign(checkpoint: Path, campaign_root: Path) -> dict[str, Any]:
    run_id, run_dir, descriptor = _run_location(checkpoint, campaign_root)
    return {"run_id": run_id, "run_dir": str(run_dir), "checkpoint_sha256": descriptor["sha256"]}


def prepare_campaign(
    *,
    checkpoint: Path,
    dataset_path: Path,
    campaign_root: Path,
    shards: int = DEFAULT_SHARDS,
    expected_tasks: int = DEFAULT_TASKS,
    profile_out: Path | None = None,
) -> dict[str, Any]:
    run_id, run_dir, checkpoint_document = _run_location(checkpoint, campaign_root)
    dataset = _read_dataset(dataset_path, expected_tasks=expected_tasks)
    groups = _partition(dataset, shards)
    shard_dir = run_dir / "shards"

    checkpoint_payload = _canonical_bytes({"schema": f"{SCHEMA_VERSION}-checkpoint", **checkpoint_document})
    reference_assets_path = run_dir / "reference_assets_fingerprint.json"
    if reference_assets_path.exists() or reference_assets_path.is_symlink():
        reference_assets_payload = _validate_reference_assets(dataset, reference_assets_path, rehash=False)
    else:
        reference_assets_document = _reference_assets_document(dataset, hash_contents=True)
        reference_assets_payload = _canonical_bytes(reference_assets_document)
    dataset_document = {
        "schema": f"{SCHEMA_VERSION}-dataset",
        "path": str(dataset.path),
        "sha256": dataset.sha256,
        "bytes": len(dataset.raw),
        "rows": len(dataset.rows),
        "unique_task_ids": len(dataset.task_ids),
        "task_ids_sha256": _task_set_sha256(dataset.task_ids),
        "reference_file_urls": {"entries": dataset.reference_count, "remote": 0, "non_absolute": 0},
        "reference_assets_path": str(reference_assets_path),
        "reference_assets_sha256": _sha256_bytes(reference_assets_payload),
    }
    dataset_payload = _canonical_bytes(dataset_document)

    shard_payloads: dict[Path, bytes] = {}
    shard_records: list[dict[str, Any]] = []
    for index, lines in enumerate(groups):
        payload = b"".join(lines)
        path = shard_dir / f"shard_{index:02d}_of_{shards:02d}.jsonl"
        ids = list(dataset.task_ids[index::shards])
        shard_payloads[path] = payload
        shard_records.append(
            {
                "index": index,
                "path": str(path),
                "rows": len(lines),
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
                "task_ids": ids,
            }
        )
    line_hashes = sorted(_sha256_bytes(line) for line in dataset.raw_lines)
    shard_manifest = {
        "schema": f"{SCHEMA_VERSION}-shards",
        "partition": f"zero-based raw-line index modulo {shards}",
        "source_path": str(dataset.path),
        "source_sha256": dataset.sha256,
        "source_rows": len(dataset.rows),
        "source_task_ids_sha256": _task_set_sha256(dataset.task_ids),
        "raw_line_multiset_sha256": _sha256_bytes(("\n".join(line_hashes) + "\n").encode()),
        "shard_count": shards,
        "shards": shard_records,
    }
    shard_manifest_path = shard_dir / "manifest.json"
    shard_manifest_payload = _canonical_bytes(shard_manifest)
    shard_sidecar_path = shard_dir / "manifest.json.sha256.json"
    shard_sidecar_payload = _canonical_bytes(_sidecar(shard_manifest_path, shard_manifest_payload))

    campaign = {
        "schema": SCHEMA_VERSION,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "checkpoint": {
            "path": checkpoint_document["resolved_path"],
            "sha256": checkpoint_document["sha256"],
            "fingerprint_path": str(run_dir / "checkpoint_fingerprint.json"),
            "fingerprint_sha256": _sha256_bytes(checkpoint_payload),
        },
        "dataset": {
            "path": str(dataset.path),
            "sha256": dataset.sha256,
            "fingerprint_path": str(run_dir / "dataset_fingerprint.json"),
            "fingerprint_sha256": _sha256_bytes(dataset_payload),
            "rows": len(dataset.rows),
        },
        "shards": {
            "count": shards,
            "manifest_path": str(shard_manifest_path),
            "manifest_sha256": _sha256_bytes(shard_manifest_payload),
        },
    }
    campaign_path = run_dir / "campaign.json"
    campaign_payload = _canonical_bytes(campaign)
    campaign_sidecar_path = run_dir / "campaign.json.sha256.json"
    campaign_sidecar_payload = _canonical_bytes(_sidecar(campaign_path, campaign_payload))

    artifacts: dict[Path, bytes] = {
        **shard_payloads,
        run_dir / "checkpoint_fingerprint.json": checkpoint_payload,
        reference_assets_path: reference_assets_payload,
        run_dir / "dataset_fingerprint.json": dataset_payload,
        shard_manifest_path: shard_manifest_payload,
        shard_sidecar_path: shard_sidecar_payload,
        campaign_path: campaign_payload,
        campaign_sidecar_path: campaign_sidecar_payload,
    }
    allowed_shard_names = {path.name for path in shard_payloads}
    allowed_shard_names.update({shard_manifest_path.name, shard_sidecar_path.name})
    if shard_dir.exists():
        unexpected = sorted(path.name for path in shard_dir.iterdir() if path.name not in allowed_shard_names)
        if unexpected:
            _fail(f"unexpected files in immutable shard directory {shard_dir}: {unexpected}")
    _publish_bundle(artifacts)

    if profile_out is not None:
        profile_path = profile_out.expanduser().resolve(strict=False)
        profile = {
            "schema": f"{SCHEMA_VERSION}-profile",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "campaign_path": str(campaign_path),
            "campaign_sha256": _sha256_bytes(campaign_payload),
            "shards_manifest_path": str(shard_manifest_path),
        }
        profile_payload = _canonical_bytes(profile)
        _publish_bundle(
            {
                profile_path: profile_payload,
                profile_path.with_name(f"{profile_path.name}.sha256.json"): _canonical_bytes(
                    _sidecar(profile_path, profile_payload)
                ),
            }
        )
    return {"run_id": run_id, "run_dir": str(run_dir), "campaign_sha256": _sha256_bytes(campaign_payload)}


def _read_json_object(path: Path, *, label: str, immutable: bool = False) -> dict[str, Any]:
    _require_regular(path, mode=FILE_MODE if immutable else None)
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} is not a JSON object: {path}")
    return value


def _validate_sidecar(path: Path, target: Path) -> None:
    document = _read_json_object(path, label="SHA-256 sidecar", immutable=True)
    target_payload = target.read_bytes()
    if set(document) != {"schema", "path", "bytes", "sha256"}:
        _fail(f"malformed SHA-256 sidecar: {path}")
    if document["schema"] != f"{SCHEMA_VERSION}-sha256":
        _fail(f"SHA-256 sidecar schema mismatch: {path}")
    if Path(str(document["path"])).resolve() != target.resolve():
        _fail(f"SHA-256 sidecar path mismatch: {path}")
    if document["bytes"] != len(target_payload) or document["sha256"] != _sha256_bytes(target_payload):
        _fail(f"SHA-256 sidecar digest mismatch: {path}")
    if path.read_bytes() != _canonical_bytes(_sidecar(target, target_payload)):
        _fail(f"SHA-256 sidecar canonical content mismatch: {path}")


def _validate_reference_assets(dataset: Dataset, path: Path, *, rehash: bool) -> bytes:
    document = _read_json_object(path, label="reference assets fingerprint", immutable=True)
    if path.read_bytes() != _canonical_bytes(document):
        _fail("reference assets fingerprint is not canonical")
    if (
        document.get("schema") != f"{SCHEMA_VERSION}-reference-assets"
        or document.get("dataset_path") != str(dataset.path)
        or document.get("occurrences") != dataset.reference_count
    ):
        _fail("reference assets fingerprint contract drift")
    records = document.get("files")
    if not isinstance(records, list) or document.get("unique_files") != len(records):
        _fail("reference assets fingerprint inventory is malformed")
    expected_paths = sorted({str(value) for value in dataset.reference_paths})
    observed_paths: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "mtime_ns", "sha256"}:
            _fail(f"reference asset record {index} is malformed")
        raw_path = record["path"]
        if not isinstance(raw_path, str):
            _fail(f"reference asset record {index} has no path")
        asset = Path(raw_path)
        if not asset.is_absolute() or asset.resolve(strict=True) != asset:
            _fail(f"reference asset path drift: {asset}")
        _require_regular(asset)
        metadata = asset.stat()
        if type(record["bytes"]) is not int or type(record["mtime_ns"]) is not int:
            _fail(f"reference asset record {index} has invalid stat fields")
        if metadata.st_size != record["bytes"] or metadata.st_mtime_ns != record["mtime_ns"]:
            _fail(f"reference asset stat drift: {asset}")
        digest = record["sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            _fail(f"reference asset record {index} has an invalid digest")
        if rehash and _sha256_file(asset) != digest:
            _fail(f"reference asset content drift: {asset}")
        observed_paths.append(raw_path)
    if observed_paths != expected_paths:
        _fail("dataset reference asset paths differ from their immutable inventory")
    return path.read_bytes()


def verify_campaign(run_dir: Path, *, rehash_reference_assets: bool = False) -> dict[str, Any]:
    resolved_run = run_dir.expanduser().resolve(strict=True)
    campaign_path = resolved_run / "campaign.json"
    _require_regular(campaign_path, mode=FILE_MODE)
    _validate_sidecar(resolved_run / "campaign.json.sha256.json", campaign_path)
    campaign = _read_json_object(campaign_path, label="campaign", immutable=True)
    if campaign.get("schema") != SCHEMA_VERSION or campaign.get("run_dir") != str(resolved_run):
        _fail("campaign schema or run directory mismatch")

    checkpoint_info = campaign.get("checkpoint")
    if not isinstance(checkpoint_info, dict):
        _fail("campaign checkpoint contract is malformed")
    checkpoint_path = Path(str(checkpoint_info.get("path", "")))
    checkpoint_document = _checkpoint_descriptor(checkpoint_path)
    checkpoint_fingerprint_path = resolved_run / "checkpoint_fingerprint.json"
    _require_regular(checkpoint_fingerprint_path, mode=FILE_MODE)
    checkpoint_payload = checkpoint_fingerprint_path.read_bytes()
    expected_checkpoint_payload = _canonical_bytes({"schema": f"{SCHEMA_VERSION}-checkpoint", **checkpoint_document})
    if checkpoint_payload != expected_checkpoint_payload:
        _fail("checkpoint inventory drift")
    if checkpoint_info.get("sha256") != checkpoint_document["sha256"]:
        _fail("checkpoint identity drift")
    if checkpoint_info.get("fingerprint_path") != str(checkpoint_fingerprint_path):
        _fail("checkpoint fingerprint path drift")
    if checkpoint_info.get("fingerprint_sha256") != _sha256_bytes(checkpoint_payload):
        _fail("checkpoint fingerprint digest mismatch")

    dataset_info = campaign.get("dataset")
    if not isinstance(dataset_info, dict) or type(dataset_info.get("rows")) is not int:
        _fail("campaign dataset contract is malformed")
    dataset = _read_dataset(Path(str(dataset_info.get("path", ""))), expected_tasks=dataset_info["rows"])
    dataset_fingerprint_path = resolved_run / "dataset_fingerprint.json"
    _require_regular(dataset_fingerprint_path, mode=FILE_MODE)
    dataset_fingerprint_payload = dataset_fingerprint_path.read_bytes()
    dataset_fingerprint = _read_json_object(dataset_fingerprint_path, label="dataset fingerprint", immutable=True)
    reference_assets_path = resolved_run / "reference_assets_fingerprint.json"
    reference_assets_payload = _validate_reference_assets(
        dataset,
        reference_assets_path,
        rehash=rehash_reference_assets,
    )
    expected_dataset_fingerprint = {
        "schema": f"{SCHEMA_VERSION}-dataset",
        "path": str(dataset.path),
        "sha256": dataset.sha256,
        "bytes": len(dataset.raw),
        "rows": len(dataset.rows),
        "unique_task_ids": len(dataset.task_ids),
        "task_ids_sha256": _task_set_sha256(dataset.task_ids),
        "reference_file_urls": {"entries": dataset.reference_count, "remote": 0, "non_absolute": 0},
        "reference_assets_path": str(reference_assets_path),
        "reference_assets_sha256": _sha256_bytes(reference_assets_payload),
    }
    if dataset_fingerprint_payload != _canonical_bytes(expected_dataset_fingerprint):
        _fail("dataset fingerprint content drift")
    if dataset_fingerprint.get("sha256") != dataset.sha256 or dataset_info.get("sha256") != dataset.sha256:
        _fail("dataset digest drift")
    if dataset_fingerprint.get("task_ids_sha256") != _task_set_sha256(dataset.task_ids):
        _fail("dataset task identity drift")
    if dataset_info.get("fingerprint_path") != str(dataset_fingerprint_path):
        _fail("dataset fingerprint path drift")
    if dataset_info.get("fingerprint_sha256") != _sha256_bytes(dataset_fingerprint_payload):
        _fail("dataset fingerprint digest mismatch")

    shard_info = campaign.get("shards")
    if not isinstance(shard_info, dict) or type(shard_info.get("count")) is not int:
        _fail("campaign shard contract is malformed")
    shard_count = shard_info["count"]
    groups = _partition(dataset, shard_count)
    manifest_path = resolved_run / "shards" / "manifest.json"
    _require_regular(manifest_path, mode=FILE_MODE)
    _validate_sidecar(resolved_run / "shards" / "manifest.json.sha256.json", manifest_path)
    if shard_info.get("manifest_path") != str(manifest_path):
        _fail("shard manifest path drift")
    if shard_info.get("manifest_sha256") != _sha256_file(manifest_path):
        _fail("shard manifest digest drift")
    manifest = _read_json_object(manifest_path, label="shard manifest", immutable=True)
    records = manifest.get("shards")
    if not isinstance(records, list) or len(records) != shard_count:
        _fail("shard manifest record count mismatch")
    expected_names: set[str] = {"manifest.json", "manifest.json.sha256.json"}
    expected_records: list[dict[str, Any]] = []
    for index, (record, lines) in enumerate(zip(records, groups, strict=True)):
        if not isinstance(record, dict) or record.get("index") != index:
            _fail(f"shard record {index} is malformed")
        expected_path = resolved_run / "shards" / f"shard_{index:02d}_of_{shard_count:02d}.jsonl"
        expected_names.add(expected_path.name)
        if record.get("path") != str(expected_path):
            _fail(f"shard {index} path drift")
        _require_regular(expected_path, mode=FILE_MODE)
        payload = expected_path.read_bytes()
        expected_payload = b"".join(lines)
        if payload != expected_payload:
            _fail(f"shard {index} does not equal the dataset modulo partition")
        if record.get("sha256") != _sha256_bytes(payload) or record.get("bytes") != len(payload):
            _fail(f"shard {index} fingerprint drift")
        if record.get("rows") != len(lines) or record.get("task_ids") != list(dataset.task_ids[index::shard_count]):
            _fail(f"shard {index} task coverage drift")
        expected_records.append(
            {
                "index": index,
                "path": str(expected_path),
                "rows": len(lines),
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
                "task_ids": list(dataset.task_ids[index::shard_count]),
            }
        )
    line_hashes = sorted(_sha256_bytes(line) for line in dataset.raw_lines)
    expected_manifest = {
        "schema": f"{SCHEMA_VERSION}-shards",
        "partition": f"zero-based raw-line index modulo {shard_count}",
        "source_path": str(dataset.path),
        "source_sha256": dataset.sha256,
        "source_rows": len(dataset.rows),
        "source_task_ids_sha256": _task_set_sha256(dataset.task_ids),
        "raw_line_multiset_sha256": _sha256_bytes(("\n".join(line_hashes) + "\n").encode()),
        "shard_count": shard_count,
        "shards": expected_records,
    }
    manifest_payload = manifest_path.read_bytes()
    if manifest_payload != _canonical_bytes(expected_manifest):
        _fail("shard manifest canonical content drift")

    expected_run_id = _run_id(_checkpoint_label(checkpoint_path), checkpoint_document["sha256"])
    expected_campaign = {
        "schema": SCHEMA_VERSION,
        "run_id": expected_run_id,
        "run_dir": str(resolved_run),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_document["sha256"],
            "fingerprint_path": str(checkpoint_fingerprint_path),
            "fingerprint_sha256": _sha256_bytes(checkpoint_payload),
        },
        "dataset": {
            "path": str(dataset.path),
            "sha256": dataset.sha256,
            "fingerprint_path": str(dataset_fingerprint_path),
            "fingerprint_sha256": _sha256_bytes(dataset_fingerprint_payload),
            "rows": len(dataset.rows),
        },
        "shards": {
            "count": shard_count,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_bytes(manifest_payload),
        },
    }
    if campaign_path.read_bytes() != _canonical_bytes(expected_campaign):
        _fail("campaign canonical content drift")
    if resolved_run.name != expected_run_id:
        _fail(f"campaign run directory name drift: {resolved_run.name} != {expected_run_id}")
    actual_names = {path.name for path in (resolved_run / "shards").iterdir()}
    if actual_names != expected_names:
        _fail(f"unexpected shard artifacts: {sorted(actual_names - expected_names)}")
    return {
        "status": "PASS",
        "run_id": campaign["run_id"],
        "run_dir": str(resolved_run),
        "dataset_rows": len(dataset.rows),
        "shards": shard_count,
    }


def _scan_deliverables(deliverables: Path) -> set[str]:
    try:
        root = deliverables.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise CampaignError(f"deliverables directory does not exist: {deliverables}") from exc
    if not root.is_dir():
        _fail(f"deliverables path is not a directory: {root}")
    completed: set[str] = set()
    for task_dir in sorted(root.glob("task_*")):
        if task_dir.is_symlink() or not task_dir.is_dir():
            continue
        marker = task_dir / "repeat_0" / "finish_params.json"
        if not marker.exists() and not marker.is_symlink():
            continue
        task_id = task_dir.name.removeprefix("task_")
        _require_regular(marker)
        try:
            value = json.loads(marker.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CampaignError(f"invalid finish marker {marker}: {exc}") from exc
        # Stirrup persists JSON null when an otherwise normal rollout reaches
        # the max-turn cap without explicit finish arguments. The marker's
        # atomic presence is still terminal; accept null or an object, while
        # rejecting corrupt JSON and every other JSON shape.
        if value is not None and not isinstance(value, dict):
            _fail(f"finish marker is neither a JSON object nor null: {marker}")
        completed.add(task_id)
    return completed


def coverage_report(*, dataset_path: Path, deliverables: Path, expected_tasks: int = DEFAULT_TASKS) -> dict[str, Any]:
    dataset = _read_dataset(dataset_path, expected_tasks=expected_tasks)
    expected = set(dataset.task_ids)
    completed = _scan_deliverables(deliverables)
    missing = sorted(expected - completed)
    extra = sorted(completed - expected)
    return {
        "status": "PASS" if not missing and not extra else "INCOMPLETE",
        "dataset": str(dataset.path),
        "expected": len(expected),
        "completed": len(completed & expected),
        "missing": missing,
        "extra": extra,
    }


def write_residue(
    *,
    dataset_path: Path,
    deliverables: Path,
    output: Path,
    shards_dir: Path,
    max_shards: int,
    expected_tasks: int = DEFAULT_TASKS,
) -> dict[str, Any]:
    if max_shards <= 0:
        _fail("max-shards must be positive")
    dataset = _read_dataset(dataset_path, expected_tasks=expected_tasks)
    completed = _scan_deliverables(deliverables)
    expected = set(dataset.task_ids)
    extra = sorted(completed - expected)
    if extra:
        _fail(f"deliverables contain completed task IDs absent from the dataset: {extra}")
    missing_indices = [index for index, task_id in enumerate(dataset.task_ids) if task_id not in completed]
    missing_ids = [dataset.task_ids[index] for index in missing_indices]
    output_path = output.expanduser().resolve(strict=False)
    shard_root = shards_dir.expanduser().resolve(strict=False)
    if output_path == shard_root or shard_root in output_path.parents:
        _fail("residue output must be outside shards-dir")
    output_payload = b"".join(dataset.raw_lines[index] for index in missing_indices)
    shard_count = min(max_shards, len(missing_indices))
    artifacts: dict[Path, bytes] = {output_path: output_payload}
    records: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        indices = missing_indices[shard_index::shard_count]
        payload = b"".join(dataset.raw_lines[index] for index in indices)
        path = shard_root / f"shard_{shard_index:02d}_of_{shard_count:02d}.jsonl"
        artifacts[path] = payload
        records.append(
            {
                "index": shard_index,
                "path": str(path),
                "rows": len(indices),
                "sha256": _sha256_bytes(payload),
                "task_ids": [dataset.task_ids[index] for index in indices],
            }
        )
    manifest_path = shard_root / "manifest.json"
    manifest = {
        "schema": f"{SCHEMA_VERSION}-residue",
        "dataset_path": str(dataset.path),
        "dataset_sha256": dataset.sha256,
        "deliverables": str(deliverables.expanduser().resolve()),
        "output": str(output_path),
        "output_sha256": _sha256_bytes(output_payload),
        "expected_tasks": len(dataset.task_ids),
        "completed_tasks": len(completed),
        "missing_tasks": len(missing_ids),
        "missing_task_ids": missing_ids,
        "shard_count": shard_count,
        "shards": records,
    }
    manifest_payload = _canonical_bytes(manifest)
    artifacts[manifest_path] = manifest_payload
    artifacts[shard_root / "manifest.json.sha256.json"] = _canonical_bytes(_sidecar(manifest_path, manifest_payload))
    allowed = {path.name for path in artifacts if path.parent == shard_root}
    if shard_root.exists():
        unexpected = sorted(path.name for path in shard_root.iterdir() if path.name not in allowed)
        if unexpected:
            _fail(f"unexpected files in immutable residue shard directory {shard_root}: {unexpected}")
    _publish_bundle(artifacts)
    return {
        "status": "PASS",
        "expected": len(dataset.task_ids),
        "completed": len(completed),
        "missing": len(missing_ids),
        "output": str(output_path),
        "shards": shard_count,
    }


def _read_jsonl_objects(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require_regular(path)
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                _fail(f"{label} has a blank line at {path}:{line_number}")
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CampaignError(f"invalid {label} JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                _fail(f"{label} row {line_number} is not an object")
            rows.append(value)
    return rows


def _exact_int(value: Any, expected: int, *, label: str) -> None:
    if type(value) is not int or value != expected:
        _fail(f"{label} is {value!r}, expected integer {expected}")


def _nonnegative_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{label} is {value!r}, expected a nonnegative integer")
    return value


def _numeric(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        _fail(f"{label} is not a finite number")
    return float(value)


def _exact_number(value: Any, expected: float, *, label: str) -> None:
    actual = _numeric(value, label=label)
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        _fail(f"{label} is {actual!r}, expected {expected!r}")


def _task_rollout_key(record: Mapping[str, Any], *, label: str) -> tuple[int, int]:
    task_index = record.get("_ng_task_index")
    rollout_index = record.get("_ng_rollout_index")
    if type(task_index) is not int or task_index < 0:
        _fail(f"{label} has an invalid _ng_task_index")
    if type(rollout_index) is not int or rollout_index < 0:
        _fail(f"{label} has an invalid _ng_rollout_index")
    return task_index, rollout_index


def _journal_key_set(record: Mapping[str, Any], field: str, *, label: str) -> set[tuple[int, int]]:
    values = record.get(field)
    if not isinstance(values, list):
        _fail(f"{label} {field} is not a list")
    keys: set[tuple[int, int]] = set()
    for index, value in enumerate(values):
        if not isinstance(value, list) or len(value) != 2:
            _fail(f"{label} {field}[{index}] is not a two-element list")
        task_index, rollout_index = value
        if type(task_index) is not int or task_index < 0 or type(rollout_index) is not int or rollout_index < 0:
            _fail(f"{label} {field}[{index}] is not a nonnegative integer pair")
        key = (task_index, rollout_index)
        if key in keys:
            _fail(f"{label} {field} contains duplicate key {key}")
        keys.add(key)
    return keys


def _elo_evidence_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    pooled: dict[str, dict[str, Any]] = {}
    for row in rows:
        for reference_id, counts in row["per_reference"].items():
            entry = pooled.setdefault(
                reference_id,
                {"wins": 0, "losses": 0, "ties": 0, "reference_elo": None},
            )
            for field in ("wins", "losses", "ties"):
                entry[field] += counts[field]
            if entry["reference_elo"] is None:
                entry["reference_elo"] = counts["reference_elo"]
    payload = json.dumps(pooled, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _calculate_mle_elo(
    battles: Sequence[tuple[float, float, float, float]],
    scale: float = 400.0,
    base: float = 10.0,
) -> tuple[float, float] | None:
    """Independently fit the anchored Bradley-Terry ELO used by GDPVal."""
    data: list[tuple[float, float, float]] = []
    for reference_elo, wins, losses, ties in battles:
        games = float(wins) + float(losses) + float(ties)
        if games <= 0:
            continue
        score = float(wins) + 0.5 * float(ties)
        data.append((float(reference_elo), score, games))

    if not data:
        return None

    total_score = sum(score for _, score, _ in data)
    total_games = sum(games for _, _, games in data)
    epsilon = 1e-3
    overall_win_rate = total_score / total_games
    if overall_win_rate <= epsilon or overall_win_rate >= 1.0 - epsilon:
        clamped = min(max(overall_win_rate, epsilon), 1.0 - epsilon)
        mean_reference_elo = sum(reference_elo * games for reference_elo, _, games in data) / total_games
        elo = mean_reference_elo - scale * (math.log10(1.0 - clamped) - math.log10(clamped))
        return elo, (elo - 500.0) / 2000.0

    def gradient(rating: float) -> float:
        total = 0.0
        for reference_elo, score, games in data:
            probability = 1.0 / (1.0 + base ** ((reference_elo - rating) / scale))
            total += score - games * probability
        return total

    lower = min(reference_elo for reference_elo, _, _ in data) - 4000.0
    upper = max(reference_elo for reference_elo, _, _ in data) + 4000.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if gradient(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    elo = 0.5 * (lower + upper)
    return elo, (elo - 500.0) / 2000.0


def _require_recomputed_metric(
    value: Any,
    expected: float,
    *,
    label: str,
    tolerance: float,
) -> float:
    actual = _numeric(value, label=label)
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        _fail(f"{label} is {actual!r}, but row evidence recomputes to {expected!r}")
    return actual


def validate_result(
    *,
    output: Path,
    journal: Path,
    dataset: Path | None = None,
    expected_tasks: int = DEFAULT_TASKS,
) -> dict[str, Any]:
    output_path = output.expanduser().resolve(strict=True)
    journal_path = journal.expanduser().resolve(strict=True)
    rows = _read_jsonl_objects(output_path, label="result output")
    by_stage: dict[int, dict[str, dict[str, Any]]] = {0: {}, 1: {}}
    task_keys_by_stage: dict[int, dict[str, tuple[int, int]]] = {0: {}, 1: {}}
    key_tasks_by_stage: dict[int, dict[tuple[int, int], str]] = {0: {}, 1: {}}
    row_reference_ids: dict[int, dict[str, str]] = {0: {}, 1: {}}
    reference_elos: dict[str, float] = {}
    stage_battles: dict[int, dict[str, dict[str, float | int]]] = {0: {}, 1: {}}
    observed_trial_judges: set[str] = set()
    av_routed_rows: dict[int, int] = {0: 0, 1: 0}
    total_wins = total_losses = total_ties = 0
    for row_number, row in enumerate(rows, 1):
        if row.get("verify_mode") != "comparison":
            _fail(f"result row {row_number} is not a comparison result")
        stage = row.get("stage_index")
        if type(stage) is not int or stage not in (0, 1):
            _fail(f"result row {row_number} has an invalid stage_index")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            _fail(f"result row {row_number} has an invalid task_id")
        if task_id in by_stage[stage]:
            _fail(f"result duplicates stage {stage} task_id {task_id}")
        task_key = _task_rollout_key(row, label=f"result row {row_number}")
        if task_key in key_tasks_by_stage[stage]:
            _fail(
                f"result duplicates stage {stage} task/rollout key {task_key} "
                f"for {key_tasks_by_stage[stage][task_key]} and {task_id}"
            )
        if row.get("_ng_failure_class") is not None or row.get("_ng_no_persist"):
            _fail(f"result row {row_number} contains a failure marker")
        if row.get("error") is not None and row.get("error") is not False:
            _fail(f"result row {row_number} has a top-level error")
        response = row.get("response")
        if isinstance(response, dict) and response.get("error") is not None:
            _fail(f"result row {row_number} has a response error")
        if row.get("invalid_judge_response") is not None and row.get("invalid_judge_response") is not False:
            _fail(f"result row {row_number} is marked invalid")
        judge = row.get("judge_response")
        if not isinstance(judge, dict):
            _fail(f"result row {row_number} has no judge_response object")
        if (
            (judge.get("error") is not None and judge.get("error") is not False)
            or (judge.get("scoring_error") is not None and judge.get("scoring_error") is not False)
            or judge.get("ref_errors") != {}
        ):
            _fail(f"result row {row_number} has judge or reference errors")
        _exact_int(judge.get("total_judged"), 4, label=f"result row {row_number} total_judged")
        _exact_int(judge.get("total_invalid"), 0, label=f"result row {row_number} total_invalid")
        references = row.get("reference_ids")
        if not isinstance(references, list) or len(references) != 1 or not isinstance(references[0], str):
            _fail(f"result row {row_number} does not have exactly one assigned reference")
        row_reference_ids[stage][task_id] = references[0]
        _exact_int(
            row.get("expected_final_stage_index"),
            1,
            label=f"result row {row_number} expected_final_stage_index",
        )
        _exact_int(
            row.get("expected_stage_row_count"),
            STAGE0_PLANNED_TASKS if stage == 0 else STAGE1_TASKS,
            label=f"result row {row_number} expected_stage_row_count",
        )

        per_reference = row.get("per_reference")
        if not isinstance(per_reference, dict) or set(per_reference) != {references[0]}:
            _fail(f"result row {row_number} top-level per_reference does not match its assignment")
        reference_counts = per_reference[references[0]]
        if not isinstance(reference_counts, dict):
            _fail(f"result row {row_number} top-level per_reference tally is malformed")
        reference_votes = 0
        for field in ("wins", "losses", "ties"):
            value = reference_counts.get(field)
            if type(value) is not int or value < 0:
                _fail(f"result row {row_number} top-level per_reference {field} is invalid")
            reference_votes += value
        if reference_votes != TRIALS_PER_TASK:
            _fail(f"result row {row_number} top-level per_reference tally does not total four")
        reference_elo = _numeric(
            reference_counts.get("reference_elo"),
            label=f"result row {row_number} reference_elo",
        )
        existing_reference_elo = reference_elos.setdefault(references[0], reference_elo)
        if existing_reference_elo != reference_elo:
            _fail(
                f"result reference {references[0]} has inconsistent reference_elo values: "
                f"{existing_reference_elo!r} and {reference_elo!r}"
            )
        battle = stage_battles[stage].setdefault(
            references[0],
            {"wins": 0, "losses": 0, "ties": 0, "reference_elo": reference_elo},
        )
        for field in ("wins", "losses", "ties"):
            battle[field] += reference_counts[field]

        matchups = judge.get("per_ref_repeat")
        if not isinstance(matchups, list) or len(matchups) != 1 or not isinstance(matchups[0], dict):
            _fail(f"result row {row_number} does not have exactly one matchup receipt")
        matchup = matchups[0]
        if matchup.get("ref_id") != references[0] or matchup.get("ref_repeat") != "repeat_0":
            _fail(f"result row {row_number} matchup reference/repeat drift")
        trial_judges = matchup.get("trial_judges")
        if (
            not isinstance(trial_judges, list)
            or len(trial_judges) != 4
            or any(name not in EXPECTED_JUDGE_MODELS for name in trial_judges)
        ):
            _fail(f"result row {row_number} has an invalid four-trial judge schedule")
        av_routed = judge.get("av_routed")
        if type(av_routed) is not bool:
            _fail(f"result row {row_number} has no explicit av_routed receipt")
        if av_routed:
            if trial_judges != ["gemini-3.1-pro"] * 4:
                _fail(f"result row {row_number} AV route is not four exact Gemini trials")
            av_routed_rows[stage] += 1
        observed_trial_judges.update(trial_judges)
        _exact_int(matchup.get("invalid_count"), 0, label=f"result row {row_number} matchup invalid_count")
        matchup_counts = {
            key: _nonnegative_int(
                matchup.get(key),
                label=f"result row {row_number} matchup {key}",
            )
            for key in ("win_count_a", "win_count_b", "tie_count")
        }
        matchup_votes = sum(matchup_counts.values())
        if matchup_votes != 4:
            _fail(f"result row {row_number} matchup has {matchup_votes} votes, expected 4")
        _exact_int(matchup.get("task_count"), 4, label=f"result row {row_number} matchup task_count")
        raw_responses = matchup.get("raw_responses")
        if raw_responses is not None and (not isinstance(raw_responses, list) or len(raw_responses) != 4):
            _fail(f"result row {row_number} matchup does not have exactly four raw responses")

        matchup_per_judge = matchup.get("per_judge")
        if not isinstance(matchup_per_judge, dict) or not matchup_per_judge:
            _fail(f"result row {row_number} matchup has a malformed per-judge tally")
        matchup_per_judge_trials = 0
        matchup_per_judge_invalid = 0
        for name, counts in matchup_per_judge.items():
            if name not in EXPECTED_JUDGE_MODELS or not isinstance(counts, dict):
                _fail(f"result row {row_number} matchup has a malformed per-judge tally")
            matchup_per_judge_trials += _nonnegative_int(
                counts.get("trials"),
                label=f"result row {row_number} matchup judge {name} trials",
            )
            matchup_per_judge_invalid += _nonnegative_int(
                counts.get("invalid_count"),
                label=f"result row {row_number} matchup judge {name} invalid_count",
            )
        if matchup_per_judge_trials != 4:
            _fail(f"result row {row_number} matchup per-judge tally does not equal four trials")
        if matchup_per_judge_invalid != 0:
            _fail(f"result row {row_number} matchup per-judge tally contains invalid trials")
        _exact_int(judge.get("ref_repeat_count"), 1, label=f"result row {row_number} ref_repeat_count")

        panel = judge.get("judge_panel")
        expected_panel_names = {"gemini-3.1-pro"} if av_routed else set(EXPECTED_JUDGE_MODELS)
        if not isinstance(panel, list) or len(panel) != len(expected_panel_names):
            expected_label = "Gemini-only AV" if av_routed else "three-member"
            _fail(f"result row {row_number} does not have the exact {expected_label} panel receipt")
        panel_names: set[str] = set()
        for member in panel:
            if not isinstance(member, dict) or member.get("name") not in EXPECTED_JUDGE_MODELS:
                _fail(f"result row {row_number} has an unexpected panel member")
            name = member["name"]
            if name in panel_names:
                _fail(f"result row {row_number} has a duplicate panel member {name}")
            panel_names.add(name)
            if member.get("model") != EXPECTED_JUDGE_MODELS[name]:
                _fail(f"result row {row_number} has a panel model/weight mismatch")
            _exact_number(member.get("weight"), 1.0, label=f"result row {row_number} panel member weight")
        if panel_names != expected_panel_names:
            expected_label = "Gemini-only AV" if av_routed else "three-member"
            _fail(f"result row {row_number} does not have the exact {expected_label} panel receipt")
        per_judge = judge.get("per_judge")
        if not isinstance(per_judge, dict) or any(name not in EXPECTED_JUDGE_MODELS for name in per_judge):
            _fail(f"result row {row_number} has a malformed per-judge tally")
        if av_routed and set(per_judge) != {"gemini-3.1-pro"}:
            _fail(f"result row {row_number} AV route has a non-Gemini per-judge tally")
        per_judge_trials = 0
        for name, counts in per_judge.items():
            if not isinstance(counts, dict):
                _fail(f"result row {row_number} judge {name} has a malformed tally")
            _exact_int(
                counts.get("invalid_count"),
                0,
                label=f"result row {row_number} judge {name} invalid_count",
            )
            trials = _nonnegative_int(
                counts.get("trials"),
                label=f"result row {row_number} judge {name} trials",
            )
            per_judge_trials += trials
        if per_judge_trials != 4:
            _fail(f"result row {row_number} per-judge tally does not equal four trials")

        judge_wins = _nonnegative_int(
            judge.get("total_wins"),
            label=f"result row {row_number} judge total_wins",
        )
        judge_losses = _nonnegative_int(
            judge.get("total_losses"),
            label=f"result row {row_number} judge total_losses",
        )
        judge_ties = _nonnegative_int(
            judge.get("total_ties"),
            label=f"result row {row_number} judge total_ties",
        )
        matchup_outcomes = (
            matchup_counts["win_count_b"],
            matchup_counts["win_count_a"],
            matchup_counts["tie_count"],
        )
        if (judge_wins, judge_losses, judge_ties) != matchup_outcomes:
            _fail(f"result row {row_number} judge outcome counts differ from its matchup")

        wins = _nonnegative_int(
            row.get("total_wins", judge_wins),
            label=f"result row {row_number} total_wins",
        )
        losses = _nonnegative_int(
            row.get("total_losses", judge_losses),
            label=f"result row {row_number} total_losses",
        )
        ties = _nonnegative_int(
            row.get("total_ties", judge_ties),
            label=f"result row {row_number} total_ties",
        )
        if wins + losses + ties != 4:
            _fail(f"result row {row_number} outcome counts do not total four")
        if (wins, losses, ties) != (judge_wins, judge_losses, judge_ties):
            _fail(f"result row {row_number} outcome counts differ from judge_response")
        if (wins, losses, ties) != matchup_outcomes:
            _fail(f"result row {row_number} outcome counts differ from its matchup")
        if (wins, losses, ties) != (
            reference_counts["wins"],
            reference_counts["losses"],
            reference_counts["ties"],
        ):
            _fail(f"result row {row_number} outcome counts differ from top-level per_reference")
        total_wins += wins
        total_losses += losses
        total_ties += ties
        by_stage[stage][task_id] = row
        task_keys_by_stage[stage][task_id] = task_key
        key_tasks_by_stage[stage][task_key] = task_id

    stage0_tasks = len(by_stage[0])
    stage1_tasks = len(by_stage[1])
    if (
        not STAGE0_MIN_PARTIAL_TASKS <= stage0_tasks <= STAGE0_PLANNED_TASKS
        or stage1_tasks != STAGE1_TASKS
        or len(rows) != stage0_tasks + stage1_tasks
    ):
        _fail(
            f"result coverage is rows={len(rows)}, stage0={stage0_tasks}, stage1={stage1_tasks}; "
            f"expected Stage0 {STAGE0_MIN_PARTIAL_TASKS}-{STAGE0_PLANNED_TASKS} and Stage1 {STAGE1_TASKS}"
        )
    if not set(by_stage[0]).issubset(by_stage[1]):
        _fail("Stage0 task IDs are not a subset of Stage1 task IDs")
    if dataset is not None:
        canonical_dataset = _read_dataset(dataset, expected_tasks=expected_tasks)
        if set(by_stage[1]) != set(canonical_dataset.task_ids):
            missing = sorted(set(canonical_dataset.task_ids) - set(by_stage[1]))
            extra = sorted(set(by_stage[1]) - set(canonical_dataset.task_ids))
            _fail(f"Stage1 task IDs differ from canonical dataset: missing={missing}, extra={extra}")
    for task_id, task_key in task_keys_by_stage[0].items():
        if task_keys_by_stage[1][task_id] != task_key:
            _fail(f"task/rollout key differs across stages for {task_id}")
    if observed_trial_judges != set(EXPECTED_JUDGE_MODELS):
        _fail(f"three-judge panel coverage drift: {sorted(observed_trial_judges)}")
    expected_judged = TRIALS_PER_TASK * (stage0_tasks + stage1_tasks)
    if total_wins + total_losses + total_ties != expected_judged:
        _fail(f"aggregate output vote count is not {expected_judged}")

    recomputed_stage_elos: dict[int, tuple[float, float]] = {}
    for stage in (0, 1):
        battles = [
            (
                float(counts["reference_elo"]),
                float(counts["wins"]),
                float(counts["losses"]),
                float(counts["ties"]),
            )
            for counts in stage_battles[stage].values()
        ]
        fit = _calculate_mle_elo(battles)
        if fit is None:
            _fail(f"Stage{stage} row evidence has no battles for an ELO fit")
        recomputed_stage_elos[stage] = fit

    journal_rows = _read_jsonl_objects(journal_path, label="multistage journal")
    fingerprints: set[str] = set()
    plans: dict[int, dict[str, Any]] = {}
    outcomes: dict[int, dict[str, Any]] = {}
    latest_attempt_dispositions: dict[int, dict[tuple[int, int], dict[str, Any]]] = {}
    for record_number, record in enumerate(journal_rows, 1):
        stage = record.get("stage_index")
        status_value = record.get("status")
        if type(stage) is not int or stage not in (0, 1) or not isinstance(status_value, str):
            _fail(f"journal row {record_number} has an invalid stage/status")
        fingerprint = record.get("fingerprint")
        if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
            _fail(f"journal row {record_number} has no fingerprint")
        fingerprints.add(fingerprint)
        if status_value == "planned":
            plans[stage] = record
        elif status_value in ("complete", "partial_complete"):
            if status_value == "partial_complete" and stage != 0:
                _fail("partial_complete is allowed only for Stage0")
            outcomes[stage] = record
        elif status_value == "attempt_dispositions":
            attempts = record.get("attempts")
            if not isinstance(attempts, list) or not attempts:
                _fail(f"journal row {record_number} has malformed attempt dispositions")
            record_keys: set[tuple[int, int]] = set()
            stage_dispositions = latest_attempt_dispositions.setdefault(stage, {})
            for attempt_index, attempt in enumerate(attempts):
                if not isinstance(attempt, dict):
                    _fail(f"journal row {record_number} attempt {attempt_index} is not an object")
                key = _task_rollout_key(attempt, label=f"journal row {record_number} attempt {attempt_index}")
                if key in record_keys:
                    _fail(f"journal row {record_number} attempt dispositions contain duplicate key {key}")
                record_keys.add(key)
                failure_class = attempt.get("_ng_failure_class")
                if failure_class is not None and not isinstance(failure_class, str):
                    _fail(f"journal row {record_number} attempt {attempt_index} has invalid failure class")
                if type(attempt.get("_ng_no_persist")) is not bool:
                    _fail(f"journal row {record_number} attempt {attempt_index} has invalid no-persist flag")
                # Match pinned PR #2588's load_latest_attempt_dispositions: records
                # are replayed in append order and the last disposition for a
                # stage/task/rollout key is authoritative.
                stage_dispositions[key] = attempt
        elif status_value == "restart_from_stage":
            plans = {index: plan for index, plan in plans.items() if index <= stage}
            outcomes = {index: outcome for index, outcome in outcomes.items() if index < stage}
            # A restart invalidates dependent stages but deliberately preserves
            # the restarted stage's latest attempts. This mirrors the exact
            # resume semantics in pinned PR #2588.
            latest_attempt_dispositions = {
                index: dispositions for index, dispositions in latest_attempt_dispositions.items() if index <= stage
            }
        elif status_value != "restart_cleanup_complete":
            _fail(f"journal row {record_number} has unsupported status {status_value!r}")
    if len(fingerprints) != 1 or set(plans) != {0, 1} or set(outcomes) != {0, 1}:
        _fail("multistage journal lifecycle or fingerprint drift")

    plan_task_ids: dict[int, list[str]] = {}
    plan_reference_ids: dict[int, list[str]] = {}
    plan_assignments: dict[int, dict[str, str]] = {}
    for stage, expected_count, expected_references in (
        (0, STAGE0_PLANNED_TASKS, 9),
        (1, STAGE1_TASKS, 4),
    ):
        plan = plans[stage]
        task_ids = plan.get("task_ids")
        if not isinstance(task_ids, list) or any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
            _fail(f"journal Stage{stage} plan task IDs are malformed")
        if len(task_ids) != expected_count or len(set(task_ids)) != expected_count:
            _fail(f"journal Stage{stage} plan does not have {expected_count} unique task IDs")
        reference_ids = plan.get("reference_ids")
        if (
            not isinstance(reference_ids, list)
            or len(reference_ids) != expected_references
            or len(set(reference_ids)) != expected_references
            or any(not isinstance(reference_id, str) or not reference_id for reference_id in reference_ids)
        ):
            _fail(f"journal Stage{stage} reference set is not exactly {expected_references} unique IDs")
        assignment = plan.get("task_reference_ids")
        if (
            not isinstance(assignment, dict)
            or set(assignment) != set(task_ids)
            or any(
                not isinstance(reference_id, str) or reference_id not in reference_ids
                for reference_id in assignment.values()
            )
        ):
            _fail(f"journal Stage{stage} reference assignment is malformed")
        plan_task_ids[stage] = task_ids
        plan_reference_ids[stage] = reference_ids
        plan_assignments[stage] = assignment

    if set(plan_task_ids[1]) != set(by_stage[1]):
        _fail("journal Stage1 task IDs differ from result output")
    if not set(plan_task_ids[0]).issubset(plan_task_ids[1]):
        _fail("journal Stage0 plan task IDs are not a subset of Stage1")
    if not set(by_stage[0]).issubset(plan_task_ids[0]):
        _fail("Stage0 output includes task IDs absent from its plan")
    for stage in (0, 1):
        observed_assignment = {task_id: plan_assignments[stage][task_id] for task_id in by_stage[stage]}
        if observed_assignment != row_reference_ids[stage]:
            _fail(f"journal Stage{stage} assignments differ from result output")

    stage0_is_partial = stage0_tasks < STAGE0_PLANNED_TASKS
    expected_stage0_status = "partial_complete" if stage0_is_partial else "complete"
    if outcomes[0].get("status") != expected_stage0_status:
        _fail(f"Stage0 outcome must be {expected_stage0_status} for {stage0_tasks} result rows")
    if outcomes[1].get("status") != "complete":
        _fail("Stage1 outcome must be complete")

    if stage0_is_partial:
        outcome = outcomes[0]
        included_keys = _journal_key_set(outcome, "included_keys", label="Stage0 partial outcome")
        omitted_keys = _journal_key_set(outcome, "omitted_keys", label="Stage0 partial outcome")
        accepted_unresolved_keys = _journal_key_set(
            outcome, "accepted_unresolved_keys", label="Stage0 partial outcome"
        )
        already_resolved_omitted_keys = _journal_key_set(
            outcome, "already_resolved_omitted_keys", label="Stage0 partial outcome"
        )
        planned_keys = {task_keys_by_stage[1][task_id] for task_id in plan_task_ids[0]}
        output_keys = set(task_keys_by_stage[0].values())
        if included_keys != output_keys:
            _fail("Stage0 partial included_keys differ from result output keys")
        if included_keys & omitted_keys or included_keys | omitted_keys != planned_keys:
            _fail("Stage0 partial included/omitted keys do not partition the 45-row plan")
        if omitted_keys != planned_keys - output_keys:
            _fail("Stage0 partial omitted_keys differ from planned keys absent in output")
        if (
            accepted_unresolved_keys & already_resolved_omitted_keys
            or accepted_unresolved_keys | already_resolved_omitted_keys != omitted_keys
        ):
            _fail("Stage0 accepted/already-resolved omissions do not partition omitted_keys")
        stage0_dispositions = latest_attempt_dispositions.get(0, {})
        for key in sorted(accepted_unresolved_keys):
            disposition = stage0_dispositions.get(key)
            if disposition is None:
                _fail(f"Stage0 accepted unresolved key {key} has no attempt disposition")
            if (
                disposition.get("_ng_failure_class") not in STAGE0_PARTIAL_POLICY["newly_waivable_failure_classes"]
                or disposition.get("_ng_no_persist") is not False
            ):
                _fail(f"Stage0 accepted unresolved key {key} latest disposition is not a persisted waivable failure")

        evidence_sha256 = outcome.get("evidence_sha256")
        if not isinstance(evidence_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", evidence_sha256) is None:
            _fail("Stage0 partial outcome has an invalid evidence_sha256")
        expected_evidence_sha256 = _elo_evidence_sha256(by_stage[0].values())
        if evidence_sha256 != expected_evidence_sha256:
            _fail("Stage0 partial outcome evidence_sha256 differs from included ELO evidence")
        policy = outcome.get("policy")
        if not isinstance(policy, dict) or set(policy) != set(STAGE0_PARTIAL_POLICY):
            _fail("Stage0 partial outcome policy fields differ from the accepted policy")
        _exact_number(
            policy.get("min_success_fraction"),
            STAGE0_PARTIAL_POLICY["min_success_fraction"],
            label="Stage0 policy min_success_fraction",
        )
        _exact_number(
            policy.get("min_per_reference_success_fraction"),
            STAGE0_PARTIAL_POLICY["min_per_reference_success_fraction"],
            label="Stage0 policy min_per_reference_success_fraction",
        )
        _exact_int(
            policy.get("min_successful_rows_per_reference"),
            STAGE0_PARTIAL_POLICY["min_successful_rows_per_reference"],
            label="Stage0 policy min_successful_rows_per_reference",
        )
        if policy.get("newly_waivable_failure_classes") != STAGE0_PARTIAL_POLICY["newly_waivable_failure_classes"]:
            _fail("Stage0 partial outcome permits an unexpected failure class")

        success_fraction = stage0_tasks / STAGE0_PLANNED_TASKS
        if success_fraction < STAGE0_PARTIAL_POLICY["min_success_fraction"]:
            _fail("Stage0 partial output is below the overall success floor")
        _exact_number(outcome.get("success_fraction"), success_fraction, label="Stage0 success_fraction")
        _exact_number(
            outcome.get("persisted_success_fraction"),
            success_fraction,
            label="Stage0 persisted_success_fraction",
        )

        planned_per_reference = Counter(plan_assignments[0].values())
        successful_per_reference = Counter(row_reference_ids[0].values())
        per_reference = outcome.get("per_reference")
        if not isinstance(per_reference, dict) or set(per_reference) != set(plan_reference_ids[0]):
            _fail("Stage0 partial per_reference keys differ from the selected references")
        for reference_id in plan_reference_ids[0]:
            planned = planned_per_reference[reference_id]
            successful = successful_per_reference[reference_id]
            if planned <= 0:
                _fail(f"Stage0 reference {reference_id} has no planned rows")
            reference_fraction = successful / planned
            if (
                successful < STAGE0_PARTIAL_POLICY["min_successful_rows_per_reference"]
                or reference_fraction < STAGE0_PARTIAL_POLICY["min_per_reference_success_fraction"]
            ):
                _fail(f"Stage0 reference {reference_id} is below its partial-completion floor")
            counts = per_reference[reference_id]
            if not isinstance(counts, dict):
                _fail(f"Stage0 reference {reference_id} coverage receipt is malformed")
            _exact_int(counts.get("planned"), planned, label=f"Stage0 {reference_id} planned")
            _exact_int(counts.get("successful"), successful, label=f"Stage0 {reference_id} successful")
            _exact_int(counts.get("judged"), successful, label=f"Stage0 {reference_id} judged")
            _exact_number(
                counts.get("success_fraction"),
                reference_fraction,
                label=f"Stage0 {reference_id} success_fraction",
            )
    elif set(plan_task_ids[0]) != set(by_stage[0]):
        _fail("complete Stage0 plan task IDs differ from result output")

    aggregate_path = output_path.with_name(f"{output_path.stem}_aggregate_metrics.json")
    _require_regular(aggregate_path)
    try:
        metrics_document = json.loads(aggregate_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"invalid aggregate metrics JSON {aggregate_path}: {exc}") from exc
    if (
        not isinstance(metrics_document, list)
        or len(metrics_document) != 1
        or not isinstance(metrics_document[0], dict)
    ):
        _fail("aggregate metrics must be a singleton JSON-object list")
    entry = metrics_document[0]
    agent_metrics = entry.get("agent_metrics")
    key_metrics = entry.get("key_metrics")
    if not isinstance(agent_metrics, dict) or not isinstance(key_metrics, dict):
        _fail("aggregate metrics agent_metrics/key_metrics are malformed")
    recomputed_stage0_elo, recomputed_stage0_normalized = recomputed_stage_elos[0]
    recomputed_stage1_elo, recomputed_stage1_normalized = recomputed_stage_elos[1]
    elo = _require_recomputed_metric(
        agent_metrics.get("comparison/eval_elo"),
        recomputed_stage1_elo,
        label="comparison/eval_elo",
        tolerance=ELO_ABS_TOLERANCE,
    )
    key_elo = _numeric(key_metrics.get("comparison/eval_elo"), label="key comparison/eval_elo")
    if elo != key_elo:
        _fail("agent_metrics and key_metrics ELO values differ")
    _exact_int(agent_metrics.get("comparison/num_stages"), 2, label="comparison/num_stages")
    _exact_int(agent_metrics.get("comparison/stage_0/num_tasks"), stage0_tasks, label="Stage0 metric task count")
    _exact_int(agent_metrics.get("comparison/stage_1/num_tasks"), stage1_tasks, label="Stage1 metric task count")
    _exact_int(agent_metrics.get("comparison/judged"), expected_judged, label="comparison/judged")
    _exact_int(agent_metrics.get("comparison/wins"), total_wins, label="comparison/wins")
    _exact_int(agent_metrics.get("comparison/losses"), total_losses, label="comparison/losses")
    _exact_int(agent_metrics.get("comparison/ties"), total_ties, label="comparison/ties")
    normalized = _require_recomputed_metric(
        agent_metrics.get("comparison/normalized_elo"),
        recomputed_stage1_normalized,
        label="comparison/normalized_elo",
        tolerance=NORMALIZED_ELO_ABS_TOLERANCE,
    )
    stage0_elo = _require_recomputed_metric(
        agent_metrics.get("comparison/stage_0/eval_elo"),
        recomputed_stage0_elo,
        label="Stage0 comparison ELO",
        tolerance=ELO_ABS_TOLERANCE,
    )
    stage0_normalized = _require_recomputed_metric(
        agent_metrics.get("comparison/stage_0/normalized_elo"),
        recomputed_stage0_normalized,
        label="Stage0 comparison normalized ELO",
        tolerance=NORMALIZED_ELO_ABS_TOLERANCE,
    )
    stage1_elo = _require_recomputed_metric(
        agent_metrics.get("comparison/stage_1/eval_elo"),
        recomputed_stage1_elo,
        label="Stage1 comparison ELO",
        tolerance=ELO_ABS_TOLERANCE,
    )
    stage1_normalized = _require_recomputed_metric(
        agent_metrics.get("comparison/stage_1/normalized_elo"),
        recomputed_stage1_normalized,
        label="Stage1 comparison normalized ELO",
        tolerance=NORMALIZED_ELO_ABS_TOLERANCE,
    )
    if not math.isclose(stage1_elo, elo, rel_tol=0.0, abs_tol=ELO_ABS_TOLERANCE) or not math.isclose(
        stage1_normalized,
        normalized,
        rel_tol=0.0,
        abs_tol=NORMALIZED_ELO_ABS_TOLERANCE,
    ):
        _fail("final headline ELO does not equal the complete Stage1 fit")
    final_stage_metrics = {
        "comparison/headline_stage_index": 1,
        "comparison/expected_final_stage_declared_rows": len(rows),
        "comparison/expected_final_stage_consistent": 1,
        "comparison/expected_final_stage_index": 1,
        "comparison/final_stage_present": 1,
        "comparison/final_stage_complete": 1,
        "comparison/final_stage_fit": 1,
        "comparison/final_stage_degraded": 0,
        "comparison/observed_final_stage_row_count": STAGE1_TASKS,
        "comparison/expected_final_stage_row_count_consistent": 1,
        "comparison/expected_final_stage_row_count": STAGE1_TASKS,
    }
    for key, expected in final_stage_metrics.items():
        _exact_int(agent_metrics.get(key), expected, label=key)
    for key in (
        "comparison/eval_elo",
        "comparison/normalized_elo",
        "comparison/num_stages",
        "comparison/stage_0/num_tasks",
        "comparison/stage_1/num_tasks",
        "comparison/judged",
        "comparison/wins",
        "comparison/losses",
        "comparison/ties",
        "comparison/stage_0/eval_elo",
        "comparison/stage_0/normalized_elo",
        "comparison/stage_1/eval_elo",
        "comparison/stage_1/normalized_elo",
        *final_stage_metrics,
    ):
        if key_metrics.get(key) != agent_metrics.get(key) or type(key_metrics.get(key)) is not type(
            agent_metrics.get(key)
        ):
            _fail(f"agent_metrics and key_metrics differ for {key}")
    return {
        "status": "PASS",
        "output": str(output_path),
        "journal": str(journal_path),
        "metrics": str(aggregate_path),
        "rows": len(rows),
        "stage0_tasks": stage0_tasks,
        "stage1_tasks": stage1_tasks,
        "stage0_trials": stage0_tasks * TRIALS_PER_TASK,
        "stage1_trials": stage1_tasks * TRIALS_PER_TASK,
        "av_routed_rows": av_routed_rows[0] + av_routed_rows[1],
        "stage0_av_routed_rows": av_routed_rows[0],
        "stage1_av_routed_rows": av_routed_rows[1],
        "stage0_partial": stage0_is_partial,
        "invalid": 0,
        "wins": total_wins,
        "losses": total_losses,
        "ties": total_ties,
        "stage0_elo": stage0_elo,
        "stage0_normalized_elo": stage0_normalized,
        "top4": plans[1]["reference_ids"],
        "eval_elo": elo,
        "normalized_elo": normalized,
    }


def _print_location(result: Mapping[str, Any]) -> None:
    print(f"RUN_ID={result['run_id']}")
    print(f"RUN_DIR={result['run_dir']}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    locate = commands.add_parser("locate", help="derive a stable run location without writing")
    locate.add_argument("--checkpoint", type=Path, required=True)
    locate.add_argument("--campaign-root", type=Path, required=True)

    prepare = commands.add_parser("prepare", help="publish an immutable campaign and modulo shards")
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--campaign-root", "--run-root", dest="campaign_root", type=Path, required=True)
    prepare.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    prepare.add_argument("--expected-tasks", type=int, default=DEFAULT_TASKS)
    prepare.add_argument("--profile-out", type=Path)

    verify = commands.add_parser("verify", help="recompute and verify an immutable campaign")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--rehash-reference-assets", action="store_true")
    verify.add_argument("--json", action="store_true", dest="json_output")

    coverage = commands.add_parser("coverage", help="compare exact dataset and completed deliverable IDs")
    coverage.add_argument("--dataset", type=Path, required=True)
    coverage.add_argument("--deliverables", type=Path, required=True)
    coverage.add_argument("--expected-tasks", type=int, default=DEFAULT_TASKS)
    coverage.add_argument("--json", action="store_true", dest="json_output")

    residue = commands.add_parser("residue", help="publish missing rows and bounded residue shards")
    residue.add_argument("--dataset", type=Path, required=True)
    residue.add_argument("--deliverables", type=Path, required=True)
    residue.add_argument("--output", type=Path, required=True)
    residue.add_argument("--shards-dir", type=Path, required=True)
    residue.add_argument("--max-shards", type=int, required=True)
    residue.add_argument("--expected-tasks", type=int, default=DEFAULT_TASKS)
    residue.add_argument("--json", action="store_true", dest="json_output")

    result = commands.add_parser("result", help="strictly validate a completed two-stage result")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--journal", type=Path, required=True)
    result.add_argument("--dataset", type=Path)
    result.add_argument("--expected-tasks", type=int, default=DEFAULT_TASKS)
    result.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "locate":
            _print_location(locate_campaign(args.checkpoint, args.campaign_root))
            return 0
        if args.command == "prepare":
            prepared = prepare_campaign(
                checkpoint=args.checkpoint,
                dataset_path=args.dataset,
                campaign_root=args.campaign_root,
                shards=args.shards,
                expected_tasks=args.expected_tasks,
                profile_out=args.profile_out,
            )
            _print_location(prepared)
            return 0
        if args.command == "verify":
            report = verify_campaign(
                args.run_dir,
                rehash_reference_assets=args.rehash_reference_assets,
            )
        elif args.command == "coverage":
            report = coverage_report(
                dataset_path=args.dataset,
                deliverables=args.deliverables,
                expected_tasks=args.expected_tasks,
            )
        elif args.command == "residue":
            report = write_residue(
                dataset_path=args.dataset,
                deliverables=args.deliverables,
                output=args.output,
                shards_dir=args.shards_dir,
                max_shards=args.max_shards,
                expected_tasks=args.expected_tasks,
            )
        elif args.command == "result":
            report = validate_result(
                output=args.output,
                journal=args.journal,
                dataset=args.dataset,
                expected_tasks=args.expected_tasks,
            )
        else:  # pragma: no cover - argparse makes this unreachable.
            raise AssertionError(args.command)
    except (CampaignError, OSError) as exc:
        print(f"CHECKPOINT_E2E_FAIL: {exc}", file=sys.stderr)
        return 64
    if getattr(args, "json_output", False):
        print(json.dumps(report, sort_keys=True))
    else:
        list_fields = {key for key in ("missing", "extra") if isinstance(report.get(key), list)}
        print(" ".join([f"{key}={value}" for key, value in report.items() if key not in list_fields]))
        for key in ("missing", "extra"):
            value = report.get(key)
            if isinstance(value, list) and value:
                print(f"{key}={','.join(str(item) for item in value)}")
    if args.command == "coverage" and report["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
