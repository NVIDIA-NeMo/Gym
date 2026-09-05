#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build immutable GDPVal transport views without changing scientific content."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Callable
from xml.etree import ElementTree

import yaml


SCHEMA = "gdpval.transport-views.v3"
DERIVATIVE_CACHE_SCHEMA = "gdpval.transport-derivative-cache.v1"
DERIVATIVE_CACHE_FORMAT = "FORMAT.json"
LOSSLESS_AUDIO_EXTENSIONS = {".wav", ".wave", ".aif", ".aiff"}
LOSSLESS_PCM_SUBTYPES = {"PCM_S8", "PCM_U8", "PCM_16", "PCM_24"}
VIDEO_EXTENSIONS = {".3gp", ".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}
OFFICE_EXTENSIONS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
OOXML_EXTENSIONS = {".docx", ".pptx", ".xlsx"}
STEP_EXTENSIONS = {".step", ".stp"}
PSD_EXTENSIONS = {".psd"}
NATIVE_REFERENCE_EXTENSIONS = (
    {
        ".aac",
        ".aif",
        ".aiff",
        ".bash",
        ".c",
        ".cpp",
        ".csv",
        ".flac",
        ".heic",
        ".heif",
        ".html",
        ".java",
        ".jpeg",
        ".jpg",
        ".js",
        ".json",
        ".m4a",
        ".md",
        ".mp3",
        ".oga",
        ".ogg",
        ".opus",
        ".pdf",
        ".png",
        ".py",
        ".sh",
        ".sol",
        ".ts",
        ".tsx",
        ".txt",
        ".webp",
        ".wma",
        ".xml",
        ".yaml",
        ".yml",
    }
    | LOSSLESS_AUDIO_EXTENSIONS
    | VIDEO_EXTENSIONS
)
FFMPEG_TIMEOUT_SECONDS = 1800
FFPROBE_TIMEOUT_SECONDS = 120
DEFAULT_MIN_VIDEO_BYTES = 8 * 1024 * 1024
DERIVATIVE_PROFILE = "reference-pdf-v1+video-h264-720p-crf26-aac128-min8m+ref-video-bundle8-v1"
VIDEO_MAX_WIDTH = 1280
VIDEO_MAX_HEIGHT = 720
VIDEO_CRF = 26
VIDEO_AUDIO_BITRATE = "128k"
MAX_REFERENCE_VIDEO_ATTACHMENTS = 8
REFERENCE_VIDEO_BUNDLE_SCHEMA = "gdpval-reference-video-bundles-v1"
REFERENCE_VIDEO_BUNDLE_MANIFEST = "__gdpval_reference_video_bundles__.txt"
REFERENCE_VIDEO_BUNDLE_WIDTH = 1280
REFERENCE_VIDEO_BUNDLE_HEIGHT = 720
REFERENCE_VIDEO_BUNDLE_FPS = 30
REFERENCE_VIDEO_BUNDLE_AUDIO_RATE = 48000
VIDEO_DURATION_ABSOLUTE_TOLERANCE_SECONDS = 0.5
VIDEO_DURATION_RELATIVE_TOLERANCE = 0.01
STEP_PAGE_WIDTH_POINTS = 612
STEP_PAGE_HEIGHT_POINTS = 792
STEP_MARGIN_POINTS = 36
STEP_FONT_SIZE_POINTS = 8
STEP_LEADING_POINTS = 10
STEP_COLUMNS = 90
PSD_FLATTEN_BACKGROUND_RGB = (255, 255, 255)
DERIVATIVE_PROFILE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-]{0,127}$")
DYNAMIC_CANDIDATE_CACHE = re.compile(r"^repeat_[0-9]+_verify_response(?:_[0-9a-f]{12,16})?\.json$")
OOXML_RELATIONSHIP_START = re.compile(rb"<Relationship(?=[\x09\x0a\x0d\x20>])")
OOXML_EXTERNAL_TARGET_MODE = re.compile(rb"\bTargetMode[\x09\x0a\x0d\x20]*=[\x09\x0a\x0d\x20]*([\"'])External\1")
OOXML_RELATIONSHIP_ID = re.compile(rb"\bId[\x09\x0a\x0d\x20]*=[\x09\x0a\x0d\x20]*(?:\"[^\"]+\"|'[^']+')")
OOXML_RELATIONSHIP_TARGET = re.compile(rb"\bTarget[\x09\x0a\x0d\x20]*=[\x09\x0a\x0d\x20]*(?:\"[^\"]+\"|'[^']+')")
OOXML_HYPERLINK_RELATIONSHIP_TYPE = re.compile(
    rb"\bType[\x09\x0a\x0d\x20]*=[\x09\x0a\x0d\x20]*"
    rb"(?:\"[^\"]*/hyperlink\"|'[^']*/hyperlink')"
)
OOXML_RELATIONSHIPS_NAMESPACE = "http://schemas.openxmlformats.org/package/2006/relationships"
OOXML_REPAIR_SCHEMA = "gdpval-ooxml-external-relationship-repair-v1"

VIDEO_FFMPEG_ARGS_TEMPLATE = (
    "ffmpeg",
    "-nostdin",
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    "{source}",
    "-map",
    "0:v:0",
    "-map",
    "0:a:0?",
    "-map_metadata",
    "-1",
    "-map_chapters",
    "-1",
    "-vf",
    ("scale=w='min(1280,iw)':h='min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2"),
    "-c:v",
    "libx264",
    "-preset",
    "medium",
    "-crf",
    str(VIDEO_CRF),
    "-pix_fmt",
    "yuv420p",
    "-threads:v",
    "1",
    "-c:a",
    "aac",
    "-b:a",
    VIDEO_AUDIO_BITRATE,
    "-ac",
    "2",
    "-movflags",
    "+faststart",
    "-fflags",
    "+bitexact",
    "-flags:v",
    "+bitexact",
    "-flags:a",
    "+bitexact",
    "-y",
    "{output}",
)
LOSSLESS_AUDIO_FFMPEG_ARGS_TEMPLATE = (
    "ffmpeg",
    "-nostdin",
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    "{source}",
    "-map",
    "0:a:0",
    "-c:a",
    "flac",
    "-compression_level",
    "12",
    "-y",
    "{output}",
)
VIDEO_FFPROBE_ARGS_TEMPLATE = (
    "ffprobe",
    "-v",
    "error",
    "-show_entries",
    "format=duration:stream=index,codec_type,codec_name,width,height,pix_fmt,duration,channels,sample_rate",
    "-of",
    "json",
    "{input}",
)
REFERENCE_VIDEO_BUNDLE_FILTER = (
    "scale=w=1280:h=720:force_original_aspect_ratio=decrease:force_divisible_by=2,"
    "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,fps=30,setsar=1"
)


@dataclass(frozen=True)
class _CachedDerivative:
    path: Path | None
    receipt: dict[str, Any] | None


class _CacheCorruption(RuntimeError):
    """A cache object exists but cannot satisfy its immutable receipt."""


class _DerivativeCache:
    """Immutable content-addressed cache for deterministic derivatives."""

    def __init__(
        self,
        root: Path,
        derivative_profile: str,
        converter_identity: dict[str, Any],
        *,
        persistent: bool = False,
        cache_label: str | None = None,
    ):
        self.root = root.expanduser().absolute()
        self.derivative_profile = derivative_profile
        self.converter_identity = copy.deepcopy(converter_identity)
        self.persistent = persistent
        self.cache_label = cache_label if cache_label is not None else str(self.root)
        # Fail before writing cache state if the identity cannot be encoded
        # canonically.  The exact identity is part of every object key.
        json.dumps(self.converter_identity, sort_keys=True, separators=(",", ":"))
        self._hits = 0
        self._misses = 0
        self._negative_hits = 0
        self._corruptions = 0
        self._bytes_reused = 0
        self._bytes_generated = 0
        self._initialize_root()

    @staticmethod
    def _ensure_directory(path: Path, mode: int = 0o700) -> None:
        path.mkdir(parents=True, exist_ok=True, mode=mode)
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"derivative cache path is not a directory: {path}")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _open_lock(path: Path):
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise ValueError(f"derivative cache lock is not a regular file: {path}")
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "a+b")

    def _initialize_root(self) -> None:
        self._ensure_directory(self.root)
        format_lock = self.root / ".format.lock"
        with self._open_lock(format_lock) as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            marker = self.root / DERIVATIVE_CACHE_FORMAT
            expected = {"schema": DERIVATIVE_CACHE_SCHEMA}
            if marker.exists() or marker.is_symlink():
                if marker.is_symlink() or not marker.is_file():
                    raise ValueError(f"derivative cache format marker is invalid: {marker}")
                if marker.stat().st_mode & 0o222:
                    raise ValueError(f"derivative cache format marker is writable: {marker}")
                try:
                    actual = json.loads(marker.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"derivative cache format marker is unreadable: {marker}") from exc
                if actual != expected:
                    raise ValueError(f"derivative cache format mismatch: {marker}")
            else:
                _atomic_json(marker, expected)
            for name in ("objects", "locks", "quarantine", "tmp"):
                self._ensure_directory(self.root / name)

    def _key(
        self,
        operation: str,
        source_sha256: str,
        source_suffix: str,
        output_suffix: str,
        parameters: dict[str, Any],
    ) -> str:
        payload = json.dumps(
            {
                "schema": DERIVATIVE_CACHE_SCHEMA,
                "operation": operation,
                "source_sha256": source_sha256,
                "source_suffix": source_suffix,
                "output_suffix": output_suffix,
                "parameters": parameters,
                "converter_identity": self.converter_identity,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _link_or_copy(source: Path, destination: Path) -> str:
        if not source.is_file() or source.is_symlink():
            raise _CacheCorruption(f"cached derivative is not a regular file: {source}")
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"derivative destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
            method = "hardlink"
        except OSError:
            shutil.copyfile(source, destination)
            method = "copy"
        os.chmod(destination, 0o400)
        return method

    def _object_path(self, key: str) -> Path:
        return self.root / "objects" / key[:2] / key

    def _lock_path(self, key: str) -> Path:
        return self.root / "locks" / key[:2] / f"{key}.lock"

    @staticmethod
    def _cache_metadata(
        *,
        key: str,
        operation: str,
        source_sha256: str,
        source_suffix: str,
        output_suffix: str,
        parameters: dict[str, Any],
        converter_identity: dict[str, Any],
        conversion_receipt: dict[str, Any] | None,
        artifact: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "schema": DERIVATIVE_CACHE_SCHEMA,
            "key": key,
            "operation": operation,
            "source_sha256": source_sha256,
            "source_suffix": source_suffix,
            "output_suffix": output_suffix,
            "parameters": parameters,
            "converter_identity": converter_identity,
            "result": "artifact" if artifact is not None else "no-output",
            "conversion_receipt": conversion_receipt,
            "artifact": artifact,
        }

    def _load_object(self, object_path: Path, expected: dict[str, Any]) -> _CachedDerivative:
        if object_path.is_symlink() or not object_path.is_dir():
            raise _CacheCorruption(f"cache object is not a regular directory: {object_path}")
        if object_path.stat().st_mode & 0o222:
            raise _CacheCorruption(f"cache object directory is writable: {object_path}")
        receipt_path = object_path / "receipt.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise _CacheCorruption(f"cache object receipt is missing: {receipt_path}")
        if receipt_path.stat().st_mode & 0o222:
            raise _CacheCorruption(f"cache object receipt is writable: {receipt_path}")
        try:
            document = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise _CacheCorruption(f"cache object receipt is unreadable: {receipt_path}") from exc
        required = {
            "schema",
            "key",
            "operation",
            "source_sha256",
            "source_suffix",
            "output_suffix",
            "parameters",
            "converter_identity",
            "result",
            "conversion_receipt",
            "artifact",
        }
        if not isinstance(document, dict) or set(document) != required:
            raise _CacheCorruption(f"cache object receipt shape drift: {receipt_path}")
        for name, value in expected.items():
            if document.get(name) != value:
                raise _CacheCorruption(f"cache object receipt {name} drift: {receipt_path}")
        if document["result"] == "no-output":
            if document["conversion_receipt"] is not None or document["artifact"] is not None:
                raise _CacheCorruption(f"negative cache object has artifact metadata: {receipt_path}")
            if set(path.name for path in object_path.iterdir()) != {"receipt.json"}:
                raise _CacheCorruption(f"negative cache object inventory drift: {object_path}")
            return _CachedDerivative(None, None)
        if document["result"] != "artifact":
            raise _CacheCorruption(f"cache object result is invalid: {receipt_path}")
        conversion_receipt = document["conversion_receipt"]
        artifact_receipt = document["artifact"]
        if not isinstance(conversion_receipt, dict) or not isinstance(artifact_receipt, dict):
            raise _CacheCorruption(f"cache artifact receipt is invalid: {receipt_path}")
        if set(artifact_receipt) != {"name", "bytes", "sha256"}:
            raise _CacheCorruption(f"cache artifact identity shape drift: {receipt_path}")
        artifact_name = f"artifact{expected['output_suffix']}"
        if artifact_receipt.get("name") != artifact_name:
            raise _CacheCorruption(f"cache artifact name drift: {receipt_path}")
        artifact_path = object_path / artifact_name
        expected_inventory = {"receipt.json", artifact_name}
        if set(path.name for path in object_path.iterdir()) != expected_inventory:
            raise _CacheCorruption(f"cache artifact inventory drift: {object_path}")
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise _CacheCorruption(f"cache artifact is not a regular file: {artifact_path}")
        artifact_stat = artifact_path.stat()
        if artifact_stat.st_mode & 0o222:
            raise _CacheCorruption(f"cache artifact is writable: {artifact_path}")
        if artifact_stat.st_size != artifact_receipt.get("bytes") or _sha256(artifact_path) != artifact_receipt.get(
            "sha256"
        ):
            raise _CacheCorruption(f"cache artifact content drift: {artifact_path}")
        if (
            conversion_receipt.get("source_sha256") != expected["source_sha256"]
            or conversion_receipt.get("output_bytes") != artifact_receipt["bytes"]
            or conversion_receipt.get("output_sha256") != artifact_receipt["sha256"]
        ):
            raise _CacheCorruption(f"cache conversion receipt drift: {receipt_path}")
        return _CachedDerivative(artifact_path, conversion_receipt)

    def _quarantine(self, object_path: Path, key: str) -> None:
        quarantine = self.root / "quarantine" / f"{key}.{os.getpid()}.{time.time_ns()}"
        if not object_path.is_symlink() and object_path.is_dir():
            # Darwin requires the directory's owner-write bit for this rename;
            # the per-key lock prevents a reader from observing the transition.
            os.chmod(object_path, 0o700)
        os.replace(object_path, quarantine)
        if not quarantine.is_symlink() and quarantine.is_dir():
            os.chmod(quarantine, 0o500)
        self._fsync_directory(quarantine.parent)

    def _publish_object(
        self,
        object_path: Path,
        metadata: dict[str, Any],
        destination: Path | None,
    ) -> None:
        self._ensure_directory(object_path.parent)
        temporary = Path(tempfile.mkdtemp(prefix=f".{metadata['key']}.", dir=self.root / "tmp"))
        try:
            artifact = metadata["artifact"]
            if artifact is not None:
                if destination is None:
                    raise RuntimeError("cache artifact publication has no derivative destination")
                cached_artifact = temporary / artifact["name"]
                self._link_or_copy(destination, cached_artifact)
                if (
                    cached_artifact.stat().st_size != artifact["bytes"]
                    or _sha256(cached_artifact) != artifact["sha256"]
                ):
                    raise RuntimeError(f"cache artifact changed during publication: {cached_artifact}")
                _fsync_file(cached_artifact)
            _atomic_json(temporary / "receipt.json", metadata)
            if object_path.exists() or object_path.is_symlink():
                raise RuntimeError(f"cache object appeared while its key lock was held: {object_path}")
            os.replace(temporary, object_path)
            os.chmod(object_path, 0o500)
            self._fsync_directory(object_path.parent)
        finally:
            if temporary.exists():
                os.chmod(temporary, 0o700)
                shutil.rmtree(temporary)

    def summary(self) -> dict[str, Any]:
        identity_payload = json.dumps(self.converter_identity, sort_keys=True, separators=(",", ":")).encode()
        return {
            "schema": DERIVATIVE_CACHE_SCHEMA,
            "persistent": self.persistent,
            "cache_label": self.cache_label,
            "converter_identity": copy.deepcopy(self.converter_identity),
            "converter_identity_sha256": hashlib.sha256(identity_payload).hexdigest(),
            "hits": self._hits,
            "misses": self._misses,
            "negative_hits": self._negative_hits,
            "corruptions_quarantined": self._corruptions,
            "bytes_reused": self._bytes_reused,
            "bytes_generated": self._bytes_generated,
        }

    def materialize(
        self,
        operation: str,
        source: Path,
        destination: Path,
        parameters: dict[str, Any],
        converter: Callable[[Path, Path], dict[str, Any] | None],
    ) -> dict[str, Any] | None:
        source_sha256 = _sha256(source)
        source_suffix = source.suffix.lower()
        output_suffix = destination.suffix.lower()
        cache_parameters = {"derivative_profile": self.derivative_profile, **parameters}
        key = self._key(operation, source_sha256, source_suffix, output_suffix, cache_parameters)
        object_path = self._object_path(key)
        lock_path = self._lock_path(key)
        self._ensure_directory(lock_path.parent)
        expected = {
            "schema": DERIVATIVE_CACHE_SCHEMA,
            "key": key,
            "operation": operation,
            "source_sha256": source_sha256,
            "source_suffix": source_suffix,
            "output_suffix": output_suffix,
            "parameters": cache_parameters,
            "converter_identity": self.converter_identity,
        }
        repaired_corruption = False
        with self._open_lock(lock_path) as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if object_path.exists() or object_path.is_symlink():
                try:
                    cached = self._load_object(object_path, expected)
                except _CacheCorruption:
                    self._quarantine(object_path, key)
                    self._corruptions += 1
                    repaired_corruption = True
                else:
                    self._hits += 1
                    if cached.path is None:
                        self._negative_hits += 1
                        return None
                    assert cached.receipt is not None
                    method = self._link_or_copy(cached.path, destination)
                    self._bytes_reused += cached.path.stat().st_size
                    receipt = copy.deepcopy(cached.receipt)
                    receipt["cache"] = {
                        "schema": DERIVATIVE_CACHE_SCHEMA,
                        "key": key,
                        "reused": True,
                        "materialization": method,
                    }
                    return receipt

            self._misses += 1
            receipt = converter(source, destination)
            if receipt is None:
                metadata = self._cache_metadata(
                    key=key,
                    operation=operation,
                    source_sha256=source_sha256,
                    source_suffix=source_suffix,
                    output_suffix=output_suffix,
                    parameters=cache_parameters,
                    converter_identity=self.converter_identity,
                    conversion_receipt=None,
                    artifact=None,
                )
                self._publish_object(object_path, metadata, None)
                return None
            if not isinstance(receipt, dict):
                raise RuntimeError(f"converter returned a non-mapping receipt for {destination}")
            if not destination.is_file() or destination.is_symlink():
                raise RuntimeError(f"converter did not publish a regular derivative: {destination}")
            output_sha256 = _sha256(destination)
            if (
                _sha256(source) != source_sha256
                or receipt.get("source_sha256") != source_sha256
                or receipt.get("source_bytes") != source.stat().st_size
                or receipt.get("output_bytes") != destination.stat().st_size
                or receipt.get("output_sha256") != output_sha256
            ):
                raise RuntimeError(f"converter receipt or source identity mismatch: {destination}")
            os.chmod(destination, 0o400)
            artifact = {
                "name": f"artifact{output_suffix}",
                "bytes": destination.stat().st_size,
                "sha256": output_sha256,
            }
            metadata = self._cache_metadata(
                key=key,
                operation=operation,
                source_sha256=source_sha256,
                source_suffix=source_suffix,
                output_suffix=output_suffix,
                parameters=cache_parameters,
                converter_identity=self.converter_identity,
                conversion_receipt=copy.deepcopy(receipt),
                artifact=artifact,
            )
            self._publish_object(object_path, metadata, destination)
            self._bytes_generated += destination.stat().st_size
            receipt = copy.deepcopy(receipt)
            receipt["cache"] = {
                "schema": DERIVATIVE_CACHE_SCHEMA,
                "key": key,
                "reused": False,
                "materialization": "generated",
            }
            if repaired_corruption:
                receipt["cache"]["repaired_corruption"] = True
            return receipt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o400
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def _atomic_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        yaml.safe_dump(value, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o400)
    os.replace(temporary, path)


def check_tools() -> dict[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is unavailable inside the GDPVal container")
    if ffprobe is None:
        raise RuntimeError("ffprobe is unavailable inside the GDPVal container")
    import fitz
    import soundfile
    from PIL import __version__ as pillow_version

    if not getattr(soundfile, "__version__", None):
        raise RuntimeError("soundfile has no version metadata")
    ffmpeg_version = subprocess.run(
        [ffmpeg, "-version"], capture_output=True, text=True, check=True, timeout=30
    ).stdout.splitlines()[0]
    ffprobe_version = subprocess.run(
        [ffprobe, "-version"], capture_output=True, text=True, check=True, timeout=30
    ).stdout.splitlines()[0]
    libreoffice = shutil.which("libreoffice")
    libreoffice_version = None
    if libreoffice is not None:
        libreoffice_version = subprocess.run(
            [libreoffice, "--version"], capture_output=True, text=True, check=True, timeout=30
        ).stdout.strip()
    return {
        "ffmpeg": ffmpeg,
        "ffmpeg_version": ffmpeg_version,
        "ffprobe": ffprobe,
        "ffprobe_version": ffprobe_version,
        "fitz": str(getattr(fitz, "__version__", "unknown")),
        "libreoffice": libreoffice or "unavailable",
        "libreoffice_version": libreoffice_version or "unavailable",
        "pillow": str(pillow_version),
        "soundfile": str(soundfile.__version__),
    }


def _derivative_parameters(min_audio_bytes: int, min_video_bytes: int) -> dict[str, Any]:
    return {
        "lossless_audio": {
            "eligible_extensions": sorted(LOSSLESS_AUDIO_EXTENSIONS),
            "eligible_pcm_subtypes": sorted(LOSSLESS_PCM_SUBTYPES),
            "minimum_source_bytes": min_audio_bytes,
            "codec": "flac",
            "compression_level": 12,
            "ffmpeg_args": list(LOSSLESS_AUDIO_FFMPEG_ARGS_TEMPLATE),
            "identity": "decoded PCM int32 SHA-256 plus sample rate, channels, and frames",
        },
        "video_proxy": {
            "eligible_extensions": sorted(VIDEO_EXTENSIONS),
            "minimum_source_bytes": min_video_bytes,
            "container": "mp4",
            "video_codec": "libx264",
            "maximum_dimensions": [VIDEO_MAX_WIDTH, VIDEO_MAX_HEIGHT],
            "crf": VIDEO_CRF,
            "pixel_format": "yuv420p",
            "video_threads": 1,
            "audio_codec": "aac",
            "audio_bitrate": VIDEO_AUDIO_BITRATE,
            "audio_channels": 2,
            "duration_absolute_tolerance_seconds": VIDEO_DURATION_ABSOLUTE_TOLERANCE_SECONDS,
            "duration_relative_tolerance": VIDEO_DURATION_RELATIVE_TOLERANCE,
            "ffmpeg_args": list(VIDEO_FFMPEG_ARGS_TEMPLATE),
            "ffprobe_args": list(VIDEO_FFPROBE_ARGS_TEMPLATE),
        },
        "reference_video_bundles": {
            "scope": "video members of recursive reference ZIP assets only",
            "trigger": f"more than {MAX_REFERENCE_VIDEO_ATTACHMENTS} video members",
            "maximum_physical_attachments": MAX_REFERENCE_VIDEO_ATTACHMENTS,
            "logical_clip_order": "normalized ZIP member path, ascending",
            "partition": "stable contiguous balanced groups",
            "preservation": "full-duration video and source audio; silence padded when absent",
            "container": "mp4",
            "video_codec": "libx264",
            "dimensions": [REFERENCE_VIDEO_BUNDLE_WIDTH, REFERENCE_VIDEO_BUNDLE_HEIGHT],
            "frame_rate": REFERENCE_VIDEO_BUNDLE_FPS,
            "crf": VIDEO_CRF,
            "pixel_format": "yuv420p",
            "audio_codec": "aac",
            "audio_bitrate": VIDEO_AUDIO_BITRATE,
            "audio_channels": 2,
            "audio_sample_rate": REFERENCE_VIDEO_BUNDLE_AUDIO_RATE,
            "normalization_filter": REFERENCE_VIDEO_BUNDLE_FILTER,
            "manifest_schema": REFERENCE_VIDEO_BUNDLE_SCHEMA,
            "manifest_member": REFERENCE_VIDEO_BUNDLE_MANIFEST,
        },
        "reference_pdf": {
            "scope": "files nested below reference_files/<asset>/ only",
            "native_extensions_not_requiring_pdf": sorted(NATIVE_REFERENCE_EXTENSIONS),
            "sidecar_naming": "source.ext.pdf",
            "office_extensions": sorted(OFFICE_EXTENSIONS),
            "office_converter": "passed resources_servers/gdpval/preconvert.py:convert_to_pdf",
            "ooxml_staged_repair": {
                "schema": OOXML_REPAIR_SCHEMA,
                "scope": "staged copy only; source bytes remain unchanged",
                "accepted_defect": ("External Relationship start tag missing its closing />"),
                "all_other_malformed_xml": "fail closed",
            },
            "step_extensions": sorted(STEP_EXTENSIONS),
            "step_render": {
                "format": "deterministic PDF 1.4 with built-in Courier",
                "page_points": [STEP_PAGE_WIDTH_POINTS, STEP_PAGE_HEIGHT_POINTS],
                "margin_points": STEP_MARGIN_POINTS,
                "font_size_points": STEP_FONT_SIZE_POINTS,
                "leading_points": STEP_LEADING_POINTS,
                "columns": STEP_COLUMNS,
                "tabs": 4,
                "non_ascii": "ASCII backslash escapes",
            },
            "psd_extensions": sorted(PSD_EXTENSIONS),
            "psd_render": {
                "flatten_background_rgb": list(PSD_FLATTEN_BACKGROUND_RGB),
                "intermediate": "PNG",
                "png_compress_level": 9,
                "pdf_image_encoding": "lossless FlateDecode",
                "page_pixel_scale": "1 pixel = 1 point",
            },
        },
    }


def _validate_derivative_profile(value: str) -> str:
    if DERIVATIVE_PROFILE_PATTERN.fullmatch(value) is None:
        raise ValueError("derivative_profile must match [A-Za-z0-9][A-Za-z0-9._+-]{0,127}")
    if value != DERIVATIVE_PROFILE:
        raise ValueError(f"unsupported derivative_profile: {value!r}; expected {DERIVATIVE_PROFILE!r}")
    return value


def _format_args(template: tuple[str, ...], **values: str) -> list[str]:
    return [part.format(**values) for part in template]


def _probe_video(path: Path) -> dict[str, Any]:
    arguments = _format_args(VIDEO_FFPROBE_ARGS_TEMPLATE, input=str(path))
    result = subprocess.run(
        arguments,
        check=True,
        capture_output=True,
        text=True,
        timeout=FFPROBE_TIMEOUT_SECONDS,
    )
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON for {path}: {exc}") from exc
    streams = document.get("streams")
    if not isinstance(streams, list):
        raise RuntimeError(f"ffprobe returned no stream list for {path}")
    normalized_streams = []
    for raw_stream in streams:
        if not isinstance(raw_stream, dict):
            raise RuntimeError(f"ffprobe returned an invalid stream for {path}")
        stream = {
            key: raw_stream[key]
            for key in (
                "index",
                "codec_type",
                "codec_name",
                "width",
                "height",
                "pix_fmt",
                "duration",
                "channels",
                "sample_rate",
            )
            if key in raw_stream
        }
        normalized_streams.append(stream)
    normalized_streams.sort(key=lambda stream: int(stream.get("index", -1)))
    raw_duration = document.get("format", {}).get("duration")
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"ffprobe returned no finite duration for {path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"ffprobe returned no finite duration for {path}")
    return {"duration_seconds": duration, "streams": normalized_streams}


def _validate_video_identities(source: dict[str, Any], output: dict[str, Any], label: str) -> None:
    source_video = [stream for stream in source["streams"] if stream.get("codec_type") == "video"]
    source_audio = [stream for stream in source["streams"] if stream.get("codec_type") == "audio"]
    output_video = [stream for stream in output["streams"] if stream.get("codec_type") == "video"]
    output_audio = [stream for stream in output["streams"] if stream.get("codec_type") == "audio"]
    if not source_video:
        raise RuntimeError(f"video proxy source has no video stream: {label}")
    if len(output_video) != 1 or output_video[0].get("codec_name") != "h264":
        raise RuntimeError(f"video proxy output is not exactly one H.264 stream: {label}")
    width = int(output_video[0].get("width", 0))
    height = int(output_video[0].get("height", 0))
    if width <= 0 or height <= 0 or width > VIDEO_MAX_WIDTH or height > VIDEO_MAX_HEIGHT:
        raise RuntimeError(f"video proxy dimensions are invalid: {label}: {width}x{height}")
    if output_video[0].get("pix_fmt") != "yuv420p":
        raise RuntimeError(f"video proxy pixel format is not yuv420p: {label}")
    if source_audio:
        if len(output_audio) != 1 or output_audio[0].get("codec_name") != "aac":
            raise RuntimeError(f"video proxy did not preserve one AAC audio stream: {label}")
    elif output_audio:
        raise RuntimeError(f"video proxy introduced an audio stream: {label}")
    tolerance = max(
        VIDEO_DURATION_ABSOLUTE_TOLERANCE_SECONDS,
        source["duration_seconds"] * VIDEO_DURATION_RELATIVE_TOLERANCE,
    )
    if abs(source["duration_seconds"] - output["duration_seconds"]) > tolerance:
        raise RuntimeError(
            f"video proxy duration drift: {label}: "
            f"source={source['duration_seconds']} output={output['duration_seconds']} tolerance={tolerance}"
        )


def _convert_video(source: Path, destination: Path) -> dict[str, Any]:
    before = _probe_video(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.mp4")
    arguments = _format_args(VIDEO_FFMPEG_ARGS_TEMPLATE, source=str(source), output=str(temporary))
    try:
        subprocess.run(arguments, check=True, timeout=FFMPEG_TIMEOUT_SECONDS)
        after = _probe_video(temporary)
        _validate_video_identities(before, after, str(source))
        os.chmod(temporary, 0o400)
        os.replace(temporary, destination)
        return {
            "source_sha256": _sha256(source),
            "output_sha256": _sha256(destination),
            "source_bytes": source.stat().st_size,
            "output_bytes": destination.stat().st_size,
            "ffmpeg_args": list(VIDEO_FFMPEG_ARGS_TEMPLATE),
            "ffprobe_args": list(VIDEO_FFPROBE_ARGS_TEMPLATE),
            "source_video_identity": before,
            "output_video_identity": after,
        }
    finally:
        temporary.unlink(missing_ok=True)


def _reference_video_duration(identity: dict[str, Any]) -> float:
    video_streams = [stream for stream in identity["streams"] if stream.get("codec_type") == "video"]
    if not video_streams:
        raise RuntimeError("reference video has no video stream")
    raw_duration = video_streams[0].get("duration", identity.get("duration_seconds"))
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("reference video has no finite video duration") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("reference video has no finite video duration")
    return duration


def _validate_reference_video_bundle_identity(identity: dict[str, Any], expected_duration: float, label: str) -> None:
    videos = [stream for stream in identity["streams"] if stream.get("codec_type") == "video"]
    audio = [stream for stream in identity["streams"] if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or videos[0].get("codec_name") != "h264":
        raise RuntimeError(f"reference video bundle is not exactly one H.264 stream: {label}")
    if (
        int(videos[0].get("width", 0)) != REFERENCE_VIDEO_BUNDLE_WIDTH
        or int(videos[0].get("height", 0)) != REFERENCE_VIDEO_BUNDLE_HEIGHT
        or videos[0].get("pix_fmt") != "yuv420p"
    ):
        raise RuntimeError(f"reference video bundle has an invalid video format: {label}")
    if (
        len(audio) != 1
        or audio[0].get("codec_name") != "aac"
        or int(audio[0].get("channels", 0)) != 2
        or int(audio[0].get("sample_rate", 0)) != REFERENCE_VIDEO_BUNDLE_AUDIO_RATE
    ):
        raise RuntimeError(f"reference video bundle has an invalid audio format: {label}")
    tolerance = max(
        VIDEO_DURATION_ABSOLUTE_TOLERANCE_SECONDS,
        expected_duration * VIDEO_DURATION_RELATIVE_TOLERANCE,
    )
    if abs(float(identity["duration_seconds"]) - expected_duration) > tolerance:
        raise RuntimeError(
            f"reference video bundle duration drift: {label}: "
            f"expected={expected_duration} output={identity['duration_seconds']} tolerance={tolerance}"
        )


def _normalize_reference_video_for_bundle(source: Path, destination: Path) -> dict[str, Any]:
    source_identity = _probe_video(source)
    source_duration = _reference_video_duration(source_identity)
    has_audio = any(stream.get("codec_type") == "audio" for stream in source_identity["streams"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.mp4")
    arguments = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
    ]
    if not has_audio:
        arguments.extend(
            [
                "-f",
                "lavfi",
                "-t",
                f"{source_duration:.9f}",
                "-i",
                f"anullsrc=channel_layout=stereo:sample_rate={REFERENCE_VIDEO_BUNDLE_AUDIO_RATE}",
            ]
        )
    arguments.extend(
        [
            "-map",
            "0:v:0",
            "-map",
            "0:a:0" if has_audio else "1:a:0",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-vf",
            REFERENCE_VIDEO_BUNDLE_FILTER,
            "-af",
            "apad",
            "-t",
            f"{source_duration:.9f}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(VIDEO_CRF),
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(REFERENCE_VIDEO_BUNDLE_FPS),
            "-threads:v",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            VIDEO_AUDIO_BITRATE,
            "-ar",
            str(REFERENCE_VIDEO_BUNDLE_AUDIO_RATE),
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            "-flags:a",
            "+bitexact",
            "-y",
            str(temporary),
        ]
    )
    try:
        subprocess.run(arguments, check=True, timeout=FFMPEG_TIMEOUT_SECONDS)
        output_identity = _probe_video(temporary)
        _validate_reference_video_bundle_identity(output_identity, source_duration, str(source))
        os.chmod(temporary, 0o400)
        os.replace(temporary, destination)
        return {
            "source_sha256": _sha256(source),
            "source_bytes": source.stat().st_size,
            "source_video_identity": source_identity,
            "source_duration_seconds": source_duration,
            "normalized_sha256": _sha256(destination),
            "normalized_bytes": destination.stat().st_size,
            "normalized_video_identity": output_identity,
            "normalization_audio": "source" if has_audio else "deterministic_silence",
            "normalization_filter": REFERENCE_VIDEO_BUNDLE_FILTER,
            "normalization_ffmpeg_args": [
                "ffmpeg",
                "-i",
                "{source}",
                *([] if has_audio else ["-f", "lavfi", "-i", "anullsrc"]),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0" if has_audio else "1:a:0",
                "-vf",
                REFERENCE_VIDEO_BUNDLE_FILTER,
                "-af",
                "apad",
                "-t",
                "{source_video_duration}",
                "-c:v",
                "libx264",
                "-crf",
                str(VIDEO_CRF),
                "-c:a",
                "aac",
                "-y",
                "{output}",
            ],
        }
    finally:
        temporary.unlink(missing_ok=True)


def _concatenate_reference_video_bundle(
    normalized: list[tuple[Path, dict[str, Any]]], destination: Path
) -> dict[str, Any]:
    if not normalized:
        raise ValueError("reference video bundle cannot be empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    concat_file = destination.with_suffix(".ffconcat")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.mp4")
    concat_file.write_text(
        "ffconcat version 1.0\n" + "".join(f"file '{path.name}'\n" for path, _receipt in normalized)
    )
    arguments = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "1",
        "-i",
        str(concat_file),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        "-fflags",
        "+bitexact",
        "-y",
        str(temporary),
    ]
    expected_duration = sum(
        float(receipt["normalized_video_identity"]["duration_seconds"]) for _path, receipt in normalized
    )
    try:
        subprocess.run(arguments, check=True, timeout=FFMPEG_TIMEOUT_SECONDS)
        output_identity = _probe_video(temporary)
        _validate_reference_video_bundle_identity(output_identity, expected_duration, str(destination))
        os.chmod(temporary, 0o400)
        os.replace(temporary, destination)
        return {
            "output_sha256": _sha256(destination),
            "output_bytes": destination.stat().st_size,
            "output_video_identity": output_identity,
            "expected_duration_seconds": expected_duration,
            "concat_ffmpeg_args": [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "1",
                "-i",
                "{ffconcat}",
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c",
                "copy",
                "-y",
                "{output}",
            ],
        }
    finally:
        concat_file.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def _balanced_contiguous_groups(values: list[Any], limit: int) -> list[list[Any]]:
    if limit <= 0:
        raise ValueError("reference video bundle limit must be positive")
    group_count = min(limit, len(values))
    if group_count == 0:
        return []
    minimum, extra = divmod(len(values), group_count)
    groups: list[list[Any]] = []
    cursor = 0
    for index in range(group_count):
        size = minimum + (1 if index < extra else 0)
        groups.append(values[cursor : cursor + size])
        cursor += size
    return groups


def _reference_video_bundle_manifest(receipt: dict[str, Any]) -> bytes:
    lines = [
        "GDPVal deterministic reference-video bundle manifest",
        f"Schema: {receipt['schema']}",
        (
            f"All {receipt['source_video_count']} logical source clips are retained in full, "
            f"in stable path order, across {receipt['output_video_count']} physical MP4 attachments."
        ),
        "Time ranges are half-open [start_seconds, end_seconds).",
        "",
    ]
    for bundle in receipt["bundles"]:
        lines.append(f"Bundle: {bundle['output_member']}")
        for clip in bundle["clips"]:
            lines.append(
                f"  [{clip['start_seconds']:.6f}, {clip['end_seconds']:.6f}) "
                f"{clip['source_member']} sha256={clip['source_sha256']}"
            )
        lines.append("")
    return ("\n".join(lines).rstrip() + "\n").encode()


def _build_reference_video_bundles(
    source_zip: zipfile.ZipFile,
    video_infos: list[zipfile.ZipInfo],
    workspace: Path,
    derivative_profile: str,
) -> tuple[dict[str, Any], bytes, list[tuple[Path, dict[str, Any]]]]:
    ordered_infos = sorted(
        video_infos,
        key=lambda info: _safe_zip_name(info.filename).as_posix(),
    )
    normalized: list[tuple[Path, dict[str, Any]]] = []
    for index, info in enumerate(ordered_infos):
        member = _safe_zip_name(info.filename)
        source_path = workspace / f"bundle_source_{index}{member.suffix.lower()}"
        normalized_path = workspace / f"bundle_clip_{index}.normalized.mp4"
        with source_zip.open(info) as source_handle, source_path.open("wb") as output_handle:
            shutil.copyfileobj(source_handle, output_handle, 1024 * 1024)
        if source_path.stat().st_size != info.file_size:
            raise ValueError(f"ZIP member size drift: {info.filename!r}")
        clip_receipt = _normalize_reference_video_for_bundle(source_path, normalized_path)
        clip_receipt["source_member"] = member.as_posix()
        normalized.append((normalized_path, clip_receipt))

    groups = _balanced_contiguous_groups(normalized, MAX_REFERENCE_VIDEO_ATTACHMENTS)
    bundles: list[tuple[Path, dict[str, Any]]] = []
    total = len(groups)
    for index, group in enumerate(groups):
        output_member = f"__gdpval_reference_video_bundle_{index + 1:02d}_of_{total:02d}__.mp4"
        output_path = workspace / output_member
        bundle = _concatenate_reference_video_bundle(group, output_path)
        cursor = 0.0
        clips = []
        for _normalized_path, clip in group:
            duration = float(clip["normalized_video_identity"]["duration_seconds"])
            start = cursor
            cursor += duration
            clips.append({**clip, "start_seconds": start, "end_seconds": cursor})
        bundle.update(
            {
                "output_member": output_member,
                "clips": clips,
                "clip_count": len(clips),
                "derivative_profile": derivative_profile,
            }
        )
        bundles.append((output_path, bundle))

    receipt = {
        "schema": REFERENCE_VIDEO_BUNDLE_SCHEMA,
        "derivative_profile": derivative_profile,
        "source_video_count": len(ordered_infos),
        "source_order": [_safe_zip_name(info.filename).as_posix() for info in ordered_infos],
        "maximum_physical_attachments": MAX_REFERENCE_VIDEO_ATTACHMENTS,
        "output_video_count": len(bundles),
        "manifest_member": REFERENCE_VIDEO_BUNDLE_MANIFEST,
        "bundles": [bundle for _path, bundle in bundles],
    }
    manifest = _reference_video_bundle_manifest(receipt)
    receipt.update(
        {
            "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "manifest_bytes": len(manifest),
        }
    )
    return receipt, manifest, bundles


def _pdf_identity(path: Path) -> dict[str, Any]:
    import fitz

    if not path.is_file() or path.is_symlink() or path.stat().st_size < 5:
        raise RuntimeError(f"PDF derivative is not a regular nonempty file: {path}")
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise RuntimeError(f"PDF derivative has no PDF header: {path}")
    try:
        with fitz.open(path) as document:
            if not document.is_pdf or document.page_count <= 0:
                raise RuntimeError(f"PDF derivative has no pages: {path}")
            pages = [
                [round(float(document[index].rect.width), 6), round(float(document[index].rect.height), 6)]
                for index in range(document.page_count)
            ]
    except Exception as exc:
        raise RuntimeError(f"PDF derivative cannot be opened: {path}: {exc}") from exc
    return {"page_count": len(pages), "page_sizes_points": pages}


def _load_preconvert_module(path: Path) -> tuple[ModuleType, str, Path]:
    module_path = path.expanduser().resolve(strict=True)
    if not module_path.is_file() or module_path.is_symlink():
        raise RuntimeError(f"preconvert module is not a regular file: {module_path}")
    module_sha256 = _sha256(module_path)
    module_name = f"_gdpval_transport_preconvert_{module_sha256[:16]}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import preconvert module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    if not callable(getattr(module, "convert_to_pdf", None)):
        raise RuntimeError(f"preconvert module has no callable convert_to_pdf: {module_path}")
    return module, module_sha256, module_path


def _has_unquoted_tag_close(value: bytes) -> bool:
    quote: int | None = None
    for byte in value:
        if quote is not None:
            if byte == quote:
                quote = None
        elif byte in (ord('"'), ord("'")):
            quote = byte
        elif byte == ord(">"):
            return True
    return False


def _repair_external_relationship_xml(value: bytes) -> tuple[bytes, list[int]] | None:
    """Close only missing ``/>`` on External OOXML Relationship elements."""

    starts = [match.start() for match in OOXML_RELATIONSHIP_START.finditer(value)]
    if not starts:
        return None
    root_close = value.find(b"</Relationships", starts[-1])
    if root_close < 0:
        return None
    boundaries = starts[1:] + [root_close]
    insertions: list[int] = []
    for start, boundary in zip(starts, boundaries, strict=True):
        segment = value[start:boundary]
        if _has_unquoted_tag_close(segment):
            continue
        stripped = segment.rstrip(b"\x09\x0a\x0d\x20")
        if (
            b"<" in stripped[len(b"<Relationship") :]
            or stripped.endswith((b"/", b">"))
            or OOXML_EXTERNAL_TARGET_MODE.search(stripped) is None
            or OOXML_RELATIONSHIP_ID.search(stripped) is None
            or OOXML_RELATIONSHIP_TARGET.search(stripped) is None
            or OOXML_HYPERLINK_RELATIONSHIP_TYPE.search(stripped) is None
        ):
            return None
        insertions.append(start + len(stripped))
    if not insertions:
        return None

    repaired = bytearray(value)
    for offset in reversed(insertions):
        repaired[offset:offset] = b"/>"
    repaired_bytes = bytes(repaired)
    try:
        root = ElementTree.fromstring(repaired_bytes)
    except ElementTree.ParseError:
        return None
    relationship_tag = f"{{{OOXML_RELATIONSHIPS_NAMESPACE}}}Relationship"
    if root.tag != f"{{{OOXML_RELATIONSHIPS_NAMESPACE}}}Relationships":
        return None
    if any(child.tag != relationship_tag for child in root):
        return None
    return repaired_bytes, insertions


def _stage_ooxml_relationship_repair(staged_source: Path) -> dict[str, Any] | None:
    """Repair one audited OOXML structural defect in a staged package.

    The live source is never modified. Any malformed XML that is not exactly
    an External ``Relationship`` start tag missing its closing ``/>`` is
    rejected rather than passed through to an alternate content renderer.
    """

    if staged_source.suffix.lower() not in OOXML_EXTENSIONS:
        return None
    # Keep non-ZIP legacy inputs on the converter's pre-existing path. This
    # repair is deliberately limited to structurally valid OOXML containers.
    if not zipfile.is_zipfile(staged_source):
        return None
    before_package_sha256 = _sha256(staged_source)
    repaired_members: dict[str, bytes] = {}
    member_audits: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(staged_source, "r") as source_zip:
            infos = source_zip.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise RuntimeError(f"duplicate OOXML member in {staged_source.name}")
            for info in infos:
                if info.is_dir() or not info.filename.lower().endswith((".xml", ".rels")):
                    continue
                payload = source_zip.read(info)
                try:
                    ElementTree.fromstring(payload)
                    continue
                except ElementTree.ParseError as exc:
                    parse_error = exc
                if not info.filename.lower().endswith(".rels"):
                    raise RuntimeError(
                        f"unsupported malformed OOXML XML member {info.filename!r} "
                        f"in {staged_source.name}: {parse_error}"
                    ) from parse_error
                repaired = _repair_external_relationship_xml(payload)
                if repaired is None:
                    raise RuntimeError(
                        f"unsupported malformed OOXML relationships member {info.filename!r} "
                        f"in {staged_source.name}: {parse_error}"
                    ) from parse_error
                repaired_payload, insertion_offsets = repaired
                repaired_members[info.filename] = repaired_payload
                member_audits.append(
                    {
                        "member": info.filename,
                        "repair": "close_missing_external_relationship_empty_element",
                        "inserted_bytes_hex": "2f3e",
                        "insertion_count": len(insertion_offsets),
                        "insertion_offsets": insertion_offsets,
                        "before_bytes": len(payload),
                        "after_bytes": len(repaired_payload),
                        "before_sha256": hashlib.sha256(payload).hexdigest(),
                        "after_sha256": hashlib.sha256(repaired_payload).hexdigest(),
                    }
                )
            if not repaired_members:
                return None

            temporary = staged_source.with_name(f".{staged_source.name}.relationship-repair.tmp")
            try:
                with zipfile.ZipFile(temporary, "w") as output_zip:
                    output_zip.comment = source_zip.comment
                    for info in infos:
                        payload = b"" if info.is_dir() else source_zip.read(info)
                        payload = repaired_members.get(info.filename, payload)
                        output_zip.writestr(copy.copy(info), payload)
                with zipfile.ZipFile(temporary, "r") as repaired_zip:
                    if repaired_zip.testzip() is not None:
                        raise RuntimeError(f"repaired OOXML CRC validation failed: {staged_source.name}")
                    repaired_infos = repaired_zip.infolist()
                    if [info.filename for info in repaired_infos] != names:
                        raise RuntimeError(f"repaired OOXML inventory changed: {staged_source.name}")
                    for info in repaired_infos:
                        if info.is_dir():
                            continue
                        repaired_payload = repaired_zip.read(info)
                        if info.filename not in repaired_members:
                            if repaired_payload != source_zip.read(info.filename):
                                raise RuntimeError(f"unrelated OOXML member changed during repair: {info.filename!r}")
                            continue
                        ElementTree.fromstring(repaired_payload)
                os.replace(temporary, staged_source)
            finally:
                temporary.unlink(missing_ok=True)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"OOXML source is not a valid ZIP package: {staged_source}") from exc

    after_package_sha256 = _sha256(staged_source)
    if after_package_sha256 == before_package_sha256:
        raise RuntimeError(f"OOXML repair did not change staged package: {staged_source}")
    return {
        "schema": OOXML_REPAIR_SCHEMA,
        "repair_scope": "staged_copy_only",
        "before_package_sha256": before_package_sha256,
        "after_package_sha256": after_package_sha256,
        "members": member_audits,
        "member_count": len(member_audits),
        "insertion_count": sum(member["insertion_count"] for member in member_audits),
    }


def _convert_office_pdf(
    source: Path,
    destination: Path,
    preconvert_module: ModuleType,
    preconvert_module_sha256: str,
) -> dict[str, Any]:
    converter = preconvert_module.convert_to_pdf
    workspace = Path(tempfile.mkdtemp(prefix="gdpval_office_transport_"))
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.pdf")
    try:
        staged_source = workspace / source.name
        shutil.copyfile(source, staged_source)
        source_sha256 = _sha256(source)
        if _sha256(staged_source) != source_sha256:
            raise RuntimeError(f"Office staging hash mismatch: {source}")
        ooxml_repair = _stage_ooxml_relationship_repair(staged_source)
        if ooxml_repair is not None and ooxml_repair["before_package_sha256"] != source_sha256:
            raise RuntimeError(f"Office staged OOXML repair provenance mismatch: {source}")
        if _sha256(source) != source_sha256:
            raise RuntimeError(f"Office source changed during staged OOXML repair: {source}")
        requested_output = workspace / f"{staged_source.name}.pdf"
        libreoffice_output = staged_source.with_suffix(".pdf")
        parameters = inspect.signature(converter).parameters
        if "output_pdf" in parameters:
            result = converter(staged_source, output_pdf=requested_output)
        else:
            result = converter(staged_source)
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError(f"convert_to_pdf returned an invalid result for {source}")
        _converted_path, ok, message = result

        # LibreOffice batch conversion always chooses ``<input stem>.pdf``.
        # Some converter adapters accept an injective ``output_pdf`` request
        # but still leave that natural LibreOffice output behind.  Perform the
        # handoff here, inside a one-source workspace, so same-stem Office
        # inputs can never claim one another's PDF.  If an adapter produced
        # both spellings, accept them only when their bytes are identical.
        expected_outputs = (requested_output, libreoffice_output)
        unexpected_outputs = sorted(
            path.name for path in workspace.iterdir() if path.suffix.lower() == ".pdf" and path not in expected_outputs
        )
        if unexpected_outputs:
            raise RuntimeError(f"Office converter produced unexpected PDF outputs for {source}: {unexpected_outputs}")
        available_outputs = [path for path in expected_outputs if path.exists() or path.is_symlink()]
        if not available_outputs:
            raise RuntimeError(
                f"Office converter reported success without producing "
                f"{requested_output.name} or {libreoffice_output.name} for {source}"
            )
        output_identities = {path: _pdf_identity(path) for path in available_outputs}
        output_hashes = {path: _sha256(path) for path in available_outputs}
        if len(set(output_hashes.values())) != 1:
            raise RuntimeError(
                f"Office converter produced ambiguous PDF outputs for {source}: "
                f"{[path.name for path in available_outputs]}"
            )
        staged_output = available_outputs[0]
        identity = output_identities[staged_output]
        rc0_stem_handoff = (
            not ok
            and staged_output == libreoffice_output
            and f"libreoffice rc=0 did not produce {requested_output.name}" in str(message)
        )
        if not ok and not rc0_stem_handoff:
            raise RuntimeError(f"Office conversion failed for {source}: {message}")
        if _sha256(source) != source_sha256:
            raise RuntimeError(f"Office source changed during conversion: {source}")
        shutil.copyfile(staged_output, temporary)
        os.chmod(temporary, 0o400)
        os.replace(temporary, destination)
        return {
            "source_sha256": source_sha256,
            "output_sha256": _sha256(destination),
            "source_bytes": source.stat().st_size,
            "output_bytes": destination.stat().st_size,
            "preconvert_module_sha256": preconvert_module_sha256,
            "converter_output_handoff": (
                "requested_injective" if staged_output == requested_output else "libreoffice_stem"
            ),
            "converter_status_override": ("libreoffice_rc0_stem_handoff" if rc0_stem_handoff else None),
            "ooxml_repair": ooxml_repair,
            "pdf_identity": identity,
        }
    finally:
        temporary.unlink(missing_ok=True)
        shutil.rmtree(workspace)


def _step_display_lines(source: Path) -> tuple[list[bytes], str, int]:
    raw = source.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"STEP source is not UTF-8 text: {source}: {exc}") from exc
    display = text.encode("ascii", "backslashreplace").decode("ascii").expandtabs(4)
    logical_lines = display.split("\n")
    lines: list[bytes] = []
    for line in logical_lines:
        if line.endswith("\r"):
            line = line[:-1]
        chunks = [line[index : index + STEP_COLUMNS] for index in range(0, len(line), STEP_COLUMNS)]
        lines.extend((chunks or [""]))
    return (
        [line.encode("ascii") for line in lines],
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
        len(logical_lines),
    )


def _pdf_literal(value: bytes) -> bytes:
    rendered = bytearray()
    for byte in value:
        if byte in (ord("("), ord(")"), ord("\\")):
            rendered.extend(b"\\" + bytes((byte,)))
        elif 32 <= byte <= 126:
            rendered.append(byte)
        else:
            rendered.extend(f"\\{byte:03o}".encode("ascii"))
    return bytes(rendered)


def _deterministic_text_pdf(lines: list[bytes]) -> bytes:
    lines_per_page = max(
        1,
        int((STEP_PAGE_HEIGHT_POINTS - (2 * STEP_MARGIN_POINTS)) // STEP_LEADING_POINTS),
    )
    pages = [lines[index : index + lines_per_page] for index in range(0, len(lines), lines_per_page)] or [[]]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>",
    }
    kids = []
    for index, page_lines in enumerate(pages):
        page_id = 4 + index * 2
        content_id = page_id + 1
        kids.append(f"{page_id} 0 R")
        content = bytearray(
            (
                f"BT\n/F1 {STEP_FONT_SIZE_POINTS} Tf\n{STEP_LEADING_POINTS} TL\n"
                f"{STEP_MARGIN_POINTS} {STEP_PAGE_HEIGHT_POINTS - STEP_MARGIN_POINTS} Td\n"
            ).encode("ascii")
        )
        for line in page_lines:
            content.extend(b"(" + _pdf_literal(line) + b") Tj\nT*\n")
        content.extend(b"ET\n")
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {STEP_PAGE_WIDTH_POINTS} "
            f"{STEP_PAGE_HEIGHT_POINTS}] /Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("ascii")
        objects[content_id] = f"<< /Length {len(content)} >>\nstream\n".encode("ascii") + bytes(content) + b"endstream"
    objects[2] = f"<< /Type /Pages /Count {len(pages)} /Kids [{' '.join(kids)}] >>".encode("ascii")

    document = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id in range(1, max(objects) + 1):
        offsets.append(len(document))
        document.extend(f"{object_id} 0 obj\n".encode("ascii"))
        document.extend(objects[object_id])
        document.extend(b"\nendobj\n")
    xref = len(document)
    document.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    return bytes(document)


def _convert_step_pdf(source: Path, destination: Path) -> dict[str, Any]:
    lines, text_sha256, logical_line_count = _step_display_lines(source)
    payload = _deterministic_text_pdf(lines)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.pdf")
    try:
        temporary.write_bytes(payload)
        identity = _pdf_identity(temporary)
        os.chmod(temporary, 0o400)
        os.replace(temporary, destination)
        return {
            "source_sha256": _sha256(source),
            "output_sha256": _sha256(destination),
            "source_bytes": source.stat().st_size,
            "output_bytes": destination.stat().st_size,
            "source_text_sha256": text_sha256,
            "logical_line_count": logical_line_count,
            "rendered_line_count": len(lines),
            "pdf_identity": identity,
        }
    finally:
        temporary.unlink(missing_ok=True)


def _psd_embedded_identity(path: Path) -> dict[str, Any]:
    import fitz
    from PIL import Image

    with fitz.open(path) as document:
        if document.page_count != 1:
            raise RuntimeError(f"PSD PDF derivative must have exactly one page: {path}")
        images = document[0].get_images(full=True)
        if len(images) != 1:
            raise RuntimeError(f"PSD PDF derivative must contain exactly one image: {path}")
        extracted = document.extract_image(images[0][0])
    with Image.open(io.BytesIO(extracted["image"])) as image:
        flattened = image.convert("RGB")
        flattened.load()
        return {
            "width": flattened.width,
            "height": flattened.height,
            "rgb_sha256": hashlib.sha256(flattened.tobytes()).hexdigest(),
        }


def _convert_psd_pdf(source: Path, destination: Path) -> dict[str, Any]:
    import fitz
    from PIL import Image

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.pdf")
    try:
        with Image.open(source) as layered:
            if layered.format != "PSD":
                raise RuntimeError(f"PSD source was not decoded as PSD: {source}")
            layered.load()
            rgba = layered.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (*PSD_FLATTEN_BACKGROUND_RGB, 255))
            background.alpha_composite(rgba)
            flattened = background.convert("RGB")
        pixels_sha256 = hashlib.sha256(flattened.tobytes()).hexdigest()
        png_stream = io.BytesIO()
        flattened.save(png_stream, format="PNG", compress_level=9, optimize=False)
        png_bytes = png_stream.getvalue()
        with Image.open(io.BytesIO(png_bytes)) as roundtrip:
            roundtrip_rgb = roundtrip.convert("RGB")
            roundtrip_rgb.load()
            if roundtrip_rgb.size != flattened.size or roundtrip_rgb.tobytes() != flattened.tobytes():
                raise RuntimeError(f"PSD flattened PNG identity check failed: {source}")

        width, height = flattened.size
        if width <= 0 or height <= 0 or max(width, height) > 14_400:
            raise RuntimeError(f"PSD dimensions exceed deterministic PDF page bounds: {source}: {width}x{height}")
        document = fitz.open()
        page = document.new_page(width=width, height=height)
        page.insert_image(page.rect, stream=png_bytes)
        document.set_metadata({})
        payload = document.tobytes(
            garbage=4,
            clean=True,
            deflate=True,
            linear=False,
            no_new_id=True,
        )
        document.close()
        temporary.write_bytes(payload)
        pdf_identity = _pdf_identity(temporary)
        embedded = _psd_embedded_identity(temporary)
        if embedded != {"width": width, "height": height, "rgb_sha256": pixels_sha256}:
            raise RuntimeError(f"PSD PDF embedded image identity check failed: {source}")
        os.chmod(temporary, 0o400)
        os.replace(temporary, destination)
        return {
            "source_sha256": _sha256(source),
            "output_sha256": _sha256(destination),
            "source_bytes": source.stat().st_size,
            "output_bytes": destination.stat().st_size,
            "flattened_png_sha256": hashlib.sha256(png_bytes).hexdigest(),
            "flattened_image_identity": embedded,
            "pdf_identity": pdf_identity,
        }
    finally:
        temporary.unlink(missing_ok=True)


def _audio_identity(path: Path) -> dict[str, Any]:
    import soundfile as sf

    digest = hashlib.sha256()
    with sf.SoundFile(path) as audio:
        metadata = {
            "samplerate": int(audio.samplerate),
            "channels": int(audio.channels),
            "frames": int(audio.frames),
            "subtype": str(audio.subtype),
        }
        while True:
            samples = audio.read(65536, dtype="int32", always_2d=True)
            if not len(samples):
                break
            digest.update(samples.tobytes(order="C"))
    metadata["decoded_pcm_int32_sha256"] = digest.hexdigest()
    return metadata


def _convert_lossless(source: Path, destination: Path) -> dict[str, Any] | None:
    before = _audio_identity(source)
    if before["subtype"] not in LOSSLESS_PCM_SUBTYPES:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.flac")
    try:
        arguments = _format_args(
            LOSSLESS_AUDIO_FFMPEG_ARGS_TEMPLATE,
            source=str(source),
            output=str(temporary),
        )
        subprocess.run(
            arguments,
            check=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
        after = _audio_identity(temporary)
        comparable = ("samplerate", "channels", "frames", "decoded_pcm_int32_sha256")
        if any(before[key] != after[key] for key in comparable):
            raise RuntimeError(f"lossless audio identity check failed for {source}")
        if temporary.stat().st_size >= source.stat().st_size:
            return None
        os.chmod(temporary, 0o400)
        os.replace(temporary, destination)
        return {
            "source_sha256": _sha256(source),
            "output_sha256": _sha256(destination),
            "source_bytes": source.stat().st_size,
            "output_bytes": destination.stat().st_size,
            "audio_identity": before,
            "ffmpeg_args": list(LOSSLESS_AUDIO_FFMPEG_ARGS_TEMPLATE),
        }
    finally:
        temporary.unlink(missing_ok=True)


def _safe_zip_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or (len(path.parts[0]) >= 2 and path.parts[0][1] == ":")
    ):
        raise ValueError(f"unsafe ZIP member: {name!r}")
    return path


def _zip_reference_pdf_plan(
    infos: list[zipfile.ZipInfo],
) -> tuple[dict[str, str], set[str]]:
    """Return missing sidecar operations and existing PDFs trusted by provenance."""

    members = {
        _safe_zip_name(info.filename).as_posix(): _safe_zip_name(info.filename) for info in infos if not info.is_dir()
    }
    sources_by_directory_stem: dict[tuple[PurePosixPath, str], list[PurePosixPath]] = {}
    office_by_directory_stem: dict[tuple[PurePosixPath, str], list[PurePosixPath]] = {}
    for member in members.values():
        extension = member.suffix.lower()
        if extension in OFFICE_EXTENSIONS:
            office_by_directory_stem.setdefault((member.parent, member.stem), []).append(member)
        if extension not in NATIVE_REFERENCE_EXTENSIONS and extension != ".zip":
            sources_by_directory_stem.setdefault((member.parent, member.stem), []).append(member)
        if extension == ".zip":
            raise RuntimeError(f"nested ZIP member is unsupported in recursive references: {member.as_posix()}")
        if extension not in (NATIVE_REFERENCE_EXTENSIONS | OFFICE_EXTENSIONS | STEP_EXTENSIONS | PSD_EXTENSIONS):
            raise RuntimeError(f"unsupported recursive reference ZIP member: {member.as_posix()}")

    missing: dict[str, str] = {}
    trusted_pdfs: set[str] = set()
    for member in members.values():
        extension = member.suffix.lower()
        if extension not in OFFICE_EXTENSIONS | STEP_EXTENSIONS | PSD_EXTENSIONS:
            continue
        injective = member.with_name(f"{member.name}.pdf").as_posix()
        plain = member.with_suffix(".pdf").as_posix()
        if injective in members:
            trusted_pdfs.add(injective)
            continue
        if plain in members:
            if extension in OFFICE_EXTENSIONS:
                same_stem = office_by_directory_stem[(member.parent, member.stem)]
            else:
                same_stem = sources_by_directory_stem[(member.parent, member.stem)]
            if len(same_stem) == 1 and same_stem[0] == member:
                trusted_pdfs.add(plain)
                continue
        if extension in OFFICE_EXTENSIONS:
            missing[member.as_posix()] = "office_pdf"
        elif extension in STEP_EXTENSIONS:
            missing[member.as_posix()] = "step_text_pdf"
        else:
            missing[member.as_posix()] = "psd_flattened_pdf"
    return missing, trusted_pdfs


def _validate_zip_pdf_members(
    source_zip: zipfile.ZipFile,
    member_names: set[str],
    workspace: Path,
) -> None:
    for index, member_name in enumerate(sorted(member_names)):
        extracted = workspace / f"trusted_pdf_{index}.pdf"
        with source_zip.open(member_name) as source_handle, extracted.open("wb") as output_handle:
            shutil.copyfileobj(source_handle, output_handle, 1024 * 1024)
        _pdf_identity(extracted)
        extracted.unlink()


def _convert_zip_lossless(
    source: Path,
    destination: Path,
    min_audio_bytes: int,
    *,
    min_video_bytes: int | None = None,
    derivative_cache: _DerivativeCache | None = None,
    reference_pdf_derivatives: bool = False,
    preconvert_module: ModuleType | None = None,
    preconvert_module_sha256: str | None = None,
    derivative_profile: str = DERIVATIVE_PROFILE,
) -> dict[str, Any] | None:
    """Rewrite a ZIP when an eligible audio or video member needs a safe derivative."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="gdpval_zip_transport_"))
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.zip")
    conversions: list[dict[str, Any]] = []
    video_threshold = sys.maxsize if min_video_bytes is None else min_video_bytes
    derivative_profile = _validate_derivative_profile(derivative_profile)
    try:
        with zipfile.ZipFile(source, "r") as source_zip:
            infos = source_zip.infolist()
            seen: set[str] = set()
            for info in infos:
                member = _safe_zip_name(info.filename)
                normalized_name = member.as_posix()
                if normalized_name in seen or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f"duplicate or symlink ZIP member: {info.filename!r}")
                if reference_pdf_derivatives and info.filename.rstrip("/") != normalized_name:
                    raise ValueError(f"non-canonical recursive reference ZIP member: {info.filename!r}")
                seen.add(normalized_name)
            pdf_plan: dict[str, str] = {}
            trusted_pdf_members: set[str] = set()
            if reference_pdf_derivatives:
                pdf_plan, trusted_pdf_members = _zip_reference_pdf_plan(infos)
                _validate_zip_pdf_members(source_zip, trusted_pdf_members, workspace)
            reference_video_infos = [
                info
                for info in infos
                if not info.is_dir()
                and PurePosixPath(info.filename.replace("\\", "/")).suffix.lower() in VIDEO_EXTENSIONS
            ]
            requires_reference_video_bundles = bool(
                reference_pdf_derivatives and len(reference_video_infos) > MAX_REFERENCE_VIDEO_ATTACHMENTS
            )
            has_av_conversion = requires_reference_video_bundles or any(
                not info.is_dir()
                and (
                    (
                        PurePosixPath(info.filename.replace("\\", "/")).suffix.lower() in LOSSLESS_AUDIO_EXTENSIONS
                        and info.file_size >= min_audio_bytes
                    )
                    or (
                        PurePosixPath(info.filename.replace("\\", "/")).suffix.lower() in VIDEO_EXTENSIONS
                        and info.file_size >= video_threshold
                    )
                )
                for info in infos
            )
            if not has_av_conversion and not pdf_plan:
                return None

            reference_video_bundles: dict[str, Any] | None = None
            reference_video_manifest: bytes | None = None
            bundle_outputs: list[tuple[Path, dict[str, Any]]] = []
            if requires_reference_video_bundles:
                (
                    reference_video_bundles,
                    reference_video_manifest,
                    bundle_outputs,
                ) = _build_reference_video_bundles(
                    source_zip,
                    reference_video_infos,
                    workspace,
                    derivative_profile,
                )
                generated_names = {
                    reference_video_bundles["manifest_member"],
                    *(bundle["output_member"] for _path, bundle in bundle_outputs),
                }
                collisions = seen & generated_names
                if collisions:
                    raise ValueError(
                        f"reference video bundle output collides with source member: {sorted(collisions)!r}"
                    )

            with zipfile.ZipFile(temporary, "w") as output_zip:
                for index, info in enumerate(infos):
                    member = _safe_zip_name(info.filename)
                    if info.is_dir():
                        output_zip.writestr(info, b"")
                        continue
                    if requires_reference_video_bundles and member.suffix.lower() in VIDEO_EXTENSIONS:
                        continue
                    extracted = workspace / f"member_{index}{member.suffix}"
                    with source_zip.open(info, "r") as source_handle, extracted.open("wb") as output_handle:
                        shutil.copyfileobj(source_handle, output_handle, 1024 * 1024)
                    if extracted.stat().st_size != info.file_size:
                        raise ValueError(f"ZIP member size drift: {info.filename!r}")

                    normalized_member = member.as_posix()
                    pdf_operation = pdf_plan.get(normalized_member)
                    if pdf_operation is not None:
                        with extracted.open("rb") as input_handle, output_zip.open(info, "w") as output_handle:
                            shutil.copyfileobj(input_handle, output_handle, 1024 * 1024)
                        output_name = f"{normalized_member}.pdf"
                        converted = workspace / f"member_{index}.{pdf_operation}.pdf"
                        conversion_source = workspace / "reference_sources" / Path(*member.parts)
                        conversion_source.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(extracted, conversion_source)
                        reference_parameters = _derivative_parameters(min_audio_bytes, video_threshold)[
                            "reference_pdf"
                        ]
                        if pdf_operation == "office_pdf":
                            if preconvert_module is None or preconvert_module_sha256 is None:
                                raise RuntimeError(
                                    f"missing preconvert module for recursive reference ZIP member: {normalized_member}"
                                )
                            parameters = {
                                **reference_parameters,
                                "preconvert_module_sha256": preconvert_module_sha256,
                            }
                            converter = lambda input_path, output_path: _convert_office_pdf(
                                input_path,
                                output_path,
                                preconvert_module,
                                preconvert_module_sha256,
                            )
                        elif pdf_operation == "step_text_pdf":
                            parameters = reference_parameters["step_render"]
                            converter = _convert_step_pdf
                        else:
                            parameters = reference_parameters["psd_render"]
                            converter = _convert_psd_pdf
                        if derivative_cache is None:
                            conversion = converter(conversion_source, converted)
                        else:
                            conversion = derivative_cache.materialize(
                                pdf_operation,
                                conversion_source,
                                converted,
                                parameters,
                                converter,
                            )
                        if conversion is None:
                            raise RuntimeError(
                                f"reference PDF converter returned no output for ZIP member: {normalized_member}"
                            )
                        output_info = zipfile.ZipInfo(output_name, date_time=info.date_time)
                        output_info.compress_type = zipfile.ZIP_STORED
                        output_info.comment = info.comment
                        output_info.external_attr = info.external_attr
                        output_info.create_system = info.create_system
                        with converted.open("rb") as input_handle, output_zip.open(output_info, "w") as output_handle:
                            shutil.copyfileobj(input_handle, output_handle, 1024 * 1024)
                        conversion.update(
                            {
                                "source_member": info.filename,
                                "output_member": output_name,
                                "kind": pdf_operation,
                                "derivative_profile": derivative_profile,
                            }
                        )
                        conversions.append(conversion)
                        extracted.unlink()
                        converted.unlink(missing_ok=True)
                        continue

                    conversion: dict[str, Any] | None = None
                    converted = workspace / f"member_{index}.derived.flac"
                    if member.suffix.lower() in LOSSLESS_AUDIO_EXTENSIONS and info.file_size >= min_audio_bytes:
                        if derivative_cache is None:
                            conversion = _convert_lossless(extracted, converted)
                        else:
                            conversion = derivative_cache.materialize(
                                "lossless_audio",
                                extracted,
                                converted,
                                _derivative_parameters(min_audio_bytes, video_threshold)["lossless_audio"],
                                _convert_lossless,
                            )
                    if conversion is not None:
                        output_name = member.with_name(f"{member.stem}.__gdpval_lossless__.flac").as_posix()
                        conversion_kind = "lossless_flac"
                    elif member.suffix.lower() in VIDEO_EXTENSIONS and info.file_size >= video_threshold:
                        converted = workspace / f"member_{index}.derived.mp4"
                        if derivative_cache is None:
                            conversion = _convert_video(extracted, converted)
                        else:
                            conversion = derivative_cache.materialize(
                                "h264_video",
                                extracted,
                                converted,
                                _derivative_parameters(min_audio_bytes, video_threshold)["video_proxy"],
                                _convert_video,
                            )
                        if conversion is None:
                            raise RuntimeError(f"video proxy converter returned no output for {info.filename!r}")
                        output_name = member.with_name(f"{member.stem}.__gdpval_h264__.mp4").as_posix()
                        conversion_kind = "h264_video"
                    if conversion is not None:
                        if output_name != member.as_posix() and output_name in seen:
                            raise ValueError(f"derived ZIP member collides with source member: {output_name!r}")
                        output_info = zipfile.ZipInfo(output_name, date_time=info.date_time)
                        output_info.compress_type = zipfile.ZIP_STORED
                        output_info.comment = info.comment
                        output_info.external_attr = info.external_attr
                        output_info.create_system = info.create_system
                        with converted.open("rb") as input_handle, output_zip.open(output_info, "w") as output_handle:
                            shutil.copyfileobj(input_handle, output_handle, 1024 * 1024)
                        conversion.update(
                            {
                                "source_member": info.filename,
                                "output_member": output_name,
                                "kind": conversion_kind,
                                "derivative_profile": derivative_profile,
                            }
                        )
                        conversions.append(conversion)
                    else:
                        with extracted.open("rb") as input_handle, output_zip.open(info, "w") as output_handle:
                            shutil.copyfileobj(input_handle, output_handle, 1024 * 1024)
                    extracted.unlink()
                    converted.unlink(missing_ok=True)
                if reference_video_bundles is not None:
                    assert reference_video_manifest is not None
                    manifest_info = zipfile.ZipInfo(
                        reference_video_bundles["manifest_member"],
                        date_time=(1980, 1, 1, 0, 0, 0),
                    )
                    manifest_info.compress_type = zipfile.ZIP_STORED
                    manifest_info.create_system = 3
                    manifest_info.external_attr = (stat.S_IFREG | 0o400) << 16
                    output_zip.writestr(manifest_info, reference_video_manifest)
                    for bundle_path, bundle in bundle_outputs:
                        output_info = zipfile.ZipInfo(
                            bundle["output_member"],
                            date_time=(1980, 1, 1, 0, 0, 0),
                        )
                        output_info.compress_type = zipfile.ZIP_STORED
                        output_info.create_system = 3
                        output_info.external_attr = (stat.S_IFREG | 0o400) << 16
                        with (
                            bundle_path.open("rb") as input_handle,
                            output_zip.open(output_info, "w") as output_handle,
                        ):
                            shutil.copyfileobj(input_handle, output_handle, 1024 * 1024)
        if not conversions:
            if reference_video_bundles is None:
                return None
        os.chmod(temporary, 0o400)
        os.replace(temporary, destination)
        conversion_counts = {
            kind: sum(item["kind"] == kind for item in conversions)
            for kind in sorted({item["kind"] for item in conversions})
        }
        if reference_video_bundles is not None:
            conversion_counts["reference_video_bundle"] = reference_video_bundles["output_video_count"]
        return {
            "source_sha256": _sha256(source),
            "output_sha256": _sha256(destination),
            "source_bytes": source.stat().st_size,
            "output_bytes": destination.stat().st_size,
            "member_conversions": conversions,
            "member_conversion_counts": conversion_counts,
            **({"reference_video_bundles": reference_video_bundles} if reference_video_bundles is not None else {}),
        }
    finally:
        temporary.unlink(missing_ok=True)
        shutil.rmtree(workspace)


def _reference_models(document: Any) -> tuple[list[str], dict[str, Any]]:
    matches: list[tuple[list[str], dict[str, Any]]] = []

    def visit(value: Any, path: list[str]) -> None:
        if not isinstance(value, dict):
            return
        models = value.get("reference_models")
        if isinstance(models, dict) and models:
            matches.append((path + ["reference_models"], models))
        for key, child in value.items():
            visit(child, path + [str(key)])

    visit(document, [])
    if len(matches) != 1:
        raise ValueError(f"expected one reference_models mapping, found {len(matches)}")
    return matches[0]


def _nested_overlay(path: list[str], models: dict[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = models
    for key in reversed(path):
        value = {key: value}
    return value


def _is_nested_reference_asset(relative: Path) -> bool:
    try:
        index = relative.parts.index("reference_files")
    except ValueError:
        return False
    return len(relative.parts) - index >= 3


def _existing_provenance_pdf(source: Path) -> bool:
    sidecar = source.with_name(f"{source.name}.pdf")
    if sidecar.exists() or sidecar.is_symlink():
        if not sidecar.is_file():
            raise RuntimeError(f"provenance sidecar is not a file: {sidecar}")
        _pdf_identity(sidecar.resolve(strict=True))
        return True

    plain = source.with_suffix(".pdf")
    if not plain.exists() and not plain.is_symlink():
        return False
    if not plain.is_file():
        raise RuntimeError(f"plain provenance PDF is not a file: {plain}")
    _pdf_identity(plain.resolve(strict=True))
    if source.suffix.lower() in OFFICE_EXTENSIONS:
        same_stem_sources = [
            sibling
            for sibling in source.parent.iterdir()
            if sibling.is_file() and sibling.suffix.lower() in OFFICE_EXTENSIONS and sibling.stem == source.stem
        ]
    else:
        # The runtime accepts a plain fallback PDF only when the stem is
        # injective among unknown sources. Count all non-PDF/non-ZIP siblings
        # conservatively so we never bless an ambiguous legacy render.
        same_stem_sources = [
            sibling
            for sibling in source.parent.iterdir()
            if sibling.is_file() and sibling.suffix.lower() not in {".pdf", ".zip"} and sibling.stem == source.stem
        ]
    return len(same_stem_sources) == 1 and same_stem_sources[0] == source


def _mirror_tree(
    source_root: Path,
    output_root: Path,
    min_audio_bytes: int,
    *,
    min_video_bytes: int = DEFAULT_MIN_VIDEO_BYTES,
    derivative_cache: _DerivativeCache | None = None,
    preconvert_module: ModuleType | None = None,
    preconvert_module_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    entries: list[dict[str, Any]] = []
    directories = ["."]
    parameters = _derivative_parameters(min_audio_bytes, min_video_bytes)
    for source in sorted(source_root.rglob("*")):
        relative = source.relative_to(source_root)
        target = output_root / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            directories.append(relative.as_posix())
            continue
        if not source.is_file():
            raise ValueError(f"unsupported source entry: {source}")
        stat = source.stat()
        record: dict[str, Any] = {
            "relative_path": relative.as_posix(),
            "source": str(source.resolve()),
            "source_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "kind": "symlink",
        }
        if source.suffix.lower() in LOSSLESS_AUDIO_EXTENSIONS and stat.st_size >= min_audio_bytes:
            converted_relative = relative.with_name(f"{relative.stem}.__gdpval_lossless__.flac")
            converted_target = output_root / converted_relative
            if derivative_cache is None:
                conversion = _convert_lossless(source, converted_target)
            else:
                conversion = derivative_cache.materialize(
                    "lossless_audio",
                    source,
                    converted_target,
                    parameters["lossless_audio"],
                    _convert_lossless,
                )
            if conversion is not None:
                record.update(conversion)
                record["kind"] = "lossless_flac"
                record["output_relative_path"] = converted_relative.as_posix()
                entries.append(record)
                continue
        if source.suffix.lower() in VIDEO_EXTENSIONS and stat.st_size >= min_video_bytes:
            converted_relative = relative.with_name(f"{relative.stem}.__gdpval_h264__.mp4")
            converted_target = output_root / converted_relative
            if derivative_cache is None:
                conversion = _convert_video(source, converted_target)
            else:
                conversion = derivative_cache.materialize(
                    "h264_video",
                    source,
                    converted_target,
                    parameters["video_proxy"],
                    _convert_video,
                )
            if conversion is None:
                raise RuntimeError(f"video proxy converter returned no output for {source}")
            record.update(conversion)
            record["kind"] = "h264_video"
            record["output_relative_path"] = converted_relative.as_posix()
            entries.append(record)
            continue
        if source.suffix.lower() == ".zip":
            reference_zip = _is_nested_reference_asset(relative)
            converted_relative = relative.with_name(f"{relative.stem}.__gdpval_transport__.zip")
            converted_target = output_root / converted_relative
            if derivative_cache is None:
                conversion = _convert_zip_lossless(
                    source,
                    converted_target,
                    min_audio_bytes,
                    min_video_bytes=min_video_bytes,
                    reference_pdf_derivatives=reference_zip,
                    preconvert_module=preconvert_module,
                    preconvert_module_sha256=preconvert_module_sha256,
                )
            else:
                conversion = derivative_cache.materialize(
                    "transport_zip",
                    source,
                    converted_target,
                    {
                        "lossless_audio": parameters["lossless_audio"],
                        "video_proxy": parameters["video_proxy"],
                        "reference_video_bundles": (parameters["reference_video_bundles"] if reference_zip else None),
                        "reference_pdf": parameters["reference_pdf"] if reference_zip else None,
                        "preconvert_module_sha256": preconvert_module_sha256 if reference_zip else None,
                    },
                    lambda source_path, output_path: _convert_zip_lossless(
                        source_path,
                        output_path,
                        min_audio_bytes,
                        min_video_bytes=min_video_bytes,
                        derivative_cache=derivative_cache,
                        reference_pdf_derivatives=reference_zip,
                        preconvert_module=preconvert_module,
                        preconvert_module_sha256=preconvert_module_sha256,
                        derivative_profile=derivative_cache.derivative_profile,
                    ),
                )
            if conversion is not None:
                record.update(conversion)
                record["kind"] = "transport_zip"
                record["output_relative_path"] = converted_relative.as_posix()
                entries.append(record)
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve())
        record["output_relative_path"] = relative.as_posix()
        entries.append(record)

        extension = source.suffix.lower()
        if not _is_nested_reference_asset(relative):
            continue
        if extension in NATIVE_REFERENCE_EXTENSIONS or extension == ".zip":
            continue
        if extension not in OFFICE_EXTENSIONS | STEP_EXTENSIONS | PSD_EXTENSIONS:
            raise RuntimeError(f"unsupported recursive reference asset: {relative.as_posix()}")
        if _existing_provenance_pdf(source):
            continue
        sidecar_relative = relative.with_name(f"{relative.name}.pdf")
        sidecar_target = output_root / sidecar_relative
        if sidecar_target.exists() or sidecar_target.is_symlink():
            raise RuntimeError(f"provenance sidecar output collision: {sidecar_target}")
        derivative_record: dict[str, Any] = {
            "relative_path": relative.as_posix(),
            "source": str(source.resolve()),
            "source_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "output_relative_path": sidecar_relative.as_posix(),
        }
        if extension in OFFICE_EXTENSIONS:
            if preconvert_module is None or preconvert_module_sha256 is None:
                raise RuntimeError(f"missing --preconvert-py for nested Office reference asset: {source}")
            operation = "office_pdf"
            derivative_parameters = {
                **parameters["reference_pdf"],
                "preconvert_module_sha256": preconvert_module_sha256,
            }
            converter = lambda input_path, output_path: _convert_office_pdf(
                input_path,
                output_path,
                preconvert_module,
                preconvert_module_sha256,
            )
        elif extension in STEP_EXTENSIONS:
            operation = "step_text_pdf"
            derivative_parameters = parameters["reference_pdf"]["step_render"]
            converter = _convert_step_pdf
        else:
            operation = "psd_flattened_pdf"
            derivative_parameters = parameters["reference_pdf"]["psd_render"]
            converter = _convert_psd_pdf
        if derivative_cache is None:
            derivative = converter(source, sidecar_target)
        else:
            derivative = derivative_cache.materialize(
                operation,
                source,
                sidecar_target,
                derivative_parameters,
                converter,
            )
        if derivative is None:
            raise RuntimeError(f"reference PDF derivative converter returned no output for {source}")
        derivative_record.update(derivative)
        derivative_record["kind"] = operation
        entries.append(derivative_record)
    return entries, directories


def _validate_view_inventory(output_root: Path, group: dict[str, Any]) -> None:
    view_root = output_root / group["view"]
    expected_directories = {Path(value) for value in group.get("directories", [])}
    if Path(".") not in expected_directories:
        raise ValueError(f"transport view has no root directory receipt: {view_root}")
    expected_outputs = {Path(entry["output_relative_path"]) for entry in group["entries"]}
    if len(expected_outputs) != len(group["entries"]):
        raise ValueError(f"transport view has duplicate output receipts: {view_root}")

    actual_directories = {Path(".")}
    actual_outputs: set[Path] = set()
    for path in view_root.rglob("*"):
        relative = path.relative_to(view_root)
        if path.is_dir() and not path.is_symlink():
            actual_directories.add(relative)
        else:
            actual_outputs.add(relative)
    if actual_directories != expected_directories:
        unexpected = sorted(str(path) for path in actual_directories - expected_directories)
        missing = sorted(str(path) for path in expected_directories - actual_directories)
        raise ValueError(
            f"transport directory inventory drift in {view_root}: unexpected={unexpected} missing={missing}"
        )

    extras = actual_outputs - expected_outputs
    if group["name"] == "candidate":
        invalid = []
        for relative in sorted(extras):
            output = view_root / relative
            if (
                len(relative.parts) != 2
                or not relative.parts[0].startswith("task_")
                or DYNAMIC_CANDIDATE_CACHE.fullmatch(relative.name) is None
                or not output.is_file()
                or output.is_symlink()
            ):
                invalid.append(str(relative))
        extras = {Path(value) for value in invalid}
    if extras:
        raise ValueError(f"unexpected transport outputs in {view_root}: {sorted(str(path) for path in extras)}")
    missing_outputs = expected_outputs - actual_outputs
    if missing_outputs:
        raise ValueError(f"missing transport outputs in {view_root}: {sorted(str(path) for path in missing_outputs)}")


def _validate_derived_file(entry: dict[str, Any], output: Path) -> None:
    source = Path(entry["source"])
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"transport derivative source is not a regular file: {source}")
    source_stat = source.stat()
    if source_stat.st_size != entry["source_bytes"] or source_stat.st_mtime_ns != entry["source_mtime_ns"]:
        raise ValueError(f"transport derivative source stat drift: {source}")
    if _sha256(source) != entry["source_sha256"]:
        raise ValueError(f"transport derivative source hash drift: {source}")
    if not output.is_file() or output.is_symlink():
        raise ValueError(f"transport derivative output is not a regular file: {output}")
    if output.stat().st_mode & 0o222:
        raise ValueError(f"transport derivative output is writable: {output}")
    if output.stat().st_size != entry["output_bytes"] or _sha256(output) != entry["output_sha256"]:
        raise ValueError(f"transport derivative output drift: {output}")

    kind = entry["kind"]
    if kind == "lossless_flac":
        if _audio_identity(output) != entry["audio_identity"]:
            raise ValueError(f"lossless audio output identity drift: {output}")
    elif kind == "h264_video":
        source_identity = _probe_video(source)
        output_identity = _probe_video(output)
        if source_identity != entry["source_video_identity"] or output_identity != entry["output_video_identity"]:
            raise ValueError(f"video proxy identity receipt drift: {output}")
        _validate_video_identities(source_identity, output_identity, str(output))
    elif kind in {"office_pdf", "step_text_pdf", "psd_flattened_pdf"}:
        if _pdf_identity(output) != entry["pdf_identity"]:
            raise ValueError(f"reference PDF identity drift: {output}")
        if kind == "step_text_pdf":
            lines, text_sha256, logical_line_count = _step_display_lines(source)
            if text_sha256 != entry["source_text_sha256"] or logical_line_count != entry["logical_line_count"]:
                raise ValueError(f"STEP source text identity drift: {source}")
            if hashlib.sha256(_deterministic_text_pdf(lines)).hexdigest() != entry["output_sha256"]:
                raise ValueError(f"STEP deterministic PDF drift: {output}")
        elif kind == "psd_flattened_pdf":
            if _psd_embedded_identity(output) != entry["flattened_image_identity"]:
                raise ValueError(f"PSD embedded image identity drift: {output}")
    elif kind == "transport_zip":
        _validate_transport_zip(entry, source, output)
    else:
        raise ValueError(f"unknown transport derivative kind: {kind!r}")


def _validate_reference_video_bundles(
    receipt: dict[str, Any],
    source_zip: zipfile.ZipFile,
    output_zip: zipfile.ZipFile,
    output: Path,
    workspace: Path,
) -> None:
    if receipt.get("schema") != REFERENCE_VIDEO_BUNDLE_SCHEMA:
        raise ValueError(f"reference video bundle schema drift: {output}")
    if receipt.get("derivative_profile") != DERIVATIVE_PROFILE:
        raise ValueError(f"reference video bundle derivative profile drift: {output}")
    if receipt.get("maximum_physical_attachments") != MAX_REFERENCE_VIDEO_ATTACHMENTS:
        raise ValueError(f"reference video bundle attachment limit drift: {output}")

    source_video_names = sorted(
        _safe_zip_name(info.filename).as_posix()
        for info in source_zip.infolist()
        if not info.is_dir() and _safe_zip_name(info.filename).suffix.lower() in VIDEO_EXTENSIONS
    )
    if receipt.get("source_order") != source_video_names:
        raise ValueError(f"reference video bundle source inventory drift: {output}")
    if receipt.get("source_video_count") != len(source_video_names):
        raise ValueError(f"reference video bundle source count drift: {output}")

    bundles = receipt.get("bundles")
    if not isinstance(bundles, list) or not 1 <= len(bundles) <= MAX_REFERENCE_VIDEO_ATTACHMENTS:
        raise ValueError(f"reference video bundle output count is invalid: {output}")
    if receipt.get("output_video_count") != len(bundles):
        raise ValueError(f"reference video bundle output receipt drift: {output}")
    bundle_names = [bundle.get("output_member") for bundle in bundles]
    if any(not isinstance(name, str) for name in bundle_names) or len(set(bundle_names)) != len(bundle_names):
        raise ValueError(f"reference video bundle output names are invalid: {output}")

    output_names = output_zip.namelist()
    if len(output_names) != len(set(output_names)):
        raise ValueError(f"transport ZIP has duplicate output members: {output}")
    output_video_names = sorted(
        name
        for name in output_names
        if not name.endswith("/") and _safe_zip_name(name).suffix.lower() in VIDEO_EXTENSIONS
    )
    if output_video_names != sorted(bundle_names):
        raise ValueError(f"reference video bundle physical inventory drift: {output}")

    manifest_name = receipt.get("manifest_member")
    if manifest_name != REFERENCE_VIDEO_BUNDLE_MANIFEST or manifest_name not in output_names:
        raise ValueError(f"reference video bundle manifest is missing: {output}")
    manifest = output_zip.read(manifest_name)
    if (
        len(manifest) != receipt.get("manifest_bytes")
        or hashlib.sha256(manifest).hexdigest() != receipt.get("manifest_sha256")
        or manifest != _reference_video_bundle_manifest(receipt)
    ):
        raise ValueError(f"reference video bundle manifest drift: {output}")

    flattened_source_order: list[str] = []
    source_receipt_index = 0
    for bundle_index, bundle in enumerate(bundles):
        if bundle.get("derivative_profile") != DERIVATIVE_PROFILE:
            raise ValueError(f"reference video bundle member profile drift: {output}")
        clips = bundle.get("clips")
        if not isinstance(clips, list) or not clips or bundle.get("clip_count") != len(clips):
            raise ValueError(f"reference video bundle clip count drift: {output}")
        cursor = 0.0
        for clip in clips:
            source_name = _safe_zip_name(clip["source_member"]).as_posix()
            flattened_source_order.append(source_name)
            source_member = workspace / f"bundle_source_receipt_{source_receipt_index}"
            source_receipt_index += 1
            with source_zip.open(source_name) as handle:
                source_member.write_bytes(handle.read())
            if (
                source_member.stat().st_size != clip.get("source_bytes")
                or _sha256(source_member) != clip.get("source_sha256")
                or _probe_video(source_member) != clip.get("source_video_identity")
            ):
                raise ValueError(f"reference video bundle source receipt drift: {output}: {source_name}")
            normalized_identity = clip.get("normalized_video_identity")
            if not isinstance(normalized_identity, dict):
                raise ValueError(f"reference video bundle normalized receipt is invalid: {output}")
            source_duration = float(clip.get("source_duration_seconds", 0))
            _validate_reference_video_bundle_identity(
                normalized_identity,
                source_duration,
                f"{output}!/{source_name}",
            )
            normalized_sha256 = clip.get("normalized_sha256")
            if (
                not isinstance(normalized_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", normalized_sha256) is None
                or not isinstance(clip.get("normalized_bytes"), int)
                or clip["normalized_bytes"] <= 0
            ):
                raise ValueError(f"reference video bundle normalized hash receipt is invalid: {output}")
            start = float(clip.get("start_seconds", -1))
            end = float(clip.get("end_seconds", -1))
            normalized_duration = float(normalized_identity["duration_seconds"])
            if abs(start - cursor) > 1e-6 or end <= start or abs((end - start) - normalized_duration) > 1e-6:
                raise ValueError(f"reference video bundle boundary receipt drift: {output}")
            cursor = end
            source_member.unlink()

        output_name = bundle["output_member"]
        bundle_member = workspace / f"bundle_output_receipt_{bundle_index}.mp4"
        with output_zip.open(output_name) as handle:
            bundle_member.write_bytes(handle.read())
        if (
            bundle_member.stat().st_size != bundle.get("output_bytes")
            or _sha256(bundle_member) != bundle.get("output_sha256")
            or _probe_video(bundle_member) != bundle.get("output_video_identity")
            or abs(float(bundle.get("expected_duration_seconds", -1)) - cursor) > 1e-6
        ):
            raise ValueError(f"reference video bundle output receipt drift: {output}: {output_name}")
        _validate_reference_video_bundle_identity(
            bundle["output_video_identity"],
            cursor,
            f"{output}!/{output_name}",
        )
        bundle_member.unlink()

    if flattened_source_order != source_video_names:
        raise ValueError(f"reference video bundle clip order drift: {output}")


def _validate_transport_zip(entry: dict[str, Any], source: Path, output: Path) -> None:
    workspace = Path(tempfile.mkdtemp(prefix="gdpval_zip_validation_"))
    try:
        with zipfile.ZipFile(source) as source_zip, zipfile.ZipFile(output) as output_zip:
            output_names = set(output_zip.namelist())
            for index, conversion in enumerate(entry["member_conversions"]):
                if conversion.get("derivative_profile") != DERIVATIVE_PROFILE:
                    raise ValueError(f"ZIP member derivative profile drift: {output}")
                source_name = _safe_zip_name(conversion["source_member"]).as_posix()
                output_name = _safe_zip_name(conversion["output_member"]).as_posix()
                if output_name not in output_names:
                    raise ValueError(f"derived ZIP member is missing: {output}: {output_name}")
                source_member = workspace / f"source_{index}{PurePosixPath(source_name).suffix}"
                output_member = workspace / f"output_{index}{PurePosixPath(output_name).suffix}"
                with source_zip.open(conversion["source_member"]) as handle:
                    source_member.write_bytes(handle.read())
                with output_zip.open(conversion["output_member"]) as handle:
                    output_member.write_bytes(handle.read())
                if (
                    source_member.stat().st_size != conversion["source_bytes"]
                    or _sha256(source_member) != conversion["source_sha256"]
                    or output_member.stat().st_size != conversion["output_bytes"]
                    or _sha256(output_member) != conversion["output_sha256"]
                ):
                    raise ValueError(f"derived ZIP member receipt drift: {output}: {output_name}")
                if conversion["kind"] == "lossless_flac":
                    if _audio_identity(output_member) != conversion["audio_identity"]:
                        raise ValueError(f"lossless ZIP audio identity drift: {output}: {output_name}")
                elif conversion["kind"] == "h264_video":
                    source_identity = _probe_video(source_member)
                    output_identity = _probe_video(output_member)
                    if (
                        source_identity != conversion["source_video_identity"]
                        or output_identity != conversion["output_video_identity"]
                    ):
                        raise ValueError(f"ZIP video identity receipt drift: {output}: {output_name}")
                    _validate_video_identities(source_identity, output_identity, f"{output}!/{output_name}")
                elif conversion["kind"] in {"office_pdf", "step_text_pdf", "psd_flattened_pdf"}:
                    if _pdf_identity(output_member) != conversion["pdf_identity"]:
                        raise ValueError(f"ZIP PDF identity drift: {output}: {output_name}")
                    if conversion["kind"] == "step_text_pdf":
                        lines, text_sha256, logical_line_count = _step_display_lines(source_member)
                        if (
                            text_sha256 != conversion["source_text_sha256"]
                            or logical_line_count != conversion["logical_line_count"]
                            or hashlib.sha256(_deterministic_text_pdf(lines)).hexdigest()
                            != conversion["output_sha256"]
                        ):
                            raise ValueError(f"ZIP STEP derivative identity drift: {output}: {output_name}")
                    elif conversion["kind"] == "psd_flattened_pdf":
                        if _psd_embedded_identity(output_member) != conversion["flattened_image_identity"]:
                            raise ValueError(f"ZIP PSD derivative identity drift: {output}: {output_name}")
                else:
                    raise ValueError(f"unknown ZIP member derivative kind: {conversion['kind']!r}")
            reference_video_bundles = entry.get("reference_video_bundles")
            if reference_video_bundles is not None:
                if not isinstance(reference_video_bundles, dict):
                    raise ValueError(f"reference video bundle receipt is invalid: {output}")
                _validate_reference_video_bundles(
                    reference_video_bundles,
                    source_zip,
                    output_zip,
                    output,
                    workspace,
                )
    finally:
        shutil.rmtree(workspace)


def _validate_cache_summary(manifest: dict[str, Any]) -> None:
    summary = manifest.get("derivative_cache")
    required = {
        "schema",
        "persistent",
        "cache_label",
        "converter_identity",
        "converter_identity_sha256",
        "hits",
        "misses",
        "negative_hits",
        "corruptions_quarantined",
        "bytes_reused",
        "bytes_generated",
    }
    if not isinstance(summary, dict) or set(summary) != required:
        raise ValueError("transport derivative cache receipt is malformed")
    if summary["schema"] != DERIVATIVE_CACHE_SCHEMA or type(summary["persistent"]) is not bool:
        raise ValueError("transport derivative cache schema or persistence drift")
    if not isinstance(summary["cache_label"], str) or not summary["cache_label"]:
        raise ValueError("transport derivative cache label is invalid")
    for name in (
        "hits",
        "misses",
        "negative_hits",
        "corruptions_quarantined",
        "bytes_reused",
        "bytes_generated",
    ):
        if type(summary[name]) is not int or summary[name] < 0:
            raise ValueError(f"transport derivative cache counter is invalid: {name}")
    identity = summary["converter_identity"]
    if not isinstance(identity, dict) or set(identity) != {
        "schema",
        "transport_views_sha256",
        "preconvert_module_sha256",
        "container_sha256",
        "tools",
    }:
        raise ValueError("transport derivative converter identity is malformed")
    if identity["schema"] != "gdpval.transport-converter-identity.v1":
        raise ValueError("transport derivative converter identity schema drift")
    if identity["transport_views_sha256"] != _sha256(Path(__file__).resolve(strict=True)):
        raise ValueError("transport derivative converter source drift")
    if identity["preconvert_module_sha256"] != manifest.get("preconvert_module_sha256"):
        raise ValueError("transport derivative preconvert identity drift")
    if identity["tools"] != manifest.get("tools"):
        raise ValueError("transport derivative tool identity drift")
    container_sha256 = identity["container_sha256"]
    if summary["persistent"]:
        if not isinstance(container_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", container_sha256) is None:
            raise ValueError("persistent derivative cache has no container digest")
    elif container_sha256 is not None and (
        not isinstance(container_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", container_sha256) is None
    ):
        raise ValueError("ephemeral derivative cache container digest is invalid")
    identity_payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(identity_payload).hexdigest() != summary["converter_identity_sha256"]:
        raise ValueError("transport derivative converter identity digest mismatch")


def _validate(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"transport manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("transport manifest schema mismatch")
    try:
        _validate_derivative_profile(manifest["derivative_profile"])
        min_audio_bytes = int(manifest["min_audio_bytes"])
        min_video_bytes = int(manifest["min_video_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"transport derivative profile is invalid: {exc}") from exc
    if min_audio_bytes < 0 or min_video_bytes != DEFAULT_MIN_VIDEO_BYTES:
        raise ValueError("transport derivative thresholds do not match the declared profile")
    if manifest.get("derivative_parameters") != _derivative_parameters(min_audio_bytes, min_video_bytes):
        raise ValueError("transport derivative parameters drift")
    tools = manifest.get("tools")
    if not isinstance(tools, dict) or not {"ffmpeg", "ffprobe", "fitz", "pillow", "soundfile"} <= set(tools):
        raise ValueError("transport tool identity is incomplete")
    _validate_cache_summary(manifest)
    overlay = output_root / "reference_views.yaml"
    if not overlay.is_file() or overlay.is_symlink():
        raise ValueError(f"transport reference overlay is invalid: {overlay}")
    if _sha256(overlay) != manifest["reference_overlay_sha256"]:
        raise ValueError("transport reference overlay digest mismatch")
    expected_root_entries = {"candidate", "references", "manifest.json", "reference_views.yaml"}
    actual_root_entries = {path.name for path in output_root.iterdir()}
    if actual_root_entries != expected_root_entries:
        raise ValueError(
            "transport root inventory drift: "
            f"unexpected={sorted(actual_root_entries - expected_root_entries)} "
            f"missing={sorted(expected_root_entries - actual_root_entries)}"
        )
    reference_groups = {Path(group["view"]).name for group in manifest["groups"] if group["name"] != "candidate"}
    actual_reference_groups = {path.name for path in (output_root / "references").iterdir()}
    if actual_reference_groups != reference_groups:
        raise ValueError(
            "transport reference-group inventory drift: "
            f"unexpected={sorted(actual_reference_groups - reference_groups)} "
            f"missing={sorted(reference_groups - actual_reference_groups)}"
        )
    for group in manifest["groups"]:
        view_root = output_root / group["view"]
        _validate_view_inventory(output_root, group)
        for entry in group["entries"]:
            output = view_root / entry["output_relative_path"]
            if entry["kind"] != "symlink":
                _validate_derived_file(entry, output)
            else:
                if not output.is_symlink() or output.resolve() != Path(entry["source"]):
                    raise ValueError(f"transport symlink drift: {output}")
                stat = Path(entry["source"]).stat()
                if stat.st_size != entry["source_bytes"] or stat.st_mtime_ns != entry["source_mtime_ns"]:
                    raise ValueError(f"transport source drift: {entry['source']}")
    return manifest


def build(
    candidate_root: Path,
    reference_overlay: Path,
    output_root: Path,
    min_audio_bytes: int,
    *,
    derivative_profile: str,
    min_video_bytes: int = DEFAULT_MIN_VIDEO_BYTES,
    preconvert_py: Path | None = None,
    derivative_cache_root: Path | None = None,
    derivative_cache_label: str | None = None,
    container_sha256: str | None = None,
) -> dict[str, Any]:
    derivative_profile = _validate_derivative_profile(derivative_profile)
    if min_audio_bytes < 0:
        raise ValueError("min_audio_bytes must be nonnegative")
    if min_video_bytes != DEFAULT_MIN_VIDEO_BYTES:
        raise ValueError(f"profile {derivative_profile!r} requires min_video_bytes={DEFAULT_MIN_VIDEO_BYTES}")
    candidate_root = candidate_root.resolve(strict=True)
    reference_overlay = reference_overlay.resolve(strict=True)
    output_root = output_root.resolve(strict=False)
    preconvert_module: ModuleType | None = None
    preconvert_module_sha256: str | None = None
    preconvert_module_path: Path | None = None
    if preconvert_py is not None:
        preconvert_module, preconvert_module_sha256, preconvert_module_path = _load_preconvert_module(preconvert_py)
    tools = check_tools()
    persistent_cache = derivative_cache_root is not None
    if persistent_cache and (
        not isinstance(container_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", container_sha256) is None
    ):
        raise ValueError("persistent derivative cache requires a lowercase SHA-256 container identity")
    converter_identity = {
        "schema": "gdpval.transport-converter-identity.v1",
        "transport_views_sha256": _sha256(Path(__file__).resolve(strict=True)),
        "preconvert_module_sha256": preconvert_module_sha256,
        "container_sha256": container_sha256,
        "tools": tools,
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_root.parent / f".{output_root.name}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (output_root / "manifest.json").is_file():
            existing = _validate(output_root)
            requested = {
                "candidate_source": str(candidate_root),
                "reference_overlay_source": str(reference_overlay),
                "reference_overlay_source_sha256": _sha256(reference_overlay),
                "derivative_profile": derivative_profile,
                "min_audio_bytes": min_audio_bytes,
                "min_video_bytes": min_video_bytes,
                "preconvert_module_sha256": preconvert_module_sha256,
            }
            actual = {key: existing.get(key) for key in requested}
            if actual != requested:
                raise ValueError(f"existing transport view was built from different inputs: {output_root}")
            return existing

        document = yaml.safe_load(reference_overlay.read_text())
        mapping_path, reference_models = _reference_models(document)
        source_references = {
            str(reference_id): Path(str(config["deliverables_dir"])).resolve(strict=True)
            for reference_id, config in reference_models.items()
        }
        temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
        cleanup_cache = derivative_cache_root is None
        cache_root = (
            Path(tempfile.mkdtemp(prefix=f".{output_root.name}.derivative_cache.", dir=output_root.parent))
            if cleanup_cache
            else derivative_cache_root
        )
        assert cache_root is not None
        derivative_cache = _DerivativeCache(
            cache_root,
            derivative_profile,
            converter_identity,
            persistent=persistent_cache,
            cache_label=(derivative_cache_label if persistent_cache else "ephemeral"),
        )
        try:
            groups = []
            candidate_view = temporary / "candidate"
            candidate_view.mkdir()
            candidate_entries, candidate_directories = _mirror_tree(
                candidate_root,
                candidate_view,
                min_audio_bytes,
                min_video_bytes=min_video_bytes,
                derivative_cache=derivative_cache,
                preconvert_module=preconvert_module,
                preconvert_module_sha256=preconvert_module_sha256,
            )
            groups.append(
                {
                    "name": "candidate",
                    "source": str(candidate_root),
                    "view": "candidate",
                    "entries": candidate_entries,
                    "directories": candidate_directories,
                }
            )
            rewritten_models: dict[str, Any] = {}
            for reference_id, source in sorted(source_references.items()):
                if not reference_id or reference_id in {".", ".."} or Path(reference_id).name != reference_id:
                    raise ValueError(f"unsafe reference model id: {reference_id!r}")
                relative_view = Path("references") / reference_id
                view = temporary / relative_view
                view.mkdir(parents=True)
                reference_entries, reference_directories = _mirror_tree(
                    source,
                    view,
                    min_audio_bytes,
                    min_video_bytes=min_video_bytes,
                    derivative_cache=derivative_cache,
                    preconvert_module=preconvert_module,
                    preconvert_module_sha256=preconvert_module_sha256,
                )
                groups.append(
                    {
                        "name": reference_id,
                        "source": str(source),
                        "view": relative_view.as_posix(),
                        "entries": reference_entries,
                        "directories": reference_directories,
                    }
                )
                rewritten = dict(reference_models[reference_id])
                rewritten["deliverables_dir"] = str(output_root / relative_view)
                rewritten_models[reference_id] = rewritten
            overlay_document = _nested_overlay(mapping_path, rewritten_models)
            overlay_path = temporary / "reference_views.yaml"
            _atomic_yaml(overlay_path, overlay_document)
            manifest = {
                "schema": SCHEMA,
                "tools": tools,
                "candidate_source": str(candidate_root),
                "reference_overlay_source": str(reference_overlay),
                "reference_overlay_source_sha256": _sha256(reference_overlay),
                "reference_overlay_sha256": _sha256(overlay_path),
                "derivative_profile": derivative_profile,
                "derivative_parameters": _derivative_parameters(min_audio_bytes, min_video_bytes),
                "min_audio_bytes": min_audio_bytes,
                "min_video_bytes": min_video_bytes,
                "preconvert_module": str(preconvert_module_path) if preconvert_module_path is not None else None,
                "preconvert_module_sha256": preconvert_module_sha256,
                "derivative_cache": derivative_cache.summary(),
                "groups": groups,
            }
            _atomic_json(temporary / "manifest.json", manifest)
            if output_root.exists():
                raise ValueError(f"incomplete transport output already exists: {output_root}")
            os.replace(temporary, output_root)
            temporary = None
            return _validate(output_root)
        finally:
            if temporary is not None and temporary.exists():
                shutil.rmtree(temporary)
            if cleanup_cache and cache_root.exists():
                shutil.rmtree(cache_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "validate", "check-tools"))
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--reference-overlay", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--min-audio-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--min-video-bytes", type=int, default=DEFAULT_MIN_VIDEO_BYTES)
    parser.add_argument("--derivative-profile")
    parser.add_argument("--preconvert-py", type=Path)
    parser.add_argument("--derivative-cache-root", type=Path)
    parser.add_argument("--derivative-cache-label")
    parser.add_argument("--container-sha256")
    args = parser.parse_args()
    if args.action == "check-tools":
        print(json.dumps(check_tools(), sort_keys=True))
        return
    if args.output_root is None:
        parser.error("--output-root is required")
    if args.action == "validate":
        manifest = _validate(args.output_root.resolve(strict=True))
    else:
        if args.candidate_root is None or args.reference_overlay is None:
            parser.error("build requires --candidate-root and --reference-overlay")
        if args.derivative_profile is None:
            parser.error("build requires --derivative-profile")
        manifest = build(
            args.candidate_root,
            args.reference_overlay,
            args.output_root,
            args.min_audio_bytes,
            derivative_profile=args.derivative_profile,
            min_video_bytes=args.min_video_bytes,
            preconvert_py=args.preconvert_py,
            derivative_cache_root=args.derivative_cache_root,
            derivative_cache_label=args.derivative_cache_label,
            container_sha256=args.container_sha256,
        )
    converted = sum(entry["kind"] != "symlink" for group in manifest["groups"] for entry in group["entries"])
    print(f"TRANSPORT_VIEWS_PASS groups={len(manifest['groups'])} derived_outputs={converted}")


if __name__ == "__main__":
    main()
