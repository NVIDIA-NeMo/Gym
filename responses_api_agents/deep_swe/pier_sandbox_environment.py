# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pier environment adapter backed by NeMo Gym's provider-neutral sandbox API."""

import asyncio
import gzip
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import stat
import sys
import tarfile
import tempfile
import uuid
from collections.abc import Awaitable
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any

from pier.environments.base import BaseEnvironment, ExecResult
from pier.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from pier.models.trial.config import ResourceMode

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.utils import rewrite_image


_RUNTIME_CONFIGS: dict[str, dict[str, Any]] = {}
_RUNTIME_CONFIGS_LOCK = asyncio.Lock()
_DOCKERFILE_FROM_RE = re.compile(r"^\s*FROM(?:\s+--platform=\S+)?\s+(\S+)", re.IGNORECASE)
_AGENT_VERSION_RE = re.compile(r"(?<![0-9A-Za-z.])v?(\d+\.\d+\.\d+)(?![0-9A-Za-z.])", re.IGNORECASE)
_AGENT_ARTIFACTS_DIR = PurePosixPath("/logs/artifacts")
_AGENT_VERSION_PROOF_NAME = "gym-agent-version.json"
_SANDBOX_OBSERVATIONS_DIR_NAME = "gym-sandbox-observations"
_DEEPSWE_VERIFIER_INSTRUCTIONS = (
    "COPY test.sh /tests/test.sh",
    "COPY test.patch /tests/test.patch",
    "COPY grader.py /tests/grader.py",
    "COPY config.json /tests/config.json",
    "RUN chmod +x /tests/test.sh",
)
_DEFAULT_TRANSFER_TIMEOUT_S = 1800.0
_DEFAULT_ARTIFACT_MAX_FILES = 2048
_DEFAULT_ARTIFACT_MAX_FILE_BYTES = 64 * 1024 * 1024
_DEFAULT_ARTIFACT_MAX_TOTAL_BYTES = 256 * 1024 * 1024
_DEFAULT_ARCHIVE_OVERHEAD_BYTES = 8 * 1024 * 1024
_MAX_DOWNLOAD_FILE_BYTES = _DEFAULT_ARTIFACT_MAX_FILE_BYTES
_MAX_DOWNLOAD_ARCHIVE_BYTES = _DEFAULT_ARTIFACT_MAX_TOTAL_BYTES + _DEFAULT_ARCHIVE_OVERHEAD_BYTES
_MAX_DOWNLOAD_MEMBERS = _DEFAULT_ARTIFACT_MAX_FILES * 4
_MAX_DOWNLOAD_MEMBER_BYTES = _DEFAULT_ARTIFACT_MAX_FILE_BYTES
_MAX_DOWNLOAD_EXPANDED_BYTES = _DEFAULT_ARTIFACT_MAX_TOTAL_BYTES
_MAX_DOWNLOAD_DECOMPRESSED_TAR_BYTES = _MAX_DOWNLOAD_ARCHIVE_BYTES
_MAX_DOWNLOAD_METADATA_BYTES = 1024 * 1024
_MAX_DOWNLOAD_PATH_BYTES = 4096
_MAX_DOWNLOAD_PATH_COMPONENT_BYTES = 255
_MAX_DOWNLOAD_PATH_COMPONENTS = 256
_EXTRACTION_COPY_CHUNK_BYTES = 1024 * 1024
_TAR_BLOCK_BYTES = 512
_TAR_EXTENSION_TYPES = frozenset(
    {
        tarfile.XHDTYPE,
        tarfile.XGLTYPE,
        tarfile.SOLARIS_XHDTYPE,
        tarfile.GNUTYPE_LONGNAME,
        tarfile.GNUTYPE_LONGLINK,
    }
)


class _BoundedDecompressedTarReader:
    """Read a gzip stream in small chunks while enforcing a total output cap."""

    def __init__(self, stream: Any, max_bytes: int = _MAX_DOWNLOAD_DECOMPRESSED_TAR_BYTES) -> None:
        self._stream = stream
        self._total = 0
        self._max_bytes = max_bytes

    def _account(self, chunk: bytes) -> None:
        self._total += len(chunk)
        if self._total > self._max_bytes:
            raise RuntimeError(
                f"Sandbox download archive exceeds decompressed tar budget: {self._total} > {self._max_bytes} bytes"
            )

    def read_exact(self, size: int, *, allow_eof: bool = False) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = self._stream.read(min(_EXTRACTION_COPY_CHUNK_BYTES, remaining))
            if not chunk:
                if allow_eof and remaining == size:
                    return b""
                raise RuntimeError("Sandbox download archive is truncated")
            self._account(chunk)
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def skip_exact(self, size: int) -> None:
        remaining = size
        while remaining:
            chunk = self._stream.read(min(_EXTRACTION_COPY_CHUNK_BYTES, remaining))
            if not chunk:
                raise RuntimeError("Sandbox download archive is truncated")
            self._account(chunk)
            remaining -= len(chunk)


def _padded_tar_size(size: int) -> int:
    return (size + _TAR_BLOCK_BYTES - 1) // _TAR_BLOCK_BYTES * _TAR_BLOCK_BYTES


def _validate_pax_payload(payload: bytes) -> None:
    position = 0
    while position < len(payload):
        separator = payload.find(b" ", position)
        if separator < 0 or not payload[position:separator].isdigit():
            raise RuntimeError("Sandbox download archive has malformed PAX metadata")
        try:
            record_size = int(payload[position:separator])
        except ValueError as error:
            raise RuntimeError("Sandbox download archive has malformed PAX metadata") from error
        record_end = position + record_size
        if record_size <= separator - position + 3 or record_end > len(payload) or payload[record_end - 1] != 0x0A:
            raise RuntimeError("Sandbox download archive has malformed PAX metadata")
        body = payload[separator + 1 : record_end - 1]
        key, marker, value = body.partition(b"=")
        if not marker or not key:
            raise RuntimeError("Sandbox download archive has malformed PAX metadata")
        if key == b"size" or key.startswith(b"GNU.sparse"):
            raise RuntimeError("Sandbox download archive contains unsupported sparse or size-override metadata")
        if key in {b"path", b"linkpath"} and len(value) > _MAX_DOWNLOAD_PATH_BYTES:
            raise RuntimeError("Sandbox download archive member path exceeds metadata budget")
        position = record_end


def _validate_raw_tar_stream(
    archive: Path,
    *,
    max_members: int | None = None,
    max_member_bytes: int | None = None,
    max_expanded_bytes: int | None = None,
    max_decompressed_tar_bytes: int | None = None,
) -> None:
    """Inspect every raw tar header before ``tarfile`` can allocate extension data."""
    max_members = _MAX_DOWNLOAD_MEMBERS if max_members is None else max_members
    max_member_bytes = _MAX_DOWNLOAD_MEMBER_BYTES if max_member_bytes is None else max_member_bytes
    max_expanded_bytes = _MAX_DOWNLOAD_EXPANDED_BYTES if max_expanded_bytes is None else max_expanded_bytes
    max_decompressed_tar_bytes = (
        _MAX_DOWNLOAD_DECOMPRESSED_TAR_BYTES if max_decompressed_tar_bytes is None else max_decompressed_tar_bytes
    )
    raw_member_count = 0
    expanded_size = 0
    try:
        with gzip.open(archive, "rb") as compressed:
            reader = _BoundedDecompressedTarReader(compressed, max_decompressed_tar_bytes)
            while True:
                header = reader.read_exact(_TAR_BLOCK_BYTES, allow_eof=True)
                if not header:
                    return
                if header == b"\0" * _TAR_BLOCK_BYTES:
                    if reader.read_exact(_TAR_BLOCK_BYTES) != b"\0" * _TAR_BLOCK_BYTES:
                        raise RuntimeError("Sandbox download archive has an invalid end marker")
                    return
                raw_member_count += 1
                if raw_member_count > max_members:
                    raise RuntimeError(
                        "Sandbox download archive exceeds member-count budget (raw headers): "
                        f"{raw_member_count} > {max_members}"
                    )
                try:
                    member = tarfile.TarInfo.frombuf(header, "utf-8", "surrogateescape")
                except tarfile.HeaderError as error:
                    raise RuntimeError("Sandbox download archive has an invalid tar header") from error
                if member.size < 0:
                    raise RuntimeError("Sandbox download archive has a negative member size")
                if member.type == tarfile.GNUTYPE_SPARSE:
                    raise RuntimeError("Sandbox download archive contains an unsupported sparse member")

                padded_size = _padded_tar_size(member.size)
                if member.type in _TAR_EXTENSION_TYPES:
                    if member.size > _MAX_DOWNLOAD_METADATA_BYTES:
                        raise RuntimeError(
                            "Sandbox download archive extension metadata exceeds budget: "
                            f"{member.size} > {_MAX_DOWNLOAD_METADATA_BYTES} bytes"
                        )
                    payload = reader.read_exact(member.size)
                    reader.skip_exact(padded_size - member.size)
                    if member.type in {tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.SOLARIS_XHDTYPE}:
                        _validate_pax_payload(payload)
                    continue

                if member.size > max_member_bytes:
                    raise RuntimeError(
                        "Sandbox download archive member exceeds expanded-size budget: "
                        f"{member.name!r} ({member.size} > {max_member_bytes} bytes)"
                    )
                expanded_size += member.size
                if expanded_size > max_expanded_bytes:
                    raise RuntimeError(
                        "Sandbox download archive exceeds cumulative expanded-size budget: "
                        f"{expanded_size} > {max_expanded_bytes} bytes"
                    )
                reader.skip_exact(padded_size)
    except (gzip.BadGzipFile, EOFError) as error:
        raise RuntimeError("Sandbox download archive is not a valid gzip stream") from error


def _archive_destination(member: tarfile.TarInfo, extraction_root: Path) -> Path | None:
    encoded_name = member.name.encode("utf-8", errors="surrogateescape")
    if len(encoded_name) > _MAX_DOWNLOAD_PATH_BYTES:
        raise RuntimeError(f"Sandbox archive member path is too long: {member.name!r}")
    path = PurePosixPath(member.name)
    if not member.name or "\x00" in member.name or path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"Unsafe sandbox archive member path: {member.name!r}")
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if len(parts) > _MAX_DOWNLOAD_PATH_COMPONENTS or any(
        len(part.encode("utf-8", errors="surrogateescape")) > _MAX_DOWNLOAD_PATH_COMPONENT_BYTES for part in parts
    ):
        raise RuntimeError(f"Sandbox archive member path has unsafe components: {member.name!r}")
    if not parts:
        if member.isdir():
            return None
        raise RuntimeError(f"Unsafe sandbox archive member path: {member.name!r}")
    destination = extraction_root.joinpath(*parts)
    if destination != extraction_root and not destination.is_relative_to(extraction_root):
        raise RuntimeError(f"Sandbox archive member escapes extraction root: {member.name!r}")
    return destination


def _ensure_extraction_parent(extraction_root: Path, destination: Path) -> None:
    current = extraction_root
    for component in destination.relative_to(extraction_root).parts[:-1]:
        current /= component
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            metadata = current.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(f"Sandbox archive destination parent is not a directory: {current}") from None


def _extract_download_archive(
    archive: Path,
    target: Path,
    *,
    max_archive_bytes: int | None = None,
    max_members: int | None = None,
    max_member_bytes: int | None = None,
    max_expanded_bytes: int | None = None,
    max_decompressed_tar_bytes: int | None = None,
) -> None:
    """Validate and incrementally extract a bounded archive into ``target``."""
    max_archive_bytes = _MAX_DOWNLOAD_ARCHIVE_BYTES if max_archive_bytes is None else max_archive_bytes
    max_members = _MAX_DOWNLOAD_MEMBERS if max_members is None else max_members
    max_member_bytes = _MAX_DOWNLOAD_MEMBER_BYTES if max_member_bytes is None else max_member_bytes
    max_expanded_bytes = _MAX_DOWNLOAD_EXPANDED_BYTES if max_expanded_bytes is None else max_expanded_bytes
    max_decompressed_tar_bytes = (
        _MAX_DOWNLOAD_DECOMPRESSED_TAR_BYTES if max_decompressed_tar_bytes is None else max_decompressed_tar_bytes
    )
    try:
        archive_size = archive.stat().st_size
    except OSError as error:
        raise RuntimeError("Sandbox download archive is unavailable") from error
    if archive_size > max_archive_bytes:
        raise RuntimeError(
            f"Sandbox download archive exceeds compressed transfer budget: {archive_size} > {max_archive_bytes} bytes"
        )
    _validate_raw_tar_stream(
        archive,
        max_members=max_members,
        max_member_bytes=max_member_bytes,
        max_expanded_bytes=max_expanded_bytes,
        max_decompressed_tar_bytes=max_decompressed_tar_bytes,
    )

    target = Path(os.path.abspath(os.path.expanduser(target)))
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target_metadata = target.lstat()
    except FileNotFoundError:
        target_metadata = None
    if target_metadata is not None:
        if not stat.S_ISDIR(target_metadata.st_mode):
            raise RuntimeError(f"Sandbox download target is not a real directory: {target}")

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.extract-", dir=target.parent))
    try:
        if target_metadata is not None:
            shutil.copytree(target, staging, dirs_exist_ok=True, symlinks=True)
        member_count = 0
        expanded_size = 0
        with tarfile.open(archive, mode="r|gz") as tf:
            for member in tf:
                member_count += 1
                if member_count > max_members:
                    raise RuntimeError(
                        f"Sandbox download archive exceeds member-count budget: {member_count} > {max_members}"
                    )
                if not member.isdir() and not member.isreg():
                    raise RuntimeError(f"Sandbox download archive contains a link or special file: {member.name!r}")
                if member.size < 0 or member.size > max_member_bytes:
                    raise RuntimeError(
                        "Sandbox download archive member exceeds expanded-size budget: "
                        f"{member.name!r} ({member.size} > {max_member_bytes} bytes)"
                    )
                expanded_size += member.size
                if expanded_size > max_expanded_bytes:
                    raise RuntimeError(
                        "Sandbox download archive exceeds cumulative expanded-size budget: "
                        f"{expanded_size} > {max_expanded_bytes} bytes"
                    )

                destination = _archive_destination(member, staging)
                if destination is None:
                    continue
                _ensure_extraction_parent(staging, destination)
                if member.isdir():
                    try:
                        destination.mkdir(mode=0o700)
                    except FileExistsError:
                        metadata = destination.lstat()
                        if not stat.S_ISDIR(metadata.st_mode):
                            raise RuntimeError(
                                f"Sandbox archive directory collides with a file: {member.name!r}"
                            ) from None
                    continue

                source = tf.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Sandbox archive member has no readable payload: {member.name!r}")
                remaining = member.size
                try:
                    try:
                        existing = destination.lstat()
                    except FileNotFoundError:
                        existing = None
                    if existing is not None:
                        if not stat.S_ISREG(existing.st_mode):
                            raise RuntimeError(f"Sandbox archive file collides with a non-file: {member.name!r}")
                        destination.unlink()
                    with destination.open("xb") as output:
                        while remaining:
                            chunk = source.read(min(_EXTRACTION_COPY_CHUNK_BYTES, remaining))
                            if not chunk:
                                raise RuntimeError(f"Sandbox archive member is truncated: {member.name!r}")
                            output.write(chunk)
                            remaining -= len(chunk)
                finally:
                    source.close()
                os.chmod(destination, 0o700 if member.mode & 0o100 else 0o600)

        if target_metadata is None:
            os.replace(staging, target)
        else:
            backup = target.with_name(f".{target.name}.replaced-{uuid.uuid4().hex}")
            os.replace(target, backup)
            try:
                os.replace(staging, target)
            except BaseException:
                os.replace(backup, target)
                raise
            shutil.rmtree(backup, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_local_json_proof(target: Path, name: str, proof: dict[str, Any]) -> None:
    """Atomically replace a sandbox-supplied claim with an owner-only host observation."""
    if not name or PurePosixPath(name).name != name or "/" in name or "\x00" in name:
        raise ValueError("Local proof name must be one safe path component")
    target = Path(os.path.abspath(os.path.expanduser(target)))
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_descriptor = os.open(target, directory_flags)
    temporary_name = f".{name}.{uuid.uuid4().hex}.tmp"
    temporary_created = False
    try:
        metadata = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise RuntimeError("Sandbox artifact target must be a current-user-owned real directory")
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        temporary_created = True
        try:
            payload = (json.dumps(proof, sort_keys=True, separators=(",", ":")) + "\n").encode()
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("Could not write the local agent version proof")
                offset += written
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        temporary_created = False
        os.fsync(directory_descriptor)
    finally:
        if temporary_created:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_descriptor)
        os.close(directory_descriptor)


def _write_local_agent_version_proof(target: Path, proof: dict[str, str]) -> None:
    _write_local_json_proof(target, _AGENT_VERSION_PROOF_NAME, proof)


def _write_local_sandbox_observation(target: Path, proof: dict[str, Any]) -> None:
    target = Path(os.path.abspath(os.path.expanduser(target)))
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    target_metadata = target.lstat()
    if (
        not stat.S_ISDIR(target_metadata.st_mode)
        or target_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(target_metadata.st_mode) & 0o022
    ):
        raise RuntimeError("Sandbox observation root is not a current-user-owned real directory")
    observations = target / _SANDBOX_OBSERVATIONS_DIR_NAME
    try:
        observations.mkdir(mode=0o700)
    except FileExistsError:
        metadata = observations.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise RuntimeError("Sandbox observation destination is not an owner-only real directory") from None
    observation_id = proof.get("observation_id")
    if not isinstance(observation_id, str) or re.fullmatch(r"[0-9a-f]{32}", observation_id) is None:
        raise ValueError("Sandbox observation has an invalid observation_id")
    _write_local_json_proof(observations, f"sandbox-{observation_id}.json", proof)


async def register_runtime_config(config: dict[str, Any]) -> str:
    """Store secret-bearing sandbox config outside Pier's persisted job config."""
    config_id = uuid.uuid4().hex
    async with _RUNTIME_CONFIGS_LOCK:
        _RUNTIME_CONFIGS[config_id] = config
    return config_id


async def unregister_runtime_config(config_id: str) -> None:
    async with _RUNTIME_CONFIGS_LOCK:
        _RUNTIME_CONFIGS.pop(config_id, None)


def _runtime_config(
    config_id: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    if config_path is not None:
        value = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Pier sandbox runtime config file must contain a JSON object")
        return value
    if config_id is not None:
        try:
            return _RUNTIME_CONFIGS[config_id]
        except KeyError as e:
            raise RuntimeError(
                "Pier sandbox runtime config is unavailable. Start this job through "
                "the NeMo Gym DeepSWE agent rather than replaying Pier's redacted config directly."
            ) from e
    raise ValueError("PierSandboxEnvironment requires runtime_config_id or runtime_config_path")


def _positive_runtime_int(config: dict[str, Any], name: str, default: int) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Pier sandbox {name} must be a positive integer")
    return value


def _image_from_dockerfile(path: Path) -> str:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"Empty Dockerfile: {path}")
    match = _DOCKERFILE_FROM_RE.match(lines[0])
    if match is None:
        raise ValueError(f"Verifier Dockerfile must begin with a concrete FROM image: {path}")
    image = match.group(1)
    if image.startswith("$"):
        raise ValueError(f"Dockerfile base image must be concrete, got {image!r}")
    if tuple(lines[1:]) != _DEEPSWE_VERIFIER_INSTRUCTIONS:
        raise ValueError(
            "Unsupported verifier Dockerfile. The Gym sandbox adapter only accepts "
            "DeepSWE v1.1's audited FROM + four COPY + chmod template; refusing to "
            f"skip unrecognized build instructions in {path}."
        )
    return image


class PierSandboxEnvironment(BaseEnvironment):
    """Run a Pier task or verifier in a Gym ``AsyncSandbox``."""

    def __init__(
        self,
        *args: Any,
        runtime_config_id: str | None = None,
        runtime_config_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        runtime_config = _runtime_config(runtime_config_id, runtime_config_path)
        self._provider = runtime_config["provider"]
        self._spec_config = dict(runtime_config.get("spec") or {})
        self._provider_metadata = dict(runtime_config.get("provider_metadata") or {})
        self._supports_disable_internet = bool(runtime_config.get("supports_disable_internet", False))
        self._supports_filtered_egress = bool(runtime_config.get("supports_filtered_egress", False))
        self._supports_gpus = bool(runtime_config.get("supports_gpus", False))
        self._supports_disk_resource = bool(runtime_config.get("supports_disk_resource", True))
        self._preinstall_agent_in_image = bool(runtime_config.get("preinstall_agent_in_image", False))
        expected_agent_name = runtime_config.get("expected_agent_name")
        if (
            not isinstance(expected_agent_name, str)
            or not expected_agent_name
            or expected_agent_name.strip() != expected_agent_name
        ):
            raise ValueError("Pier sandbox expected_agent_name must be a nonempty trimmed string")
        expected_agent_version = runtime_config.get("expected_agent_version")
        if (
            not isinstance(expected_agent_version, str)
            or re.fullmatch(r"\d+\.\d+\.\d+", expected_agent_version) is None
        ):
            raise ValueError("Pier sandbox expected_agent_version must have x.y.z form")
        self._expected_agent_name = expected_agent_name
        self._expected_agent_version = expected_agent_version
        transfer_timeout_s = runtime_config.get("transfer_timeout_s", _DEFAULT_TRANSFER_TIMEOUT_S)
        if (
            isinstance(transfer_timeout_s, bool)
            or not isinstance(transfer_timeout_s, (int, float))
            or not math.isfinite(transfer_timeout_s)
            or transfer_timeout_s <= 0
        ):
            raise ValueError("Pier sandbox transfer_timeout_s must be a positive finite number")
        self._transfer_timeout_s = float(transfer_timeout_s)
        self._artifact_max_files = _positive_runtime_int(
            runtime_config,
            "artifact_max_files",
            _DEFAULT_ARTIFACT_MAX_FILES,
        )
        self._artifact_max_file_bytes = _positive_runtime_int(
            runtime_config,
            "artifact_max_file_bytes",
            _DEFAULT_ARTIFACT_MAX_FILE_BYTES,
        )
        self._artifact_max_total_bytes = _positive_runtime_int(
            runtime_config,
            "artifact_max_total_bytes",
            _DEFAULT_ARTIFACT_MAX_TOTAL_BYTES,
        )
        if self._artifact_max_file_bytes > self._artifact_max_total_bytes:
            raise ValueError("Pier sandbox artifact_max_file_bytes cannot exceed artifact_max_total_bytes")
        archive_overhead_bytes = min(
            32 * 1024 * 1024,
            max(_DEFAULT_ARCHIVE_OVERHEAD_BYTES, self._artifact_max_files * 4096),
        )
        self._max_download_members = self._artifact_max_files * 4
        self._max_download_archive_bytes = self._artifact_max_total_bytes + archive_overhead_bytes
        self._max_download_decompressed_tar_bytes = self._max_download_archive_bytes
        self._sandbox: AsyncSandbox | None = None
        self._sandbox_observation: dict[str, Any] | None = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return "nemo-gym-sandbox"

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(
            cpu_limit=True,
            cpu_request=True,
            memory_limit=True,
            memory_request=True,
        )

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            gpus=self._supports_gpus,
            disable_internet=self._supports_disable_internet,
            filtered_egress=self._supports_filtered_egress,
            preinstall_agents=True,
        )

    @classmethod
    def preflight(cls) -> None:
        return None

    def _validate_definition(self) -> None:
        if self.task_env_config.docker_image:
            return
        dockerfile = self.environment_dir / "Dockerfile"
        if not dockerfile.is_file():
            raise FileNotFoundError(f"{dockerfile} not found and the task does not declare environment.docker_image")
        _image_from_dockerfile(dockerfile)

    def _image(self) -> str:
        image = self.task_env_config.docker_image
        if image is None:
            image = _image_from_dockerfile(self.environment_dir / "Dockerfile")
        return rewrite_image(image, self._spec_config.get("image_rewrites", []))

    def _sandbox_spec(self) -> SandboxSpec:
        cfg = dict(self._spec_config)
        resources = dict(cfg.pop("resources", {}) or {})
        provider_options = dict(cfg.pop("provider_options", {}) or {})
        provider_name = next(iter(self._provider))
        if self._effective_cpus is not None:
            resources["cpu"] = self._effective_cpus
            if provider_name == "modal" and self._cpu_resource_mode in {
                ResourceMode.AUTO,
                ResourceMode.GUARANTEE,
                ResourceMode.LIMIT,
            }:
                provider_options["cpu_limit"] = self._effective_cpus
                if self._cpu_resource_mode == ResourceMode.LIMIT:
                    resources["cpu"] = min(0.125, self._effective_cpus)
        if self._effective_memory_mb is not None:
            resources["memory_mib"] = self._effective_memory_mb
            if provider_name == "modal" and self._memory_resource_mode in {
                ResourceMode.GUARANTEE,
                ResourceMode.LIMIT,
            }:
                provider_options["memory_limit_mib"] = self._effective_memory_mb
                if self._memory_resource_mode == ResourceMode.LIMIT:
                    resources["memory_mib"] = min(128, self._effective_memory_mb)
        if self._supports_disk_resource and self._effective_storage_mb is not None:
            resources["disk_gib"] = math.ceil(self._effective_storage_mb / 1024)
        if self._effective_gpus:
            resources["gpu"] = self._effective_gpus
            if self.task_env_config.gpu_types:
                resources["gpu_type"] = self.task_env_config.gpu_types[0]

        metadata = {
            **self._provider_metadata,
            **dict(cfg.pop("metadata", {}) or {}),
            "nemo_gym_agent": "deep_swe",
            "pier_environment": self.environment_name[:63],
            "pier_session": self.session_id[:63],
        }
        env = {**dict(cfg.pop("env", {}) or {}), **self._persistent_env}
        if self._preinstall_agent_in_image and self.agent_install_spec is not None:
            setup_steps = []
            for step in self.agent_install_spec.steps:
                setup_step = step.model_dump(mode="json", exclude_none=True)
                setup_step["user"] = "root" if step.user == "root" else self.default_user or "root"
                setup_step["shell"] = "/bin/bash"
                setup_steps.append(setup_step)
            provider_options["image_setup_steps"] = setup_steps
        if not self.task_env_config.allow_internet:
            task_domains = tuple(self.network_allowlist.domains)
            configured_domains = provider_options.get("network_allowlist")
            outbound_domains = provider_options.get("outbound_domain_allowlist")
            if configured_domains is not None and outbound_domains is not None:
                raise ValueError("Configure only one sandbox domain allowlist")
            configured_domains = configured_domains if configured_domains is not None else outbound_domains
            if configured_domains is not None and set(configured_domains) != set(task_domains):
                raise ValueError("Sandbox domain allowlist conflicts with Pier's task-derived network policy")
            if provider_options.get("outbound_cidr_allowlist") is not None:
                raise ValueError("Sandbox CIDR egress cannot override Pier's task-derived network policy")
            if provider_options.get("inbound_cidr_allowlist") is not None:
                raise ValueError("Sandbox inbound access is not permitted for an offline Pier task")
            block_network = not task_domains
            if "block_network" in provider_options and provider_options["block_network"] is not block_network:
                raise ValueError("Sandbox block_network conflicts with Pier's task-derived network policy")
            provider_options.pop("outbound_domain_allowlist", None)
            provider_options["block_network"] = block_network
            if task_domains:
                provider_options["network_allowlist"] = list(task_domains)
            else:
                provider_options.pop("network_allowlist", None)

        known = {
            "ttl_s",
            "ready_timeout_s",
            "workdir",
            "files",
            "entrypoint",
        }
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"Unknown Pier sandbox spec keys: {sorted(unknown)}")

        return SandboxSpec(
            image=self._image(),
            ttl_s=cfg.get("ttl_s"),
            ready_timeout_s=cfg.get("ready_timeout_s"),
            workdir=cfg.get("workdir") or self.task_env_config.workdir,
            env=env,
            files=dict(cfg.get("files", {}) or {}),
            metadata=metadata,
            resources=SandboxResources.from_mapping(resources),
            entrypoint=cfg.get("entrypoint"),
            provider_options=provider_options,
        )

    async def _verify_agent_version(self) -> dict[str, str]:
        install_spec = self.agent_install_spec
        if install_spec is None:
            raise RuntimeError("Cannot verify an agent environment without an install specification")
        if (
            install_spec.agent_name != self._expected_agent_name
            or install_spec.version != self._expected_agent_version
        ):
            raise RuntimeError(
                "Pier agent install specification does not match the Gym-requested agent identity: "
                f"expected {self._expected_agent_name} {self._expected_agent_version}"
            )
        verification_command = install_spec.verification_command
        if not verification_command:
            raise RuntimeError(f"{self._expected_agent_name} has no install verification command")

        verification = await self.exec(
            f"/bin/bash -c {shlex.quote(verification_command)}",
            timeout_sec=120,
        )
        if verification.return_code != 0:
            raise RuntimeError(
                f"{self._expected_agent_name} install verification failed with code {verification.return_code}"
            )
        observed_versions = _AGENT_VERSION_RE.findall(verification.stdout or "")
        if len(observed_versions) != 1:
            raise RuntimeError(
                f"{self._expected_agent_name} install verification did not report exactly one x.y.z version on stdout"
            )
        observed_version = observed_versions[0]
        if observed_version != self._expected_agent_version:
            raise RuntimeError(
                f"{self._expected_agent_name} reported version {observed_version}, "
                f"expected {self._expected_agent_version}"
            )
        return {
            "agent": self._expected_agent_name,
            "expected": self._expected_agent_version,
            "observed": observed_version,
            "verification_command": verification_command,
        }

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError("Gym Sandbox providers start immutable images; force_build is unsupported")
        self._sandbox = AsyncSandbox(self._provider)
        try:
            sandbox_spec = self._sandbox_spec()
            await self._sandbox.start(sandbox_spec)
            self._sandbox_observation = {
                "schema_version": 1,
                "observation_id": uuid.uuid4().hex,
                "provider": self._sandbox.provider_name,
                "sandbox_id": self._sandbox.sandbox_id,
                "requested_image": sandbox_spec.image,
                "resolved_image_digest": None,
                "environment_role": "agent" if self.agent_install_spec is not None else "verifier",
                "session_id": self.session_id,
            }
            await self._verify_default_disk_capacity()
            result = await self.exec(
                "mkdir -p /logs/agent /logs/verifier /logs/artifacts && "
                "chmod 777 /logs/agent /logs/verifier /logs/artifacts",
                user="root",
                timeout_sec=120,
            )
            if result.return_code != 0:
                raise RuntimeError(f"Failed to initialize Pier log directories: {result.stderr or result.stdout}")

            if self.task_env_config.docker_image is None:
                await self.upload_dir(self.environment_dir, "/tests")

            if self.agent_install_spec is not None and not self._preinstall_agent_in_image:
                for step in self.agent_install_spec.steps:
                    user = "root" if step.user == "root" else self.default_user
                    install_result = await self.exec(
                        f"/bin/bash -c {shlex.quote(step.run)}",
                        env=step.env,
                        user=user,
                        timeout_sec=1800,
                    )
                    if install_result.return_code != 0:
                        output = install_result.stderr or install_result.stdout or "no output"
                        raise RuntimeError(
                            f"{self.agent_install_spec.agent_name} install failed "
                            f"with code {install_result.return_code}: {output}"
                        )
            if self.agent_install_spec is not None:
                await self._verify_agent_version()
        except BaseException as startup_error:
            try:
                await self.stop(delete=True)
            except BaseException:
                startup_error.add_note("Sandbox cleanup also failed; cleanup details were omitted.")
            raise

    async def _verify_default_disk_capacity(self) -> None:
        if self._supports_disk_resource or self._effective_storage_mb is None:
            return
        result = await self.exec("df -Pk / | awk 'NR == 2 {print $4}'", user="root", timeout_sec=120)
        try:
            available_kib = int((result.stdout or "").strip())
        except ValueError:
            available_kib = -1
        required_kib = self._effective_storage_mb * 1024
        if result.return_code != 0 or available_kib < required_kib:
            raise RuntimeError(
                "Sandbox provider cannot request task disk capacity and its live default is insufficient: "
                f"required_mib={self._effective_storage_mb}, available_kib={available_kib}"
            )

    async def stop(self, delete: bool) -> None:
        del delete
        if self._sandbox is None:
            return
        sandbox = self._sandbox
        stop_error: BaseException | None = None
        try:
            await sandbox.stop()
        except BaseException as error:
            stop_error = error
        observation_error: BaseException | None = None
        if self._sandbox_observation is not None:
            try:
                _write_local_sandbox_observation(
                    self.trial_paths.artifacts_dir,
                    self._sandbox_observation,
                )
            except BaseException as error:
                observation_error = error
        if self._sandbox is sandbox:
            self._sandbox = None
        self._sandbox_observation = None
        if stop_error is not None:
            if observation_error is not None:
                stop_error.add_note(
                    f"Writing the host sandbox observation also failed: {type(observation_error).__name__}"
                )
            raise stop_error
        if observation_error is not None:
            raise observation_error

    def _require_sandbox(self) -> AsyncSandbox:
        if self._sandbox is None:
            raise RuntimeError("Pier sandbox environment has not been started")
        return self._sandbox

    async def _run_transfer(self, operation: Awaitable[None], *, name: str) -> None:
        try:
            async with asyncio.timeout(self._transfer_timeout_s):
                await operation
        except TimeoutError:
            raise TimeoutError(f"Sandbox {name} timed out after {self._transfer_timeout_s:g} seconds") from None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        result = await self._require_sandbox().exec(
            command,
            cwd=cwd or self.task_env_config.workdir,
            env=self._merge_env(env),
            timeout_s=timeout_sec,
            user=self._resolve_user(user),
        )
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.return_code,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        parent = str(PurePosixPath(target_path).parent)
        mkdir_result = await self.exec(
            f"mkdir -p {shlex.quote(parent)}",
            user="root",
            timeout_sec=120,
        )
        if mkdir_result.return_code != 0:
            raise RuntimeError(f"Failed to create sandbox directory {parent}: {mkdir_result.stderr}")
        await self._run_transfer(
            self._require_sandbox().upload(Path(source_path), target_path),
            name="upload",
        )

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        if not source.is_dir():
            raise FileNotFoundError(f"Source directory does not exist: {source}")
        archive_name = f"pier-upload-{uuid.uuid4().hex}.tar.gz"
        remote_archive = f"/tmp/{archive_name}"
        with tempfile.TemporaryDirectory(prefix="pier-sandbox-upload-") as tmp_dir:
            local_archive = Path(tmp_dir) / archive_name
            with tarfile.open(local_archive, "w:gz") as tf:
                for child in source.iterdir():
                    tf.add(child, arcname=child.name)
            await self.upload_file(local_archive, remote_archive)
        command = (
            f"mkdir -p {shlex.quote(target_dir)} && "
            f"tar xzf {shlex.quote(remote_archive)} -C {shlex.quote(target_dir)} && "
            f"rm -f {shlex.quote(remote_archive)}"
        )
        result = await self.exec(command, user="root", timeout_sec=300)
        if result.return_code != 0:
            raise RuntimeError(f"Failed to extract upload into {target_dir}: {result.stderr or result.stdout}")

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.download-", dir=target.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            await self._run_transfer(
                self._require_sandbox().download(
                    source_path,
                    temporary,
                    max_bytes=self._artifact_max_file_bytes,
                ),
                name="download",
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    async def _remove_remote_archive(self, remote_archive: str) -> None:
        cleanup = await self.exec(
            f"rm -f {shlex.quote(remote_archive)}",
            user="root",
            timeout_sec=120,
        )
        if cleanup.return_code != 0:
            self.logger.warning("Failed to remove sandbox transfer archive %s", remote_archive)

    def _observe_remote_archive_cleanup(self, task: asyncio.Task[None], remote_archive: str) -> None:
        try:
            task.result()
        except BaseException:
            self.logger.warning("Deferred sandbox transfer archive cleanup failed for %s", remote_archive)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        target = Path(target_dir)
        archive_name = f"pier-download-{uuid.uuid4().hex}.tar.gz"
        remote_archive = f"/tmp/{archive_name}"
        try:
            agent_version_proof = None
            if self.agent_install_spec is not None and PurePosixPath(source_dir) == _AGENT_ARTIFACTS_DIR:
                agent_version_proof = await self._verify_agent_version()
            result = await self.exec(
                f"tar czf {shlex.quote(remote_archive)} -C {shlex.quote(source_dir)} . && "
                f"chmod 400 {shlex.quote(remote_archive)} && "
                f"wc -c < {shlex.quote(remote_archive)} && "
                f"sha256sum -- {shlex.quote(remote_archive)}",
                user="root",
                timeout_sec=300,
            )
            if result.return_code != 0:
                raise RuntimeError(f"Failed to archive {source_dir}: {result.stderr or result.stdout}")
            metadata_lines = (result.stdout or "").splitlines()
            try:
                remote_archive_size = int(metadata_lines[0].strip())
                remote_archive_sha256 = metadata_lines[1].split()[0].lower()
            except (IndexError, ValueError) as error:
                raise RuntimeError("Sandbox did not report valid transfer archive metadata") from error
            if re.fullmatch(r"[0-9a-f]{64}", remote_archive_sha256) is None:
                raise RuntimeError("Sandbox did not report valid transfer archive metadata")
            if remote_archive_size < 0 or remote_archive_size > self._max_download_archive_bytes:
                raise RuntimeError(
                    "Sandbox download archive exceeds compressed transfer budget: "
                    f"{remote_archive_size} > {self._max_download_archive_bytes} bytes"
                )

            with tempfile.TemporaryDirectory(prefix="pier-sandbox-download-") as tmp_dir:
                local_archive = Path(tmp_dir) / archive_name
                await self._run_transfer(
                    self._require_sandbox().download(
                        remote_archive,
                        local_archive,
                        max_bytes=self._max_download_archive_bytes,
                    ),
                    name="download",
                )
                if (
                    local_archive.stat().st_size != remote_archive_size
                    or _file_sha256(local_archive) != remote_archive_sha256
                ):
                    raise RuntimeError("Sandbox transfer archive changed during download")
                _extract_download_archive(
                    local_archive,
                    target,
                    max_archive_bytes=self._max_download_archive_bytes,
                    max_members=self._max_download_members,
                    max_member_bytes=self._artifact_max_file_bytes,
                    max_expanded_bytes=self._artifact_max_total_bytes,
                    max_decompressed_tar_bytes=self._max_download_decompressed_tar_bytes,
                )
                if agent_version_proof is not None:
                    # The archive is model-controlled. Replace any claimed proof locally,
                    # after bounded extraction and before Pier manifests the artifacts.
                    _write_local_agent_version_proof(target, agent_version_proof)
        finally:
            active_error = sys.exception()
            cleanup_task = asyncio.create_task(self._remove_remote_archive(remote_archive))
            try:
                await asyncio.shield(cleanup_task)
            except BaseException:
                if not cleanup_task.done():
                    cleanup_task.add_done_callback(
                        lambda task: self._observe_remote_archive_cleanup(task, remote_archive)
                    )
                if active_error is None:
                    raise
                active_error.add_note("Sandbox transfer archive cleanup also failed; details were omitted.")

    async def attach(self) -> None:
        raise NotImplementedError("Gym Sandbox-backed Pier environments do not support attach")


__all__ = [
    "PierSandboxEnvironment",
    "register_runtime_config",
    "unregister_runtime_config",
]
