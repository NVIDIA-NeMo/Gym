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

"""NeMo Gym agent for pinned DeepSWE tasks executed by Pier."""

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse
from uuid import uuid4

from fastapi import Body
from pydantic import ConfigDict, Field, SecretStr, model_validator

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import ServerClient
from responses_api_agents.deep_swe.benchmark import (
    ensure_checkout,
    materialize_task_snapshot,
    resolve_task,
    task_tree_digest,
)
from responses_api_agents.deep_swe.secure_paths import default_private_path, ensure_private_directory
from responses_api_agents.deep_swe.setup_pier import (
    PIER_RUNTIME_LAYOUT_VERSION,
    PIER_RUNTIME_MODAL_VERSION,
    PIER_SOURCE_COMMIT,
    PIER_SOURCE_URL,
    PIER_VERSION,
    ensure_pier_runtime,
    pier_constraints_sha256,
    pier_direct_url_sha256,
)
from responses_api_agents.deep_swe.trajectory import (
    ArtifactLimitError,
    ArtifactLimits,
    artifact_snapshot,
    empty_response,
    read_bytes,
    read_json,
    read_text,
    response_from_atif,
)


DEFAULT_DEEP_SWE_COMMIT = "8cae5984d5dd0ee37445beff0e928dc10c331116"  # pragma: allowlist secret
DEFAULT_DEEP_SWE_TASK_COUNT = 113
PIER_SANDBOX_ENVIRONMENT = "responses_api_agents.deep_swe.pier_sandbox_environment:PierSandboxEnvironment"
PIER_CLAUDE_CODE_NPM_AGENT = "responses_api_agents.deep_swe.pier_claude_code:ClaudeCodeNpmInstall"
PIER_CLAUDE_CODE_AGENT_NAME = "claude-code"
GYM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_CACHE_DIR = default_private_path("checkouts")
DEFAULT_WORK_ROOT = default_private_path("jobs")
DEFAULT_PIER_RUNTIME_DIR = default_private_path(
    "runtime",
    f"pier-{PIER_VERSION}-v{PIER_RUNTIME_LAYOUT_VERSION}-{hashlib.sha256(str(GYM_ROOT).encode()).hexdigest()[:12]}",
)
_SENSITIVE_KEY_RE = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|AUTH)", re.IGNORECASE)
_PIER_PASSTHROUGH_SECRET_ENV = frozenset({"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"})
_PIER_INHERITED_ENV_DENY_PREFIXES = (
    "ANTHROPIC_",
    "AWS_BEDROCK_",
    "BEDROCK_",
    "CLAUDE_CODE_",
    "PIER_",
)
_PIER_INHERITED_ENV_DENY_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_REGION",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "CLAUDE_CONFIG_DIR",
        "DISABLE_PROMPT_CACHING",
        "MAX_THINKING_TOKENS",
    }
)


@dataclass(frozen=True)
class PierRunError(RuntimeError):
    """A Pier launcher failure with any retained partial trial provenance."""

    message: str
    job_dir: Path
    trial: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self.message


class UnsafeArtifactError(RuntimeError):
    """A sanitized failure raised after unsafe retained evidence is isolated."""

    def __init__(
        self,
        message: str,
        *,
        pending_cancellation: asyncio.CancelledError | None = None,
    ) -> None:
        super().__init__(message)
        self.pending_cancellation = pending_cancellation


class DeepSWEAgentConfig(BaseResponsesAPIAgentConfig):
    model_base_url: str
    model_api_key: SecretStr
    model_name: str
    claude_code_version: str = Field(default="2.1.153", pattern=r"^\d+\.\d+\.\d+$")
    claude_code_install_method: Literal["pier", "npm"] = "pier"
    claude_code_kwargs: dict[str, Any] = Field(default_factory=dict)
    claude_code_env: dict[str, str] = Field(default_factory=dict)

    benchmark_git_url: str = "https://github.com/datacurve-ai/deep-swe.git"
    benchmark_git_commit: str = DEFAULT_DEEP_SWE_COMMIT
    benchmark_expected_task_count: int | None = DEFAULT_DEEP_SWE_TASK_COUNT
    benchmark_cache_dir: str = str(DEFAULT_BENCHMARK_CACHE_DIR)
    benchmark_path: str | None = None

    sandbox_provider: str | dict[str, Any]
    sandbox_required_provider: str | None = None
    sandbox_spec: dict[str, Any] = Field(default_factory=dict)
    sandbox_supports_disable_internet: bool = False
    sandbox_supports_filtered_egress: bool = False
    sandbox_supports_gpus: bool = False
    sandbox_supports_disk_resource: bool = True
    sandbox_preinstall_agent_in_image: bool = False
    sandbox_transfer_timeout_s: float = Field(default=1800, gt=0, allow_inf_nan=False)

    work_root: str = str(DEFAULT_WORK_ROOT)
    pier_runtime_dir: str = str(DEFAULT_PIER_RUNTIME_DIR)
    pier_cancel_grace_s: float = Field(default=120, gt=0)
    max_concurrent: int = Field(default=4, ge=1)
    max_concurrent_assembly: int = Field(
        default=2,
        ge=1,
        description="Maximum concurrent bounded artifact response assemblies per server process.",
    )
    artifact_max_files: int = Field(
        default=2048,
        ge=1,
        description="Maximum retained files inspected for one response.",
    )
    artifact_max_file_bytes: int = Field(
        default=64 * 1024 * 1024,
        ge=1,
        description="Maximum bytes read or hashed from any one retained artifact.",
    )
    artifact_max_total_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=1,
        description="Maximum cumulative retained artifact bytes inspected for one response.",
    )
    trajectory_max_bytes: int = Field(
        default=64 * 1024 * 1024,
        ge=1,
        description="Maximum ATIF trajectory size returned inline.",
    )
    patch_max_bytes: int = Field(
        default=16 * 1024 * 1024,
        ge=1,
        description="Maximum model.patch size returned inline.",
    )

    @model_validator(mode="after")
    def validate_environment_selection(self) -> "DeepSWEAgentConfig":
        if not self.sandbox_provider:
            raise ValueError("DeepSWE requires sandbox_provider")
        sensitive_env_keys = sorted(key for key in self.claude_code_env if _SENSITIVE_KEY_RE.search(key))
        if sensitive_env_keys:
            raise ValueError(
                "claude_code_env accepts non-secret values only; use runtime secret handling for: "
                + ", ".join(sensitive_env_keys)
            )
        if self.artifact_max_file_bytes > self.artifact_max_total_bytes:
            raise ValueError("artifact_max_file_bytes cannot exceed artifact_max_total_bytes")
        if self.trajectory_max_bytes > self.artifact_max_file_bytes:
            raise ValueError("trajectory_max_bytes cannot exceed artifact_max_file_bytes")
        if self.patch_max_bytes > self.artifact_max_file_bytes:
            raise ValueError("patch_max_bytes cannot exceed artifact_max_file_bytes")
        return self


class DeepSWERunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class DeepSWEVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    task_id: str
    status: str
    error_type: str | None = None
    error_message: str | None = None
    raw_reward: float | None = None
    benchmark_metadata: dict[str, Any]
    pier_result: dict[str, Any] | None = None
    verifier_result: dict[str, Any] | None = None
    raw_rollout: dict[str, Any]
    model_patch: str | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


def _git_command(root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _safe_repository_url(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value or len(value) > 4096 or any(ord(character) < 32 for character in value):
        return None
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https", "ssh", "git", "git+ssh"} and parsed.hostname:
        try:
            port = parsed.port
        except ValueError:
            return None
        hostname = parsed.hostname.lower()
        if hostname == "github.com" and port not in {None, 22, 443}:
            return None
        authority = hostname if hostname == "github.com" or port is None else f"{hostname}:{port}"
        repository_path = parsed.path.strip("/")
        if repository_path.endswith(".git"):
            repository_path = repository_path[:-4]
        return f"https://{authority}/{repository_path}" if repository_path else None
    scp_match = None if "://" in value else re.fullmatch(r"(?:[^@/:]+@)?([^@/:]+):([^?#]+)", value)
    if scp_match is not None:
        hostname = scp_match.group(1).lower()
        repository_path = scp_match.group(2).strip("/")
        if repository_path.endswith(".git"):
            repository_path = repository_path[:-4]
        return f"https://{hostname}/{repository_path}" if repository_path else None
    return None


def _gym_source_provenance(root: Path) -> dict[str, Any]:
    commit_result = _git_command(root, "rev-parse", "--verify", "HEAD")
    commit = commit_result.stdout.strip() if commit_result is not None and commit_result.returncode == 0 else None
    if commit is None or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        commit = None

    remote_result = _git_command(root, "remote", "get-url", "origin")
    repository_url = _safe_repository_url(
        remote_result.stdout if remote_result is not None and remote_result.returncode == 0 else None
    )

    status_result = _git_command(root, "status", "--porcelain=v1", "--untracked-files=all")
    working_tree_clean = (
        status_result.stdout == "" if status_result is not None and status_result.returncode == 0 else None
    )

    try:
        lock_bytes = read_bytes(
            root / "uv.lock",
            max_bytes=16 * 1024 * 1024,
            description="NeMo Gym uv.lock",
        )
    except (ArtifactLimitError, OSError):
        uv_lock_sha256 = None
    else:
        uv_lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()

    return {
        "repository_url": repository_url,
        "commit": commit,
        "uv_lock_sha256": uv_lock_sha256,
        "working_tree_clean": working_tree_clean,
    }


def _trial_path(trial_uri: str, job_dir: Path) -> Path:
    parsed = urlparse(trial_uri)
    if parsed.scheme != "file" or parsed.netloc:
        raise ValueError("Pier returned a trial URI that is not a local file URI")
    path = Path(unquote(parsed.path)).resolve()
    try:
        path.relative_to(job_dir.resolve())
    except ValueError as error:
        raise ValueError("Pier trial URI escapes its retained job directory") from error
    return path


def _redacted_secret_variants(secret: str) -> tuple[str, ...]:
    if not secret:
        return ()
    values = [secret]
    if len(secret) > 8:
        values.append(secret[:4] + "****" + secret[-3:])
    return tuple(values)


def _redact_text(text: str, secrets: tuple[str, ...]) -> str:
    for secret in sorted(set(secrets), key=len, reverse=True):
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


def _sanitize_persisted_value(value: Any, secrets: tuple[str, ...], *, key: str | None = None) -> Any:
    if key and _SENSITIVE_KEY_RE.search(key):
        if isinstance(value, str) and value.startswith("${"):
            return value
        return "<redacted>"
    if isinstance(value, dict):
        return {
            _redact_text(str(item_key), secrets): _sanitize_persisted_value(item, secrets, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_persisted_value(item, secrets) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_persisted_value(item, secrets) for item in value]
    if isinstance(value, str):
        sanitized = _redact_text(value, secrets)
        return sanitized
    return value


def _redact_stream_buffer(
    buffer: bytes,
    replacements: tuple[tuple[bytes, bytes], ...],
    *,
    final: bool,
) -> tuple[bytes, bytes, bool]:
    patterns = tuple(sorted((item for item in replacements if item[0]), key=lambda item: len(item[0]), reverse=True))
    if not patterns:
        return buffer, b"", False

    max_secret_length = max(len(secret) for secret, _ in patterns)
    patterns_by_first_byte: dict[int, list[tuple[bytes, bytes]]] = {}
    for secret, replacement in patterns:
        patterns_by_first_byte.setdefault(secret[0], []).append((secret, replacement))
    output = bytearray()
    cursor = 0
    replaced = False
    while cursor < len(buffer):
        remaining_length = len(buffer) - cursor
        candidates = patterns_by_first_byte.get(buffer[cursor], ())
        if (
            not final
            and remaining_length < max_secret_length
            and any(
                len(secret) > remaining_length and buffer.startswith(secret[:remaining_length], cursor)
                for secret, _ in candidates
            )
        ):
            break
        match = next(
            ((secret, replacement) for secret, replacement in candidates if buffer.startswith(secret, cursor)),
            None,
        )
        if match is not None:
            secret, replacement = match
            output.extend(replacement)
            cursor += len(secret)
            replaced = True
            continue
        output.append(buffer[cursor])
        cursor += 1
    return bytes(output), buffer[cursor:], replaced


def _assert_safe_artifact_paths(entry_paths: tuple[str, ...], secrets: tuple[str, ...]) -> None:
    for relative in entry_paths:
        if _redact_text(relative, secrets) != relative:
            raise UnsafeArtifactError("Retained Pier evidence used secret material in an artifact path")


def _discard_or_quarantine_job(job_dir: Path) -> str:
    """Remove unsafe evidence or atomically hide it under the private jobs root."""

    def make_removable(function: Any, path: str, _: BaseException) -> None:
        candidate = Path(path)
        parent_metadata = candidate.parent.lstat()
        if stat.S_ISDIR(parent_metadata.st_mode):
            os.chmod(candidate.parent, 0o700, follow_symlinks=False)
        metadata = candidate.lstat()
        if not stat.S_ISLNK(metadata.st_mode):
            os.chmod(candidate, 0o700 if stat.S_ISDIR(metadata.st_mode) else 0o600, follow_symlinks=False)
        function(path)

    try:
        shutil.rmtree(job_dir, onexc=make_removable)
        return "discarded"
    except OSError:
        quarantine = job_dir.parent / f".unsafe-{uuid4().hex}"
        try:
            job_dir.replace(quarantine)
            os.chmod(quarantine, 0o700)
            return "quarantined"
        except OSError:
            return "isolated under the private jobs root"


def _artifact_metadata_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _open_artifact_parent(root: Path, relative: Path) -> tuple[int, str]:
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ArtifactLimitError("DeepSWE artifact path is not a safe relative path")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(root, flags)
    try:
        for component in relative.parts[:-1]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor, relative.name
    except BaseException:
        os.close(descriptor)
        raise


def _open_regular_artifact(
    parent_descriptor: int,
    name: str,
    *,
    max_bytes: int,
) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise ArtifactLimitError(f"DeepSWE artifact is not a private regular file: {name}")
    if metadata.st_size > max_bytes:
        os.close(descriptor)
        raise ArtifactLimitError(f"DeepSWE artifact exceeds {max_bytes} bytes: {name}")
    return descriptor, metadata


def _verify_artifact_entry(parent_descriptor: int, name: str, metadata: os.stat_result) -> None:
    current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if _artifact_metadata_signature(current) != _artifact_metadata_signature(metadata):
        raise ArtifactLimitError(f"DeepSWE artifact changed while scrubbing: {name}")


def _write_all(descriptor: int, value: bytes) -> None:
    view = memoryview(value)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("Could not write finalized Pier evidence")
        view = view[written:]


def _scrub_file(
    path: Path,
    replacements: tuple[tuple[bytes, bytes], ...],
    *,
    max_bytes: int = 64 * 1024 * 1024,
    root: Path | None = None,
) -> None:
    root = path.parent if root is None else root
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ArtifactLimitError("DeepSWE artifact escapes its retained job directory") from error
    parent_descriptor, name = _open_artifact_parent(root, relative)
    temporary_name: str | None = None
    try:
        descriptor, opened = _open_regular_artifact(parent_descriptor, name, max_bytes=max_bytes)
    except BaseException:
        os.close(parent_descriptor)
        raise
    try:
        changed = False
        if replacements:
            carry = b""
            size = 0
            while chunk := os.read(descriptor, min(1024 * 1024, max_bytes - size + 1)):
                size += len(chunk)
                if size > max_bytes:
                    raise ArtifactLimitError(f"DeepSWE artifact exceeds {max_bytes} bytes: {name}")
                _, new_carry, replaced = _redact_stream_buffer(carry + chunk, replacements, final=False)
                changed = changed or replaced
                carry = new_carry
            _, _, replaced = _redact_stream_buffer(carry, replacements, final=True)
            changed = changed or replaced
        after_scan = os.fstat(descriptor)
        if _artifact_metadata_signature(after_scan) != _artifact_metadata_signature(opened):
            raise ArtifactLimitError(f"DeepSWE artifact changed while scrubbing: {name}")
        _verify_artifact_entry(parent_descriptor, name, opened)

        if not changed:
            os.fchmod(descriptor, stat.S_IMODE(opened.st_mode) & 0o600)
            return

        os.lseek(descriptor, 0, os.SEEK_SET)
        temporary_name = f".scrub-{uuid4().hex}"
        temporary_descriptor = os.open(
            temporary_name,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            carry = b""
            size = 0
            while chunk := os.read(descriptor, min(1024 * 1024, max_bytes - size + 1)):
                size += len(chunk)
                if size > max_bytes:
                    raise ArtifactLimitError(f"DeepSWE artifact exceeds {max_bytes} bytes: {name}")
                output, carry, _ = _redact_stream_buffer(carry + chunk, replacements, final=False)
                _write_all(temporary_descriptor, output)
            output, _, _ = _redact_stream_buffer(carry, replacements, final=True)
            _write_all(temporary_descriptor, output)
            os.fchmod(temporary_descriptor, stat.S_IMODE(opened.st_mode) & 0o600)
        finally:
            os.close(temporary_descriptor)
        after_write = os.fstat(descriptor)
        if _artifact_metadata_signature(after_write) != _artifact_metadata_signature(opened):
            raise ArtifactLimitError(f"DeepSWE artifact changed while scrubbing: {name}")
        _verify_artifact_entry(parent_descriptor, name, opened)
        assert temporary_name is not None
        os.replace(
            temporary_name,
            name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        temporary_name = None
    finally:
        os.close(descriptor)
        if temporary_name is not None:
            with suppress(OSError):
                os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.close(parent_descriptor)


def _seal_job_tree(job_dir: Path, *, entry_paths: tuple[str, ...], file_paths: set[str]) -> None:
    for relative in sorted(entry_paths, key=lambda value: value.count("/"), reverse=True):
        path = job_dir / relative
        os.chmod(path, 0o400 if relative in file_paths else 0o500, follow_symlinks=False)
    os.chmod(job_dir, 0o500, follow_symlinks=False)


def _scrub_job_secrets(
    job_dir: Path,
    secrets: tuple[str, ...],
    limits: ArtifactLimits | None = None,
) -> None:
    limits = limits or ArtifactLimits(
        max_files=2048,
        max_file_bytes=64 * 1024 * 1024,
        max_total_bytes=256 * 1024 * 1024,
    )
    replacements = tuple((secret.encode(), b"<redacted>") for secret in secrets if secret)
    before = artifact_snapshot(job_dir, limits=limits)
    _assert_safe_artifact_paths(before.entry_paths, secrets)
    failures = 0
    for item in before.manifest:
        path = job_dir / item["path"]
        try:
            _scrub_file(path, replacements, max_bytes=limits.max_file_bytes, root=job_dir)
        except (ArtifactLimitError, OSError):
            failures += 1
    if failures:
        raise RuntimeError(f"Failed to scrub or restrict {failures} retained Pier artifact(s)")
    after = artifact_snapshot(job_dir, limits=limits)
    _assert_safe_artifact_paths(after.entry_paths, secrets)
    if set(after.entry_paths) != set(before.entry_paths):
        raise RuntimeError("Retained Pier artifact tree changed during finalization")
    _seal_job_tree(
        job_dir,
        entry_paths=after.entry_paths,
        file_paths={item["path"] for item in after.manifest},
    )
    artifact_snapshot(job_dir, limits=limits)


def _finalize_job_files(
    job_dir: Path,
    stdout_path: Path,
    stdout: bytes,
    secrets: tuple[str, ...],
    limits: ArtifactLimits,
) -> None:
    if not stdout_path.exists():
        output = _redact_text(stdout.decode("utf-8", errors="replace"), secrets).encode()
        if len(output) > limits.max_file_bytes:
            raise ArtifactLimitError(f"DeepSWE Pier stdout exceeds {limits.max_file_bytes} bytes")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(stdout_path, flags, 0o600)
        try:
            _write_all(descriptor, output)
        finally:
            os.close(descriptor)
    _scrub_job_secrets(job_dir, secrets, limits)


async def _settle_security_task(
    awaitable: Any,
    *,
    initial_cancellation: asyncio.CancelledError | None = None,
) -> tuple[Any, BaseException | None, asyncio.CancelledError | None]:
    """Settle a security-critical worker without allowing cancellation to orphan it."""

    task = asyncio.create_task(awaitable)
    cancellation = initial_cancellation
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
        except BaseException:
            # The terminal exception is collected below after the task reports done.
            pass
    try:
        return task.result(), None, cancellation
    except BaseException as error:
        return None, error, cancellation


class DeepSWEAgent(SimpleResponsesAPIAgent):
    config: DeepSWEAgentConfig

    def model_post_init(self, context: Any) -> None:
        self._sem = asyncio.Semaphore(self.config.max_concurrent)
        self._assembly_sem = asyncio.Semaphore(self.config.max_concurrent_assembly)
        self._checkout_lock = asyncio.Lock()
        self._checkout_cache: tuple[Path, str] | None = None
        self._gym_source_provenance = _gym_source_provenance(GYM_ROOT)
        super().model_post_init(context)

    def _artifact_limits(self) -> ArtifactLimits:
        return ArtifactLimits(
            max_files=self.config.artifact_max_files,
            max_file_bytes=self.config.artifact_max_file_bytes,
            max_total_bytes=self.config.artifact_max_total_bytes,
        )

    async def _finalize_job_evidence(
        self,
        job_dir: Path,
        stdout_path: Path,
        stdout: bytes,
        *,
        initial_cancellation: asyncio.CancelledError | None = None,
    ) -> None:
        _, scrub_error, cancellation = await _settle_security_task(
            asyncio.to_thread(
                _finalize_job_files,
                job_dir,
                stdout_path,
                stdout,
                self._secret_variants(),
                self._artifact_limits(),
            ),
            initial_cancellation=initial_cancellation,
        )
        unsafe_error: UnsafeArtifactError | None = None
        if scrub_error is not None:
            disposition, disposition_error, disposition_cancellation = await _settle_security_task(
                asyncio.to_thread(_discard_or_quarantine_job, job_dir),
            )
            if cancellation is None:
                cancellation = disposition_cancellation
            if disposition_error is not None:
                disposition = "isolated under the private jobs root"
            unsafe_error = UnsafeArtifactError(
                "Pier artifact finalization failed "
                f"({type(scrub_error).__name__}); unsafe retained evidence was {disposition}"
            )
        if cancellation is not None:
            if unsafe_error is not None:
                cancellation.add_note(str(unsafe_error))
            raise cancellation
        if unsafe_error is not None:
            raise unsafe_error

    async def _assemble_response(self, function: Any, /, **kwargs: Any) -> DeepSWEVerifyResponse:
        async with self._assembly_sem:
            result, error, cancellation = await _settle_security_task(asyncio.to_thread(function, **kwargs))
            if cancellation is not None:
                if isinstance(error, UnsafeArtifactError):
                    error.pending_cancellation = cancellation
                    raise error
                if error is not None:
                    cancellation.add_note(f"DeepSWE response assembly also failed ({type(error).__name__})")
                raise cancellation
            if error is not None:
                raise error
            assert isinstance(result, DeepSWEVerifyResponse)
            return result

    async def _isolate_unsafe_job(self, job_dir: Path) -> str:
        disposition, error, cancellation = await _settle_security_task(
            asyncio.to_thread(_discard_or_quarantine_job, job_dir)
        )
        if cancellation is not None:
            raise cancellation
        if error is not None:
            return "isolated under the private jobs root"
        return str(disposition)

    async def responses(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        raise NotImplementedError("DeepSWE is an external benchmark; use /run")

    async def _checkout(self) -> Path:
        benchmark_path = Path(self.config.benchmark_path) if self.config.benchmark_path is not None else None
        cache_dir = Path(self.config.benchmark_cache_dir)
        if benchmark_path is None:
            cache_dir = ensure_private_directory(cache_dir)
        return await ensure_checkout(
            git_url=self.config.benchmark_git_url,
            commit=self.config.benchmark_git_commit,
            cache_dir=cache_dir,
            benchmark_path=benchmark_path,
            expected_task_count=self.config.benchmark_expected_task_count,
        )

    async def _checkout_with_digest(self) -> tuple[Path, str]:
        if self._checkout_cache is not None:
            return self._checkout_cache
        async with self._checkout_lock:
            if self._checkout_cache is None:
                checkout = await self._checkout()
                digest = await asyncio.to_thread(task_tree_digest, checkout)
                snapshot_parent = ensure_private_directory(default_private_path("snapshots"))
                snapshot = await asyncio.to_thread(materialize_task_snapshot, checkout, snapshot_parent)
                self._checkout_cache = (snapshot, digest)
        return self._checkout_cache

    def _agent_config(self) -> dict[str, Any]:
        kwargs = dict(self.config.claude_code_kwargs)
        kwargs.setdefault("version", self.config.claude_code_version)
        env = {
            "ANTHROPIC_BASE_URL": self.config.model_base_url,
            **self.config.claude_code_env,
            # Pier resolves this from the child process. Keeping the template in
            # AgentConfig prevents token values or redacted fragments from being
            # persisted in config.json, lock.json, result.json, or rollouts.
            "ANTHROPIC_AUTH_TOKEN": "${ANTHROPIC_AUTH_TOKEN}",
        }
        config = {
            "model_name": self.config.model_name,
            "kwargs": kwargs,
            "env": env,
        }
        if self.config.claude_code_install_method == "npm":
            config["import_path"] = PIER_CLAUDE_CODE_NPM_AGENT
        else:
            config["name"] = PIER_CLAUDE_CODE_AGENT_NAME
        return config

    def _pier_process_env(self) -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if key in _PIER_PASSTHROUGH_SECRET_ENV
            or (
                not _SENSITIVE_KEY_RE.search(key)
                and key not in _PIER_INHERITED_ENV_DENY_NAMES
                and not key.startswith(_PIER_INHERITED_ENV_DENY_PREFIXES)
            )
        }
        env["ANTHROPIC_AUTH_TOKEN"] = self.config.model_api_key.get_secret_value()
        return env

    def _secret_variants(self) -> tuple[str, ...]:
        secrets = [self.config.model_api_key.get_secret_value()]
        secrets.extend(os.environ.get(name, "") for name in _PIER_PASSTHROUGH_SECRET_ENV)
        secrets.extend(value for key, value in self.config.claude_code_env.items() if _SENSITIVE_KEY_RE.search(key))
        variants = {variant for secret in secrets for variant in _redacted_secret_variants(secret)}
        return tuple(sorted(variants, key=len, reverse=True))

    def _environment_config(self) -> tuple[dict[str, Any], dict[str, Any]]:
        global_config = (
            ServerClient.load_from_global_config().global_config_dict
            if isinstance(self.config.sandbox_provider, str)
            else {}
        )
        provider = resolve_provider_config(self.config.sandbox_provider, global_config)
        provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)
        if not isinstance(provider, dict) or len(provider) != 1:
            raise ValueError("DeepSWE sandbox_provider must resolve to exactly one provider")
        resolved_provider_name = next(iter(provider))
        if (
            self.config.sandbox_required_provider is not None
            and resolved_provider_name != self.config.sandbox_required_provider
        ):
            raise ValueError(
                "DeepSWE requires sandbox provider "
                f"{self.config.sandbox_required_provider!r}, resolved {resolved_provider_name!r}"
            )
        if resolved_provider_name == "modal":
            from nemo_gym.sandbox.providers.modal import ModalProvider

            ModalProvider.preflight()
        runtime_config = {
            "provider": provider,
            "provider_metadata": provider_metadata,
            "spec": dict(self.config.sandbox_spec),
            "supports_disable_internet": self.config.sandbox_supports_disable_internet,
            "supports_filtered_egress": self.config.sandbox_supports_filtered_egress,
            "supports_gpus": self.config.sandbox_supports_gpus,
            "supports_disk_resource": self.config.sandbox_supports_disk_resource,
            "preinstall_agent_in_image": self.config.sandbox_preinstall_agent_in_image,
            "transfer_timeout_s": self.config.sandbox_transfer_timeout_s,
            "artifact_max_files": self.config.artifact_max_files,
            "artifact_max_file_bytes": self.config.artifact_max_file_bytes,
            "artifact_max_total_bytes": self.config.artifact_max_total_bytes,
            "expected_agent_name": PIER_CLAUDE_CODE_AGENT_NAME,
            "expected_agent_version": self.config.claude_code_version,
        }
        return (
            {
                "type": None,
                "import_path": PIER_SANDBOX_ENVIRONMENT,
                "kwargs": {},
            },
            runtime_config,
        )

    def _runtime_provenance(
        self,
        environment: dict[str, Any],
        runtime_config: dict[str, Any],
    ) -> dict[str, Any]:
        del environment
        provider_config = runtime_config["provider"]
        provider_name = (
            next(iter(provider_config))
            if isinstance(provider_config, dict) and len(provider_config) == 1
            else "configured"
        )
        provenance = {
            "schema_version": 2,
            "adapter": PIER_SANDBOX_ENVIRONMENT,
            "gym_source": dict(self._gym_source_provenance),
            "pier_runtime": {
                "version": PIER_VERSION,
                "source_url": PIER_SOURCE_URL,
                "source_commit": PIER_SOURCE_COMMIT,
                "direct_url_sha256": pier_direct_url_sha256(),
                "constraints_sha256": pier_constraints_sha256(),
                "layout_version": PIER_RUNTIME_LAYOUT_VERSION,
                "modal_version": PIER_RUNTIME_MODAL_VERSION,
            },
            "provider": provider_name,
            "provider_config": _sanitize_persisted_value(provider_config, self._secret_variants()),
            "provider_metadata": _sanitize_persisted_value(
                runtime_config.get("provider_metadata"), self._secret_variants()
            ),
            "sandbox_spec": _sanitize_persisted_value(runtime_config.get("spec"), self._secret_variants()),
            "transfer_timeout_s": runtime_config["transfer_timeout_s"],
            "artifact_limits": {
                "max_files": runtime_config["artifact_max_files"],
                "max_file_bytes": runtime_config["artifact_max_file_bytes"],
                "max_total_bytes": runtime_config["artifact_max_total_bytes"],
            },
            "expected_agent": {
                "name": runtime_config["expected_agent_name"],
                "version": runtime_config["expected_agent_version"],
            },
            "capabilities": {
                "supports_disable_internet": runtime_config["supports_disable_internet"],
                "supports_filtered_egress": runtime_config["supports_filtered_egress"],
                "supports_gpus": runtime_config["supports_gpus"],
                "supports_disk_resource": runtime_config["supports_disk_resource"],
                "preinstall_agent_in_image": runtime_config["preinstall_agent_in_image"],
            },
        }
        canonical = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
        provenance["sha256"] = hashlib.sha256(canonical).hexdigest()
        return provenance

    def _response_runtime_provenance(self, job_dir: Path | None) -> dict[str, Any] | None:
        if job_dir is not None:
            try:
                persisted = read_json(
                    job_dir / "gym-runtime-provenance.json",
                    max_bytes=self.config.artifact_max_file_bytes,
                    description="sandbox runtime provenance",
                )
            except (ArtifactLimitError, OSError, UnicodeError, json.JSONDecodeError):
                persisted = None
            if isinstance(persisted, dict):
                return persisted
        try:
            environment, runtime_config = self._environment_config()
            return self._runtime_provenance(environment, runtime_config)
        except Exception:
            return None

    def _read_partial_trial(self, job_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
        result_paths: list[Path] = []
        for path in job_dir.glob("*/result.json"):
            if path.is_file():
                result_paths.append(path)
                if len(result_paths) > 1:
                    break
        if len(result_paths) != 1:
            return None, f"expected one Pier trial result, found {len(result_paths)}"
        try:
            trial = json.loads(
                read_text(
                    result_paths[0],
                    max_bytes=self.config.artifact_max_file_bytes,
                    description="Pier trial result",
                )
            )
        except (ArtifactLimitError, OSError, UnicodeError, json.JSONDecodeError) as error:
            return None, f"could not parse Pier trial result ({type(error).__name__})"
        if not isinstance(trial, dict):
            return None, "Pier trial result is not a JSON object"
        return trial, None

    async def _capture_pier_output(
        self,
        process: asyncio.subprocess.Process,
        log_path: Path,
    ) -> bytes:
        replacements = tuple((secret.encode(), b"<redacted>") for secret in self._secret_variants() if secret)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(log_path, flags, 0o600)
        tail = bytearray()
        persisted = 0

        def write_output(stream: Any, output: bytes) -> None:
            nonlocal persisted
            if not output:
                return
            if persisted + len(output) > self.config.artifact_max_file_bytes:
                raise ArtifactLimitError(f"DeepSWE Pier stdout exceeds {self.config.artifact_max_file_bytes} bytes")
            stream.write(output)
            stream.flush()
            persisted += len(output)
            tail.extend(output)
            if len(tail) > 64 * 1024:
                del tail[: len(tail) - 64 * 1024]

        with os.fdopen(descriptor, "wb") as log:
            reader = getattr(process, "stdout", None)
            if reader is None:
                stdout, _ = await process.communicate()
                output, _, _ = _redact_stream_buffer(stdout, replacements, final=True)
                write_output(log, output)
                return bytes(tail)

            carry = b""
            while chunk := await reader.read(64 * 1024):
                output, carry, _ = _redact_stream_buffer(carry + chunk, replacements, final=False)
                write_output(log, output)
            output, _, _ = _redact_stream_buffer(carry, replacements, final=True)
            write_output(log, output)
            await process.wait()
        return bytes(tail)

    async def _finish_cancelled_process(
        self,
        process: asyncio.subprocess.Process,
        communicate_task: asyncio.Task[bytes],
    ) -> bytes:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
        try:
            return await asyncio.wait_for(
                asyncio.shield(communicate_task),
                timeout=self.config.pier_cancel_grace_s,
            )
        except asyncio.TimeoutError:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
            try:
                return await asyncio.wait_for(
                    asyncio.shield(communicate_task),
                    timeout=self.config.pier_cancel_grace_s,
                )
            except asyncio.TimeoutError:
                communicate_task.cancel()
                with suppress(BaseException):
                    await communicate_task
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=self.config.pier_cancel_grace_s)
                return b""
        except BaseException:
            await self._terminate_failed_process(process)
            raise

    async def _communicate_with_cancellation_cleanup(
        self,
        process: asyncio.subprocess.Process,
        log_path: Path,
    ) -> bytes:
        communicate_task = asyncio.create_task(self._capture_pier_output(process, log_path))
        try:
            return await asyncio.shield(communicate_task)
        except asyncio.CancelledError as cancellation:
            stdout, cleanup_error, _ = await _settle_security_task(
                self._finish_cancelled_process(process, communicate_task),
                initial_cancellation=cancellation,
            )
            # Preserve cancellation after Pier has had a bounded grace period to
            # run its SIGTERM cleanup and close any live remote sandboxes.
            if cleanup_error is not None:
                cancellation.add_note(f"DeepSWE Pier cancellation cleanup failed ({type(cleanup_error).__name__})")
                stdout = b""
            setattr(cancellation, "pier_stdout", stdout)
            raise cancellation
        except Exception as error:
            _, cleanup_error, cancellation = await _settle_security_task(self._terminate_failed_process(process))
            if cancellation is not None:
                if cleanup_error is not None:
                    cancellation.add_note(f"DeepSWE Pier failure cleanup failed ({type(cleanup_error).__name__})")
                raise cancellation
            if cleanup_error is not None:
                error.add_note(f"DeepSWE Pier failure cleanup failed ({type(cleanup_error).__name__})")
            raise

    async def _terminate_failed_process(self, process: asyncio.subprocess.Process) -> None:
        """Bound cleanup when subprocess I/O fails without cancelling the caller."""
        if process.returncode is not None:
            return
        with suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=self.config.pier_cancel_grace_s)
        except asyncio.TimeoutError:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=self.config.pier_cancel_grace_s)

    async def _run_pier(self, *, task_path: Path, run_id: str) -> tuple[dict[str, Any], Path]:
        environment, runtime_config = self._environment_config()
        jobs_dir = ensure_private_directory(Path(self.config.work_root))
        job_dir = jobs_dir / run_id
        job_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        try:
            runtime_provenance = self._runtime_provenance(environment, runtime_config)
            (job_dir / "gym-runtime-provenance.json").write_text(
                json.dumps(runtime_provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            pier = await ensure_pier_runtime(Path(self.config.pier_runtime_dir), GYM_ROOT)
        except BaseException as error:
            _, cleanup_error, cancellation = await _settle_security_task(
                asyncio.to_thread(_discard_or_quarantine_job, job_dir),
                initial_cancellation=error if isinstance(error, asyncio.CancelledError) else None,
            )
            if cancellation is not None:
                if cleanup_error is not None:
                    cancellation.add_note(f"DeepSWE unlaunched-job cleanup failed ({type(cleanup_error).__name__})")
                raise cancellation
            if cleanup_error is not None:
                error.add_note(f"DeepSWE unlaunched-job cleanup failed ({type(cleanup_error).__name__})")
            raise
        process: asyncio.subprocess.Process | None = None
        stdout = b""
        launch_error: BaseException | None = None
        stdout_path = job_dir / "gym-pier-stdout.log"
        with tempfile.TemporaryDirectory(prefix="deep-swe-launch-", dir=jobs_dir) as launch_dir_value:
            launch_dir = Path(launch_dir_value)
            try:
                runtime_config_path = launch_dir / "sandbox-runtime.json"
                runtime_config_path.write_text(json.dumps(runtime_config), encoding="utf-8")
                os.chmod(runtime_config_path, 0o600)
                environment["kwargs"] = {"runtime_config_path": str(runtime_config_path)}
                config = {
                    "job_name": run_id,
                    "jobs_dir": str(jobs_dir),
                    "n_attempts": 1,
                    "n_concurrent_trials": 1,
                    "quiet": True,
                    "retry": {"max_retries": 0},
                    "environment": environment,
                    "agents": [self._agent_config()],
                    # A local path is intentionally not assigned a synthetic
                    # ``source``. Pier treats source as a dataset key and its metric
                    # hook expects registered dataset metrics for every non-null key.
                    "tasks": [{"path": str(task_path)}],
                }
                config_path = launch_dir / "job.json"
                config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
                process = await asyncio.create_subprocess_exec(
                    str(pier),
                    "run",
                    "--config",
                    str(config_path),
                    "--yes",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=self._pier_process_env(),
                )
                stdout = await self._communicate_with_cancellation_cleanup(process, stdout_path)
            except asyncio.CancelledError as error:
                stdout = getattr(error, "pier_stdout", stdout)
                raise
            except Exception as error:
                launch_error = error
            finally:
                active_error = sys.exception()
                await self._finalize_job_evidence(
                    job_dir,
                    stdout_path,
                    stdout,
                    initial_cancellation=active_error if isinstance(active_error, asyncio.CancelledError) else None,
                )

        trial, trial_error = self._read_partial_trial(job_dir)
        if launch_error is not None:
            raise PierRunError(
                f"Pier launch failed ({type(launch_error).__name__})",
                job_dir=job_dir,
                trial=trial,
            ) from launch_error
        assert process is not None
        if process.returncode != 0:
            raise PierRunError(f"Pier exited with code {process.returncode}", job_dir=job_dir, trial=trial)
        if trial_error is not None:
            raise PierRunError(trial_error, job_dir=job_dir, trial=trial)
        assert trial is not None
        return trial, job_dir

    def _success_response(
        self,
        *,
        record: dict[str, Any],
        task_id: str,
        instruction: str,
        trial: dict[str, Any],
        job_dir: Path,
        harness_error: BaseException | None = None,
        benchmark_task_tree_sha256: str | None = None,
    ) -> DeepSWEVerifyResponse:
        trial_uri = str(trial["trial_uri"])
        trial_dir = _trial_path(trial_uri, job_dir)
        trajectory_path = trial_dir / "agent" / "trajectory.json"
        snapshot = artifact_snapshot(
            trial_dir,
            limits=self._artifact_limits(),
            capture_limits={
                "agent/trajectory.json": self.config.trajectory_max_bytes,
                "artifacts/model.patch": self.config.patch_max_bytes,
            },
        )
        _assert_safe_artifact_paths(snapshot.entry_paths, self._secret_variants())
        artifacts = snapshot.manifest
        trajectory_error: tuple[str, str] | None = None
        try:
            trajectory_bytes = snapshot.captured.get("agent/trajectory.json")
            if trajectory_bytes is None:
                trajectory = None
            else:
                trajectory_value = json.loads(trajectory_bytes.decode("utf-8"))
                trajectory = trajectory_value if isinstance(trajectory_value, dict) else {"value": trajectory_value}
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            trajectory = None
            trajectory_error = (
                "InvalidTrajectory",
                f"Pier ATIF trajectory could not be parsed ({type(error).__name__})",
            )
        if trajectory_error is None:
            if trajectory is None:
                trajectory_error = ("MissingTrajectory", "Pier trial did not retain an ATIF trajectory")
            elif (
                not isinstance(trajectory.get("schema_version"), str)
                or not trajectory["schema_version"].startswith("ATIF-v")
                or not isinstance(trajectory.get("steps"), list)
                or not trajectory["steps"]
                or not all(isinstance(step, dict) for step in trajectory["steps"])
                or not any(step.get("source") == "agent" for step in trajectory["steps"])
            ):
                trajectory_error = (
                    "InvalidTrajectory",
                    "Pier ATIF trajectory must have an ATIF-v schema and at least one structured agent step",
                )
        response = empty_response(task_id=task_id, model_name=self.config.model_name)
        if trajectory is not None and trajectory_error is None:
            try:
                response = response_from_atif(
                    trajectory,
                    task_id=task_id,
                    model_name=self.config.model_name,
                )
            except (AttributeError, TypeError, ValueError) as error:
                trajectory_error = (
                    "InvalidTrajectory",
                    f"Pier ATIF trajectory content is invalid ({type(error).__name__})",
                )
        verifier_result = trial.get("verifier_result")
        rewards = verifier_result.get("rewards", {}) if verifier_result else {}
        raw_reward_value = rewards.get("reward") if verifier_result else None
        try:
            raw_reward = float(raw_reward_value) if raw_reward_value is not None else None
        except (TypeError, ValueError):
            raw_reward = None
        exception = trial.get("exception_info")
        invalid_reward = raw_reward is None or not math.isfinite(raw_reward) or raw_reward not in {0.0, 1.0}
        if harness_error is not None:
            status = "harness_error"
            error_type = type(harness_error).__name__
            error_message = str(harness_error)
        elif exception is not None:
            status = "error"
            error_type = _redact_text(
                str(exception.get("exception_type") or "PierTrialError"),
                self._secret_variants(),
            )
            error_message = exception.get("exception_message")
        elif verifier_result is None:
            status = "error"
            error_type = "MissingVerifierResult"
            error_message = "Pier trial did not produce a verifier result"
        elif invalid_reward:
            status = "error"
            error_type = "InvalidReward"
            error_message = f"DeepSWE verifier reward must be binary and finite, got {raw_reward_value!r}"
        elif trajectory_error is not None:
            status = "harness_error"
            error_type, error_message = trajectory_error
        else:
            status = "success"
            error_type = None
            error_message = None
        if error_message is not None:
            error_message = _redact_text(str(error_message), self._secret_variants())
        reward = raw_reward if status == "success" and raw_reward is not None else 0.0
        model_patch_bytes = snapshot.captured.get("artifacts/model.patch")
        model_patch = model_patch_bytes.decode("utf-8", errors="replace") if model_patch_bytes is not None else None
        record = dict(record)
        record.pop("task_id", None)
        record["responses_create_params"] = dict(record["responses_create_params"])
        record["responses_create_params"]["input"] = [{"role": "user", "content": instruction}]
        sanitized_trial = _sanitize_persisted_value(trial, self._secret_variants())
        sanitized_verifier = _sanitize_persisted_value(verifier_result, self._secret_variants())
        return DeepSWEVerifyResponse(
            **record,
            reward=reward,
            raw_reward=raw_reward,
            response=response,
            task_id=task_id,
            status=status,
            error_type=error_type,
            error_message=error_message,
            benchmark_metadata={
                "benchmark": "datacurve-ai/deep-swe",
                "benchmark_git_url": self.config.benchmark_git_url,
                "benchmark_git_commit": self.config.benchmark_git_commit,
                "benchmark_task_tree_sha256": benchmark_task_tree_sha256,
                "gym_source": dict(self._gym_source_provenance),
                "task_checksum": _sanitize_persisted_value(trial.get("task_checksum"), self._secret_variants()),
                "pier_version": PIER_VERSION,
                "pier_source_url": PIER_SOURCE_URL,
                "pier_source_commit": PIER_SOURCE_COMMIT,
                "pier_direct_url_sha256": pier_direct_url_sha256(),
                "pier_constraints_sha256": pier_constraints_sha256(),
                "pier_runtime_modal_version": PIER_RUNTIME_MODAL_VERSION,
                "pier_runtime_layout_version": PIER_RUNTIME_LAYOUT_VERSION,
                "agent": "claude-code",
                "agent_version": self.config.claude_code_version,
                "agent_install_method": self.config.claude_code_install_method,
                "model": self.config.model_name,
                "job_dir": str(job_dir),
                "trial_uri": _redact_text(trial_uri, self._secret_variants()),
                "sandbox_runtime": self._response_runtime_provenance(job_dir),
                "sandbox_runtime_path": str(job_dir / "gym-runtime-provenance.json"),
            },
            pier_result=sanitized_trial,
            verifier_result=sanitized_verifier,
            raw_rollout={
                "format": trajectory.get("schema_version") if trajectory else None,
                "trajectory": trajectory,
                "trajectory_path": str(trajectory_path),
            },
            model_patch=model_patch,
            artifacts=artifacts,
        )

    def _error_response(
        self,
        *,
        record: dict[str, Any],
        task_id: str,
        error: BaseException,
        instruction: str | None = None,
        job_dir: Path | None = None,
        trial: dict[str, Any] | None = None,
        benchmark_task_tree_sha256: str | None = None,
    ) -> DeepSWEVerifyResponse:
        if isinstance(error, UnsafeArtifactError):
            job_dir = None
            trial = None
            instruction = None
        evidence_allowed = True
        if trial is not None and job_dir is not None and instruction is not None:
            try:
                return self._success_response(
                    record=record,
                    task_id=task_id,
                    instruction=instruction,
                    trial=trial,
                    job_dir=job_dir,
                    harness_error=error,
                    benchmark_task_tree_sha256=benchmark_task_tree_sha256,
                )
            except ArtifactLimitError as partial_error:
                error = partial_error
                trial = None
                evidence_allowed = False
            except Exception as partial_error:
                error = PierRunError(
                    f"{error}; failed to read partial Pier result ({type(partial_error).__name__})",
                    job_dir=job_dir,
                )

        artifacts: list[dict[str, Any]] = []
        snapshot = None
        trajectory_path: Path | None = None
        trajectory_key: str | None = None
        if evidence_allowed and trial is not None and job_dir is not None and isinstance(trial.get("trial_uri"), str):
            try:
                trial_dir = _trial_path(str(trial["trial_uri"]), job_dir)
                trajectory_path = trial_dir / "agent" / "trajectory.json"
                trajectory_key = trajectory_path.relative_to(job_dir).as_posix()
            except ValueError:
                trajectory_path = None
                trajectory_key = None
        if evidence_allowed and job_dir is not None and job_dir.is_dir():
            try:
                snapshot = artifact_snapshot(
                    job_dir,
                    limits=self._artifact_limits(),
                    capture_limits=(
                        {trajectory_key: self.config.trajectory_max_bytes} if trajectory_key is not None else None
                    ),
                )
                _assert_safe_artifact_paths(snapshot.entry_paths, self._secret_variants())
                artifacts = snapshot.manifest
            except ArtifactLimitError as artifact_error:
                error = artifact_error
                trial = None
                evidence_allowed = False
            except OSError as artifact_error:
                error = PierRunError(
                    f"{error}; failed to inventory partial Pier evidence ({type(artifact_error).__name__})",
                    job_dir=job_dir,
                )
                trial = None
                evidence_allowed = False

        record = dict(record)
        record.pop("task_id", None)
        sanitized_trial = _sanitize_persisted_value(trial, self._secret_variants()) if trial is not None else None
        verifier_result = trial.get("verifier_result") if trial is not None else None
        sanitized_verifier = _sanitize_persisted_value(verifier_result, self._secret_variants())
        raw_reward: float | None = None
        if isinstance(verifier_result, dict):
            raw_reward_value = (verifier_result.get("rewards") or {}).get("reward")
            try:
                candidate_reward = float(raw_reward_value)
                if math.isfinite(candidate_reward):
                    raw_reward = candidate_reward
            except (TypeError, ValueError):
                pass
        trajectory: dict[str, Any] | None = None
        if evidence_allowed and trial is not None and trajectory_key is not None and snapshot is not None:
            try:
                trajectory_bytes = snapshot.captured.get(trajectory_key)
                if trajectory_bytes is not None:
                    trajectory_value = json.loads(trajectory_bytes.decode("utf-8"))
                    trajectory = (
                        trajectory_value if isinstance(trajectory_value, dict) else {"value": trajectory_value}
                    )
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                trajectory = None
        return DeepSWEVerifyResponse(
            **record,
            reward=0.0,
            raw_reward=raw_reward,
            response=empty_response(task_id=task_id, model_name=self.config.model_name),
            task_id=task_id,
            status="harness_error",
            error_type=type(error).__name__,
            error_message=_redact_text(str(error), self._secret_variants()),
            benchmark_metadata={
                "benchmark": "datacurve-ai/deep-swe",
                "benchmark_git_url": self.config.benchmark_git_url,
                "benchmark_git_commit": self.config.benchmark_git_commit,
                "benchmark_task_tree_sha256": benchmark_task_tree_sha256,
                "gym_source": dict(self._gym_source_provenance),
                "pier_version": PIER_VERSION,
                "pier_source_url": PIER_SOURCE_URL,
                "pier_source_commit": PIER_SOURCE_COMMIT,
                "pier_direct_url_sha256": pier_direct_url_sha256(),
                "pier_constraints_sha256": pier_constraints_sha256(),
                "pier_runtime_modal_version": PIER_RUNTIME_MODAL_VERSION,
                "pier_runtime_layout_version": PIER_RUNTIME_LAYOUT_VERSION,
                "agent": "claude-code",
                "agent_version": self.config.claude_code_version,
                "agent_install_method": self.config.claude_code_install_method,
                "model": self.config.model_name,
                "sandbox_runtime": self._response_runtime_provenance(job_dir),
                **({"job_dir": str(job_dir)} if job_dir is not None else {}),
            },
            pier_result=sanitized_trial,
            verifier_result=sanitized_verifier,
            raw_rollout={
                "format": trajectory.get("schema_version") if trajectory else None,
                "trajectory": trajectory,
                "trajectory_path": str(trajectory_path) if trajectory_path is not None else None,
            },
            artifacts=artifacts,
        )

    async def run(self, body: DeepSWERunRequest = Body()) -> DeepSWEVerifyResponse:
        record = body.model_dump(mode="json")
        metadata = record.get("verifier_metadata") or {}
        task_id = metadata.get("task_id") or record.get("task_id")
        if not task_id:
            raise ValueError("DeepSWE record is missing verifier_metadata.task_id")

        instruction: str | None = None
        benchmark_task_tree_sha256: str | None = None
        trial: dict[str, Any] | None = None
        job_dir: Path | None = None
        try:
            checkout, benchmark_task_tree_sha256 = await self._checkout_with_digest()
            task_path = resolve_task(checkout, str(task_id))
            instruction = (task_path / "instruction.md").read_text(encoding="utf-8", errors="replace")
            run_id = f"{task_id}-{uuid4().hex}"
            async with self._sem:
                trial, job_dir = await self._run_pier(
                    task_path=task_path,
                    run_id=run_id,
                )
            return await self._assemble_response(
                self._success_response,
                record=record,
                task_id=str(task_id),
                instruction=instruction,
                trial=trial,
                job_dir=job_dir,
                benchmark_task_tree_sha256=benchmark_task_tree_sha256,
            )
        except Exception as error:
            pier_error = error if isinstance(error, PierRunError) else None
            if pier_error is not None:
                trial = pier_error.trial
                job_dir = pier_error.job_dir
            if isinstance(error, UnsafeArtifactError):
                pending_cancellation = error.pending_cancellation
                if job_dir is not None:
                    disposition = await self._isolate_unsafe_job(job_dir)
                    error = UnsafeArtifactError(
                        f"{error}; unsafe retained evidence was {disposition}",
                        pending_cancellation=pending_cancellation,
                    )
                job_dir = None
                trial = None
                instruction = None
                if pending_cancellation is not None:
                    pending_cancellation.add_note(str(error))
                    raise pending_cancellation
            try:
                return await self._assemble_response(
                    self._error_response,
                    record=record,
                    task_id=str(task_id),
                    error=error,
                    instruction=instruction,
                    job_dir=job_dir,
                    trial=trial,
                    benchmark_task_tree_sha256=benchmark_task_tree_sha256,
                )
            except UnsafeArtifactError as unsafe_error:
                pending_cancellation = unsafe_error.pending_cancellation
                if job_dir is not None:
                    disposition = await self._isolate_unsafe_job(job_dir)
                    unsafe_error = UnsafeArtifactError(
                        f"{unsafe_error}; unsafe retained evidence was {disposition}",
                        pending_cancellation=pending_cancellation,
                    )
                if pending_cancellation is not None:
                    pending_cancellation.add_note(str(unsafe_error))
                    raise pending_cancellation
                return await self._assemble_response(
                    self._error_response,
                    record=record,
                    task_id=str(task_id),
                    error=unsafe_error,
                    instruction=None,
                    job_dir=None,
                    trial=None,
                    benchmark_task_tree_sha256=benchmark_task_tree_sha256,
                )


if __name__ == "__main__":
    DeepSWEAgent.run_webserver()
