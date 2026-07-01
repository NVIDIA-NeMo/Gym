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

"""ATIF trajectory conversion and artifact inventory helpers."""

import hashlib
import json
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)


@dataclass(frozen=True)
class ArtifactLimits:
    """Hard limits for retained evidence inspected during response assembly."""

    max_files: int
    max_file_bytes: int
    max_total_bytes: int
    max_entries: int | None = None

    @property
    def entry_limit(self) -> int:
        """Bound directory and special-entry traversal as well as regular files."""

        return self.max_entries if self.max_entries is not None else max(16, self.max_files * 4)


@dataclass(frozen=True)
class ArtifactSnapshot:
    """One descriptor-stable view of retained evidence."""

    manifest: list[dict[str, Any]]
    captured: dict[str, bytes]
    entry_paths: tuple[str, ...]


class ArtifactLimitError(RuntimeError):
    """Raised before an artifact operation would exceed its configured budget."""


def _file_open_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _directory_open_flags() -> int:
    return _file_open_flags() | getattr(os, "O_DIRECTORY", 0)


def _metadata_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _validate_regular_metadata(metadata: os.stat_result, path: Path, *, description: str) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise ArtifactLimitError(f"DeepSWE {description} is not a regular file: {path.name}")
    if metadata.st_nlink != 1:
        raise ArtifactLimitError(f"DeepSWE {description} has unsafe hard links: {path.name}")


def _verify_open_path(path: Path, opened: os.stat_result, *, description: str) -> None:
    try:
        current = path.lstat()
    except OSError as error:
        raise ArtifactLimitError(f"DeepSWE {description} changed while reading: {path.name}") from error
    if _metadata_signature(current) != _metadata_signature(opened):
        raise ArtifactLimitError(f"DeepSWE {description} changed while reading: {path.name}")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[str] = []
    for item in content:
        if hasattr(item, "model_dump"):
            item = item.model_dump(mode="json", exclude_none=True)
        if isinstance(item, dict):
            if item.get("type") == "text" and item.get("text"):
                parts.append(str(item["text"]))
            elif item.get("type") == "image":
                parts.append(f"[image: {item.get('source', {}).get('path', 'unknown')}]")
        elif item is not None:
            parts.append(str(item))
    return "\n".join(parts)


def empty_response(*, task_id: str, model_name: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id=task_id,
        created_at=time.time(),
        model=model_name,
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg_0",
                content=[NeMoGymResponseOutputText(text="", annotations=[])],
            )
        ],
        parallel_tool_calls=False,
        tools=[],
        tool_choice="auto",
    )


def response_from_atif(
    trajectory: dict[str, Any],
    *,
    task_id: str,
    model_name: str,
) -> NeMoGymResponse:
    output: list[Any] = []
    for step in trajectory.get("steps") or []:
        if step.get("source") != "agent":
            continue
        step_id = step.get("step_id", len(output) + 1)
        reasoning = step.get("reasoning_content")
        if reasoning:
            output.append(
                NeMoGymResponseReasoningItem(
                    id=f"rs_{step_id}",
                    summary=[NeMoGymSummary(text=str(reasoning), type="summary_text")],
                )
            )
        message = _content_text(step.get("message"))
        if message:
            output.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{step_id}",
                    content=[NeMoGymResponseOutputText(text=message, annotations=[])],
                )
            )
        for call in step.get("tool_calls") or []:
            call_id = str(call.get("tool_call_id") or f"call_{step_id}_{len(output)}")
            output.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=json.dumps(call.get("arguments") or {}, ensure_ascii=False),
                    call_id=call_id,
                    name=str(call.get("function_name") or ""),
                    id=call_id,
                    status="completed",
                )
            )
        observation = step.get("observation") or {}
        for index, result in enumerate(observation.get("results") or []):
            call_id = str(result.get("source_call_id") or f"observation_{step_id}_{index}")
            output.append(
                NeMoGymFunctionCallOutput(
                    call_id=call_id,
                    output=_content_text(result.get("content")),
                    id=f"out_{call_id}",
                    status="completed",
                )
            )

    if not output:
        return empty_response(task_id=task_id, model_name=model_name)

    metrics = trajectory.get("final_metrics") or {}
    input_tokens = int(metrics.get("total_prompt_tokens") or 0)
    output_tokens = int(metrics.get("total_completion_tokens") or 0)
    usage = NeMoGymResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=NeMoGymResponseInputTokensDetails(
            cached_tokens=int(metrics.get("total_cached_tokens") or 0)
        ),
        output_tokens=output_tokens,
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
        total_tokens=input_tokens + output_tokens,
    )
    return NeMoGymResponse(
        id=str(trajectory.get("session_id") or task_id),
        created_at=time.time(),
        model=model_name,
        object="response",
        output=output,
        parallel_tool_calls=False,
        tools=[],
        tool_choice="auto",
        usage=usage,
    )


def read_bytes(path: Path, *, max_bytes: int, description: str = "artifact") -> bytes:
    """Read one file without ever allocating beyond ``max_bytes + 1`` bytes."""
    try:
        descriptor = os.open(path, _file_open_flags())
    except OSError as error:
        raise ArtifactLimitError(f"DeepSWE {description} is not a regular file: {path.name}") from error
    try:
        opened = os.fstat(descriptor)
        _validate_regular_metadata(opened, path, description=description)
        if opened.st_size > max_bytes:
            raise ArtifactLimitError(f"DeepSWE {description} exceeds {max_bytes} bytes: {path.name}")
        value = bytearray()
        while chunk := os.read(descriptor, min(1024 * 1024, max_bytes - len(value) + 1)):
            value.extend(chunk)
            if len(value) > max_bytes:
                raise ArtifactLimitError(f"DeepSWE {description} exceeds {max_bytes} bytes: {path.name}")
        after = os.fstat(descriptor)
        if _metadata_signature(after) != _metadata_signature(opened):
            raise ArtifactLimitError(f"DeepSWE {description} changed while reading: {path.name}")
        _verify_open_path(path, opened, description=description)
        return bytes(value)
    finally:
        os.close(descriptor)


def read_text(path: Path, *, max_bytes: int, description: str = "artifact", errors: str = "strict") -> str:
    return read_bytes(path, max_bytes=max_bytes, description=description).decode("utf-8", errors=errors)


def read_json(
    path: Path, *, max_bytes: int | None = None, description: str = "JSON artifact"
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    text = (
        path.read_text(encoding="utf-8")
        if max_bytes is None
        else read_text(path, max_bytes=max_bytes, description=description)
    )
    value = json.loads(text)
    return value if isinstance(value, dict) else {"value": value}


def artifact_snapshot(
    root: Path,
    *,
    limits: ArtifactLimits | None = None,
    capture_limits: dict[str, int] | None = None,
) -> ArtifactSnapshot:
    """Capture a bounded manifest and selected bytes from one stable descriptor walk."""

    if limits is None:
        limits = ArtifactLimits(max_files=2048, max_file_bytes=64 * 1024 * 1024, max_total_bytes=256 * 1024 * 1024)
    capture_limits = capture_limits or {}

    try:
        root_descriptor = os.open(root, _directory_open_flags())
    except OSError as error:
        raise ArtifactLimitError(f"DeepSWE artifact root is not a real directory: {root}") from error
    root_metadata = os.fstat(root_descriptor)

    stack: list[tuple[int, str]] = [(root_descriptor, "")]
    manifest: list[dict[str, Any]] = []
    captured: dict[str, bytes] = {}
    entry_paths: list[str] = []
    entry_signatures: dict[str, tuple[int, int, int, int, int, int]] = {}
    entry_count = 0
    file_count = 0
    declared_total = 0
    actual_total = 0
    try:
        while stack:
            directory_descriptor, prefix = stack.pop()
            try:
                with os.scandir(directory_descriptor) as entries:
                    for entry in entries:
                        entry_count += 1
                        if entry_count > limits.entry_limit:
                            raise ArtifactLimitError(f"DeepSWE artifact entry count exceeds {limits.entry_limit}")
                        relative = f"{prefix}/{entry.name}" if prefix else entry.name
                        entry_paths.append(relative)
                        metadata = entry.stat(follow_symlinks=False)
                        if stat.S_ISDIR(metadata.st_mode):
                            try:
                                child_descriptor = os.open(
                                    entry.name,
                                    _directory_open_flags(),
                                    dir_fd=directory_descriptor,
                                )
                            except OSError as error:
                                raise ArtifactLimitError(
                                    f"DeepSWE artifact directory is unsafe: {relative}"
                                ) from error
                            child_metadata = os.fstat(child_descriptor)
                            if not stat.S_ISDIR(child_metadata.st_mode) or (
                                child_metadata.st_dev,
                                child_metadata.st_ino,
                            ) != (metadata.st_dev, metadata.st_ino):
                                os.close(child_descriptor)
                                raise ArtifactLimitError(
                                    f"DeepSWE artifact directory changed while walking: {relative}"
                                )
                            entry_signatures[relative] = _metadata_signature(child_metadata)
                            stack.append((child_descriptor, relative))
                            continue
                        if not stat.S_ISREG(metadata.st_mode):
                            raise ArtifactLimitError(f"DeepSWE artifact entry is unsafe: {relative}")
                        if metadata.st_nlink != 1:
                            raise ArtifactLimitError(f"DeepSWE artifact has unsafe hard links: {relative}")

                        file_count += 1
                        if file_count > limits.max_files:
                            raise ArtifactLimitError(f"DeepSWE artifact count exceeds {limits.max_files}")
                        if metadata.st_size > limits.max_file_bytes:
                            raise ArtifactLimitError(
                                f"DeepSWE artifact exceeds {limits.max_file_bytes} bytes: {relative}"
                            )
                        capture_limit = capture_limits.get(relative)
                        if capture_limit is not None and metadata.st_size > capture_limit:
                            raise ArtifactLimitError(
                                f"DeepSWE captured artifact exceeds {capture_limit} bytes: {relative}"
                            )
                        declared_total += metadata.st_size
                        if declared_total > limits.max_total_bytes:
                            raise ArtifactLimitError(f"DeepSWE artifact total exceeds {limits.max_total_bytes} bytes")

                        try:
                            file_descriptor = os.open(
                                entry.name,
                                _file_open_flags(),
                                dir_fd=directory_descriptor,
                            )
                        except OSError as error:
                            raise ArtifactLimitError(f"DeepSWE artifact is unsafe: {relative}") from error
                        try:
                            opened = os.fstat(file_descriptor)
                            if (
                                not stat.S_ISREG(opened.st_mode)
                                or opened.st_nlink != 1
                                or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
                            ):
                                raise ArtifactLimitError(f"DeepSWE artifact changed before hashing: {relative}")
                            digest = hashlib.sha256()
                            size = 0
                            capture = bytearray() if capture_limit is not None else None
                            while chunk := os.read(
                                file_descriptor,
                                min(1024 * 1024, limits.max_file_bytes - size + 1),
                            ):
                                size += len(chunk)
                                if size > limits.max_file_bytes:
                                    raise ArtifactLimitError(
                                        "DeepSWE artifact exceeds "
                                        f"{limits.max_file_bytes} bytes while hashing: {relative}"
                                    )
                                actual_total += len(chunk)
                                if actual_total > limits.max_total_bytes:
                                    raise ArtifactLimitError(
                                        f"DeepSWE artifact total exceeds {limits.max_total_bytes} bytes while hashing"
                                    )
                                digest.update(chunk)
                                if capture is not None:
                                    capture.extend(chunk)
                                    assert capture_limit is not None
                                    if len(capture) > capture_limit:
                                        raise ArtifactLimitError(
                                            f"DeepSWE captured artifact exceeds {capture_limit} bytes: {relative}"
                                        )
                            after = os.fstat(file_descriptor)
                            if _metadata_signature(after) != _metadata_signature(opened):
                                raise ArtifactLimitError(f"DeepSWE artifact changed while hashing: {relative}")
                            current = os.stat(
                                entry.name,
                                dir_fd=directory_descriptor,
                                follow_symlinks=False,
                            )
                            if _metadata_signature(current) != _metadata_signature(opened):
                                raise ArtifactLimitError(f"DeepSWE artifact changed while hashing: {relative}")
                        finally:
                            os.close(file_descriptor)
                        if size != metadata.st_size:
                            raise ArtifactLimitError(f"DeepSWE artifact changed while hashing: {relative}")
                        entry_signatures[relative] = _metadata_signature(opened)
                        manifest.append(
                            {
                                "path": relative,
                                "bytes": size,
                                "sha256": digest.hexdigest(),
                            }
                        )
                        if capture is not None:
                            captured[relative] = bytes(capture)
            finally:
                os.close(directory_descriptor)
    finally:
        for descriptor, _ in stack:
            os.close(descriptor)

    try:
        current_root = root.lstat()
    except OSError as error:
        raise ArtifactLimitError("DeepSWE artifact root changed while walking") from error
    if _metadata_signature(current_root) != _metadata_signature(root_metadata):
        raise ArtifactLimitError("DeepSWE artifact root changed while walking")
    for relative, signature in entry_signatures.items():
        try:
            current = (root / relative).lstat()
        except OSError as error:
            raise ArtifactLimitError(f"DeepSWE artifact changed while walking: {relative}") from error
        if _metadata_signature(current) != signature:
            raise ArtifactLimitError(f"DeepSWE artifact changed while walking: {relative}")

    manifest.sort(key=lambda item: item["path"])
    return ArtifactSnapshot(
        manifest=manifest,
        captured=captured,
        entry_paths=tuple(sorted(entry_paths)),
    )


def artifact_manifest(root: Path, *, limits: ArtifactLimits | None = None) -> list[dict[str, Any]]:
    return artifact_snapshot(root, limits=limits).manifest


__all__ = [
    "ArtifactLimitError",
    "ArtifactLimits",
    "ArtifactSnapshot",
    "artifact_manifest",
    "artifact_snapshot",
    "empty_response",
    "read_bytes",
    "read_json",
    "read_text",
    "response_from_atif",
]
