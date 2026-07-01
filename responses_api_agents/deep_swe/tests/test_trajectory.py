# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from responses_api_agents.deep_swe import trajectory as trajectory_module
from responses_api_agents.deep_swe.trajectory import (
    ArtifactLimitError,
    ArtifactLimits,
    _content_text,
    artifact_manifest,
    artifact_snapshot,
    empty_response,
    read_bytes,
    read_json,
    read_text,
    response_from_atif,
)


class _DumpableText:
    def model_dump(self, **_: object) -> dict[str, str]:
        return {"type": "text", "text": "dumped"}


def _trajectory() -> dict:
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "session-1",
        "steps": [
            {"step_id": 1, "source": "user", "message": "fix it"},
            {
                "step_id": 2,
                "source": "agent",
                "reasoning_content": "inspect first",
                "message": [{"type": "text", "text": "Running tests"}],
                "tool_calls": [
                    {
                        "tool_call_id": "call-1",
                        "function_name": "bash",
                        "arguments": {"command": "pytest"},
                    }
                ],
                "observation": {
                    "results": [
                        {"source_call_id": "call-1", "content": "1 passed"},
                        {
                            "content": [
                                {
                                    "type": "image",
                                    "source": {"path": "/tmp/result.png"},
                                }
                            ]
                        },
                    ]
                },
            },
        ],
        "final_metrics": {
            "total_prompt_tokens": 100,
            "total_completion_tokens": 20,
            "total_cached_tokens": 7,
        },
    }


def test_response_from_atif_preserves_reasoning_tools_and_usage() -> None:
    response = response_from_atif(_trajectory(), task_id="task", model_name="model")
    dumped = response.model_dump(mode="json")

    assert response.id == "session-1"
    assert dumped["output"][0]["type"] == "reasoning"
    assert any(item["type"] == "function_call" for item in dumped["output"])
    assert any(item["type"] == "function_call_output" for item in dumped["output"])
    assert response.usage is not None
    assert response.usage.total_tokens == 120
    assert response.usage.input_tokens_details.cached_tokens == 7


def test_response_from_atif_falls_back_to_empty() -> None:
    response = response_from_atif(
        {"steps": [{"source": "system", "message": "system"}]},
        task_id="task",
        model_name="model",
    )
    assert response.output[0].type == "message"
    assert response.output[0].content[0].text == ""
    assert empty_response(task_id="other", model_name="model").id == "other"


def test_content_text_handles_structured_and_unknown_content() -> None:
    assert _content_text({"unexpected": True}) == "{'unexpected': True}"
    assert (
        _content_text(
            [
                _DumpableText(),
                {"type": "image", "source": {"path": "/tmp/image.png"}},
                42,
            ]
        )
        == "dumped\n[image: /tmp/image.png]\n42"
    )


def test_read_json_and_artifact_manifest(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "value.json").write_text(json.dumps({"ok": True}))
    (tmp_path / "plain.txt").write_text("payload")

    assert read_json(nested / "value.json") == {"ok": True}
    assert read_json(tmp_path / "missing.json") is None
    manifest = artifact_manifest(tmp_path)
    assert [item["path"] for item in manifest] == ["nested/value.json", "plain.txt"]
    assert all(len(item["sha256"]) == 64 and item["bytes"] > 0 for item in manifest)

    (tmp_path / "plain-link.txt").symlink_to(tmp_path / "plain.txt")
    with pytest.raises(ArtifactLimitError, match="entry is unsafe"):
        artifact_manifest(tmp_path)


def test_bounded_reads_reject_oversized_files_before_loading(tmp_path: Path) -> None:
    artifact = tmp_path / "large.json"
    artifact.write_bytes(b"{" + b"x" * 32 + b"}")

    with pytest.raises(ArtifactLimitError, match="ATIF trajectory exceeds 16 bytes"):
        read_json(artifact, max_bytes=16, description="ATIF trajectory")
    with pytest.raises(ArtifactLimitError, match="model patch exceeds 16 bytes"):
        read_text(artifact, max_bytes=16, description="model patch")
    with pytest.raises(ArtifactLimitError, match="artifact exceeds 16 bytes"):
        read_bytes(artifact, max_bytes=16)

    link = tmp_path / "linked.json"
    link.symlink_to(artifact)
    with pytest.raises(ArtifactLimitError, match="is not a regular file"):
        read_bytes(link, max_bytes=1024)

    outside = tmp_path / "outside.json"
    outside.write_text("safe")
    hard_link = tmp_path / "hard-link.json"
    os.link(outside, hard_link)
    with pytest.raises(ArtifactLimitError, match="unsafe hard links"):
        read_bytes(hard_link, max_bytes=1024)


def test_read_bytes_detects_same_size_path_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"original")
    displaced = tmp_path / "displaced.bin"
    real_read = os.read
    replaced = False

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        chunk = real_read(descriptor, size)
        if chunk and not replaced:
            replaced = True
            artifact.rename(displaced)
            artifact.write_bytes(b"replaced")
        return chunk

    monkeypatch.setattr("responses_api_agents.deep_swe.trajectory.os.read", replacing_read)
    with pytest.raises(ArtifactLimitError, match="changed while reading"):
        read_bytes(artifact, max_bytes=16)


def test_read_bytes_rejects_nonregular_growth_and_metadata_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ArtifactLimitError, match="not a regular file"):
        read_bytes(directory, max_bytes=16)

    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"x")
    real_read = os.read
    with monkeypatch.context() as context:
        context.setattr(trajectory_module.os, "read", lambda descriptor, _: real_read(descriptor, 1) + b"x")
        with pytest.raises(ArtifactLimitError, match="exceeds 1 bytes"):
            read_bytes(artifact, max_bytes=1)

    opened = artifact.lstat()
    with monkeypatch.context() as context:
        context.setattr(
            Path,
            "lstat",
            lambda self: (_ for _ in ()).throw(OSError("gone")) if self == artifact else opened,
        )
        with pytest.raises(ArtifactLimitError, match="changed while reading"):
            trajectory_module._verify_open_path(artifact, opened, description="artifact")

    changed = SimpleNamespace(
        st_dev=opened.st_dev,
        st_ino=opened.st_ino,
        st_size=opened.st_size,
        st_mtime_ns=opened.st_mtime_ns,
        st_ctime_ns=opened.st_ctime_ns + 1,
        st_nlink=opened.st_nlink,
    )
    with monkeypatch.context() as context:
        context.setattr(Path, "lstat", lambda self: changed if self == artifact else opened)
        with pytest.raises(ArtifactLimitError, match="changed while reading"):
            trajectory_module._verify_open_path(artifact, opened, description="artifact")


def test_artifact_snapshot_hashes_the_exact_captured_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "trajectory.json"
    contents = b'{"schema_version":"ATIF-v1.7","steps":[]}'
    artifact.write_bytes(contents)

    snapshot = artifact_snapshot(
        tmp_path,
        capture_limits={"trajectory.json": len(contents)},
    )

    assert snapshot.captured == {"trajectory.json": contents}
    assert snapshot.manifest == [
        {
            "path": "trajectory.json",
            "bytes": len(contents),
            "sha256": hashlib.sha256(contents).hexdigest(),
        }
    ]


def test_artifact_snapshot_bounds_all_directory_entries(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"empty-{index}").mkdir()

    with pytest.raises(ArtifactLimitError, match="entry count exceeds 2"):
        artifact_snapshot(
            tmp_path,
            limits=ArtifactLimits(
                max_files=10,
                max_file_bytes=16,
                max_total_bytes=16,
                max_entries=2,
            ),
        )


def test_artifact_snapshot_rejects_filesystem_races(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def changed_metadata(metadata: os.stat_result, **overrides: int) -> SimpleNamespace:
        values = {
            "st_dev": metadata.st_dev,
            "st_ino": metadata.st_ino,
            "st_size": metadata.st_size,
            "st_mtime_ns": metadata.st_mtime_ns,
            "st_ctime_ns": metadata.st_ctime_ns,
            "st_nlink": metadata.st_nlink,
            "st_mode": metadata.st_mode,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    with pytest.raises(ArtifactLimitError, match="root is not a real directory"):
        artifact_snapshot(tmp_path / "missing")

    hard_root = tmp_path / "hard-root"
    hard_root.mkdir()
    source = tmp_path / "source"
    source.write_bytes(b"x")
    os.link(source, hard_root / "linked")
    with pytest.raises(ArtifactLimitError, match="unsafe hard links"):
        artifact_snapshot(hard_root)

    capture_root = tmp_path / "capture-root"
    capture_root.mkdir()
    (capture_root / "captured").write_bytes(b"xx")
    with pytest.raises(ArtifactLimitError, match="captured artifact exceeds 1 bytes"):
        artifact_snapshot(capture_root, capture_limits={"captured": 1})

    real_open = trajectory_module.os.open
    directory_root = tmp_path / "directory-root"
    (directory_root / "nested").mkdir(parents=True)

    def fail_child_open(path: Any, *args: Any, dir_fd: int | None = None, **kwargs: Any) -> int:
        if path == "nested" and dir_fd is not None:
            raise OSError("swapped")
        return real_open(path, *args, dir_fd=dir_fd, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(trajectory_module.os, "open", fail_child_open)
        with pytest.raises(ArtifactLimitError, match="directory is unsafe"):
            artifact_snapshot(directory_root)

    file_root = tmp_path / "file-root"
    file_root.mkdir()
    file_path = file_root / "file"
    file_path.write_bytes(b"x")

    def fail_file_open(path: Any, *args: Any, dir_fd: int | None = None, **kwargs: Any) -> int:
        if path == "file" and dir_fd is not None:
            raise OSError("swapped")
        return real_open(path, *args, dir_fd=dir_fd, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(trajectory_module.os, "open", fail_file_open)
        with pytest.raises(ArtifactLimitError, match="artifact is unsafe"):
            artifact_snapshot(file_root)

    real_fstat = trajectory_module.os.fstat
    fstat_calls = 0

    def mismatch_before_hash(descriptor: int) -> os.stat_result | SimpleNamespace:
        nonlocal fstat_calls
        fstat_calls += 1
        metadata = real_fstat(descriptor)
        if fstat_calls == 2:
            return changed_metadata(metadata, st_ino=metadata.st_ino + 1)
        return metadata

    with monkeypatch.context() as context:
        context.setattr(trajectory_module.os, "fstat", mismatch_before_hash)
        with pytest.raises(ArtifactLimitError, match="changed before hashing"):
            artifact_snapshot(file_root)

    def oversized_read(_: int, __: int) -> bytes:
        return b"xx"

    with monkeypatch.context() as context:
        context.setattr(trajectory_module.os, "read", oversized_read)
        with pytest.raises(ArtifactLimitError, match="total exceeds 1 bytes while hashing"):
            artifact_snapshot(
                file_root,
                limits=ArtifactLimits(max_files=1, max_file_bytes=4, max_total_bytes=1),
            )

    with monkeypatch.context() as context:
        context.setattr(trajectory_module.os, "read", oversized_read)
        with pytest.raises(ArtifactLimitError, match="captured artifact exceeds 1 bytes"):
            artifact_snapshot(
                file_root,
                limits=ArtifactLimits(max_files=1, max_file_bytes=4, max_total_bytes=4),
                capture_limits={"file": 1},
            )

    fstat_calls = 0

    def mismatch_after_hash(descriptor: int) -> os.stat_result | SimpleNamespace:
        nonlocal fstat_calls
        fstat_calls += 1
        metadata = real_fstat(descriptor)
        if fstat_calls == 3:
            return changed_metadata(metadata, st_ctime_ns=metadata.st_ctime_ns + 1)
        return metadata

    with monkeypatch.context() as context:
        context.setattr(trajectory_module.os, "fstat", mismatch_after_hash)
        with pytest.raises(ArtifactLimitError, match="changed while hashing"):
            artifact_snapshot(file_root)

    real_stat = trajectory_module.os.stat

    def mismatch_current(path: Any, *args: Any, **kwargs: Any) -> SimpleNamespace:
        metadata = real_stat(path, *args, **kwargs)
        return changed_metadata(metadata, st_ctime_ns=metadata.st_ctime_ns + 1)

    with monkeypatch.context() as context:
        context.setattr(trajectory_module.os, "stat", mismatch_current)
        with pytest.raises(ArtifactLimitError, match="changed while hashing"):
            artifact_snapshot(file_root)

    with monkeypatch.context() as context:
        context.setattr(trajectory_module.os, "read", lambda *_args, **_kwargs: b"")
        with pytest.raises(ArtifactLimitError, match="changed while hashing"):
            artifact_snapshot(file_root)

    root_metadata = file_root.lstat()
    with monkeypatch.context() as context:
        context.setattr(
            Path,
            "lstat",
            lambda self: (
                changed_metadata(root_metadata, st_ctime_ns=root_metadata.st_ctime_ns + 1)
                if self == file_root
                else self.stat()
            ),
        )
        with pytest.raises(ArtifactLimitError, match="root changed while walking"):
            artifact_snapshot(file_root)

    file_metadata = file_path.lstat()
    with monkeypatch.context() as context:
        context.setattr(
            Path,
            "lstat",
            lambda self: (
                changed_metadata(file_metadata, st_ctime_ns=file_metadata.st_ctime_ns + 1)
                if self == file_path
                else self.stat()
            ),
        )
        with pytest.raises(ArtifactLimitError, match="artifact changed while walking"):
            artifact_snapshot(file_root)


@pytest.mark.parametrize(
    ("files", "limits", "message"),
    [
        (
            {"one": b"1", "two": b"2"},
            ArtifactLimits(max_files=1, max_file_bytes=8, max_total_bytes=16),
            "artifact count exceeds 1",
        ),
        (
            {"large": b"12345"},
            ArtifactLimits(max_files=2, max_file_bytes=4, max_total_bytes=16),
            "artifact exceeds 4 bytes",
        ),
        (
            {"one": b"123", "two": b"456"},
            ArtifactLimits(max_files=2, max_file_bytes=4, max_total_bytes=5),
            "artifact total exceeds 5 bytes",
        ),
    ],
)
def test_artifact_manifest_enforces_count_per_file_and_total_limits(
    tmp_path: Path,
    files: dict[str, bytes],
    limits: ArtifactLimits,
    message: str,
) -> None:
    for name, contents in files.items():
        (tmp_path / name).write_bytes(contents)

    with pytest.raises(ArtifactLimitError, match=message):
        artifact_manifest(tmp_path, limits=limits)
