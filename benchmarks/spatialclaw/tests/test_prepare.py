# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from benchmarks.spatialclaw import prepare


def test_instruction_matches_native_runner_choice_order() -> None:
    benchmark = SimpleNamespace(data_specific_prompt="Answer with one letter.")
    sample = SimpleNamespace(question="Where?", choices={"A": "Left", "B": "Right"})
    assert prepare._instruction(benchmark, sample) == "Where?\n\nAnswer with one letter.\nA. Left\nB. Right"


def test_sample_metadata_is_relative_and_preserves_groups(tmp_path) -> None:
    data_root = tmp_path / "data"
    first = data_root / "bench" / "first.png"
    second = data_root / "bench" / "second.png"
    reference = data_root / "bench" / "reference.png"
    for path in (first, second, reference):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    sample = SimpleNamespace(
        video=None,
        video_paths=None,
        images=[str(first), str(second)],
        image_groups=[[str(first)], [str(second)]],
        ref_images=[str(reference)],
        frame_indices_groups=[[0], [0]],
        fps_per_video=[1.0, 1.0],
        total_frames_per_video=[1, 1],
        duration_per_video=[0.0, 0.0],
        video_names=["first.mp4", "second.mp4"],
    )
    metadata = prepare._sample_metadata(sample, data_root.resolve())
    assert metadata["image_groups"] == '[["bench/first.png"],["bench/second.png"]]'
    assert metadata["ref_image_paths"] == '["bench/reference.png"]'
    assert metadata["frame_indices_groups"] == "[[0],[0]]"


def test_portable_path_rejects_media_outside_data_root(tmp_path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"image")
    with pytest.raises(ValueError, match="outside data_root"):
        prepare._portable_path(str(outside), data_root.resolve())


def test_server_venv_setup_is_locked_and_reused(monkeypatch, tmp_path) -> None:
    server_dir = tmp_path / "resources_servers" / "spatialclaw"
    server_dir.mkdir(parents=True)
    calls = []

    def fake_run(command, *, check, cwd):
        assert check is True
        assert cwd == server_dir
        calls.append(command)
        if command[1] == "venv":
            python = server_dir / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()

    monkeypatch.setattr(prepare, "SERVER_DIR", server_dir)
    monkeypatch.setattr(prepare.subprocess, "run", fake_run)
    first = prepare._ensure_server_venv()
    second = prepare._ensure_server_venv()
    assert first == second == server_dir / ".venv" / "bin" / "python"
    assert [command[1] for command in calls] == ["venv", "pip"]
    assert calls[0][3] == prepare.sys.executable
    assert (server_dir / ".venv" / ".spatialclaw-requirements-installed").is_file()
