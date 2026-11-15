# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task staging separates trusted rubrics from model files and rejects unsafe output paths."""

import json
import stat
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
from resources_servers.gdpval_disaggregated.artifact_io import ArtifactIO
from resources_servers.gdpval_disaggregated.stage_gdpval_task import stage


def test_stage_hides_rubric_from_agent(tmp_path: Path):
    references = tmp_path / "references"
    references.mkdir()
    (references / "brief.txt").write_text("Inputs")
    task = stage(
        {
            "task_id": "sample",
            "prompt": "Write a report",
            "reference_files": ["brief.txt"],
            "rubric_pretty": "TRUSTED_RUBRIC",
            "rubric_json": {},
        },
        references,
        tmp_path / "task",
        "synthetic-v1",
    )
    assert (task / "agent/reference_files/brief.txt").read_text() == "Inputs"
    assert "/workspace/deliverables" in (task / "agent/prompt.txt").read_text()
    assert not any("TRUSTED_RUBRIC" in path.read_text() for path in (task / "agent").rglob("*") if path.is_file())
    assert json.loads((task / "judge.json").read_text())["rubric_pretty"] == "TRUSTED_RUBRIC"


@pytest.mark.parametrize("name", ["../secret", "/absolute", "a/../../secret", "a\\b"])
def test_stage_rejects_traversal(tmp_path: Path, name):
    with pytest.raises(ValueError, match="reference filename|Reference filename"):
        stage(
            {"task_id": "sample", "prompt": "test", "reference_files": [name], "rubric_pretty": "test"},
            tmp_path,
            tmp_path / "task",
            "synthetic-v1",
        )


@pytest.mark.parametrize("name", ["../outside", "/outside", "trace/../../outside", "trace\\outside"])
def test_output_archive_rejects_traversal(tmp_path: Path, name):
    archive = tmp_path / "output.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(name, "malicious")
    with pytest.raises(ValueError, match="Unsafe sandbox output path"):
        ArtifactIO._extract_results(archive, tmp_path / "task")
    assert not (tmp_path / "outside").exists()


def test_output_archive_rejects_symlink(tmp_path: Path):
    archive = tmp_path / "output.zip"
    member = zipfile.ZipInfo("workspace/deliverables/link")
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, "/etc/passwd")
    with pytest.raises(ValueError, match="symlink"):
        ArtifactIO._extract_results(archive, tmp_path / "task")


async def test_exec_error_is_not_a_successful_upload(tmp_path: Path):
    (tmp_path / "file.txt").write_text("data")
    sandbox = MagicMock(spec=AsyncSandbox)
    sandbox.exec = AsyncMock(
        return_value=SandboxExecResult(return_code=0, error_type="timeout", stdout="", stderr="timeout")
    )
    with pytest.raises(RuntimeError, match="create sandbox input directories"):
        await ArtifactIO._upload_tree(sandbox, tmp_path, "/workspace")
    sandbox.upload.assert_not_called()
