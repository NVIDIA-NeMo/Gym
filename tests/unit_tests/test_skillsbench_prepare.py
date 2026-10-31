# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract checks for converting official SkillsBench tasks to Harbor tasks."""

import json
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

import benchmarks.skillsbench.prepare as prepare_module
from benchmarks.skillsbench.prepare import _adapt_task


def test_adapt_task_preserves_prompt_and_sandbox_policy(tmp_path: Path) -> None:
    source = tmp_path / "example-task"
    (source / "environment" / "skills" / "example").mkdir(parents=True)
    (source / "verifier").mkdir()
    (source / "oracle").mkdir()
    (source / "task.md").write_text(
        """---
schema_version: '1.3'
metadata: {difficulty: easy, category: test}
verifier: {type: test-script, timeout_sec: 90, env: {CHECK: ready}}
agent: {timeout_sec: 120}
sandbox: {network_mode: no-network, build_timeout_sec: 60, os: linux, cpus: 1, memory_mb: 2048, storage_mb: 4096, gpus: 0}
---
Solve the task without changing this prompt.
"""
    )
    dockerfile = "FROM ubuntu:24.04\nCOPY skills/ /root/.agents/skills/\nRUN echo keep-task-dependencies\n"
    (source / "environment" / "Dockerfile").write_text(dockerfile)
    (source / "environment" / "skills" / "example" / "SKILL.md").write_text("# Example\n")
    (source / "verifier" / "test.sh").write_text("python /verifier/test_outputs.py\n")
    (source / "verifier" / "test_outputs.py").write_text("print('ok')\n")
    (source / "oracle" / "solve.sh").write_text("echo solved\n")

    with_skill = tmp_path / "with-skill"
    _adapt_task(source, with_skill, "with-skill", "registry.example/skillsbench/example:abc123")
    converted = tomllib.loads((with_skill / "task.toml").read_text())
    assert (with_skill / "instruction.md").read_text() == "Solve the task without changing this prompt.\n"
    assert converted["agent"]["timeout_sec"] == 120
    assert converted["verifier"]["timeout_sec"] == 90
    assert converted["verifier"]["env"] == {"CHECK": "ready"}
    assert converted["environment"]["allow_internet"] is False
    assert converted["environment"]["memory_mb"] == 2048
    assert converted["environment"]["docker_image"] == "registry.example/skillsbench/example:abc123"
    assert "COPY skills/ /root/.agents/skills/" in (with_skill / "environment" / "Dockerfile").read_text()
    assert "ln -s /tests /verifier" in (with_skill / "tests" / "test.sh").read_text()
    assert (with_skill / "tests" / "test.original.sh").read_text() == "python /verifier/test_outputs.py\n"
    assert "ln -s /solution /oracle" in (with_skill / "solution" / "solve.sh").read_text()

    no_skill = tmp_path / "no-skill"
    _adapt_task(source, no_skill, "no-skill")
    # Preserve task setup; no-skill removes baked-in skills at agent setup.
    assert (no_skill / "environment" / "Dockerfile").read_text() == dockerfile

    # The image-pinned official revision stores the same resource settings
    # under `environment`, while newer task packages use `sandbox`.
    task_definition = (source / "task.md").read_text().replace("sandbox:", "environment:")
    (source / "task.md").write_text(task_definition)
    pinned = tmp_path / "pinned-revision"
    _adapt_task(source, pinned, "no-skill", "ghcr.io/benchflow-ai/skillsbench-task-env@sha256:abc")
    pinned_toml = tomllib.loads((pinned / "task.toml").read_text())
    assert pinned_toml["environment"]["memory_mb"] == 2048
    assert pinned_toml["environment"]["docker_image"].endswith("@sha256:abc")


def _write_task(task: Path) -> None:
    (task / "environment" / "skills" / "example").mkdir(parents=True)
    (task / "verifier").mkdir()
    (task / "oracle").mkdir()
    (task / "task.md").write_text(
        """---
schema_version: '1.3'
verifier: {type: test-script, timeout_sec: 90}
agent: {timeout_sec: 120}
environment: {network_mode: public, build_timeout_sec: 60, os: linux, cpus: 1, memory_mb: 2048, storage_mb: 4096, gpus: 0}
---
Do it.
"""
    )
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "environment" / "skills" / "example" / "SKILL.md").write_text("# Example\n")
    (task / "verifier" / "test.sh").write_text("true\n")
    (task / "oracle" / "solve.sh").write_text("true\n")


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_prepare_clones_pinned_tasks_and_applies_image_manifest(tmp_path: Path, monkeypatch) -> None:
    upstream = tmp_path / "upstream"
    _write_task(upstream / "tasks" / "alpha")
    (upstream / "README.md").write_text("outside the sparse checkout\n")
    _git("init", "--quiet", cwd=upstream)
    _git("add", ".", cwd=upstream)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "--quiet", "-m", "tasks", cwd=upstream)
    revision = _git("rev-parse", "HEAD", cwd=upstream)

    data_dir = tmp_path / "data"
    monkeypatch.setattr(prepare_module, "SOURCE_REPO", str(upstream))
    monkeypatch.setattr(prepare_module, "SOURCE_REVISION", revision)
    monkeypatch.setattr(prepare_module, "SOURCE_CHECKOUT", data_dir / "source")
    monkeypatch.setattr(prepare_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(prepare_module, "TASKS_DIR", data_dir / "tasks")
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", data_dir / "index.jsonl")
    monkeypatch.setattr(prepare_module, "GYM_ROOT", tmp_path)
    (tmp_path / "images.json").write_text(
        json.dumps(
            {
                "source_revision": revision,
                "tasks": [{"task_name": "alpha", "docker_image": "ghcr.io/x/y@sha256:abc", "success": True}],
            }
        )
    )
    monkeypatch.delenv(prepare_module.SOURCE_ENV_VAR, raising=False)
    monkeypatch.setenv(prepare_module.IMAGE_MANIFEST_ENV_VAR, "images.json")
    monkeypatch.setenv(prepare_module.SKILL_MODE_ENV_VAR, "with-skill")

    output = prepare_module.prepare()

    assert [json.loads(line)["task_name"] for line in output.read_text().splitlines()] == ["alpha"]
    task_toml = tomllib.loads((data_dir / "tasks" / "alpha" / "task.toml").read_text())
    assert task_toml["environment"]["docker_image"] == "ghcr.io/x/y@sha256:abc"
    assert not (data_dir / "source" / "README.md").exists()

    # A second prepare reuses the checkout at the pinned revision instead of cloning again.
    marker = data_dir / "source" / "tasks" / "alpha" / "reused"
    marker.write_text("")
    prepare_module.prepare()
    assert marker.exists()


def test_pinned_image_manifest_covers_pinned_revision() -> None:
    manifest = json.loads((Path(prepare_module.__file__).parent / "images.json").read_text())
    assert manifest["source_revision"] == prepare_module.SOURCE_REVISION
    assert len(manifest["tasks"]) == manifest["total"] == 87
    assert all(task["success"] and "@sha256:" in task["docker_image"] for task in manifest["tasks"])
