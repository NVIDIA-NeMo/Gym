# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract checks for converting official SkillsBench tasks to dataset entries for the `harbor` environment server."""

import json
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

import benchmarks.skillsbench.prepare as source_module
import benchmarks.skillsbench.prepare_harbor_env as prepare_module
from benchmarks.skillsbench.prepare_harbor_env import (
    AGENT_SKILL_PATHS,
    DATASETS,
    ENTRIES_DIR,
    ORACLE_WRAPPER,
    SKILLS_DIR,
    _adapt_task,
)
from nemo_gym.data.tasks.harbor.dataset_config import read_dataset_config
from nemo_gym.data.tasks.harbor.entries import find_entry
from nemo_gym.data.tasks.harbor.models import HarborTaskConfig


def _write_source(source: Path, dockerfile: str = "FROM ubuntu:24.04\nRUN echo keep-task-dependencies\n") -> None:
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
    (source / "environment" / "Dockerfile").write_text(dockerfile)
    (source / "environment" / "skills" / "example" / "SKILL.md").write_text("# Example\n")
    (source / "verifier" / "test.sh").write_text("python /verifier/test_outputs.py\n")
    (source / "verifier" / "test_outputs.py").write_text("print('ok')\n")
    (source / "oracle" / "solve.sh").write_text("echo solved\n")


def test_adapt_task_preserves_prompt_and_sandbox_policy(tmp_path: Path) -> None:
    source = tmp_path / "example-task"
    _write_source(source)

    with_skill = tmp_path / "with-skill"
    _adapt_task(source, with_skill, "with-skill", "registry.example/skillsbench/example:abc123")
    converted = tomllib.loads((with_skill / "task.toml").read_text())
    assert (with_skill / "instruction.md").read_text() == "Solve the task without changing this prompt.\n"
    assert converted["agent"]["timeout_sec"] == 120
    assert converted["verifier"]["timeout_sec"] == 90
    assert converted["verifier"]["env"] == {"CHECK": "ready"}
    assert converted["environment"]["network_mode"] == "no-network"
    assert converted["environment"]["memory_mb"] == 2048
    assert converted["environment"]["docker_image"] == "registry.example/skillsbench/example:abc123"
    # The skills stay in the build context; the environment server uploads them to skills_dir at seed.
    assert converted["environment"]["skills_dir"] == SKILLS_DIR
    assert (with_skill / "environment" / "skills" / "example" / "SKILL.md").is_file()
    assert (with_skill / "environment" / "Dockerfile").read_text() == (
        source / "environment" / "Dockerfile"
    ).read_text()
    assert "ln -s /tests /verifier" in (with_skill / "tests" / "test.sh").read_text()
    assert (with_skill / "tests" / "test.original.sh").read_text() == "python /verifier/test_outputs.py\n"
    # The oracle also finds the skills at /app/skills while solve.sh runs, and the link is gone after it.
    assert (with_skill / "solution" / "solve.sh").read_text() == ORACLE_WRAPPER
    assert "ln -s /solution /oracle" in ORACLE_WRAPPER
    assert "ln -s /skills /app/skills" in ORACLE_WRAPPER and "trap 'rm -f /app/skills' EXIT" in ORACLE_WRAPPER
    assert (with_skill / "solution" / "solve.original.sh").read_text() == "echo solved\n"
    # Gym's Harbor loader reads the file with its own field names, nothing deprecated.
    config = HarborTaskConfig.model_validate(converted)
    assert config.environment.skills_dir == SKILLS_DIR
    assert config.environment.network_mode == "no-network"

    no_skill = tmp_path / "no-skill"
    _adapt_task(source, no_skill, "no-skill", "registry.example/skillsbench/example:abc123")
    no_skill_toml = tomllib.loads((no_skill / "task.toml").read_text())
    assert "skills_dir" not in no_skill_toml["environment"]
    assert no_skill_toml["environment"]["docker_image"] == "registry.example/skillsbench/example:abc123"
    # Nothing builds from the skills, so the no-skill task does not carry them, and its oracle gets no link.
    assert not (no_skill / "environment" / "skills").exists()
    assert "/app/skills" not in (no_skill / "solution" / "solve.sh").read_text()

    # The image-pinned official revision stores the same resource settings
    # under `environment`, while newer task packages use `sandbox`.
    task_definition = (source / "task.md").read_text().replace("sandbox:", "environment:")
    (source / "task.md").write_text(task_definition)
    pinned = tmp_path / "pinned-revision"
    _adapt_task(source, pinned, "with-skill", "ghcr.io/benchflow-ai/skillsbench-task-env@sha256:abc")
    pinned_toml = tomllib.loads((pinned / "task.toml").read_text())
    assert pinned_toml["environment"]["memory_mb"] == 2048
    assert pinned_toml["environment"]["docker_image"].endswith("@sha256:abc")


def test_no_skill_clears_skills_an_image_builds_in(tmp_path: Path) -> None:
    source = tmp_path / "bakes-skills"
    dockerfile = "FROM ubuntu:24.04\nWORKDIR /root\nCOPY skills /root/.claude/skills\n"
    _write_source(source, dockerfile)
    removal = f"RUN rm -rf {' '.join(AGENT_SKILL_PATHS)}\n"

    # A published image runs with an overlay RUN that clears the agent skill locations, and no docker_image.
    published = tmp_path / "published"
    _adapt_task(source, published, "no-skill", "ghcr.io/x/y@sha256:abc")
    assert (published / "environment" / "Dockerfile").read_text() == f"FROM ghcr.io/x/y@sha256:abc\n{removal}"
    assert "docker_image" not in tomllib.loads((published / "task.toml").read_text())["environment"]
    assert not (published / "environment" / "skills").exists()

    # Without one, the removal ends the task's own build.
    built = tmp_path / "built"
    _adapt_task(source, built, "no-skill")
    assert (built / "environment" / "Dockerfile").read_text().endswith(removal)
    assert (built / "environment" / "Dockerfile").read_text().startswith(dockerfile)
    # That build still copies the skills before removing them, so its context keeps them.
    assert (built / "environment" / "skills" / "example" / "SKILL.md").is_file()

    # The with-skill condition leaves the image as built, without the OpenHands path's Dockerfile skill copies.
    with_skill = tmp_path / "with-skill"
    _adapt_task(source, with_skill, "with-skill", "ghcr.io/x/y@sha256:abc")
    assert (with_skill / "environment" / "Dockerfile").read_text() == dockerfile


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
def test_prepare_clones_pinned_tasks_and_writes_both_conditions(tmp_path: Path, monkeypatch) -> None:
    upstream = tmp_path / "upstream"
    _write_task(upstream / "tasks" / "alpha")
    (upstream / "README.md").write_text("outside the sparse checkout\n")
    _git("init", "--quiet", cwd=upstream)
    _git("add", ".", cwd=upstream)
    identity = ("-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false")
    _git(*identity, "commit", "--quiet", "-m", "tasks", cwd=upstream)
    revision = _git("rev-parse", "HEAD", cwd=upstream)

    data_dir = tmp_path / "data"
    # The checkout is the OpenHands path's, shared by both scripts.
    monkeypatch.setattr(source_module, "SOURCE_REPO", str(upstream))
    monkeypatch.setattr(source_module, "SOURCE_REVISION", revision)
    monkeypatch.setattr(source_module, "SOURCE_CHECKOUT", data_dir / "source")
    monkeypatch.setattr(prepare_module, "STAGING_DIR", data_dir)
    entries = tmp_path / "environments" / "harbor"
    monkeypatch.setattr(prepare_module, "ENTRIES_DIR", entries)
    # The committed entry files stay, as .gitignore keeps them (README.md, dataset.toml, *.yaml); tasks and loader
    # files from an earlier prepare are replaced.
    for name in DATASETS.values():
        (entries / name / "stale-task").mkdir(parents=True)
        (entries / name / "manifest.toml").write_text("stale\n")
        (entries / name / "README.md").write_text(f"# {name}\n")
        (entries / name / "dataset.toml").write_text('[gym]\ntype = "benchmark"\n')
        (entries / name / "harbor_sandbox.yaml").write_text("overlay: true\n")
    (tmp_path / "images.json").write_text(
        json.dumps(
            {
                "source_revision": revision,
                "tasks": [{"task_name": "alpha", "docker_image": "ghcr.io/x/y@sha256:abc", "success": True}],
            }
        )
    )
    # The pinned manifest is the default for the pinned checkout.
    monkeypatch.setattr(prepare_module, "PINNED_IMAGE_MANIFEST", tmp_path / "images.json")
    monkeypatch.delenv(source_module.SOURCE_ENV_VAR, raising=False)
    monkeypatch.delenv(source_module.IMAGE_MANIFEST_ENV_VAR, raising=False)
    monkeypatch.delenv(source_module.SKILL_MODE_ENV_VAR, raising=False)

    folders = prepare_module.prepare()

    assert folders == [entries / "skillsbench", entries / "skillsbench_no_skill"]
    with_skill = tomllib.loads((entries / "skillsbench" / "alpha" / "task.toml").read_text())
    assert with_skill["environment"]["docker_image"] == "ghcr.io/x/y@sha256:abc"
    assert with_skill["environment"]["skills_dir"] == SKILLS_DIR
    no_skill = tomllib.loads((entries / "skillsbench_no_skill" / "alpha" / "task.toml").read_text())
    assert "skills_dir" not in no_skill["environment"]
    for folder in folders:
        assert sorted(path.name for path in folder.iterdir()) == [
            "README.md",
            "alpha",
            "dataset.toml",
            "harbor_sandbox.yaml",
        ]
        assert (folder / "README.md").read_text() == f"# {folder.name}\n"
        assert (folder / "harbor_sandbox.yaml").read_text() == "overlay: true\n"
    assert sorted(path.name for path in data_dir.iterdir()) == ["source"]
    assert not (data_dir / "source" / "README.md").exists()
    # The OpenHands path's task tree and index are left to its own prepare.
    assert not (data_dir / "tasks").exists()
    assert not (data_dir / "skillsbench_official.jsonl").exists()

    # A second prepare of one condition reuses the checkout at the pinned revision instead of cloning again,
    # and leaves the other condition's folder alone.
    marker = data_dir / "source" / "tasks" / "alpha" / "reused"
    marker.write_text("")
    monkeypatch.setenv(source_module.SKILL_MODE_ENV_VAR, "no-skill")
    assert prepare_module.prepare() == [entries / "skillsbench_no_skill"]
    assert marker.exists()
    assert (entries / "skillsbench" / "alpha").is_dir()

    # A preparation stopped between its two renames left the entry aside; the next one puts it back first, so its
    # committed files survive.
    no_skill_entry = entries / "skillsbench_no_skill"
    no_skill_entry.rename(data_dir / ".skillsbench_no_skill.previous")
    prepare_module.prepare()
    assert (no_skill_entry / "harbor_sandbox.yaml").read_text() == "overlay: true\n"
    assert sorted(path.name for path in data_dir.iterdir()) == ["source"]

    # `none` builds from the Dockerfiles.
    monkeypatch.setenv(source_module.IMAGE_MANIFEST_ENV_VAR, "none")
    prepare_module.prepare()
    assert (
        "docker_image"
        not in tomllib.loads((entries / "skillsbench_no_skill" / "alpha" / "task.toml").read_text())["environment"]
    )


@pytest.mark.parametrize("name", sorted(DATASETS.values()))
def test_committed_entries_resolve_by_name(name: str) -> None:
    # `gym eval run skillsbench ...` finds the entry this repository commits; its tasks are prepared into it.
    entry = find_entry(name)
    assert entry is not None and entry.source is None
    assert entry.folder == (ENTRIES_DIR / name).resolve()
    config = read_dataset_config(entry.folder)
    assert config.type == "benchmark"
    assert config.license == "Apache 2.0"
