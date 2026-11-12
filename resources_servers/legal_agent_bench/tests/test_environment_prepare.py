# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real Harbor task loading and materialization from LAB cache inputs."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nemo_gym.data.tasks.harbor import DIGEST_KEY, load_task
from resources_servers.legal_agent_bench.environment_prepare import prepare_environment
from resources_servers.legal_agent_bench.prepare import LAB_SOURCE_REVISION


@pytest.fixture
def assets(tmp_path):
    cache = tmp_path / "cache"
    shared = cache / "_corpora/firm"
    shared.mkdir(parents=True)
    (shared / "document.txt").write_text("shared legal source")
    for name in ("one", "two"):
        task = cache / name
        (task / "tests").mkdir(parents=True)
        data = {
            "title": name,
            "instructions": "Write a memo to /workspace/output",
            "criteria": [{"text": "PRIVATE RUBRIC"}],
            "docs_dir": "../_corpora/firm",
            "metadata": {"lab_source_revision": LAB_SOURCE_REVISION},
        }
        (task / "task.json").write_text(json.dumps(data))
        (task / "tests/task.json").write_text(json.dumps(data))
        (task / "tests/test.sh").write_text("exit 0")
    skills = tmp_path / "skills"
    (skills / "docx").mkdir(parents=True)
    (skills / "docx/SKILL.md").write_text("Create documents with this skill.")
    return cache, skills


def test_materialization_preserves_identity_params_and_isolates_rubric(assets, tmp_path):
    cache, skills = assets
    output = tmp_path / "rows.jsonl"
    rows = [
        {
            "instance_id": f"legal_agent_bench::{name}",
            "responses_create_params": {
                "input": [{"role": "user", "content": "Keep caller text"}],
                "temperature": 0.4,
                "max_output_tokens": 1234,
            },
        }
        for name in ("one", "two")
    ]
    prepare_environment(tasks_dir=cache, skills_dir=skills, rows=rows, output=output, image="python:3.13")
    result = [json.loads(line) for line in output.read_text().splitlines()]
    mapping = json.loads(Path(str(output) + ".tasksets.json").read_text())["legal_agent_bench"]
    folder = Path(mapping["folder"])
    for row in result:
        name = row["task_id"]["task_id"]
        task = load_task(folder / name, root=folder)
        assert row["task_id"]["taskset"] == "legal_agent_bench"
        assert row["task_input"]["task_data"]["instance_id"] == f"legal_agent_bench::{name}"
        assert row["task_input"]["task_data"][DIGEST_KEY] == task.digest == mapping["tasks"][name]
        params = row["task_input"]["responses_create_params"]
        assert params["temperature"] == 0.4 and params["max_output_tokens"] == 1234
        assert params["input"][-1]["content"] == "Keep caller text"
        assert "PRIVATE RUBRIC" not in json.dumps(params)
        assert not (folder / name / "environment/task.json").exists()
        assert task.python_verifier is not None
        assert task.config.verifier.environment_mode == "separate"
        assert task.config.verifier.env["LAB_JUDGES"] == "${LAB_JUDGES}"
    source = cache / "_corpora/firm/document.txt"
    assert os.path.samefile(folder / "one/environment/vdr/document.txt", source)
    assert os.path.samefile(folder / "two/environment/vdr/document.txt", source)


def test_preparation_refuses_external_symlinks_without_publishing(assets, tmp_path):
    cache, skills = assets
    outside = tmp_path / "credentials"
    outside.write_text("test-only-host-value")
    (skills / "docx/leak").symlink_to(outside)
    output = tmp_path / "rows.jsonl"
    with pytest.raises(ValueError, match="symbolic link"):
        prepare_environment(
            tasks_dir=cache,
            skills_dir=skills,
            rows=[{"instance_id": "legal_agent_bench::one"}],
            output=output,
            image="python:3.13",
        )
    assert not output.exists()


def test_canonical_rows_round_trip_without_duplicating_instructions(assets, tmp_path):
    cache, skills = assets
    output = tmp_path / "rows.jsonl"
    prepare_environment(
        tasks_dir=cache,
        skills_dir=skills,
        rows=[{"instance_id": "legal_agent_bench::one", "responses_create_params": {"temperature": 0.2}}],
        output=output,
        image="python:3.13",
    )
    first = output.read_text()
    rows = [json.loads(first)]
    prepare_environment(tasks_dir=cache, skills_dir=skills, rows=rows, output=output, image="python:3.13")
    assert output.read_text() == first
    rows[0]["instance_id"] = "legal_agent_bench::two"
    with pytest.raises(ValueError, match="Conflicting"):
        prepare_environment(tasks_dir=cache, skills_dir=skills, rows=rows, output=output, image="python:3.13")
    assert output.read_text() == first


def test_another_process_waits_for_preparation_lock(assets, tmp_path):
    import fcntl

    cache, skills = assets
    output = tmp_path / "rows.jsonl"
    prepare_environment(
        tasks_dir=cache,
        skills_dir=skills,
        rows=[{"instance_id": "legal_agent_bench::one"}],
        output=output,
        image="python:3.13",
    )
    previous = output.read_text()
    mapping = json.loads(Path(str(output) + ".tasksets.json").read_text())["legal_agent_bench"]
    root = Path(mapping["folder"])
    ready = tmp_path / "waiting"
    script = """
import sys
from pathlib import Path
from resources_servers.legal_agent_bench import environment_prepare as preparation
cache, skills, output, ready = map(Path, sys.argv[1:])
original = preparation._lock_cache
def observed_lock(handle):
    ready.touch()
    original(handle)
preparation._lock_cache = observed_lock
preparation.prepare_environment(tasks_dir=cache, skills_dir=skills,
    rows=[{"instance_id": "legal_agent_bench::two"}], output=output, image="python:3.13")
"""
    with (root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        child = subprocess.Popen([sys.executable, "-c", script, str(cache), str(skills), str(output), str(ready)])
        try:
            deadline = time.monotonic() + 20
            while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert ready.exists(), "child did not reach the preparation lock"
            with pytest.raises(subprocess.TimeoutExpired):
                child.wait(timeout=0.2)
            assert output.read_text() == previous
            assert not (root / "two").exists()
            fcntl.flock(lock, fcntl.LOCK_UN)
            assert child.wait(timeout=20) == 0
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()
    row = json.loads(output.read_text())
    current = json.loads(Path(str(output) + ".tasksets.json").read_text())["legal_agent_bench"]
    assert row["task_id"]["task_id"] == "two"
    assert row["task_input"]["task_data"][DIGEST_KEY] == current["tasks"]["two"]
    assert load_task(root / "one", root=root).digest == mapping["tasks"]["one"]
