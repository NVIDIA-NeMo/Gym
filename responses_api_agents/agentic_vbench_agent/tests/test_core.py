# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess

import pytest

from responses_api_agents.agentic_vbench_agent import core


def fixture_result(tmp_path, reward="0", status="OK", trajectory=True):
    job = tmp_path / "jobs/trial/repair1"
    step = job / "steps/solve"
    (step / "agent").mkdir(parents=True)
    (step / "verifier").mkdir()
    (job / "result.json").write_text(json.dumps({"task_name": "repair1"}))
    if trajectory:
        (step / "agent/trajectory.json").write_text(json.dumps({"steps": [{"source": "agent", "message": "done"}]}))
    if status != "FAIL":
        (step / "verifier/reward.json").write_text(json.dumps({"reward": float(reward)}))
    return {"task_id": "repair1", "family": "repair"}


def test_valid_zero_is_preserved(tmp_path):
    task = fixture_result(tmp_path)
    assert core.read_result(tmp_path, task)["reward"] == 0


@pytest.mark.parametrize("reward", ["nan", "inf", "-0.1", "1.1"])
def test_invalid_reward_fails(tmp_path, reward):
    task = fixture_result(tmp_path, reward=reward)
    with pytest.raises(ValueError, match="Invalid verifier"):
        core.read_result(tmp_path, task)


def test_infra_failure_does_not_become_zero(tmp_path):
    task = fixture_result(tmp_path, reward="", status="FAIL")
    with pytest.raises(RuntimeError, match="Missing native verifier"):
        core.read_result(tmp_path, task)


def test_reward_without_model_trajectory_rejected(tmp_path):
    task = fixture_result(tmp_path, trajectory=False)
    with pytest.raises(RuntimeError, match="one model trajectory"):
        core.read_result(tmp_path, task)


def test_wrong_task_rejected(tmp_path):
    task = fixture_result(tmp_path)
    task["task_id"] = "different"
    with pytest.raises(ValueError, match="mismatched"):
        core.read_result(tmp_path, task)


def test_dataset_keeps_verbatim_prompt_and_no_media():
    prompt = "\nRead /workspace/input.mp4.\n\n"
    rows = core.dataset_rows({"x": {"task_id": "x", "family": "repair", "prompt": prompt}})
    assert rows[0]["responses_create_params"] == {"input": [{"role": "user", "content": prompt}]}
    assert "prompt" not in rows[0]["verifier_metadata"]
    with pytest.raises(ValueError):
        core.dataset_rows({}, "typo")


def test_multiple_families_and_overlap_rejection():
    tasks = {family: {"task_id": family + "1", "family": family, "prompt": "prompt"} for family in core.FAMILIES}
    tasks = {task["task_id"]: task for task in tasks.values()}
    rows = core.dataset_rows(tasks, "agentic_vbench_repair assembly sequencing")
    assert {row["verifier_metadata"]["family"] for row in rows} == {"repair", "assembly", "sequencing"}
    with pytest.raises(ValueError, match="overlaps"):
        core.dataset_rows(tasks, "repair repair1")
    with pytest.raises(ValueError, match="empty"):
        core.dataset_rows(tasks, " ")


def test_equal_family_weights_recover_leaderboard_mean():
    rows = []
    for family, size in core.FAMILIES.items():
        rows.extend({"family": family, "reward": 1.0 if family == "repair" else 0.0} for _ in range(size))
    weighted = sum(row["reward"] * core.equal_family_weight(row["family"]) for row in rows) / len(rows)
    assert weighted == pytest.approx(0.25)
    assert core.equal_family_mean(rows)["mean/equal_family_reward"] == pytest.approx(0.25)


def test_ensure_checkout_fetches_pinned_revision(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=upstream, check=True)
    (upstream / "tasks").mkdir()
    (upstream / "tasks/README").write_text("pinned")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"}
    subprocess.run(["git", "add", "."], cwd=upstream, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "pin"], cwd=upstream, check=True, env={**env, "PATH": "/usr/bin:/bin"}
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    monkeypatch.setattr(core, "REVISION", head)
    root = tmp_path / "cache" / "agentic-vbench"
    assert core.ensure_checkout(root, repo_url=str(upstream)) == root
    assert (root / "tasks/README").read_text() == "pinned"
    # Idempotent, and a checkout at another revision is refused rather than reused.
    assert core.ensure_checkout(root, repo_url=str(upstream)) == root
    monkeypatch.setattr(core, "REVISION", "0" * 40)
    with pytest.raises(ValueError, match="expected"):
        core.ensure_checkout(root, repo_url=str(upstream))


def test_default_checkout_root_follows_hf_home(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/cache/huggingface")
    monkeypatch.delenv("AGENTIC_VBENCH_CACHE_ROOT", raising=False)
    assert str(core.default_checkout_root()) == f"/cache/agentic_vbench/agentic-vbench-{core.REVISION[:12]}"


def test_backend_environment_reads_client_env_and_rebinds_cert_path(tmp_path):
    (tmp_path / "client.env").write_text(
        "export DOCKER_HOST=tcp://nsc-svg-slurm-1-cpu-26:18443\n"
        "export DOCKER_TLS_VERIFY=1\n"
        "export DOCKER_CERT_PATH=/somewhere/on/the/backend\n"
        "export DOCKER_BUILDKIT=0\n"
    )
    env = core.backend_environment(tmp_path, wait_seconds=0)
    assert env == {
        "DOCKER_HOST": "tcp://nsc-svg-slurm-1-cpu-26:18443",
        "DOCKER_TLS_VERIFY": "1",
        "DOCKER_CERT_PATH": str(tmp_path),
        "DOCKER_BUILDKIT": "0",
    }


def test_backend_environment_times_out_without_backend(tmp_path):
    with pytest.raises(TimeoutError, match="serve_rootless_backend"):
        core.backend_environment(tmp_path / "missing", wait_seconds=0, poll_seconds=0)


def test_malformed_client_env_is_rejected():
    with pytest.raises(ValueError, match="Malformed"):
        core.parse_client_env("DOCKER_HOST")
    assert core.parse_client_env("# comment\nexport A='x y'\n") == {"A": "x y"}


def test_lock_falls_back_when_the_filesystem_refuses_flock(tmp_path, monkeypatch):
    def refuse(handle, operation):
        raise OSError(38, "Function not implemented")

    monkeypatch.setattr(core.fcntl, "flock", refuse)
    with core._locked(tmp_path / "x.lock"):
        pass
    assert (tmp_path / "x.lock").exists()
