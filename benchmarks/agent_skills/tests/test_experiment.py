# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.agent_skills.scripts.compare_variants import compare_rollouts, render_report
from benchmarks.agent_skills.scripts.run_experiment import (
    build_arm_command,
    load_manifest,
    resolve_model_access,
    run_experiment,
    snapshot_arm_skills,
    unsafe_uv_project_ancestor,
    validate_execution_health,
    validate_skills_provenance,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "benchmarks/agent_skills/configs/experiments/create_environment_v1.yaml"
SMOKE_MANIFEST_PATH = REPO_ROOT / "benchmarks/agent_skills/configs/experiments/create_environment_smoke.yaml"


def _row(task_id: str, rollout_index: int, reward: float, *, skills_hash: str | None = None):
    row = {
        "task_id": task_id,
        "_ng_rollout_index": rollout_index,
        "reward": reward,
        "correctness": reward,
        "completeness": reward,
        "convention_compliance": 1.0,
        "sandbox_elapsed_seconds": 10.0,
        "sandbox_return_code": 0,
        "sandbox_error_type": None,
        "workspace_patch": "diff --git a/file b/file\n",
        "verifier_elapsed_seconds": 2.0,
        "response": {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
            }
        },
    }
    if skills_hash:
        row["skills_ref"] = {"hash": skills_hash}
    return row


def test_manifest_and_arm_commands(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST_PATH)
    snapshots = snapshot_arm_skills(manifest, tmp_path)

    assert set(snapshots) == {"treatment"}
    treatment_hash = snapshots["treatment"]["skills_ref"]["hash"]
    assert len(treatment_hash) == 12
    assert (tmp_path / "skill-bundles/treatment/nemo-gym-create-environment/SKILL.md").is_file()

    control_command = build_arm_command(
        manifest,
        arm_name="discovery_control",
        arm=manifest.arms["discovery_control"],
        output_dir=tmp_path,
        skill_snapshot=None,
        execution_image="sha256:image",
        execution_model="test-model",
    )
    treatment_command = build_arm_command(
        manifest,
        arm_name="treatment",
        arm=manifest.arms["treatment"],
        output_dir=tmp_path,
        skill_snapshot=snapshots["treatment"],
        execution_image="sha256:image",
        execution_model="test-model",
    )

    assert "--benchmark" in control_command
    assert "agent_skills" in control_command
    assert any(value.endswith(".bare=false") for value in control_command)
    assert any(value.endswith(".max_turns=40") for value in control_command)
    assert any(value.endswith(".timeout=1200") for value in control_command)
    assert not any("skills.path" in value for value in control_command)
    assert any("skills.path" in value for value in treatment_command)


def test_checked_smoke_manifest_is_minimal() -> None:
    manifest = load_manifest(SMOKE_MANIFEST_PATH)

    assert set(manifest.arms) == {"discovery_control", "treatment"}
    assert manifest.sampling.repeats == 1
    assert manifest.sampling.concurrency == 1
    assert manifest.agent.max_turns == 30
    assert manifest.agent.timeout == 600


def test_paired_comparison_and_report() -> None:
    control = [
        _row("task-1", 0, 0),
        _row("task-1", 1, 1),
        _row("task-2", 0, 0),
        _row("task-2", 1, 0),
    ]
    treatment = [
        _row("task-1", 0, 1),
        _row("task-1", 1, 1),
        _row("task-2", 0, 0),
        _row("task-2", 1, 1),
    ]

    comparison = compare_rollouts(
        control,
        treatment,
        control_arm="control",
        treatment_arm="treatment",
        seed=7,
        bootstrap_samples=200,
    )

    assert comparison["paired_rollouts"] == 4
    assert comparison["outcomes"] == {"wins": 2, "losses": 0, "ties": 2}
    assert comparison["metrics"]["reward"]["delta"] == pytest.approx(0.5)
    assert comparison["pass_at_k"] == {"control": 0.5, "treatment": 1.0}
    report = render_report(comparison)
    assert "Wins / losses / ties: 2 / 0 / 2" in report
    assert "| reward |" in report


def test_skills_provenance_validation() -> None:
    snapshot = {"skills_ref": {"hash": "abc123"}}
    validate_skills_provenance(
        [_row("task", 0, 1, skills_hash="abc123")],
        arm_name="treatment",
        snapshot=snapshot,
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_skills_provenance(
            [_row("task", 0, 1, skills_hash="different")],
            arm_name="treatment",
            snapshot=snapshot,
        )
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_skills_provenance(
            [
                _row("task", 0, 1, skills_hash="abc123"),
                _row("task", 1, 1),
            ],
            arm_name="treatment",
            snapshot=snapshot,
        )
    with pytest.raises(ValueError, match="unexpectedly contains"):
        validate_skills_provenance(
            [_row("task", 0, 1, skills_hash="abc123")],
            arm_name="control",
            snapshot=None,
        )


def test_execution_health_rejects_sandbox_failure() -> None:
    row = _row("task", 0, 0)
    row["sandbox_return_code"] = 1
    row["workspace_patch"] = ""
    row["response"]["output"] = [{"content": [{"text": "Invalid API key"}]}]

    with pytest.raises(RuntimeError, match="refusing to report them as model scores"):
        validate_execution_health([row], arm_name="control")

    row["workspace_patch"] = "diff --git a/file b/file\n"
    validate_execution_health([row], arm_name="control")


def test_dry_run_writes_lock_without_running_gym(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST_PATH)
    output_dir = tmp_path / "experiment"
    with (
        patch(
            "benchmarks.agent_skills.scripts.run_experiment.git_state",
            return_value={"revision": "a" * 40, "dirty": False},
        ),
        patch(
            "benchmarks.agent_skills.scripts.run_experiment.resolve_image_digest",
            return_value="sha256:image",
        ),
        patch(
            "benchmarks.agent_skills.scripts.run_experiment.resolve_model_access",
            return_value={"name": "test-model", "base_url": "https://endpoint.example"},
        ),
        patch("benchmarks.agent_skills.scripts.run_experiment.subprocess.run") as run,
    ):
        result = run_experiment(
            manifest,
            manifest_path=MANIFEST_PATH,
            output_dir=output_dir,
            dry_run=True,
            allow_dirty=False,
        )

    assert result is None
    run.assert_not_called()
    lock = json.loads((output_dir / "experiment.lock.json").read_text())
    assert lock["dataset_sha256"]
    assert lock["sandbox_image"]["execution_reference"] == "sha256:image"
    assert lock["model_access"] == {
        "name": "test-model",
        "base_url": "https://endpoint.example",
    }
    assert set(lock["commands"]) == set(manifest.arms)
    assert lock["skill_snapshots"]["treatment"]["skills_ref"]["hash"]

    with (
        patch(
            "benchmarks.agent_skills.scripts.run_experiment.git_state",
            return_value={"revision": "a" * 40, "dirty": False},
        ),
        patch(
            "benchmarks.agent_skills.scripts.run_experiment.resolve_model_access",
            return_value={"name": "test-model", "base_url": "https://endpoint.example"},
        ),
        patch("benchmarks.agent_skills.scripts.run_experiment.subprocess.run") as resumed_run,
    ):
        resumed = run_experiment(
            manifest,
            manifest_path=MANIFEST_PATH,
            output_dir=output_dir,
            dry_run=True,
            allow_dirty=False,
            resume=True,
        )
    assert resumed is None
    resumed_run.assert_not_called()


def test_resume_forwards_gym_resume_for_existing_arm_outputs(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST_PATH)
    output_dir = tmp_path / "experiment"
    with (
        patch(
            "benchmarks.agent_skills.scripts.run_experiment.git_state",
            return_value={"revision": "a" * 40, "dirty": False},
        ),
        patch(
            "benchmarks.agent_skills.scripts.run_experiment.resolve_image_digest",
            return_value="sha256:image",
        ),
        patch(
            "benchmarks.agent_skills.scripts.run_experiment.resolve_model_access",
            return_value={"name": "test-model", "base_url": "https://endpoint.example"},
        ),
    ):
        run_experiment(
            manifest,
            manifest_path=MANIFEST_PATH,
            output_dir=output_dir,
            dry_run=True,
            allow_dirty=False,
        )

    lock = json.loads((output_dir / "experiment.lock.json").read_text())
    treatment_hash = lock["skill_snapshots"]["treatment"]["skills_ref"]["hash"]
    for arm_name in manifest.arms:
        arm_dir = output_dir / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        row = _row(
            "task",
            0,
            1,
            skills_hash=treatment_hash if arm_name == "treatment" else None,
        )
        (arm_dir / "rollouts.jsonl").write_text(json.dumps(row) + "\n")

    with (
        patch(
            "benchmarks.agent_skills.scripts.run_experiment.git_state",
            return_value={"revision": "a" * 40, "dirty": False},
        ),
        patch(
            "benchmarks.agent_skills.scripts.run_experiment.resolve_model_access",
            return_value={"name": "test-model", "base_url": "https://endpoint.example"},
        ),
        patch("benchmarks.agent_skills.scripts.run_experiment.subprocess.run") as run,
    ):
        run_experiment(
            manifest,
            manifest_path=MANIFEST_PATH,
            output_dir=output_dir,
            dry_run=False,
            allow_dirty=False,
            resume=True,
        )

    assert run.call_count == len(manifest.arms)
    assert all("--resume" in call.args[0] for call in run.call_args_list)


def test_fresh_run_rejects_nonempty_unlocked_output(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST_PATH)
    output_dir = tmp_path / "experiment"
    output_dir.mkdir()
    (output_dir / "unrelated.txt").write_text("keep me")

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        run_experiment(
            manifest,
            manifest_path=MANIFEST_PATH,
            output_dir=output_dir,
            dry_run=True,
            allow_dirty=True,
        )


def test_detects_project_aware_uv_ancestor() -> None:
    process = MagicMock()
    project_uv = MagicMock()
    project_uv.cmdline.return_value = ["uv", "run", "python", "driver.py"]
    process.parents.return_value = [project_uv]
    with patch(
        "benchmarks.agent_skills.scripts.run_experiment.psutil.Process",
        return_value=process,
    ):
        assert unsafe_uv_project_ancestor() is True

    no_project_uv = MagicMock()
    no_project_uv.cmdline.return_value = ["uv", "run", "--no-project", "python", "driver.py"]
    process.parents.return_value = [no_project_uv]
    with patch(
        "benchmarks.agent_skills.scripts.run_experiment.psutil.Process",
        return_value=process,
    ):
        assert unsafe_uv_project_ancestor() is False


def test_model_access_reads_env_yaml_key() -> None:
    with patch(
        "benchmarks.agent_skills.scripts.run_experiment.load_env_yaml",
        return_value={
            "anthropic_model_name": "allowed-model",
            "anthropic_base_url": "https://endpoint.example",
        },
    ):
        access = resolve_model_access(load_manifest(MANIFEST_PATH).model)

    assert access == {
        "name": "allowed-model",
        "base_url": "https://endpoint.example",
    }


def test_comparison_rejects_partial_arm() -> None:
    control = [_row("task-1", 0, 0), _row("task-2", 0, 0)]
    treatment = [_row("task-1", 0, 1)]

    with pytest.raises(ValueError, match="rollout keys differ"):
        compare_rollouts(
            control,
            treatment,
            control_arm="control",
            treatment_arm="treatment",
            seed=7,
        )


def test_single_task_report_does_not_claim_confidence_interval() -> None:
    comparison = compare_rollouts(
        [_row("task", 0, 0)],
        [_row("task", 0, 1)],
        control_arm="control",
        treatment_arm="treatment",
        seed=7,
    )

    assert comparison["metrics"]["reward"]["paired_task_ci_95"] == [None, None]
    assert "n/a" in render_report(comparison)
