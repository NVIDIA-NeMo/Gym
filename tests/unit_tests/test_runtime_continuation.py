# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real persisted Store and installed reader; scheduler/container transport is explicitly simulated."""

import hashlib
import json
import shlex
import sys
from pathlib import Path

import pytest

from nemo_gym.orchestration.api import ResumeConfig, SubmitConfig
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script
from nemo_gym.rollout_recovery import RunManifest
from nemo_gym.rollout_store import RolloutStore
from tests.unit_tests.test_managed_recovery_selection import prepare_managed
from tests.unit_tests.test_resume_script import _run_prologue
from tests.unit_tests.test_rollout_recovery import runner_config as runner_config


@pytest.mark.parametrize(
    "case,exit_code,continues,debit",
    [
        ("missing", "75:0", True, False),
        ("missing", "1:0", True, True),
        ("infra", "75:0", True, True),
        ("infra", "1:0", True, True),
        ("healthy", "75:0", False, False),
        ("wrong", "75:0", False, False),
        ("final_zero", "75:0", False, False),
        ("judge", "75:0", False, False),
        ("judge_then_reservation", "75:0", False, False),
        ("prior_reservation", "75:0", False, False),
        ("unknown_kill", "75:0", False, False),
        ("exhausted", "75:0", False, False),
        ("absent", "75:0", False, False),
        ("secondary_cleanup", "75:0", False, False),
        ("malformed", "75:0", False, False),
    ],
)
def test_installed_runtime_authorizes_only_selected_remaining_work(
    tmp_path, monkeypatch, case, exit_code, continues, debit
):
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    if case != "absent":
        output = prepare_managed(
            tmp_path,
            "infra" if case == "malformed" else "healthy" if case == "secondary_cleanup" else case,
            n=2 if case == "missing" else 1,
        )
        if case == "malformed":
            sidecar = output.with_name("rollouts_failures.jsonl")
            rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
            rows[-1]["_ng_failure_id"] = "missing-selected"
            rows[-1]["_ng_failure_events"] = {}
            sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows))
        if case == "secondary_cleanup":
            row = json.loads(output.read_text())
            row["_ng_failure_events"] = {
                "cleanup": {
                    "failure_kind": "session_release_failed",
                    "boundary": "environment",
                    "stage": "cleanup",
                    "operation": "episode.cleanup",
                }
            }
            output.write_text(json.dumps(row) + "\n")
        before = {
            str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (tmp_path / "artifacts").glob("*")
            if f.is_file()
        }
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=3), "FAILED", exit_code=exit_code)
    assert ("REACHED_WORK" in result.stdout) is continues, result.stdout + result.stderr
    retry = tmp_path / ".gym_infra_retries"
    assert retry.exists() is debit
    if debit:
        assert retry.read_text().strip() == "1"
    assert (tmp_path / ".gym_accumulated_walltime").read_text().strip() == "60"
    if case != "absent":
        assert before == {
            str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (tmp_path / "artifacts").glob("*")
            if f.is_file()
        }


def test_mixed_saved_judge_and_missing_never_runs_agent(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(2)]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def prepare():
        return rows, RunManifest.create(source, rows, {}, {}).model_copy(update={"max_rollout_attempts": 3})

    with RolloutStore.start_or_resume(artifacts / "rollouts.jsonl", prepare, resume=False) as store:
        row = store.pending(3)[0]
        store.record_dispatch(row)
        store.record_outcome(
            row
            | {
                "_ng_failure_class": "judge_failed",
                "failure_kind": "judge_failed",
                "response": {"status": "completed", "output": []},
            }
        )
    result = _run_prologue(tmp_path, ResumeConfig(), "FAILED", exit_code="75:0")
    assert "REACHED_WORK" not in result.stdout
    assert not (tmp_path / "sbatch_calls").exists()


def test_generated_container_command_invokes_preinstalled_reader_before_services(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    prepare_managed(tmp_path, "missing", n=2)
    config = SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {"cluster": {"type": "slurm", "account": "test"}},
            "driver": {"container": "installed-gym:test", "benchmarks": {"task": {"resumable": True}}},
            "job": {"output_path": str(tmp_path.parent)},
        }
    )
    bench = config.driver.benchmarks["task"]
    script = build_sbatch_script(config, "task", bench, config.compute["cluster"], tmp_path)
    start = script.index("_gym_admission_output=$(") + len("_gym_admission_output=$(")
    command = script[start : script.index(") || _gym_stop_incomplete", start)]
    # The generated command and entrypoint are unchanged. Only srun/container transport is substituted;
    # this process is the preinstalled Gym image boundary and imports the candidate source.
    stub = tmp_path / "srun-stub"
    stub.write_text(
        "#!/bin/bash\nwhile [[ $1 == --* ]]; do shift; done\nexport PATH="
        + shlex.quote(str(Path(sys.executable).parent))
        + ":$PATH\nexport PYTHONPATH="
        + shlex.quote(str(Path(__file__).resolve().parents[2]))
        + '\nexec "$@"\n'
    )
    stub.chmod(0o755)
    command = command.replace(" srun ", f" {shlex.quote(str(stub))} ", 1)
    result = _run_prologue(
        tmp_path, ResumeConfig(max_retries=3), "FAILED", exit_code="75:0", admission_command=command
    )
    assert "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert not (tmp_path / ".gym_infra_retries").exists()
    assert "--container-image=installed-gym:test" in script
    assert "--container-mounts=" in command


def test_exit75_without_installed_runtime_cannot_authorize_work(tmp_path):
    prepare_managed(tmp_path, "missing", n=2)
    result = _run_prologue(tmp_path, ResumeConfig(), "FAILED", exit_code="75:0", admission_command="false")
    assert "REACHED_WORK" not in result.stdout
    assert not (tmp_path / "sbatch_calls").exists()


@pytest.mark.parametrize(
    "mutation",
    ["missing_binding", "stale_allocation", "zero_progress", "bad_progress", "wrong_coverage", "missing_report"],
)
def test_stale_or_unbound_progress_cannot_waive_spent_retry_budget(tmp_path, monkeypatch, mutation):
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    prepare_managed(tmp_path, "missing", n=2)
    report = tmp_path / "artifacts" / "rollouts_coverage.json"
    data = json.loads(report.read_text())
    if mutation == "missing_binding":
        data.pop("allocation_id")
    elif mutation == "stale_allocation":
        data["allocation_id"] = "99"
    elif mutation == "zero_progress":
        data["progress_outcomes_recorded"] = 0
    elif mutation == "bad_progress":
        data["progress_outcomes_recorded"] = True
    elif mutation == "wrong_coverage":
        data["expected"] += 1
    report.write_text(json.dumps(data))
    if mutation == "missing_report":
        report.unlink()
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=0), "FAILED", exit_code="75:0")
    assert "REACHED_WORK" not in result.stdout, result.stdout + result.stderr
    assert not (tmp_path / ".gym_infra_retries").exists()
    assert not (tmp_path / "sbatch_calls").exists()


def test_owned_infrastructure_cannot_exceed_carried_retry_budget(tmp_path):
    prepare_managed(tmp_path, "infra")
    (tmp_path / ".gym_infra_retries").write_text("3\n")
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=3), "FAILED", exit_code="75:0")
    assert "REACHED_WORK" not in result.stdout
    assert (tmp_path / ".gym_infra_retries").read_text() == "3\n"


def test_explicit_attempt_reset_does_not_reopen_unknown_selected_execution(tmp_path):
    output = prepare_managed(tmp_path, "infra")
    sidecar = output.with_name("rollouts_failures.jsonl")
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    rows[-1]["_ng_failure_id"] = "absent"
    rows[-1]["_ng_failure_events"] = {}
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def current_inputs():
        rows = [{"_ng_task_index": 0, "_ng_rollout_index": 0}]
        return rows, RunManifest.create(tmp_path / "source.jsonl", rows, {}, {}).model_copy(
            update={"max_rollout_attempts": 3}
        )

    with RolloutStore.start_or_resume(output, current_inputs, resume=True, resubmit=True) as store:
        assert store.pending(3) == []
    result = _run_prologue(tmp_path, ResumeConfig(), "FAILED", exit_code="75:0")
    assert "REACHED_WORK" not in result.stdout


def test_generated_reader_uses_same_pinned_checkout_as_driver(tmp_path):
    config = SubmitConfig.model_validate(
        {
            "services": {},
            "compute": {"cluster": {"type": "slurm", "account": "test"}},
            "driver": {
                "container": "installed-gym:test",
                "gym_install": {
                    "repo": "https://github.com/NVIDIA-NeMo/Gym.git",
                    "ref": "0123456789012345678901234567890123456789",
                },
                "benchmarks": {"task": {"resumable": True}},
            },
            "job": {"output_path": str(tmp_path.parent)},
        }
    )
    script = build_sbatch_script(config, "task", config.driver.benchmarks["task"], config.compute["cluster"], tmp_path)
    admission = script[
        script.index("_gym_admission_output=$(") : script.index(
            ") || _gym_stop_incomplete", script.index("_gym_admission_output=$(")
        )
    ]
    assert "git checkout 0123456789012345678901234567890123456789" in admission
    assert "source .venv/bin/activate" in admission
    assert "continuation_admission" in admission
    assert "gym eval run" not in admission


async def test_actual_collection_failure_reaches_installed_runtime_and_retry_budget(
    tmp_path, monkeypatch, runner_config
):
    from unittest.mock import AsyncMock

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from omegaconf import OmegaConf
    from pydantic import BaseModel

    from nemo_gym.rollout_collection import RolloutCollectionHelper
    from nemo_gym.rollout_recovery import IncompleteEvaluationError
    from nemo_gym.server_runtime.episode_types import BaseEpisodeRequest, BaseEpisodeResponse
    from tests.unit_tests.test_rollout_recovery import FakeResponse, install_fake_server_client

    class TaskInput(BaseModel):
        pass

    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "3")
    (tmp_path / "artifacts").mkdir()
    runner_config.output_jsonl_fpath = str(tmp_path / "artifacts" / "rollouts.jsonl")
    runner_config.environment_server_routes = {"native": "environment"}
    runner_config.require_complete = True
    Path(runner_config.input_jsonl_fpath).write_text(
        json.dumps({"task_id": {"taskset": "native", "task_id": "0"}, "task_input": {}}) + "\n"
    )
    app = FastAPI()
    calls = []

    @app.post("/run")
    async def run_episode(payload: dict):
        request = BaseEpisodeRequest[TaskInput].model_validate(payload)
        calls.append(request.episode_id.attempt)
        return BaseEpisodeResponse[dict].model_validate(
            {
                "episode_id": request.episode_id,
                "task_id": request.task.task_id,
                "failure": {
                    "failure_reason": "Infrastructure unavailable before answer",
                    "terminal": False,
                    "failure_kind": "transport_timeout",
                    "stage": "agent",
                },
            }
        )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://native-test") as http:

        async def post(*, server_name, url_path, json, **kwargs):
            assert server_name == "environment"
            payload = json.model_dump(mode="json") if isinstance(json, BaseModel) else json
            response = await http.post(url_path, json=payload)
            return FakeResponse(response.status_code, response.json())

        client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        client.global_config_dict = OmegaConf.create({"environment": {"environment_servers": {"custom": {}}}})
        with pytest.raises(IncompleteEvaluationError) as error:
            await RolloutCollectionHelper().run_from_config(runner_config)
    assert calls == [0]
    assert error.value.exit_code == 75
    output = Path(runner_config.output_jsonl_fpath)
    before = output.with_name("rollouts_failures.jsonl").read_bytes()
    result = _run_prologue(tmp_path, ResumeConfig(max_retries=3), "FAILED", exit_code="75:0")
    assert "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert (tmp_path / ".gym_infra_retries").read_text().strip() == "1"
    assert output.with_name("rollouts_failures.jsonl").read_bytes() == before
    assert RolloutStore.read(output).attempt_count({"_ng_task_index": 0, "_ng_rollout_index": 0}) == 1
    assert calls == [0]  # Admission starts no agent/model/verifier work.


@pytest.mark.parametrize("state", ["TIMEOUT", "PREEMPTED", "NODE_FAIL", "BOOT_FAIL"])
@pytest.mark.parametrize("case", ["judge", "unknown_kill", "prior_reservation", "healthy"])
def test_known_scheduler_event_cannot_bypass_saved_outcome_guards(tmp_path, state, case):
    prepare_managed(tmp_path, case)
    result = _run_prologue(tmp_path, ResumeConfig(), state)
    assert "REACHED_WORK" not in result.stdout, result.stdout + result.stderr
    assert not (tmp_path / "sbatch_calls").exists()


@pytest.mark.parametrize("state,debit", [("TIMEOUT", False), ("PREEMPTED", True)])
def test_known_scheduler_event_preserves_untouched_work_semantics(tmp_path, state, debit):
    prepare_managed(tmp_path, "missing", n=2)
    result = _run_prologue(tmp_path, ResumeConfig(), state)
    assert "REACHED_WORK" in result.stdout, result.stdout + result.stderr
    assert (tmp_path / ".gym_infra_retries").exists() is debit
