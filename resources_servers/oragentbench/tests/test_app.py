# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Server behaviour with the container faked: task parsing, the pass predicate, step aggregation,
attribution of failures, and the metric selection consumers actually read."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient

import resources_servers.oragentbench.app as app_module
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.server_utils import ServerClient
from resources_servers.oragentbench.app import (
    HARNESS_FAULTS,
    ORAgentBenchResourcesServer,
    ORAgentBenchResourcesServerConfig,
    ORAgentBenchVerifyRequest,
    Status,
    StepResult,
    TaskSpec,
    is_pass,
    load_task,
    scalar_reward,
)


SERVER_DIR = Path(__file__).resolve().parents[1]
FIXTURE_TASK = SERVER_DIR / "tests" / "fixtures" / "toy_assignment"
CONFIG_PATH = SERVER_DIR / "configs" / "oragentbench.yaml"

EMPTY_RESPONSE = {
    "id": "r",
    "created_at": 0,
    "model": "m",
    "object": "response",
    "output": [],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}


class FakeExecResult:
    def __init__(self, return_code: int = 0, stdout: str = "", stderr: str = "", error_type: Optional[str] = None):
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.error_type = error_type


class FakeSandbox:
    """Scripted container: ``verifier`` decides what running test.sh leaves in /logs/verifier."""

    def __init__(self, verifier=None, exec_timeout_on_tests: bool = False, setup_fails: bool = False):
        self.commands: List[str] = []
        self.uploads: List[str] = []
        self.files: Dict[str, str] = {}
        self.verifier = verifier
        self.exec_timeout_on_tests = exec_timeout_on_tests
        self.setup_fails = setup_fails
        self.stopped = False
        self._handle = MagicMock(sandbox_id="fake-sandbox")

    async def start_with_setup(self, spec, setup):
        self.spec = spec
        await setup(self)
        return self

    async def exec(self, command: str, **kwargs: Any) -> FakeExecResult:
        self.commands.append(command)
        if command == "pwd":
            return FakeExecResult(stdout="/app\n")
        if command.startswith("find /app/submissions"):
            return FakeExecResult(stdout="\n".join(self.files))
        if command.endswith("/setup.sh"):
            return FakeExecResult(return_code=1 if self.setup_fails else 0, stderr="boom" if self.setup_fails else "")
        if command.startswith("bash /solution/solve.sh"):
            self.files["/app/submissions/solution.csv"] = "worker,job\n"
            return FakeExecResult(stdout="solved")
        if command.startswith("mv "):
            return FakeExecResult()
        if command == "bash /tests/test.sh":
            if self.exec_timeout_on_tests:
                return FakeExecResult(return_code=-1, stderr="timed out", error_type="timeout")
            if self.verifier is not None:
                for name, payload in self.verifier(self).items():
                    self.files[f"/logs/verifier/{name}"] = json.dumps(payload)
            return FakeExecResult(stdout="verifier ran")
        return FakeExecResult()

    async def upload(self, local_path, remote_path: str) -> None:
        self.uploads.append(remote_path)

    async def download(self, remote_path: str, local_path) -> None:
        if remote_path not in self.files:
            raise RuntimeError(f"no such file {remote_path}")
        Path(local_path).write_text(self.files[remote_path])

    async def stop(self) -> None:
        self.stopped = True


def scored(feasible: bool, quality: float, status: str = "scored"):
    return lambda sandbox: {
        "reward.json": {"feasibility": 1.0 if feasible else 0.0, "quality": quality},
        "reward_details.json": {"quality_status": status},
    }


def make_server(sandbox: FakeSandbox, monkeypatch, **config_overrides) -> ORAgentBenchResourcesServer:
    monkeypatch.setattr(app_module, "AsyncSandbox", lambda _provider: sandbox)
    monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda *_: {"fake": {}})
    monkeypatch.setattr(app_module, "resolve_provider_metadata", lambda *_: {})
    config = ORAgentBenchResourcesServerConfig(
        host="",
        port=0,
        entrypoint="",
        name="oragentbench",
        sandbox_provider="sandbox",
        sandbox_config={"resources": {"cpu": 1, "memory_mib": 512}},
        **config_overrides,
    )
    return ORAgentBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def verify_body(task_folder: Path = FIXTURE_TASK, difficulty: Any = "easy") -> dict:
    return {
        "responses_create_params": {"input": [{"role": "user", "content": "solve"}]},
        "response": EMPTY_RESPONSE,
        "task_name": "synthetic/toy_assignment",
        "docker_image": "oragentbench-fixture:toy_assignment",
        "task_folder": str(task_folder),
        "difficulty": difficulty,
        # Provenance extras the preparer writes; ``num_steps`` deliberately collides with a computed field.
        "num_steps": 99,
        "agent_ref": {"type": "responses_api_agents", "name": "oragentbench_agent"},
    }


def multi_step_task(tmp_path: Path, step_names=("initial_plan", "step2"), min_reward=0.3) -> Path:
    """A [[steps]] task in upstream's layout, built from the single-step fixture's scripts."""
    task_dir = tmp_path / "multi"
    task_dir.mkdir()
    steps_toml = ""
    for name in step_names:
        step_dir = task_dir / "steps" / name
        shutil.copytree(FIXTURE_TASK / "tests", step_dir / "tests")
        shutil.copytree(FIXTURE_TASK / "solution", step_dir / "solution")
        (step_dir / "instruction.md").write_text(f"Instruction for {name}\n")
        if name != step_names[0]:
            (step_dir / "workdir").mkdir()
            (step_dir / "workdir" / "setup.sh").write_text("#!/bin/bash\ntrue\n")
            (step_dir / "workdir" / "EVENT_NOTICE.md").write_text("event\n")
        gate = f"min_reward = {min_reward}\n" if min_reward is not None and name != step_names[-1] else ""
        steps_toml += f'\n[[steps]]\nname = "{name}"\n{gate}[steps.agent]\ntimeout_sec = 100.0\n[steps.verifier]\ntimeout_sec = 50.0\n'
    (task_dir / "task.toml").write_text(
        'schema_version = "1.1"\nmulti_step_reward_strategy = "mean"\n[task]\nname = "synthetic/multi"\n'
        "[agent]\ntimeout_sec = 2700\n[verifier]\ntimeout_sec = 360\n"
        '[solution]\nenv = { ORCLAW_SOLVE_TIME_LIMIT_SECONDS = "300" }\n' + steps_toml
    )
    return task_dir


class TestUpstreamFormulas:
    """Values checked against upstream's summarize_results.py at the pinned commit."""

    @pytest.mark.parametrize(
        "feasibility,quality_raw,expected",
        [(1.0, 0.81, True), (1.0, 0.8, False), (1.0, 2.0, True), (0.0, 2.0, False), (1.0, 0.0, False)],
    )
    def test_pass_predicate_is_feasible_and_normalized_quality_above_0_4(self, feasibility, quality_raw, expected):
        assert is_pass(feasibility, quality_raw) is expected

    def test_scalar_reward_matches_reward_txt(self):
        assert scalar_reward(1.0, 1.0) == pytest.approx(2 / 3)
        assert scalar_reward(1.0, 0.0) == pytest.approx(1 / 3)
        assert scalar_reward(0.0, 2.0) == 0.0


class TestLoadTask:
    def test_single_step_fixture(self):
        task = load_task(FIXTURE_TASK)
        assert task.name == "synthetic/toy_assignment"
        assert len(task.steps) == 1
        step = task.steps[0]
        assert step.name is None and step.agent_timeout_s == 120 and step.verifier_timeout_s == 60
        assert step.solution_dir is not None and step.workdir_dir is None
        assert step.solution_env == {"ORCLAW_SOLVE_TIME_LIMIT_SECONDS": "10", "ORCLAW_SCIP_GAP": "0.0005"}
        assert step.instruction.startswith("You are solving a small operations-research exercise")

    def test_multi_step_task(self, tmp_path):
        task = load_task(multi_step_task(tmp_path))
        assert [s.name for s in task.steps] == ["initial_plan", "step2"]
        assert task.steps[0].min_reward == 0.3 and task.steps[1].min_reward is None
        assert task.steps[0].workdir_dir is None and task.steps[1].workdir_dir is not None
        assert task.steps[1].instruction == "Instruction for step2\n"
        assert task.steps[1].agent_timeout_s == 100.0 and task.steps[1].verifier_timeout_s == 50.0

    def test_missing_tests_fail_closed(self, tmp_path):
        broken = tmp_path / "broken"
        shutil.copytree(FIXTURE_TASK, broken)
        (broken / "tests" / "test.sh").unlink()
        with pytest.raises(FileNotFoundError):
            load_task(broken)

    def test_dict_min_reward_is_rejected(self, tmp_path):
        task_dir = multi_step_task(tmp_path)
        toml = (task_dir / "task.toml").read_text().replace("min_reward = 0.3", "min_reward = { reward = 0.3 }")
        (task_dir / "task.toml").write_text(toml)
        with pytest.raises(ValueError, match="min_reward"):
            load_task(task_dir)


class TestModelFreeVerify:
    """verify() in a validation mode drives the whole task without a model."""

    def post(self, server, body):
        client = TestClient(server.setup_webserver())
        response = client.post("/verify", json=body)
        assert response.status_code == 200, response.text
        return response.json()

    def test_reference_solution_passes(self, monkeypatch):
        sandbox = FakeSandbox(verifier=scored(True, 1.0))
        server = make_server(sandbox, monkeypatch, validation_mode="reference")
        result = self.post(server, verify_body())
        assert result["reward"] == 1.0 and result["status"] == "scored" and result["harness_failure"] == 0.0
        assert result["failure_reason"] is None
        assert result["feasibility"] == 1.0 and result["quality"] == 0.5 and result["quality_raw"] == 1.0
        assert result["upstream_scalar_reward"] == pytest.approx(2 / 3)
        assert result["difficulty"] == "easy" and result["num_steps"] == 1 == result["steps_completed"]
        assert any(cmd.startswith("bash /solution/solve.sh") for cmd in sandbox.commands)
        assert "/tests/test.sh" in sandbox.uploads and "/solution/solve.sh" in sandbox.uploads
        assert sandbox.stopped

    def test_reference_patch_applies_exactly_once_and_only_to_the_reference(self, monkeypatch):
        sandbox = FakeSandbox(verifier=scored(True, 2.0))
        server = make_server(sandbox, monkeypatch, validation_mode="reference")
        monkeypatch.setitem(
            app_module.REFERENCE_SOLUTION_PATCHES,
            "synthetic/toy_assignment",
            {"solve_reference.py": [("brute force; optimum is 12", "brute force; patched")]},
        )
        uploaded = {}

        async def upload(local_path, remote_path):
            uploaded[remote_path] = Path(local_path).read_text()

        sandbox.upload = upload
        self.post(server, verify_body())
        assert "brute force; patched" in uploaded["/solution/solve_reference.py"]
        assert all("patched" not in text for path, text in uploaded.items() if path.startswith("/tests/"))
        monkeypatch.setitem(
            app_module.REFERENCE_SOLUTION_PATCHES,
            "synthetic/toy_assignment",
            {"solve_reference.py": [("not in the file", "x")]},
        )
        with pytest.raises(RuntimeError, match="did not match exactly once"):
            self.post(server, verify_body())

    def test_reference_time_limit_override_reaches_the_solver_env_only_in_reference_mode(self, monkeypatch):
        seen = {}

        class RecordingSandbox(FakeSandbox):
            async def exec(self, command, **kwargs):
                if command.startswith("bash /solution/solve.sh"):
                    seen.update(kwargs)
                return await super().exec(command, **kwargs)

        sandbox = RecordingSandbox(verifier=scored(True, 2.0))
        server = make_server(sandbox, monkeypatch, validation_mode="reference", reference_solve_time_limit_s=1800)
        self.post(server, verify_body())
        assert seen["env"]["ORCLAW_SOLVE_TIME_LIMIT_SECONDS"] == "1800" and seen["timeout_s"] >= 1800
        seen.clear()
        default = make_server(RecordingSandbox(verifier=scored(True, 2.0)), monkeypatch, validation_mode="reference")
        self.post(default, verify_body())
        assert seen["env"]["ORCLAW_SOLVE_TIME_LIMIT_SECONDS"] == "10" and seen["timeout_s"] == 120

    def test_upstream_reference_patches_target_real_files(self):
        for task, files in app_module.REFERENCE_SOLUTION_PATCHES.items():
            assert task.startswith("oragentbench/")
            for rel, pairs in files.items():
                assert rel == "solve_reference.py" and all(old != new for old, new in pairs)

    def test_wrong_file_control_renames_every_new_artifact(self, monkeypatch):
        sandbox = FakeSandbox(verifier=scored(False, 0.0, status="missing_solution"))
        server = make_server(sandbox, monkeypatch, validation_mode="wrong_file")
        result = self.post(server, verify_body())
        assert any(cmd.startswith("mv /app/submissions/solution.csv ") for cmd in sandbox.commands)
        assert result["reward"] == 0.0 and result["status"] == "missing_solution" and result["harness_failure"] == 0.0

    def test_no_action_control_scores_zero_and_pays_no_auxiliary_metric(self, monkeypatch):
        sandbox = FakeSandbox(verifier=scored(False, 0.0, status="missing_solution"))
        server = make_server(sandbox, monkeypatch, validation_mode="no_action")
        result = self.post(server, verify_body())
        assert not any("solve.sh" in cmd for cmd in sandbox.commands)
        assert (result["reward"], result["feasibility"], result["quality"], result["upstream_scalar_reward"]) == (
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def test_hung_process_control_still_verifies_and_tears_down(self, monkeypatch):
        sandbox = FakeSandbox(verifier=scored(False, 0.0, status="missing_solution"))
        server = make_server(sandbox, monkeypatch, validation_mode="hung_process")
        result = self.post(server, verify_body())
        assert any(cmd.startswith("nohup sleep") for cmd in sandbox.commands)
        assert result["reward"] == 0.0 and result["harness_failure"] == 0.0 and sandbox.stopped

    def test_row_may_select_the_control_only_when_the_server_is_in_a_validation_mode(self, monkeypatch):
        sandbox = FakeSandbox(verifier=scored(True, 2.0))
        server = make_server(sandbox, monkeypatch, validation_mode="reference")
        self.post(server, verify_body() | {"validation_mode": "no_action"})
        assert not any("solve.sh" in cmd for cmd in sandbox.commands)
        live = make_server(FakeSandbox(), monkeypatch)
        result = self.post(live, verify_body() | {"validation_mode": "reference"})
        assert result["status"] == Status.NO_SESSION.value

    @pytest.mark.parametrize("quality_raw,expected_reward", [(0.8, 0.0), (0.81, 1.0)])
    def test_feasible_but_low_quality_fails_the_pass_gate(self, monkeypatch, quality_raw, expected_reward):
        sandbox = FakeSandbox(verifier=scored(True, quality_raw))
        server = make_server(sandbox, monkeypatch, validation_mode="reference")
        result = self.post(server, verify_body())
        assert result["reward"] == expected_reward and result["feasibility"] == 1.0

    def test_verifier_timeout_is_the_policys_problem(self, monkeypatch):
        sandbox = FakeSandbox(exec_timeout_on_tests=True)
        server = make_server(sandbox, monkeypatch, validation_mode="no_action")
        result = self.post(server, verify_body())
        assert result["status"] == Status.VERIFIER_TIMEOUT.value
        assert result["reward"] == 0.0 and result["harness_failure"] == 0.0 and result["failure_reason"] is None

    def test_missing_reward_file_is_the_policys_problem(self, monkeypatch):
        sandbox = FakeSandbox(verifier=lambda sb: {})
        server = make_server(sandbox, monkeypatch, validation_mode="no_action")
        result = self.post(server, verify_body())
        assert result["status"] == Status.VERIFIER_OUTPUT_MISSING.value
        assert result["reward"] == 0.0 and result["harness_failure"] == 0.0

    def test_bad_task_folder_is_a_harness_fault(self, monkeypatch, tmp_path):
        sandbox = FakeSandbox()
        server = make_server(sandbox, monkeypatch, validation_mode="reference")
        result = self.post(server, verify_body(task_folder=tmp_path / "nowhere"))
        assert result["status"] == Status.BAD_TASK_FOLDER.value and result["harness_failure"] == 1.0
        assert result["failure_reason"] == HARNESS_FAULTS[Status.BAD_TASK_FOLDER] and result["reward"] == 0.0

    def test_sandbox_start_failure_is_a_harness_fault(self, monkeypatch):
        sandbox = FakeSandbox()

        async def failing_start(spec, setup):
            raise RuntimeError("docker run failed")

        sandbox.start_with_setup = failing_start
        server = make_server(sandbox, monkeypatch, validation_mode="reference")
        result = self.post(server, verify_body())
        assert result["status"] == Status.SANDBOX_FAILED.value and result["harness_failure"] == 1.0

    def test_unknown_difficulty_is_dropped_not_fatal(self, monkeypatch):
        sandbox = FakeSandbox(verifier=scored(True, 2.0))
        server = make_server(sandbox, monkeypatch, validation_mode="reference")
        for difficulty in (["easy"], 3, None, "impossible"):
            result = self.post(server, verify_body(difficulty=difficulty))
            assert result["difficulty"] is None and result["reward"] == 1.0

    def test_surrogates_in_verifier_output_survive_encoding(self, monkeypatch):
        sandbox = FakeSandbox(verifier=scored(True, 2.0, status="sc\udcffored"))
        server = make_server(sandbox, monkeypatch, validation_mode="reference")
        result = self.post(server, verify_body())
        assert result["step_results"][0]["quality_status"] == "sc?ored"


class TestMultiStep:
    def post(self, server, body):
        client = TestClient(server.setup_webserver())
        response = client.post("/verify", json=body)
        assert response.status_code == 200, response.text
        return response.json()

    def test_all_steps_feasible_averages_quality(self, monkeypatch, tmp_path):
        qualities = iter([2.0, 0.0])
        sandbox = FakeSandbox(verifier=lambda sb: scored(True, next(qualities))(sb))
        server = make_server(sandbox, monkeypatch, validation_mode="reference")
        result = self.post(server, verify_body(task_folder=multi_step_task(tmp_path)))
        assert result["num_steps"] == 2 == result["steps_completed"] and result["status"] == "scored"
        assert result["feasibility"] == 1.0 and result["quality_raw"] == 1.0 and result["reward"] == 1.0
        # Harbor applies the step workdir and setup.sh before the second step only.
        assert sum(cmd.endswith("/setup.sh") for cmd in sandbox.commands) == 1
        assert sandbox.commands.count("bash /tests/test.sh") == 2

    def test_infeasible_first_step_aborts_below_min_reward(self, monkeypatch, tmp_path):
        sandbox = FakeSandbox(verifier=scored(False, 0.0))
        server = make_server(sandbox, monkeypatch, validation_mode="reference")
        result = self.post(server, verify_body(task_folder=multi_step_task(tmp_path)))
        assert result["status"] == Status.STEP_ABORTED.value and result["steps_completed"] == 1
        assert result["feasibility"] == 0.0 and result["reward"] == 0.0 and result["harness_failure"] == 0.0
        assert sandbox.commands.count("bash /tests/test.sh") == 1

    def test_feasible_step_at_zero_quality_passes_the_gate(self, monkeypatch, tmp_path):
        # (1 + 0) / 3 = 0.333 >= 0.3: feasibility alone clears upstream's per-step gate.
        sandbox = FakeSandbox(verifier=scored(True, 0.0))
        server = make_server(sandbox, monkeypatch, validation_mode="reference")
        result = self.post(server, verify_body(task_folder=multi_step_task(tmp_path)))
        assert result["steps_completed"] == 2 and result["status"] == "scored" and result["reward"] == 0.0

    def test_setup_failure_is_a_harness_fault(self, monkeypatch, tmp_path):
        sandbox = FakeSandbox(verifier=scored(True, 2.0), setup_fails=True)
        server = make_server(sandbox, monkeypatch, validation_mode="reference")
        result = self.post(server, verify_body(task_folder=multi_step_task(tmp_path)))
        assert result["status"] == Status.STEP_SETUP_FAILED.value and result["harness_failure"] == 1.0
        assert result["reward"] == 0.0 and result["steps_completed"] == 1


class TestSessionVerify:
    def test_verify_without_a_session_is_a_harness_fault_not_a_500(self, monkeypatch):
        server = make_server(FakeSandbox(), monkeypatch)
        client = TestClient(server.setup_webserver())
        response = client.post("/verify", json=verify_body())
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == Status.NO_SESSION.value and body["harness_failure"] == 1.0 and body["reward"] == 0.0

    def test_seed_prepare_verify_step_verify_round_trip(self, monkeypatch, tmp_path):
        qualities = iter([2.0, 2.0])
        sandbox = FakeSandbox(verifier=lambda sb: scored(True, next(qualities))(sb))
        server = make_server(sandbox, monkeypatch)
        client = TestClient(server.setup_webserver())
        body = verify_body(task_folder=multi_step_task(tmp_path))
        seed = client.post("/seed_session", json=body)
        assert seed.status_code == 200, seed.text
        assert seed.json()["sandbox_handle"] == "fake-sandbox"
        assert [s["name"] for s in seed.json()["steps"]] == ["initial_plan", "step2"]
        assert [s["agent_timeout_s"] for s in seed.json()["steps"]] == [100.0, 100.0]
        for index in range(2):
            prepared = client.post("/prepare_step", json={"step_index": index})
            assert prepared.status_code == 200 and prepared.json()["setup_ok"]
            assert prepared.json()["instruction"] == f"Instruction for {['initial_plan', 'step2'][index]}\n"
            verified = client.post("/verify_step", json={"step_index": index})
            assert verified.status_code == 200 and verified.json()["stop"] is False
        result = client.post("/verify", json=body)
        assert result.status_code == 200, result.text
        assert result.json()["reward"] == 1.0 and result.json()["steps_completed"] == 2 and sandbox.stopped

    def test_stock_single_step_agent_gets_its_pending_step_verified(self, monkeypatch):
        sandbox = FakeSandbox(verifier=scored(True, 2.0))
        server = make_server(sandbox, monkeypatch)
        client = TestClient(server.setup_webserver())
        body = verify_body()
        assert client.post("/seed_session", json=body).status_code == 200
        result = client.post("/verify", json=body).json()
        assert result["reward"] == 1.0 and result["status"] == "scored" and result["steps_completed"] == 1


def aggregate(*rows: StepResult, num_steps: int, aborted: bool = False, harness: Optional[Status] = None):
    server = ORAgentBenchResourcesServer(
        config=ORAgentBenchResourcesServerConfig(
            host="", port=0, entrypoint="", name="oragentbench", sandbox_provider="sandbox"
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    steps = [
        app_module.StepSpec(
            name=str(i),
            instruction="",
            agent_timeout_s=1,
            verifier_timeout_s=1,
            min_reward=None,
            tests_dir="",
            solution_dir=None,
            workdir_dir=None,
        )
        for i in range(num_steps)
    ]
    session = app_module._Session(
        sandbox=None,
        task=TaskSpec(name="t", steps=steps),
        step_results=list(rows),
        aborted=aborted,
        harness_status=harness,
    )
    return server._aggregate(ORAgentBenchVerifyRequest.model_validate(verify_body()), session, 0.0)


class TestAggregation:
    def test_missing_step_counts_as_zero_and_kills_feasibility(self):
        result = aggregate(StepResult(name="a", status="scored", feasibility=1.0, quality_raw=2.0), num_steps=2)
        assert result.status == Status.STEP_INCOMPLETE.value
        assert result.feasibility == 0.0 and result.quality_raw == 1.0 and result.reward == 0.0
        assert result.harness_failure == 0.0 and result.failure_reason is None

    def test_first_non_scored_step_status_names_the_outcome(self):
        result = aggregate(
            StepResult(name="a", status="scored", feasibility=1.0, quality_raw=2.0),
            StepResult(name="b", status="verifier_timeout"),
            num_steps=2,
        )
        assert result.status == "verifier_timeout" and result.reward == 0.0 and result.evaluation_completed is False

    @pytest.mark.parametrize("status", list(HARNESS_FAULTS))
    def test_every_harness_fault_carries_a_reason_and_zero_everything(self, status):
        result = aggregate(
            StepResult(name="a", status="scored", feasibility=1.0, quality_raw=2.0), num_steps=1, harness=status
        )
        assert result.harness_failure == 1.0 and result.failure_reason and result.reward == 0.0
        assert result.feasibility == 0.0 and result.quality == 0.0

    @pytest.mark.parametrize("status", [s for s in Status if s not in HARNESS_FAULTS])
    def test_policy_outcomes_are_never_excused(self, status):
        assert status not in HARNESS_FAULTS and HARNESS_FAULTS.get(status) is None


class TestMetrics:
    def rows(self):
        def row(task, band, reward, feasibility, quality, harness=0.0):
            return {
                "_ng_task_index": task,
                "_ng_rollout_index": 0,
                "reward": reward,
                "difficulty": band,
                "feasibility": feasibility,
                "quality": quality,
                "quality_raw": quality * 2,
                "upstream_scalar_reward": 0.5,
                "harness_failure": harness,
                "num_steps": 1,
                "response": {"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            }

        return [
            row(0, "easy", 1.0, 1.0, 1.0),
            row(1, "easy", 0.0, 1.0, 0.2),
            row(2, "medium", 0.0, 0.0, 0.0, harness=1.0),
            row(3, "hard", 1.0, 1.0, 0.6),
        ]

    def test_key_metrics_carry_per_stratum_pass_rates_through_the_aggregate_path(self):
        server = ORAgentBenchResourcesServer(
            config=ORAgentBenchResourcesServerConfig(
                host="", port=0, entrypoint="", name="oragentbench", sandbox_provider="sandbox"
            ),
            server_client=MagicMock(spec=ServerClient),
        )
        aggregate_metrics = compute_aggregate_metrics(
            self.rows(), compute_metrics_fn=server.compute_metrics, get_key_metrics_fn=server.get_key_metrics
        )
        key = aggregate_metrics.key_metrics
        assert key["pass_rate/easy"] == 0.5 and key["pass_rate/medium"] == 0.0 and key["pass_rate/hard"] == 1.0
        assert key["feasibility_rate/easy"] == 1.0 and key["mean_quality/easy"] == pytest.approx(0.6)
        assert key["count/easy"] == 2 and key["count/medium"] == 1 and key["count/hard"] == 1
        assert key["mean/reward"] == 0.5 and key["mean/harness_failure"] == 0.25
        assert key["mean/feasibility"] == 0.75 and key["mean/quality"] == pytest.approx(0.45)
        # Diagnostics that a dashboard must not read as performance stay out of the headline.
        assert not any(k.startswith("mean/quality_raw") or "upstream_scalar" in k or "num_steps" in k for k in key)
        assert "mean/upstream_scalar_reward" in aggregate_metrics.agent_metrics


class TestShippedConfig:
    def test_yaml_agrees_with_class_defaults_and_upstream_resources(self):
        config = yaml.safe_load(CONFIG_PATH.read_text())
        server = config["oragentbench"]["resources_servers"]["oragentbench"]
        defaults = ORAgentBenchResourcesServerConfig.model_fields
        for knob in ("validation_mode", "reference_solve_timeout_s", "reference_solve_time_limit_s", "debug"):
            assert server[knob] == defaults[knob].default, knob
        # Upstream task.toml: cpus = 4, memory_mb = 8192; the benchmark runs without internet.
        assert server["sandbox_config"]["resources"] == {"cpu": 4, "memory_mib": 8192, "disk_gib": 20}
        assert config["sandbox"]["docker"]["create"]["network"] == "none"
        agent = config["oragentbench_agent"]["responses_api_agents"]["terminus_2_multi_step_sandboxed_agent"]
        assert agent["skills_dir"] == "/skills" and agent["sandbox_timeout"] == 2700
        assert config["policy_model"]["responses_api_models"]["openai_model"]["extra_body"] == {
            "reasoning": {"effort": "high"}
        }
        assert list(config)[0] == "oragentbench"
