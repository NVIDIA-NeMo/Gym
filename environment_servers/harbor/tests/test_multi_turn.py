# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor ``[[steps]]`` tasks run as turns: one agent session per turn on one sandbox, then the roll-up."""

import asyncio
import io
import json
import logging
import posixpath
import shutil
import tarfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from environment_servers.harbor.app import (
    NO_TURN_RESULT_KIND,
    HarborEpisodeRequest,
    HarborTurnResult,
    min_reward_failure,
    roll_up_turns,
)
from environment_servers.harbor.entrypoint import HealthcheckFailed
from environment_servers.harbor.sandboxes import SEED_CREATE_GRACE_S
from environment_servers.harbor.steps import HEALTHCHECK_EXEC_GRACE_S
from environment_servers.harbor.tests.conftest import (
    IMAGE_PATH,
    OK,
    TASK_TOML,
    FakeSandbox,
    agent_response,
    archive_of,
    make_server,
    write_image_config,
)
from environment_servers.harbor.verifier import COLLECT_HOOK_GRACE_S
from nemo_gym.agent_outcome import AGENT_OUTCOME_METADATA_KEY
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.tasks.harbor import load_task
from nemo_gym.tasks.harbor.digest import context_digest
from nemo_gym.tasks.harbor.materialize import STEP_METADATA_KEY, materialize_task
from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE


# The top-level instruction.md `write_task` writes: in a multi-step task it may hold what the agent must not see.
HIDDEN_SPEC = "Create hello.txt"


# The task-level tests/test.sh `write_task` writes; each step's own script differs from it (`step_test_sh`).
TASK_TEST_SH = "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"


def step_test_sh(name: str) -> str:
    """The test.sh of step ``name``: its own script, so an upload of another step's tests is told apart."""
    return f"#!/bin/bash\n# step {name}\necho 1 > /logs/verifier/reward.txt\n"


def write_steps(task_dir: Path, steps: dict[str, str]) -> None:
    """``steps`` maps each step name to its instruction; every step gets a test.sh of its own (:func:`step_test_sh`)."""
    for name, instruction in steps.items():
        (task_dir / "steps" / name / "tests").mkdir(parents=True, exist_ok=True)
        (task_dir / "steps" / name / "instruction.md").write_text(instruction)
        (task_dir / "steps" / name / "tests" / "test.sh").write_text(step_test_sh(name))


def multi_turn_server(
    tmp_path, monkeypatch, steps_toml: str, steps: dict[str, str], *, head: str = "", base: str = TASK_TOML, **config
):
    """``head`` holds top-level keys, which TOML only reads before the first table; ``base`` is the task's own
    task.toml the steps are appended to."""
    task_dir = tmp_path / "datasets" / "ds" / "hello"
    write_steps(task_dir, steps)
    return make_server(tmp_path, monkeypatch, toml=head + base + steps_toml, **config)


def rewards_per_turn(server, sandbox: FakeSandbox, rewards: list[str], *, name: str = "reward.txt") -> None:
    """Make the verifier of the n-th turn write ``rewards[n]``: each activation sets what its verify reads."""
    pending = list(rewards)

    def next_reward():
        sandbox.verifier_files = {name: pending.pop(0)}

    server.server_client.turn_hooks.append(next_reward)


def request_for(task, rollout: str = "r1") -> HarborEpisodeRequest:
    return HarborEpisodeRequest.model_validate(
        {"episode_id": {"rollout_id": rollout}, "task": materialize_task(task, "ds").model_dump(mode="json")}
    )


def uploaded_files(sandbox: FakeSandbox, name: str) -> list[bytes]:
    """The contents of ``name`` in every archive the server uploaded, in upload order."""
    found: list[bytes] = []
    for data in sandbox.uploaded_archives:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and posixpath.basename(member.name) == name:
                    found.append(tar.extractfile(member).read())
    return found


THREE_STEPS = """
[[steps]]
name = "plan"
[steps.agent]
timeout_sec = 60.0

[[steps]]
name = "build"
[steps.verifier]
timeout_sec = 30.0
env = { STEP = "build" }

[[steps]]
name = "check"
"""


async def test_each_step_is_a_turn_in_a_fresh_session_on_one_sandbox(tmp_path, monkeypatch):
    server, task, sandbox, created = multi_turn_server(
        tmp_path,
        monkeypatch,
        THREE_STEPS,
        {"plan": "Agree on a plan.", "build": "Build it.", "check": "Hand off."},
        head='multi_step_reward_strategy = "final"\n',
    )
    rewards_per_turn(server, sandbox, ["1", "1", "0"])

    response = await server.run_request(request_for(task))

    result = response.result
    assert [turn.name for turn in result.turns] == ["plan", "build", "check"]
    assert [turn.reward for turn in result.turns] == [1.0, 1.0, 0.0]
    # final: the last turn decides, even though earlier turns passed.
    assert (result.reward, result.turn_reward_strategy, result.stopped_after_turn) == (0.0, "final", None)
    # One sandbox for the whole episode, one agent session per turn.
    assert len(created) == 1
    seeds = server.server_client.seed_bodies()
    assert len({seed["agent_session_id"] for seed in seeds}) == 3
    assert len({str(seed["sandbox_access"]) for seed in seeds}) == 1
    # Each turn sends its own step's instruction and budget; the top-level instruction never leaves the server.
    turns = [
        call[2]["json"].model_dump(mode="json", exclude_unset=True)
        for call in server.server_client.calls
        if call[1].endswith("/v1/responses")
    ]
    assert [turn["input"][0]["content"] for turn in turns] == ["Agree on a plan.", "Build it.", "Hand off."]
    assert [turn["metadata"]["agent_timeout_sec"] for turn in turns] == ["60.0", "300.0", "300.0"]
    # The step name stays in the row (it names the turn in the result); the harness gets only the budget, since
    # OpenCode, Hermes, Claude Code and OpenClaw refuse any other metadata key.
    assert [list(turn["metadata"]) for turn in turns] == [["agent_timeout_sec"]] * 3
    rows = request_for(task).task.task_input.turns
    assert [row.responses_create_params.metadata[STEP_METADATA_KEY] for row in rows] == ["plan", "build", "check"]
    assert HIDDEN_SPEC not in str(server.server_client.calls)
    # The build step's verifier runs with its own timeout and environment over the task's.
    # The chmod of test.sh before each run is its own exec; the verifier's run is the one under `timeout`.
    build_verify = [e for e in sandbox.execs if "test.sh" in e["command"] and "timeout" in e["command"]][1]
    assert "timeout -s KILL 30 " in build_verify["command"] and build_verify["env"] == {"STEP": "build"}
    # Each turn uploads the task's tests/ and then its own step's over it, so the step's test.sh is the one graded.
    assert uploaded_files(sandbox, "test.sh") == [
        TASK_TEST_SH.encode(),
        step_test_sh("plan").encode(),
        TASK_TEST_SH.encode(),
        step_test_sh("build").encode(),
        TASK_TEST_SH.encode(),
        step_test_sh("check").encode(),
    ]


async def test_a_task_with_step_level_tests_alone_grades_each_turn_with_its_step(tmp_path, monkeypatch):
    """Harbor reads a multi-step task's tests from its steps: the top-level tests/ is optional, and each turn then
    uploads its own step's tests/ alone and is graded by that script's reward."""
    server, task, sandbox, _ = multi_turn_server(
        tmp_path, monkeypatch, THREE_STEPS, {"plan": "P", "build": "B", "check": "C"}
    )
    from nemo_gym.tasks.harbor import load_task

    (task.path / "tests" / "test.sh").unlink()
    (task.path / "tests").rmdir()
    task = load_task(task.path)
    server.config.tasksets["ds"].tasks["hello"] = task.digest
    rewards_per_turn(server, sandbox, ["1", "0", "1"])

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert [turn.reward for turn in response.result.turns] == [1.0, 0.0, 1.0]
    assert uploaded_files(sandbox, "test.sh") == [step_test_sh(name).encode() for name in ("plan", "build", "check")]


async def test_a_turn_whose_failure_the_harness_classified_ends_the_episode_ungraded(tmp_path, monkeypatch):
    """The deployment's ``agent_outcome_policy`` applies to every turn: a transport or unknown cause on the second
    step is a retryable agent-stage failure of the episode, so the step's test.sh never runs on the half-finished
    sandbox and the first step's grading does not stand in for the task."""
    server, task, sandbox, _ = multi_turn_server(
        tmp_path, monkeypatch, THREE_STEPS, {"plan": "Agree on a plan.", "build": "Build it.", "check": "Hand off."}
    )
    rewards_per_turn(server, sandbox, ["1", "1", "1"])
    failed = agent_response().model_dump(mode="json")
    failed.update(
        status="failed",
        error={"code": "server_error", "message": "stream disconnected before completion"},
        metadata={AGENT_OUTCOME_METADATA_KEY: '{"failure_kind":"unknown"}'},
    )
    activations = iter([None, failed, None])

    def fail_the_second_turn():
        server.server_client.turn_response = next(activations)

    server.server_client.turn_hooks.append(fail_the_second_turn)

    response = await server.run_request(request_for(task))

    assert response.result is None
    failure = response.failure
    assert (failure.stage, failure.terminal) == ("agent", False)
    assert failure.failure_reason == "stream disconnected before completion"
    assert failure.partial_response.status == "failed"
    # Only the first turn was graded; the failed turn's verifier never ran and the third turn never started.
    assert len([e for e in sandbox.execs if "test.sh" in e["command"] and "timeout" in e["command"]]) == 1
    assert len(server.server_client.seed_bodies()) == 2
    assert sandbox.stopped


async def test_mean_averages_the_turns_that_produced_a_result(tmp_path, monkeypatch):
    steps = '[[steps]]\nname = "a"\n[[steps]]\nname = "b"\n'
    server, task, sandbox, _ = multi_turn_server(tmp_path, monkeypatch, steps, {"a": "A", "b": "B"})
    rewards_per_turn(server, sandbox, ["1", "0.5"])

    result = (await server.run_request(request_for(task))).result

    assert result.turn_reward_strategy == "mean"
    assert result.reward == 0.75 and result.verifier_rewards == {"reward": 0.75}
    assert result.reward_components is None and all(turn.reward_components is None for turn in result.turns)


async def test_mean_averages_the_components_every_turn_named(tmp_path, monkeypatch):
    steps = '[[steps]]\nname = "a"\n[[steps]]\nname = "b"\n'
    server, task, sandbox, _ = multi_turn_server(tmp_path, monkeypatch, steps, {"a": "A", "b": "B"})
    rewards_per_turn(server, sandbox, [json.dumps({"x": 1.0, "y": 0.0}), json.dumps({"x": 0.0})], name="reward.json")

    result = (await server.run_request(request_for(task))).result

    assert [turn.reward_components for turn in result.turns] == [{"x": 1.0, "y": 0.0}, {"x": 0.0}]
    assert result.reward == 0.5 and result.reward_components == {"x": 0.5, "y": 0.0}
    assert sum(result.reward_components.values()) == result.reward


async def test_the_harness_trajectory_rides_on_each_turn_and_the_last_one_on_the_result(tmp_path, monkeypatch):
    """Every turn's session close carries the harness's own account of that turn; the episode forwards the last
    one in ``ng_trajectory``, the slot rollout collection reads, as it forwards the last turn's observations."""
    steps = '[[steps]]\nname = "a"\n[[steps]]\nname = "b"\n'
    server, task, sandbox, _ = multi_turn_server(tmp_path, monkeypatch, steps, {"a": "A", "b": "B"})
    rewards_per_turn(server, sandbox, ["1", "1", "1", "1"])
    server.server_client.trajectory = {
        "task_id": "hello",
        "rollout_id": "r1",
        "turns": [
            {
                "invocation_id": "harness_1",
                "task_id": "hello",
                "rollout_id": "r1",
                "turn_no": 1,
                "timestamp": 1.0,
                "question": "do it",
                "answer": [],
                "step_count": 2,
                "model_calls": [],
            }
        ],
    }

    result = (await server.run_request(request_for(task))).result

    assert [turn.trajectory.turns[0].step_count for turn in result.turns] == [2, 2]
    assert result.ng_trajectory is result.turns[-1].trajectory
    dumped = result.model_dump(mode="json")
    assert dumped["ng_trajectory"]["turns"][0]["step_count"] == 2
    # A harness that returns no trajectory leaves the key out of each turn, as it does on the result.
    server.server_client.trajectory = None
    dumped = (await server.run_request(request_for(task))).result.model_dump(mode="json")
    assert "trajectory" not in dumped["turns"][0] and "ng_trajectory" not in dumped


async def test_a_missed_min_reward_stops_the_remaining_turns(tmp_path, monkeypatch):
    steps = '[[steps]]\nname = "plan"\nmin_reward = 1.0\n[[steps]]\nname = "build"\n'
    server, task, sandbox, _ = multi_turn_server(tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"})
    rewards_per_turn(server, sandbox, ["0"])

    result = (await server.run_request(request_for(task))).result

    assert [turn.name for turn in result.turns] == ["plan"]
    assert result.stopped_after_turn == "plan" and result.reward == 0.0
    assert sum(path.endswith("/v1/responses") for path in server.server_client.paths()) == 1


async def test_a_turn_without_a_reward_file_ends_the_episode_measured(tmp_path, monkeypatch):
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n'
    server, task, sandbox, _ = multi_turn_server(tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"})
    sandbox.verifier_files = {}

    result = (await server.run_request(request_for(task))).result

    assert [turn.verifier_rewards for turn in result.turns] == [None]
    assert (result.reward, result.mask_sample, result.turns_without_verifier_result) == (0.0, False, 1)
    assert result.failure_kind == "harbor:missing_reward"


# The command that empties /tests and /logs/verifier (and removes the server's transcript) in the agent's sandbox
# before the next turn's agent; the shared verifier's own reset before test.sh empties /tests alone and removes only
# the grader's files from /logs/verifier, so it does not match.
GRADING_RESET = "find /tests /logs/verifier -mindepth 1 -delete"


async def test_the_previous_turns_grading_files_are_gone_before_the_next_agent_starts(tmp_path, monkeypatch):
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n'
    server, task, sandbox, _ = multi_turn_server(tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"})
    # What had run in the sandbox when each agent turn started.
    at_activation: list[list[dict]] = []
    server.server_client.turn_hooks.append(lambda: at_activation.append(list(sandbox.execs)))

    result = (await server.run_request(request_for(task))).result

    assert [turn.reward for turn in result.turns] == [1.0, 1.0]
    first, second = at_activation
    # The first turn keeps whatever the image put in /tests: nothing is reset before it.
    assert not any(GRADING_RESET in record["command"] for record in first)
    # Before the second agent starts, the first turn's tests, reward and transcript are removed, as root, after
    # the first turn's test.sh ran.
    ran_tests = max(i for i, record in enumerate(second) if "bash /tests/test.sh" in record["command"])
    reset = max(i for i, record in enumerate(second) if GRADING_RESET in record["command"])
    assert reset > ran_tests
    assert second[reset]["user"] == "root"
    assert "rm -rf -- /logs/agent/trajectory.json" in second[reset]["command"]


async def test_a_sandbox_that_cannot_clear_the_grading_files_ends_the_episode_before_the_next_agent(
    tmp_path, monkeypatch
):
    class KeepsGradingFiles(FakeSandbox):
        resets = 0

        async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
            if GRADING_RESET in command:
                self.resets += 1
                if self.resets == 1:  # the handover; the first turn's own verification clears the grader's files only
                    return SandboxExecResult(stdout="", stderr="find: Permission denied", return_code=1)
            return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

    task_dir = tmp_path / "datasets" / "ds" / "hello"
    write_steps(task_dir, {"plan": "P", "build": "B"})
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n'
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch, KeepsGradingFiles(), toml=TASK_TOML + steps)

    response = await server.run_request(request_for(task))

    # Not scored and retried elsewhere: a provider that cannot clean is not the task's fault.
    assert response.result is None and response.failure.stage == "agent" and not response.failure.terminal
    assert "turn 'build'" in response.failure.failure_reason
    assert "Could not clear /tests, /logs/verifier and the transcript" in response.failure.failure_reason
    # The second agent session was never opened beside the first turn's grading files.
    assert len(server.server_client.seed_bodies()) == 1
    assert sandbox.stopped


async def test_rows_whose_turns_differ_from_the_task_fail_terminally(tmp_path, monkeypatch):
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n'
    server, task, _, _ = multi_turn_server(tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"})
    request = request_for(task)
    request.task.task_input.turns.pop()

    response = await server.run_request(request)

    assert response.failure.terminal and "do not match the steps" in response.failure.failure_reason


async def test_each_turn_seeds_its_own_budget_and_hands_its_own_transcript_to_the_verifier(tmp_path, monkeypatch):
    """The single-turn protocol holds per turn: the harness is seeded with the step's agent budget, and the
    verifier reads an ATIF transcript of that turn's own request, so the hidden top-level instruction never
    reaches /logs/agent/trajectory.json either."""
    server, task, sandbox, _ = multi_turn_server(
        tmp_path, monkeypatch, THREE_STEPS, {"plan": "Agree on a plan.", "build": "Build it.", "check": "Hand off."}
    )

    response = await server.run_request(request_for(task))

    result = response.result
    assert result is not None, response.failure
    assert [seed["agent_timeout_sec"] for seed in server.server_client.seed_bodies()] == [60.0, 300.0, 300.0]
    transcripts = [json.loads(data) for data in uploaded_files(sandbox, "trajectory.json")]
    assert [steps["steps"][0]["source"] for steps in transcripts] == ["user"] * 3
    assert [steps["steps"][0]["message"] for steps in transcripts] == ["Agree on a plan.", "Build it.", "Hand off."]
    assert HIDDEN_SPEC not in json.dumps(transcripts)
    # Each turn's verifier run reports as a single turn's does, under its own logs folder.
    assert [turn.verifier_network for turn in result.turns] == ["public"] * 3
    assert len({turn.verifier_logs_dir for turn in result.turns}) == 3
    assert all((Path(turn.verifier_logs_dir) / "trajectory.json").is_file() for turn in result.turns)


async def test_the_task_deadline_counts_every_turn(tmp_path, monkeypatch):
    """Each step is a harness session, the hooks and a verifier run of its own; the episode budget sums them."""
    server, task, _, _ = multi_turn_server(
        tmp_path,
        monkeypatch,
        THREE_STEPS,
        {"plan": "P", "build": "B", "check": "C"},
        harness_setup_allowance_s=10,
        verifier_grace_s=5,
    )

    # plan: agent 60 (its own), verifier 600 (Harbor's step default: the task's 120 never reaches a step); build: 300
    # and 30 (its own); check: 300 and 600.
    per_turn = (10 + 60 + 600 + 5) + (10 + 300 + 30 + 5) + (10 + 300 + 600 + 5)
    assert server._post_seed_budget_s(task) == pytest.approx(per_turn)
    assert server._task_budget_s(task) == pytest.approx(server.config.sandbox_ready_timeout_s + per_turn)


async def test_the_task_deadline_ends_an_overrunning_turn(tmp_path, monkeypatch):
    steps = """
[[steps]]
name = "plan"
[steps.agent]
timeout_sec = 0.05
[steps.verifier]
timeout_sec = 0.05

[[steps]]
name = "build"
[steps.agent]
timeout_sec = 0.05
[steps.verifier]
timeout_sec = 0.05
"""
    server, task, sandbox, _ = multi_turn_server(
        tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"}, harness_setup_allowance_s=0, verifier_grace_s=0
    )
    assert server._post_seed_budget_s(task) == pytest.approx(0.2)
    server.server_client.slow_path = "/ng-rollout/r1/v1/responses"

    response = await server.run_request(request_for(task))

    failure = response.failure
    assert failure.stage == "agent" and failure.failure_kind == "harbor:task_deadline" and not failure.terminal
    assert "time budget of task 'hello'" in failure.failure_reason and "over 2 turn(s)" in failure.failure_reason
    assert not any("test.sh" in call["command"] for call in sandbox.execs)
    # The final unwind still closed the agent session and stopped the sandbox.
    assert server.server_client.paths()[-1] == "/v1/agent_sessions/close"
    assert sandbox.stopped and not server._sandboxes and server._active == set()

    # The same slow harness finishes both turns when the run ignores task timeouts.
    server.config.enforce_task_timeouts = False
    server.server_client.slow_path = "/ng-rollout/r2/v1/responses"
    result = (await server.run_request(request_for(task, rollout="r2"))).result
    assert [turn.name for turn in result.turns] == ["plan", "build"] and result.reward == 1.0


@pytest.mark.parametrize("provider", ["docker", "opensandbox"])
async def test_a_step_verifier_user_with_a_group_follows_the_provider_at_seed(tmp_path, monkeypatch, provider):
    """Review (round 24): a step's `[steps.verifier].user = "name:group"` reached `_run_shared_verifier` as written
    through `_turn_verifier_settings` on every provider, while the task-level `[verifier].user` pair had been refused
    at seed (`_refuse_verifier_group`) where only Docker applies a group. Every step's pair is refused the same way
    now: at seed, before the sandbox exists and the turns before that step are paid for. The refusal names the step
    that carries the pair (repair round 4), as the loader's per-step refusals do, so the author of a six-step task need
    not search every `[steps.verifier]` table for it."""
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n[steps.verifier]\nuser = "tester:grp"\n'
    server, task, sandbox, created = multi_turn_server(tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"})
    assert task.config.verifier.user is None and task.turns[1].config.verifier.user == "tester:grp"
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {provider: {}}}
    )

    response = await server.run_request(request_for(task))

    runs = [call for call in sandbox.execs if "bash /tests/test.sh" in call["command"]]
    if provider == "docker":
        assert response.failure is None, response.failure
        assert [turn.name for turn in response.result.turns] == ["plan", "build"]
        # The first turn grades as the task's verifier user; the second as the step's own pair, as written.
        assert [run["user"] for run in runs] == [None, "tester:grp"]
    else:
        failure = response.failure
        assert failure is not None and failure.stage == "seed" and failure.terminal
        # The step that carries the pair is named, right after the task id; the task's own user is not blamed.
        assert "hello step 'build': user 'tester:grp' names group 'grp'" in failure.failure_reason
        assert "hello: user" not in failure.failure_reason and "'plan'" not in failure.failure_reason
        assert "'opensandbox'" in failure.failure_reason and "switches user by name only" in failure.failure_reason
        # Nothing ran: no sandbox, no agent session, no turn.
        assert created == [] and sandbox.execs == [] and server.server_client.seed_bodies() == []
        assert runs == [] and response.result is None


def solution_uploads(sandbox: FakeSandbox) -> list[dict]:
    return [c for c in sandbox.execs if "tar -xzf" in c["command"] and "-C /solution" in c["command"]]


def write_step_solution(task_dir: Path, name: str, script: str) -> None:
    (task_dir / "steps" / name / "solution").mkdir(parents=True, exist_ok=True)
    (task_dir / "steps" / name / "solution" / "solve.sh").write_text(script)


async def test_the_oracle_gets_each_steps_own_solution_before_its_turn(tmp_path, monkeypatch):
    """G7 round 24, attempt 4: the MemoryArena task keeps its reference solutions in `steps/<name>/solution/` and
    the server uploaded the task's top-level `solution/` once at seed, so the oracle had nothing to run. Harbor
    0.22.0's oracle resolves the step's own folder per step and falls back to the task's (`agents/oracle.py`
    `_resolve_solution_paths`); the server now places that folder at /solution before each turn, emptying what the
    previous turn left, so each turn's `/solution/solve.sh` is its step's and a step without any leaves none."""
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n[[steps]]\nname = "check"\n'
    task_dir = tmp_path / "datasets" / "ds" / "hello"
    write_steps(task_dir, {"plan": "P", "build": "B", "check": "C"})
    write_step_solution(task_dir, "plan", "#!/bin/bash\necho plan\n")
    write_step_solution(task_dir, "check", "#!/bin/bash\necho check\n")
    # No top-level solution/: `build` has none anywhere; the loader reads the folder as it is.
    server, task, sandbox, _ = make_server(
        tmp_path, monkeypatch, toml=TASK_TOML + steps, agent_implementation="harbor_oracle_agent"
    )
    assert not task.has_solution and not (task.path / "solution").exists()
    uploads_per_turn: list[int] = []
    server.server_client.turn_hooks.append(lambda: uploads_per_turn.append(len(solution_uploads(sandbox))))

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert [turn.name for turn in response.result.turns] == ["plan", "build", "check"]
    # Nothing at seed; one upload before the first turn and before the third, none for the step without a solution.
    assert uploads_per_turn == [1, 1, 2]
    assert [data.decode() for data in uploaded_files(sandbox, "solve.sh")] == [
        "#!/bin/bash\necho plan\n",
        "#!/bin/bash\necho check\n",
    ]
    uploads = solution_uploads(sandbox)
    assert all(upload["user"] == "root" and "chmod -R a+rX /solution" in upload["command"] for upload in uploads)
    # /solution is emptied before every turn, as root, so `build` runs with no previous step's script in place.
    resets = [i for i, c in enumerate(sandbox.execs) if c["command"] == "rm -rf -- /solution"]
    assert len(resets) == 3 and all(sandbox.execs[i]["user"] == "root" for i in resets)
    upload_indexes = [i for i, c in enumerate(sandbox.execs) if c in uploads]
    assert resets[0] < upload_indexes[0] < resets[1] < resets[2] < upload_indexes[1]

    # The task's top-level solution/ stands in for a step without its own, as Harbor's oracle falls back to it.
    (task.path / "solution").mkdir()
    (task.path / "solution" / "solve.sh").write_text("#!/bin/bash\necho task\n")
    task = load_task(task.path)
    server.config.tasksets["ds"].tasks["hello"] = task.digest
    assert task.has_solution
    sandbox.execs.clear()
    sandbox.uploaded_archives.clear()
    response = await server.run_request(request_for(task, rollout="r2"))
    assert response.failure is None, response.failure
    assert [data.decode() for data in uploaded_files(sandbox, "solve.sh")] == [
        "#!/bin/bash\necho plan\n",
        "#!/bin/bash\necho task\n",
        "#!/bin/bash\necho check\n",
    ]


async def test_other_harnesses_never_get_a_steps_solution(tmp_path, monkeypatch):
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n'
    task_dir = tmp_path / "datasets" / "ds" / "hello"
    write_steps(task_dir, {"plan": "P", "build": "B"})
    write_step_solution(task_dir, "plan", "#!/bin/bash\necho plan\n")
    server, task, sandbox, _ = make_server(
        tmp_path, monkeypatch, toml=TASK_TOML + steps, agent_implementation="hermes_agent"
    )

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert not any("/solution" in c["command"] for c in sandbox.execs)
    assert uploaded_files(sandbox, "solve.sh") == []


TWO_USERS = """
[[steps]]
name = "plan"

[[steps]]
name = "build"
[steps.agent]
user = "builder"

[[steps]]
name = "check"
[steps.agent]
user = 1000
"""


async def test_each_turn_runs_as_its_steps_agent_user(tmp_path, monkeypatch):
    """Harbor runs each step as its own `[steps.agent].user` (else the task's `[agent].user`): the sandbox's
    default user, the working directory's owner and the user the harness is handed switch with the turn, once,
    when the step's user differs from the one in place; a numeric uid is resolved to its name as at seed, and a
    step without a user runs as the task's. The verifier keeps its own fallback (the task image's user)."""
    server, task, sandbox, _ = multi_turn_server(
        tmp_path,
        monkeypatch,
        TWO_USERS,
        {"plan": "P", "build": "B", "check": "C"},
        agent_implementation="harbor_oracle_agent",
    )
    write_step_solution(task.path, "plan", "#!/bin/bash\necho plan\n")
    task = load_task(task.path)
    server.config.tasksets["ds"].tasks["hello"] = task.digest
    users_at_turn: list = []
    server.server_client.turn_hooks.append(lambda: users_at_turn.append(sandbox.default_user))

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert [turn.name for turn in response.result.turns] == ["plan", "build", "check"]
    # The task's user is root (no [agent].user, a root image): the first turn keeps the image's default.
    assert users_at_turn == [None, "builder", "agent"]
    assert [seed["sandbox_access"].get("user") for seed in server.server_client.seed_bodies()] == [
        None,
        "builder",
        "agent",
    ]
    # The working directory is handed to each new user as root, after the seed's own mkdir.
    chowns = [c for c in sandbox.execs if c["command"].startswith("chown ")]
    assert [(c["command"], c["user"]) for c in chowns] == [
        ("chown builder /app", "root"),
        ("chown agent /app", "root"),
    ]
    # test.sh never runs as the step's agent user: the verifier keeps its own user (the image's, here plain root,
    # which is the facade's `None` path). The facade runs a `None` exec as the sandbox's default, which the turn
    # switched to `builder` and `agent`, so the seed's default is put back around the verifier: the container ran
    # every test.sh as root, as Harbor's shared verifier does outside the step's `with_default_user`.
    runs = [c for c in sandbox.execs if "bash /tests/test.sh" in c["command"]]
    assert [run["user"] for run in runs] == [None, None, None]
    assert [run["ran_as"] for run in runs] == ["root", "root", "root"]
    # The step's user is back in place after its verifier, for the next turn's switch to compare with.
    assert sandbox.default_user == "agent"
    # The uid was looked up once, by name, as root.
    lookups = [c for c in sandbox.execs if c["command"].startswith("getent passwd 1000")]
    assert len(lookups) == 1
    # A step solution placed for the first turn was uploaded before any switch; the switch leaves it readable.
    assert uploaded_files(sandbox, "solve.sh") == [b"#!/bin/bash\necho plan\n"]


@pytest.mark.parametrize("image_user", ["root", "svc"])
async def test_the_turn_verifier_runs_as_its_own_user_not_the_steps(tmp_path, monkeypatch, image_user):
    """Round 24 (G11 review): with the image's user unnamed, `_verifier_user` leaves test.sh to the sandbox's
    default, and the facade substitutes whatever default the turn installed, so after a step switched to `builder`
    the verifier ran as `builder`, not as the image's user as under Harbor. The default is now the seed's again
    around every turn's verifier (hooks and test.sh), whether the image runs as root or as a non-root user."""
    image_config = {"Env": ["PATH=/usr/bin"]} | ({"User": image_user} if image_user != "root" else {})
    server, task, sandbox, _ = multi_turn_server(
        tmp_path,
        monkeypatch,
        TWO_USERS,
        {"plan": "P", "build": "B", "check": "C"},
        sandbox=FakeSandbox(image_user=image_user, passwd={"0": "root", "1000": "agent", "1001": "svc"}),
        image_config=image_config,
    )
    seed_user = None if image_user == "root" else image_user
    assert task.user == seed_user and task.entrypoint_user == seed_user
    users_at_turn: list = []
    server.server_client.turn_hooks.append(lambda: users_at_turn.append(sandbox.default_user))

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert users_at_turn == [seed_user, "builder", "agent"]
    runs = [c for c in sandbox.execs if "bash /tests/test.sh" in c["command"]]
    # A recorded non-root USER is named outright; root is the facade's default path, restored to the seed's.
    assert [run["user"] for run in runs] == [seed_user] * 3
    assert [run["ran_as"] for run in runs] == [image_user] * 3
    assert sandbox.default_user == "agent"


@pytest.mark.parametrize("image_user", ["root", "svc"])
async def test_a_step_back_on_the_images_default_user_gets_the_working_directory_back(
    tmp_path, monkeypatch, image_user
):
    """Round 24 (G11 review): a step with no user of its own on a task without `[agent].user` switches the sandbox
    back to the image's default (`None`), which `_own_workdir` took for root and skipped, so the working directory
    stayed the previous step's user's. The owner the seed left is read once, before the first switch, and the
    directory is handed back to it, by uid and gid, whether the image runs as root or an unrecorded non-root user
    (the seed created the directory as root either way, as a provider does with the spec's workdir)."""
    steps = '[[steps]]\nname = "a"\n[steps.agent]\nuser = "builder"\n[[steps]]\nname = "b"\n[[steps]]\nname = "c"\n'
    server, task, sandbox, _ = multi_turn_server(
        tmp_path,
        monkeypatch,
        steps,
        {"a": "A", "b": "B", "c": "C"},
        sandbox=FakeSandbox(image_user=image_user, passwd={"0": "root", "1000": "agent", "1001": "builder"}),
    )
    assert task.user is None
    owners_at_turn: list = []
    server.server_client.turn_hooks.append(lambda: owners_at_turn.append(sandbox.dirs["/app"]))

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert [turn.name for turn in response.result.turns] == ["a", "b", "c"]
    # `builder` owned /app for its step; `b` runs as the image's default and finds the directory as the seed left it.
    assert owners_at_turn == ["builder", "root", "root"]
    commands = [c for c in sandbox.execs if c["command"].startswith(("stat ", "chown "))]
    assert [(c["command"], c["ran_as"]) for c in commands] == [
        ("stat -c %u:%g /app", "root"),
        ("chown builder /app", "root"),
        ("chown 0:0 /app", "root"),
    ]


async def test_consecutive_steps_with_the_same_user_switch_once_and_a_later_step_switches_back(tmp_path, monkeypatch):
    steps = (
        TASK_TOML.replace("[agent]\n", '[agent]\nuser = "agent"\n')
        + '[[steps]]\nname = "a"\n[steps.agent]\nuser = "builder"\n'
        + '[[steps]]\nname = "b"\n[steps.agent]\nuser = "builder"\n'
        + '[[steps]]\nname = "c"\n'
    )
    task_dir = tmp_path / "datasets" / "ds" / "hello"
    write_steps(task_dir, {"a": "A", "b": "B", "c": "C"})
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch, toml=steps)
    assert task.user == "agent"
    users_at_turn: list = []
    server.server_client.turn_hooks.append(lambda: users_at_turn.append(sandbox.default_user))

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert users_at_turn == ["builder", "builder", "agent"]
    chowns = [c["command"] for c in sandbox.execs if c["command"].startswith("chown ") or "&& chown " in c["command"]]
    # The seed's own hand-over to the task user, then one per switch: to builder before `a`, back to agent before `c`.
    assert chowns == ["mkdir -p /app && chown agent /app", "chown builder /app", "chown agent /app"]


@pytest.mark.parametrize("provider", ["docker", "opensandbox"])
async def test_a_step_agent_user_with_a_group_follows_the_provider_at_seed(tmp_path, monkeypatch, provider):
    """A step's `[steps.agent].user = "name:group"` is as static as the task's own: Docker's `--user` takes the
    pair, every other provider switches user by name only, so the pair is refused at seed there, naming the step,
    before any turn is paid for (the rule M16/M20 set for the task's `[agent].user`)."""
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n[steps.agent]\nuser = "builder:grp"\n'
    server, task, sandbox, created = multi_turn_server(tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"})
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {provider: {}}}
    )

    response = await server.run_request(request_for(task))

    if provider == "docker":
        assert response.failure is None, response.failure
        assert [turn.name for turn in response.result.turns] == ["plan", "build"]
        assert sandbox.default_user == "builder:grp"
        assert server.server_client.seed_bodies()[1]["sandbox_access"]["user"] == "builder:grp"
    else:
        failure = response.failure
        assert failure is not None and failure.stage == "seed" and failure.terminal
        assert "hello step 'build': user 'builder:grp' names group 'grp'" in failure.failure_reason
        assert created == [] and sandbox.execs == [] and server.server_client.seed_bodies() == []


LONG_STEPS = """
[[steps]]
name = "plan"
[steps.agent]
timeout_sec = 0.3
[steps.verifier]
timeout_sec = 0.3

[[steps]]
name = "build"
[steps.agent]
timeout_sec = 0.3
[steps.verifier]
timeout_sec = 0.3
"""


async def test_a_multi_step_task_longer_than_the_default_gets_its_own_budget(tmp_path, monkeypatch, caplog):
    """G7 round 24: MemoryArena's six steps budget 21 780 s, above `default_episode_timeout_seconds` (21 600), and
    the seed only warned, so a real run would be cut off at the default and retried, never scored. With
    `enforce_task_timeouts` the episode deadline and the sandbox lifetime now follow the task's own budget when it
    is the longer one, the way the single-turn path already derives the deadline from the task."""
    server, task, sandbox, _ = multi_turn_server(
        tmp_path,
        monkeypatch,
        LONG_STEPS,
        {"plan": "P", "build": "B"},
        default_episode_timeout_seconds=0.1,
        harness_setup_allowance_s=0,
        verifier_grace_s=0,
        sandbox_ready_timeout_s=1,
        image_pull_retries=0,
    )
    # Two turns of 0.3 + 0.3 s once the sandbox is ready, and the seed's own bound before that (one attempt of
    # the ready timeout plus the create grace): well above the run-wide default of 0.1 s.
    seed_bound = 1 + SEED_CREATE_GRACE_S
    assert server._post_seed_budget_s(task) == pytest.approx(1.2)
    assert server._seed_bound_s(task) == pytest.approx(seed_bound)
    assert server._episode_bound_s(task) == pytest.approx(seed_bound + 1.2)
    # The nominal one-start allowance that sizes a sandbox lifetime within the default is unchanged.
    assert server._task_budget_s(task) == pytest.approx(2.2)
    # The episode deadline (opened before the seed) and the sandbox lifetime follow the whole bound.
    assert server._episode_budget_s(task) == pytest.approx(seed_bound + 1.2)
    assert await server.episode_timeout_seconds(request_for(task)) == pytest.approx(seed_bound + 1.2)
    # A single-turn task keeps the run-wide cap.
    single_server, single_task, _, _ = make_server(
        tmp_path / "single", monkeypatch, default_episode_timeout_seconds=0.1
    )
    assert single_server._episode_budget_s(single_task) == 0.1
    assert await single_server.episode_timeout_seconds(request_for(single_task)) == 0.1
    # The harness takes 0.2 s per turn: over the 0.1 s default on its own, and the two turns leave 0.8 s of their
    # 1.2 s budget to the verifier runs and the filesystem work (round 32: 0.5 s turns left 0.2 s and failed on a
    # loaded node), so both turns are scored. `test_the_task_deadline_ends_an_overrunning_turn` keeps the deadline
    # firing on a turn that outruns the budget.
    server.server_client.slow_path = "/ng-rollout/r1/v1/responses"
    server.server_client.slow_seconds = 0.2
    with caplog.at_level("WARNING"):
        response = await server.run_request(request_for(task))
    assert response.failure is None, response.failure
    assert [turn.name for turn in response.result.turns] == ["plan", "build"] and response.result.reward == 1.0
    assert (
        f"hello: its 2 steps need up to 1s once the sandbox is ready (harness setup, setup.sh and healthcheck "
        f"bounds, agent and verifier budgets, hooks and grace per step), 2s with one sandbox start and up to "
        f"{seed_bound + 1.2:.0f}s with every image pull retry; default_episode_timeout_seconds=0 is below that "
        f"bound, so the episode deadline and the sandbox lifetime follow the task's own bound ({seed_bound + 1.2:.0f}s)"
    ) in caplog.text
    # The derived deadline is the task's own budget, not a blank cheque: `test_the_task_deadline_ends_an_overrunning_turn`
    # keeps the task deadline firing on a turn that runs past it.


async def test_a_task_lookup_slower_than_the_default_leaves_the_episode_the_default(tmp_path, monkeypatch):
    """Review (round 32): the bound's task lookup, the seed's own shared snapshot (a copy of the task folder on a
    thread), ran before the episode deadline opened with nothing bounding it but the client's request timeout. It
    now runs within `default_episode_timeout_seconds`, the bound the episode gets when the task does not resolve:
    the hook gives the default back after it, and the snapshot it waited for is not cancelled, so the seed (under
    the deadline) and the next episode of the task find it instead of copying the folder again."""
    server, task, _, _ = multi_turn_server(
        tmp_path,
        monkeypatch,
        THREE_STEPS,
        {"plan": "P", "build": "B", "check": "C"},
        default_episode_timeout_seconds=0.2,
    )
    gate = threading.Event()
    snapshots = 0
    original = server._snapshot_task

    def slow_snapshot(*args, **kwargs):
        nonlocal snapshots
        snapshots += 1
        gate.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(server, "_snapshot_task", slow_snapshot)
    started = time.monotonic()
    bound = await asyncio.wait_for(server.episode_timeout_seconds(request_for(task)), 2)
    elapsed = time.monotonic() - started
    assert bound == 0.2 and 0.2 <= elapsed < 1.5
    # The shared snapshot outlives the bound's wait: once the folder copy ends, the seed's own lookup reads it.
    (pending,) = server._tasks.values()
    assert not pending.cancelled() and not pending.done()
    gate.set()
    request = request_for(task)
    resolved = await server._resolve_task(request.task.task_id, request.task.task_input.task_data)
    assert resolved.digest == task.digest and snapshots == 1
    # With the snapshot in hand the bound is the task's own again (its steps add up to more than 0.2 s).
    assert await server.episode_timeout_seconds(request_for(task)) == pytest.approx(server._episode_bound_s(task))


async def test_a_seed_slower_than_the_ready_timeout_leaves_the_turns_their_budget(tmp_path, monkeypatch):
    """Round 24 (G11 review): the derived deadline was `sandbox_ready_timeout_s` plus the post-seed budget and
    opens at the request, while the seed's own bounds are larger (the create grace, pull retries, a sidecar's
    start), so a seed longer than one ready timeout ate the final turns' budget and the episode died with the base
    server's generic timeout, retried and never scored. The bound now covers the seed's own: a seed of 2.5 s on a
    ready timeout of 1 s leaves two turns of 0.2 s their 1.2 s budget (round 32: the turns take 0.2 s, not 0.5 s,
    so the verifier runs keep 0.8 s of headroom on a loaded node; the seed is slower instead, so the episode still
    outlives the former bound by a margin)."""
    server, task, sandbox, _ = multi_turn_server(
        tmp_path,
        monkeypatch,
        LONG_STEPS,
        {"plan": "P", "build": "B"},
        default_episode_timeout_seconds=0.1,
        harness_setup_allowance_s=0,
        verifier_grace_s=0,
        sandbox_ready_timeout_s=1,
        image_pull_retries=0,
    )
    create = server._create_sandbox

    async def slow_create(task, workdir, *, episode_key=None):
        await asyncio.sleep(2.5)
        return await create(task, workdir, episode_key=episode_key)

    monkeypatch.setattr(server, "_create_sandbox", slow_create)
    server.server_client.slow_path = "/ng-rollout/r1/v1/responses"
    server.server_client.slow_seconds = 0.2
    started = time.monotonic()

    response = await server.run_request(request_for(task))

    elapsed = time.monotonic() - started
    assert response.failure is None, response.failure
    assert [turn.name for turn in response.result.turns] == ["plan", "build"] and response.result.reward == 1.0
    # The episode outlived the former bound (one ready timeout plus the turns, 2.2 s) because the seed alone took
    # 2.5 s of it; the whole bound (`_episode_bound_s`) still has the turns' 1.2 s after the seed's 76 s.
    assert elapsed >= 2.9 > server._task_budget_s(task) == pytest.approx(2.2)


async def test_without_enforcement_a_multi_step_task_longer_than_the_default_is_refused_naming_the_setting(
    tmp_path, monkeypatch, caplog
):
    """The quantity refused is the nominal episode (one sandbox start, then everything the deadline covers), not
    the steps' agent and verifier time alone and not the worst-case seed: the boundary case here has the steps' own
    time (1.2 s) within the default (2 s) while the harness setup allowance (0.5 s per turn) and the one start take
    the need past it, so the task is refused with both figures named. Round 24 (G11 review 2): a default that fits
    the nominal episode but not a seed that retries every image pull runs the task, warned, as Harbor runs it;
    refusing it for the worst case kept tasks that fit the default comfortably from running."""
    server, task, sandbox, created = multi_turn_server(
        tmp_path,
        monkeypatch,
        LONG_STEPS,
        {"plan": "P", "build": "B"},
        default_episode_timeout_seconds=2,
        harness_setup_allowance_s=0.5,
        verifier_grace_s=0,
        sandbox_ready_timeout_s=1,
        image_pull_retries=0,
        enforce_task_timeouts=False,
    )
    seed_bound = 1 + SEED_CREATE_GRACE_S
    assert server._post_seed_budget_s(task) == pytest.approx(2.2) and server._episode_bound_s(task) == pytest.approx(
        seed_bound + 2.2
    )
    assert server._episode_budget_s(task) == 2 and await server.episode_timeout_seconds(request_for(task)) == 2

    response = await server.run_request(request_for(task))

    failure = response.failure
    assert failure is not None and failure.stage == "seed" and failure.terminal
    assert server._task_budget_s(task) == pytest.approx(3.2)
    assert (
        f"Task 'hello': its 2 steps need up to 2s once the sandbox is ready (harness setup, setup.sh and "
        f"healthcheck bounds, agent and verifier budgets, hooks and grace per step), 3s with one sandbox start and "
        f"up to {seed_bound + 2.2:.0f}s with every image pull retry, above default_episode_timeout_seconds=2 after "
        f"one sandbox start already, and enforce_task_timeouts is off"
    ) in failure.failure_reason
    assert (
        f"raise default_episode_timeout_seconds to at least 3 ({seed_bound + 2.2:.0f} to cover a retried seed)"
        in failure.failure_reason
    )
    assert created == [] and sandbox.execs == [] and server.server_client.seed_bodies() == []
    # A default that fits the nominal episode (3.2 s) but not the worst-case seed runs the turns, with the WARNING
    # that names the gap; the deadline stays the default without enforcement.
    server.config.default_episode_timeout_seconds = 4
    assert await server.episode_timeout_seconds(request_for(task)) == 4
    with caplog.at_level("WARNING"):
        response = await server.run_request(request_for(task, rollout="r2"))
    assert response.failure is None and [turn.name for turn in response.result.turns] == ["plan", "build"]
    assert (
        f"hello: its 2 steps need up to 2s once the sandbox is ready (harness setup, setup.sh and healthcheck "
        f"bounds, agent and verifier budgets, hooks and grace per step), 3s with one sandbox start and up to "
        f"{seed_bound + 2.2:.0f}s with every image pull retry; default_episode_timeout_seconds=4 covers the episode "
        f"after one sandbox start but not after a retried seed, and enforce_task_timeouts is off, so an episode "
        f"whose seed retries its image pulls may time out and be retried; set enforce_task_timeouts: true to give "
        f"the episode the task's own bound"
    ) in caplog.text
    # A default that covers the worst case runs them without a word, as before.
    caplog.clear()
    server.config.default_episode_timeout_seconds = seed_bound + 3
    with caplog.at_level("WARNING"):
        response = await server.run_request(request_for(task, rollout="r3"))
    assert response.failure is None and [turn.name for turn in response.result.turns] == ["plan", "build"]
    assert "default_episode_timeout_seconds" not in caplog.text


MCP_TOML = """
[[environment.mcp_servers]]
name = "user-server"
transport = "streamable-http"
url = "http://user-server:8000/mcp"

[[environment.mcp_servers]]
name = "legacy"
transport = "sse"
url = "http://legacy:9000/sse"
"""


@pytest.mark.parametrize("grants", ["off", "optional", "required"])
async def test_task_mcp_servers_are_granted_on_request(tmp_path, monkeypatch, grants):
    server, task, _, _ = make_server(tmp_path, monkeypatch, toml=TASK_TOML + MCP_TOML, task_mcp_grants=grants)

    await server.run_request(request_for(task))

    (seed,) = server.server_client.seed_bodies()
    if grants == "off":
        assert seed["tool_accesses"] == []
    else:
        # Only the streamable-http entry can be granted; the URL is the sidecar's own host, as declared.
        assert seed["tool_accesses"] == [
            {
                "kind": "mcp",
                "name": "user-server",
                "required": grants == "required",
                "connection": {"transport": "streamable_http", "url": "http://user-server:8000/mcp", "headers": {}},
            }
        ]


def turn(
    name: str,
    rewards: dict[str, float] | None,
    *,
    masked: bool = False,
    components: dict[str, float] | None = None,
    reward: float | None = None,
) -> HarborTurnResult:
    if reward is None:
        reward = (rewards or {}).get("reward", 0.0)
    return HarborTurnResult(
        name=name,
        reward=reward,
        verifier_rewards=rewards,
        reward_components=components,
        mask_sample=masked,
        failure_kind="provider_unavailable" if masked else None if rewards else "harbor:missing_reward",
        failure_reason="lost" if masked else None if rewards else "no reward.json or reward.txt was written",
        response=agent_response(),
    )


class TestRollUp:
    def test_a_masked_turn_masks_the_episode(self):
        rolled = roll_up_turns([turn("a", {"reward": 1.0}), turn("b", None, masked=True)], "mean")
        assert rolled["mask_sample"] and rolled["failure_kind"] == "provider_unavailable"

    def test_mean_counts_a_missing_key_as_zero(self):
        rolled = roll_up_turns([turn("a", {"reward": 1.0, "x": 1.0}), turn("b", {"reward": 0.0})], "mean")
        assert rolled == {"reward": 0.5, "verifier_rewards": {"reward": 0.5, "x": 0.5}}

    def test_mean_without_any_result_is_measured_zero_with_the_turns_cause(self):
        rolled = roll_up_turns([turn("a", None)], "mean")
        assert (rolled["reward"], rolled["mask_sample"], rolled["failure_kind"]) == (
            0.0,
            False,
            "harbor:missing_reward",
        )
        unexplained = turn("a", None).model_copy(update={"failure_kind": None})
        assert roll_up_turns([unexplained], "mean")["failure_kind"] == NO_TURN_RESULT_KIND

    def test_final_takes_the_last_turn_that_ran(self):
        assert roll_up_turns([turn("a", {"reward": 0.0}), turn("b", {"reward": 1.0})], "final")["reward"] == 1.0
        assert roll_up_turns([turn("a", {"reward": 1.0}), turn("b", None)], "final")["reward"] == 0.0

    def test_final_keeps_the_last_turns_components(self, caplog):
        parts = {"x": 0.5, "y": 0.5}
        last = turn("b", parts, components=parts, reward=1.0)
        with caplog.at_level(logging.WARNING, logger="environment_servers.harbor.app"):
            rolled = roll_up_turns([turn("a", {"reward": 0.0}), last], "final", task_id="t")
        assert rolled == {"reward": 1.0, "verifier_rewards": parts, "reward_components": parts}
        assert not caplog.records
        # Components that do not sum to the reward are never named, and the drop is said, not silent: the
        # warning names the task, the keys, their sum and the reward.
        off = turn("b", parts, components={"x": 0.5}, reward=1.0)
        with caplog.at_level(logging.WARNING, logger="environment_servers.harbor.app"):
            assert "reward_components" not in roll_up_turns([off], "final", task_id="t")
        [record] = caplog.records
        assert record.levelno == logging.WARNING
        message = record.getMessage()
        assert "task 't'" in message and "['x']" in message and "turn 'b'" in message
        assert "sum to 0.5, not to the episode reward 1.0" in message

    def test_mean_names_components_only_when_every_turn_did(self):
        first = turn("a", {"x": 1.0}, components={"x": 1.0}, reward=1.0)
        second = turn("b", {"x": 0.0, "y": 0.5}, components={"x": 0.0, "y": 0.5}, reward=0.5)
        rolled = roll_up_turns([first, second], "mean")
        assert rolled["reward"] == 0.75 and rolled["reward_components"] == {"x": 0.5, "y": 0.25}
        # A turn that named its reward directly has no components, so the episode names none either.
        mixed = roll_up_turns([first, turn("b", {"reward": 0.5})], "mean")
        assert mixed["reward"] == 0.75 and "reward_components" not in mixed

    def test_mean_drops_components_that_do_not_sum_with_a_warning(self, caplog):
        first = turn("a", {"x": 1.0}, components={"x": 1.0}, reward=1.0)
        # A turn whose components do not sum to its reward (nothing on the verifier path builds one) pulls the
        # per-key means off the mean reward; the episode then names none and says so.
        off = turn("b", {"x": 0.0, "y": 0.5}, components={"x": 0.0}, reward=0.5)
        with caplog.at_level(logging.WARNING, logger="environment_servers.harbor.app"):
            rolled = roll_up_turns([first, off], "mean", task_id="t")
        assert rolled["reward"] == 0.75 and "reward_components" not in rolled
        [record] = caplog.records
        assert "task 't'" in record.getMessage() and "the mean over turns" in record.getMessage()
        assert "sum to 0.5, not to the episode reward 0.75" in record.getMessage()


@pytest.mark.parametrize(
    ("rewards", "min_reward", "fails"),
    [
        ({"reward": 1.0}, None, False),
        ({"reward": 0.5}, 0.5, False),
        ({"reward": 0.4}, 0.5, True),
        ({"reward": 1.0, "tests": 0.0}, {"tests": 1.0}, True),
        ({"reward": 1.0}, {"tests": 0.0}, True),
        (None, 0.0, True),
    ],
)
def test_min_reward_follows_harbor(rewards, min_reward, fails):
    assert (min_reward_failure(rewards, min_reward) is not None) == fails


# -- steps/<name>/workdir/ and setup.sh (Harbor's _upload_step_workdir and _run_step_setup) -------------------------


def write_step_workdir(task_dir: Path, name: str, files: dict[str, str]) -> None:
    """``steps/<name>/workdir/`` with ``files`` (a ``setup.sh`` among them when the step has one)."""
    folder = task_dir / "steps" / name / "workdir"
    for relative, text in files.items():
        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
        (folder / relative).write_text(text)


SETUP = "bash /app/setup.sh"


@dataclass
class ScriptedSandbox(FakeSandbox):
    """A sandbox whose runs of a scripted command (a step's ``bash /app/setup.sh``, a healthcheck command) answer
    ``scripted[command]`` in order, OK once the answers run out; every other command is the fake's own."""

    scripted: dict[str, list[SandboxExecResult]] = field(default_factory=dict)

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if command in self.scripted:
            # `ran_as` is the user the container ran the command as, the fake's rule (`FakeSandbox.exec`).
            effective = user if user is not None else self.default_user
            ran_as = str(effective) if effective is not None else self.image_user
            self.execs.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_s": timeout_s,
                    "user": user,
                    "ran_as": ran_as,
                    "seen": None,
                }
            )
            answers = self.scripted[command]
            return answers.pop(0) if answers else OK
        return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)


def SetupSandbox(setup_results: list[SandboxExecResult] | None = None) -> ScriptedSandbox:  # noqa: N802
    return ScriptedSandbox(scripted={SETUP: list(setup_results or [])})


def multi_turn_server_with_workdirs(
    tmp_path, monkeypatch, workdirs: dict[str, dict[str, str]], *, build_toml: str = "", **kwargs
):
    """Three steps, ``plan``, ``build`` (as user ``tester``, plus ``build_toml``) and ``check``; ``workdirs`` maps
    a step to the files of its ``workdir/``."""
    task_dir = tmp_path / "datasets" / "ds" / "hello"
    for name, files in workdirs.items():
        write_step_workdir(task_dir, name, files)
    return multi_turn_server(
        tmp_path,
        monkeypatch,
        '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n[steps.agent]\nuser = "tester"\n'
        + build_toml
        + '[[steps]]\nname = "check"\n',
        {"plan": "P", "build": "B", "check": "C"},
        **kwargs,
    )


async def test_a_steps_workdir_is_merged_into_the_working_directory_before_its_agent(tmp_path, monkeypatch):
    """Harbor's `_upload_step_workdir`: `steps/<name>/workdir/` lands in the environment's working directory before
    the step's agent runs, over what the previous turns left (an upload, never a wipe), with the digest's ignored
    names left out; a step without the folder uploads nothing."""
    sandbox = SetupSandbox()
    server, task, sandbox, _ = multi_turn_server_with_workdirs(
        tmp_path,
        monkeypatch,
        {"build": {"fixtures/input.txt": "data\n", "notes.md": "n\n", "__pycache__/x.pyc": "\x00"}},
        sandbox=sandbox,
    )

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert [turn.reward for turn in response.result.turns] == [1.0, 1.0, 1.0]
    assert uploaded_files(sandbox, "input.txt") == [b"data\n"] and uploaded_files(sandbox, "x.pyc") == []
    # Extracted into the working directory as root, after the first turn's verifier and before the second
    # turn's agent session opens; nothing removed the directory's contents first.
    index = next(i for i, e in enumerate(sandbox.execs) if "tar -xzf" in e["command"] and " -C /app" in e["command"])
    verifies = [i for i, e in enumerate(sandbox.execs) if "timeout -s KILL" in e["command"]]
    assert verifies[0] < index < verifies[1] and sandbox.execs[index]["user"] == "root"
    assert not any("rm -rf" in e["command"] and "/app" in e["command"] for e in sandbox.execs)
    # The one root exec clears a link the previous turn's agent planted at any shipped name before it extracts, so
    # a BusyBox tar (which follows a link at a directory member's path) lands the merge where GNU tar and Harbor's
    # docker cp do: the listing loop precedes the extraction in the same shell line (the real tars are exercised in
    # tests/unit_tests/test_sandbox_transfer.py).
    extraction = sandbox.execs[index]["command"]
    assert 'if [ -L /app/"$m" ]; then rm -- /app/"$m"; fi; done; tar -xzf' in extraction
    assert extraction.index("tar -tzf") < extraction.index("tar -xzf")
    assert len(server.server_client.seed_bodies()) == 3
    assert not any(e["command"].startswith("bash ") and e["command"].endswith("setup.sh") for e in sandbox.execs)


async def test_a_steps_setup_script_runs_as_the_steps_agent_user_in_the_working_directory(tmp_path, monkeypatch):
    """Harbor's `_run_step_setup`: `bash <workdir>/setup.sh` as the step's agent user (the sandbox's default under
    `with_default_user`), once the folder is in place and before the agent session, within the step's agent budget
    (`step_setup_timeout_s` unset; the harness allowance is not the bound)."""
    sandbox = SetupSandbox()
    server, task, sandbox, _ = multi_turn_server_with_workdirs(
        tmp_path,
        monkeypatch,
        {"build": {"setup.sh": "#!/bin/bash\nmake fixtures\n"}},
        sandbox=sandbox,
        harness_setup_allowance_s=10,
    )

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    setup = [e for e in sandbox.execs if e["command"] == "bash /app/setup.sh"]
    assert len(setup) == 1
    # No explicit user: the sandbox's default at that point, the build step's `tester`, ran it (the fake records the
    # user in effect as `ran_as`).
    assert setup[0]["cwd"] == "/app" and setup[0]["user"] is None and setup[0]["ran_as"] == "tester"
    assert setup[0]["timeout_s"] == 300  # the task's [agent].timeout_sec, not harness_setup_allowance_s=10
    # The build step's user was installed before the script ran: the default the exec falls back to.
    upload = next(i for i, e in enumerate(sandbox.execs) if "tar -xzf" in e["command"] and " -C /app" in e["command"])
    assert sandbox.execs.index(setup[0]) > upload
    switched = next(e for e in sandbox.execs if e["command"] == "chown tester /app")
    assert sandbox.execs.index(switched) < sandbox.execs.index(setup[0])
    assert uploaded_files(sandbox, "setup.sh") == [b"#!/bin/bash\nmake fixtures\n"]
    assert [turn.reward for turn in response.result.turns] == [1.0, 1.0, 1.0]


async def test_a_failing_setup_script_ends_the_episode_as_a_failed_turn_without_agent_or_verifier(
    tmp_path, monkeypatch, caplog
):
    """Harbor records a non-zero setup.sh as the step's exception_info and stops the trial after that step: here the
    turn is a failed turn (no response, no verifier result, an explicit failure kind) and the roll-up scores what
    ran before it."""
    sandbox = SetupSandbox(setup_results=[SandboxExecResult(stdout="", stderr="no fixtures\n", return_code=3)])
    server, task, sandbox, _ = multi_turn_server_with_workdirs(
        tmp_path, monkeypatch, {"build": {"setup.sh": "#!/bin/bash\nexit 3\n"}}, sandbox=sandbox
    )
    rewards_per_turn(server, sandbox, ["0.5", "1", "1"])

    with caplog.at_level("WARNING"):
        response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    result = response.result
    assert [turn.name for turn in result.turns] == ["plan", "build"]
    failed = result.turns[1]
    assert failed.response is None and failed.verifier_rewards is None and not failed.mask_sample
    assert failed.failure_kind == "harbor:step_setup_failed"
    assert "step 'build' setup.sh exited with code 3: no fixtures" in failed.failure_reason
    assert "step 'build' setup.sh exited with code 3" in caplog.text
    # mean over the turns with a result: the first turn alone; the episode stopped after the failed turn.
    assert (result.reward, result.stopped_after_turn, result.turns_without_verifier_result) == (0.5, "build", 1)
    assert not result.mask_sample
    assert result.response is not None and result.response == result.turns[0].response
    # Neither an agent session nor a verifier ran for the failed turn.
    assert len(server.server_client.seed_bodies()) == 1
    assert len([e for e in sandbox.execs if "timeout -s KILL" in e["command"]]) == 1


async def test_a_failing_setup_script_under_final_masks_the_episode_as_harbor_has_no_reward(tmp_path, monkeypatch):
    """Harbor's `_select_multi_step_reward` under `final` returns the last step's verifier result, which a step whose
    setup failed has not got: the trial has no reward. The episode is masked with the step's kind, not labelled 0
    for a turn the agent never acted in; the failed turn itself stays an unmasked record of the task's failure."""
    sandbox = SetupSandbox(setup_results=[SandboxExecResult(stdout="", stderr="no fixtures\n", return_code=3)])
    server, task, sandbox, _ = multi_turn_server_with_workdirs(
        tmp_path,
        monkeypatch,
        {"build": {"setup.sh": "#!/bin/bash\nexit 3\n"}},
        sandbox=sandbox,
        head='multi_step_reward_strategy = "final"\n',
    )
    rewards_per_turn(server, sandbox, ["0.5", "1", "1"])

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    result = response.result
    assert [turn.name for turn in result.turns] == ["plan", "build"] and result.turn_reward_strategy == "final"
    assert result.turns[0].reward == 0.5 and not result.turns[1].mask_sample
    assert result.mask_sample and result.reward == 0.0 and result.failure_kind == "harbor:step_setup_failed"
    assert result.failure_reason.startswith("turn 'build': step 'build' setup.sh exited with code 3")


def test_the_roll_up_masks_a_turn_the_agent_never_ran_and_measures_one_it_ran_without_a_reward():
    """The two turns without a verifier result: a prepared turn whose test.sh wrote nothing is measured 0 (as a single
    turn is), a turn whose preparation failed (no `response`) is masked under `final` and when nothing was measured."""
    ran = HarborTurnResult(name="a", reward=0.0, failure_kind="harbor:no_reward", failure_reason="x", response=None)
    unprepared = HarborTurnResult(name="b", reward=0.0, failure_kind="harbor:step_setup_failed", failure_reason="y")
    measured = HarborTurnResult(name="a", reward=1.0, verifier_rewards={"reward": 1.0}, response=agent_response())
    assert roll_up_turns([measured, unprepared], "final")["mask_sample"] is True
    assert roll_up_turns([measured, unprepared], "final")["failure_kind"] == "harbor:step_setup_failed"
    assert roll_up_turns([measured, unprepared], "mean") == {"reward": 1.0, "verifier_rewards": {"reward": 1.0}}
    assert roll_up_turns([unprepared], "mean")["mask_sample"] is True
    assert roll_up_turns([unprepared], "final")["mask_sample"] is True
    # `response` is None for `ran` too (the fixture carries none), so the distinction is by the agent having run:
    # a measured-0 turn in the server always carries its response; here the roll-up is asked about the shape alone.
    assert roll_up_turns([ran], "mean")["mask_sample"] is True


def test_the_budget_counts_a_steps_setup_script_at_its_bound(tmp_path, monkeypatch):
    """With `enforce_task_timeouts` the episode deadline is this sum: a step that ships a setup.sh adds its bound
    (`step_setup_timeout_s`, else the step's agent budget), one that ships none adds nothing."""
    server, task, _, _ = multi_turn_server_with_workdirs(tmp_path, monkeypatch, {})
    without = server._post_seed_budget_s(task)
    server, task, _, _ = multi_turn_server_with_workdirs(
        tmp_path / "setup", monkeypatch, {"build": {"setup.sh": "#!/bin/bash\ntrue\n"}, "check": {"notes.md": "n"}}
    )
    assert server._post_seed_budget_s(task) == without + 300  # the task's [agent].timeout_sec, build's budget
    server, task, _, _ = multi_turn_server_with_workdirs(
        tmp_path / "bound", monkeypatch, {"build": {"setup.sh": "#!/bin/bash\ntrue\n"}}, step_setup_timeout_s=42
    )
    assert server._post_seed_budget_s(task) == without + 42


async def test_a_failing_setup_script_on_the_first_turn_is_a_terminal_failure_with_the_steps_kind(
    tmp_path, monkeypatch
):
    """No turn ran an agent, so there is no rollout to score: the episode is a terminal agent-stage failure naming
    the step and carrying its failure kind, as Harbor's trial ends with no verifier result."""
    sandbox = SetupSandbox(setup_results=[SandboxExecResult(stdout="", stderr="boom", return_code=1)])
    server, task, sandbox, _ = multi_turn_server_with_workdirs(
        tmp_path, monkeypatch, {"plan": {"setup.sh": "#!/bin/bash\nexit 1\n"}}, sandbox=sandbox
    )

    response = await server.run_request(request_for(task))

    failure = response.failure
    assert failure is not None and failure.stage == "agent" and failure.terminal
    assert failure.failure_kind == "harbor:step_setup_failed"
    assert "turn 'plan': step 'plan' setup.sh exited with code 1: boom" in failure.failure_reason
    assert not server.server_client.seed_bodies()


async def test_a_setup_script_past_its_bound_fails_the_step_and_a_sandbox_error_is_retryable(tmp_path, monkeypatch):
    """The bound is `step_setup_timeout_s` when set, else the step's agent budget (never 0 or unbounded: Harbor's only
    bound is the trial timeout, the episode deadline here); a run past it is the step's failure, as a non-zero exit is.
    A sandbox that cannot run the script at all is an outage, retried elsewhere."""
    timed_out = SandboxExecResult(stdout="", stderr="", return_code=-1, error_type="timeout")
    sandbox = SetupSandbox(setup_results=[timed_out])
    server, task, sandbox, _ = multi_turn_server_with_workdirs(
        tmp_path, monkeypatch, {"build": {"setup.sh": "#!/bin/bash\nsleep 1d\n"}}, sandbox=sandbox
    )
    response = await server.run_request(request_for(task))
    assert response.failure is None, response.failure
    failed = response.result.turns[1]
    assert failed.failure_kind == "harbor:step_setup_failed" and "ran past its 300s bound" in failed.failure_reason
    assert next(e for e in sandbox.execs if e["command"] == "bash /app/setup.sh")["timeout_s"] == 300

    sandbox = SetupSandbox(setup_results=[timed_out])
    server, task, sandbox, _ = multi_turn_server_with_workdirs(
        tmp_path / "bound",
        monkeypatch,
        {"build": {"setup.sh": "#!/bin/bash\nsleep 1d\n"}},
        sandbox=sandbox,
        step_setup_timeout_s=42,
        harness_setup_allowance_s=0,
    )
    response = await server.run_request(request_for(task, rollout="r3"))
    assert response.failure is None, response.failure
    assert "ran past its 42s bound" in response.result.turns[1].failure_reason
    assert next(e for e in sandbox.execs if e["command"] == "bash /app/setup.sh")["timeout_s"] == 42

    lost = SandboxExecResult(stdout="", stderr="", return_code=-1, error_type="connection")
    sandbox = SetupSandbox(setup_results=[lost])
    server, task, sandbox, _ = multi_turn_server_with_workdirs(
        tmp_path, monkeypatch, {"build": {"setup.sh": "#!/bin/bash\ntrue\n"}}, sandbox=sandbox
    )
    response = await server.run_request(request_for(task, rollout="r2"))
    failure = response.failure
    assert failure is not None and failure.stage == "agent" and not failure.terminal
    assert "setup.sh could not run: connection" in failure.failure_reason


TAR_REFUSAL = SandboxExecResult(
    stdout="",
    stderr="tar: setup.sh: Cannot open: File exists\ntar: Exiting with failure status due to previous errors\n",
    return_code=2,
)


@dataclass
class PlantedSandbox(ScriptedSandbox):
    """The agent's sandbox after the previous turn's agent left a non-empty directory at the name of a file the next
    step's ``workdir/`` ships: the extraction into the working directory answers ``extraction``."""

    extraction: SandboxExecResult = TAR_REFUSAL

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        result = await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)
        if "tar -xzf" in command and " -C /app" in command:
            return self.extraction
        return result


async def test_a_workdir_entry_the_sandbox_refuses_to_unpack_fails_the_step_instead_of_retrying_the_episode(
    tmp_path, monkeypatch
):
    """The previous turn's agent owns the working directory and can leave a non-empty directory at the name of a file
    the next step's workdir/ ships (`mkdir -p /app/setup.sh/x`); GNU tar then refuses the entry (`Cannot open: File
    exists`, exit 2), as Harbor's `docker cp` refuses the mismatch and errors the trial. That is the step's own failure
    (an unmasked turn, the episode stops, nothing retried), never an outage retried for as long as the policy allows;
    a provider that cannot run the extraction at all is still the outage it is."""
    sandbox = PlantedSandbox(scripted={SETUP: []})
    server, task, sandbox, _ = multi_turn_server_with_workdirs(
        tmp_path, monkeypatch, {"build": {"setup.sh": "#!/bin/bash\nmake fixtures\n"}}, sandbox=sandbox
    )
    rewards_per_turn(server, sandbox, ["1", "1", "1"])

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    result = response.result
    assert [turn.name for turn in result.turns] == ["plan", "build"]
    failed = result.turns[1]
    assert failed.failure_kind == "harbor:step_workdir_failed" and failed.response is None and not failed.mask_sample
    assert "step 'build' workdir/ could not be merged into /app" in failed.failure_reason
    assert "tar: setup.sh: Cannot open: File exists" in failed.failure_reason
    assert (result.reward, result.stopped_after_turn, result.mask_sample) == (1.0, "build", False)
    # The step's script never ran over the planted tree, no agent session opened for the step, and the extraction
    # was tried exactly once: nothing retried it.
    assert not [e for e in sandbox.execs if e["command"] == SETUP]
    assert len(server.server_client.seed_bodies()) == 1
    assert len([e for e in sandbox.execs if "tar -xzf" in e["command"] and " -C /app" in e["command"]]) == 1

    lost = SandboxExecResult(stdout="", stderr="", return_code=-1, error_type="connection")
    sandbox = PlantedSandbox(scripted={SETUP: []}, extraction=lost)
    server, task, sandbox, _ = multi_turn_server_with_workdirs(
        tmp_path / "lost", monkeypatch, {"build": {"setup.sh": "#!/bin/bash\ntrue\n"}}, sandbox=sandbox
    )
    response = await server.run_request(request_for(task, rollout="r2"))
    failure = response.failure
    assert failure is not None and failure.stage == "agent" and not failure.terminal
    assert "SandboxUnavailableError" in failure.failure_reason and "connection" in failure.failure_reason


# -- [steps.healthcheck] (Harbor's _run_step_healthcheck) ----------------------------------------------------------

READY = "test -f /app/ready"
BUILD_HEALTHCHECK = (
    "[steps.healthcheck]\n"
    f'command = "{READY}"\n'
    "interval_sec = 0.01\nstart_interval_sec = 0.01\nstart_period_sec = 0.0\ntimeout_sec = 5.0\nretries = 2\n"
)
FAILS = SandboxExecResult(stdout="", stderr="no ready file", return_code=1)


async def test_a_steps_healthcheck_is_polled_after_its_setup_and_before_its_agent(tmp_path, monkeypatch):
    """Harbor's `_run_step_healthcheck`: after the step's setup.sh, before its agent, with `run_healthcheck`'s
    semantics (here one failure then a pass, within `retries`), as the turn's agent user in the working directory."""
    sandbox = ScriptedSandbox(scripted={SETUP: [], READY: [FAILS]})
    server, task, sandbox, _ = multi_turn_server_with_workdirs(
        tmp_path,
        monkeypatch,
        {"build": {"setup.sh": "#!/bin/bash\ntouch ready\n"}},
        build_toml=BUILD_HEALTHCHECK,
        sandbox=sandbox,
    )

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert [turn.reward for turn in response.result.turns] == [1.0, 1.0, 1.0]
    checks = [i for i, e in enumerate(sandbox.execs) if e["command"] == READY]
    setup = next(i for i, e in enumerate(sandbox.execs) if e["command"] == SETUP)
    assert len(checks) == 2 and setup < checks[0]
    # No explicit user: the sandbox's default at that point, the build step's `tester`, ran each attempt.
    assert all(sandbox.execs[i]["cwd"] == "/app" and sandbox.execs[i]["user"] is None for i in checks)
    assert all(sandbox.execs[i]["ran_as"] == "tester" for i in checks)
    assert sandbox.execs[checks[0]]["timeout_s"] == 10.0
    # The second turn's agent session opened after the check passed: its seed is the second of three.
    assert len(server.server_client.seed_bodies()) == 3


@pytest.mark.parametrize(
    ("start_interval", "retries"),
    [(11.99, 4), (11.99, 0), (5.0, 4), (0.5, 2)],
    ids=["straddle-tight", "straddle-tight-retries-0", "harbor-defaults", "short-start-interval"],
)
async def test_the_budget_counts_a_steps_healthcheck_at_its_worst_case(tmp_path, monkeypatch, start_interval, retries):
    """A step with a `[steps.healthcheck]` adds the most the poll may take, whatever its alignment with the start
    period: the loop (`_poll_healthcheck`) is driven here against a scripted clock in which every attempt runs to the
    provider's kill at `timeout_sec` plus the exec grace, and the budget must cover the elapsed time at the failure.
    With `start_interval_sec` chosen so that the last in-grace attempt starts 0.01 s before the period ends, the poll
    ends 0.01 s short of the budget: a budget that charged `interval_sec` for the sleep after that attempt was under
    by `start_interval_sec - interval_sec` (the straddling attempt is followed by the start interval, as in Harbor).
    Without the count the last turns of a task with slow checks died at the deadline."""
    server, task, _, _ = multi_turn_server_with_workdirs(tmp_path, monkeypatch, {})
    without = server._post_seed_budget_s(task)
    check_toml = (
        '[steps.healthcheck]\ncommand = "curl -sf localhost:8000/health"\n'
        f"start_period_sec = 20.0\ninterval_sec = 2.0\ntimeout_sec = 3.0\nretries = {retries}\n"
        f"start_interval_sec = {start_interval}\n"
    )
    server, task, _, _ = multi_turn_server_with_workdirs(tmp_path / "check", monkeypatch, {}, build_toml=check_toml)
    turn = task.turns[1]
    check = turn.config.healthcheck
    budget = server._step_healthcheck_budget_s(turn)
    assert server._post_seed_budget_s(task) == pytest.approx(without + budget)

    class Clock:
        now = 0.0

        def time(self):
            return self.now

        async def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(
        "environment_servers.harbor.entrypoint.asyncio",
        SimpleNamespace(get_running_loop=lambda: clock, sleep=clock.sleep),
    )
    started: list[float] = []

    async def attempt():
        started.append(clock.now)
        # The worst attempt: the provider kills the command at the exec's timeout_s (timeout_sec plus the grace).
        clock.now += check.timeout_sec + HEALTHCHECK_EXEC_GRACE_S
        return FAILS

    with pytest.raises(HealthcheckFailed):
        await server._poll_healthcheck(None, task, check, attempt)
    elapsed = clock.now
    assert budget >= elapsed, (budget, elapsed, started)
    # `retries` attempts began after the start period (one for retries = 0), none counted twice.
    assert len([t for t in started if t >= check.start_period_sec]) == max(retries, 1)
    if start_interval == 11.99:
        # 0 -> 8, sleep 11.99 -> 19.99 (in grace) -> 27.99, sleep 11.99 -> 39.98, then the counted attempts.
        assert started[1] == pytest.approx(19.99) and started[2] == pytest.approx(39.98)
        assert budget - elapsed == pytest.approx(0.01)


async def test_a_steps_healthcheck_that_never_passes_fails_the_turn_without_agent_or_verifier(tmp_path, monkeypatch):
    """Harbor records the HealthcheckError in the step's exception_info and stops the trial: a failed turn here,
    with its own failure kind, the previous turns scored."""
    sandbox = ScriptedSandbox(scripted={READY: [FAILS, FAILS, FAILS]})
    server, task, sandbox, _ = multi_turn_server_with_workdirs(
        tmp_path, monkeypatch, {}, build_toml=BUILD_HEALTHCHECK, sandbox=sandbox
    )
    rewards_per_turn(server, sandbox, ["1", "1", "1"])

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    result = response.result
    failed = result.turns[1]
    assert [turn.name for turn in result.turns] == ["plan", "build"]
    assert failed.failure_kind == "harbor:step_healthcheck_failed" and failed.response is None
    assert "step 'build' healthcheck failed: Healthcheck failed 2 times: 'test -f /app/ready' (step 'build')" in (
        failed.failure_reason
    )
    assert (result.reward, result.stopped_after_turn) == (1.0, "build")
    # retries = 2 consecutive failures after the (empty) start period: two attempts, then the step failed.
    assert len([e for e in sandbox.execs if e["command"] == READY]) == 2
    assert len(server.server_client.seed_bodies()) == 1


async def test_a_steps_healthcheck_does_not_run_after_its_setup_failed(tmp_path, monkeypatch):
    sandbox = ScriptedSandbox(scripted={SETUP: [SandboxExecResult(stdout="", stderr="", return_code=2)], READY: []})
    server, task, sandbox, _ = multi_turn_server_with_workdirs(
        tmp_path,
        monkeypatch,
        {"build": {"setup.sh": "#!/bin/bash\nexit 2\n"}},
        build_toml=BUILD_HEALTHCHECK,
        sandbox=sandbox,
    )

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert response.result.turns[1].failure_kind == "harbor:step_setup_failed"
    assert not [e for e in sandbox.execs if e["command"] == READY]


async def test_a_sandbox_that_cannot_run_a_steps_healthcheck_is_an_outage_not_the_steps_failure(tmp_path, monkeypatch):
    """Only a check that ran and never passed is the step's failure (Harbor's HealthcheckError). A sandbox that was
    stopped under the poll (the facade's `RuntimeError('Sandbox has not been started')`), or a provider that could not
    run the command, is a retryable failure, as the same conditions are for setup.sh and the collect hooks. A command
    killed at its own timeout ran and failed, and the loop counts it as one failed attempt, as Docker's HEALTHCHECK
    counts a timed-out probe and as the seed's task-level loop does; Harbor 0.22.0 differs there: its Docker exec
    raises on the timeout, run_healthcheck does not catch it and _run_step_healthcheck catches HealthcheckError only,
    so the first attempt that overruns timeout_sec errors the whole trial (no step result, no further attempt). The
    third case below asserts the server's lenient count, not Harbor's."""

    class Stopped(ScriptedSandbox):
        async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
            if command == READY:
                raise RuntimeError("Sandbox has not been started")
            return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

    server, task, sandbox, _ = multi_turn_server_with_workdirs(
        tmp_path, monkeypatch, {}, build_toml=BUILD_HEALTHCHECK, sandbox=Stopped()
    )
    response = await server.run_request(request_for(task))
    failure = response.failure
    assert failure is not None and failure.stage == "agent" and not failure.terminal
    assert "turn 'build': RuntimeError: Sandbox has not been started" in failure.failure_reason
    assert failure.failure_kind != "harbor:step_healthcheck_failed" and response.result is None

    lost = SandboxExecResult(stdout="", stderr="", return_code=-1, error_type="connection")
    sandbox = ScriptedSandbox(scripted={READY: [lost]})
    server, task, sandbox, _ = multi_turn_server_with_workdirs(
        tmp_path / "lost", monkeypatch, {}, build_toml=BUILD_HEALTHCHECK, sandbox=sandbox
    )
    response = await server.run_request(request_for(task, rollout="r2"))
    failure = response.failure
    assert failure is not None and failure.stage == "agent" and not failure.terminal
    assert "step 'build' healthcheck could not run: connection" in failure.failure_reason
    assert len([e for e in sandbox.execs if e["command"] == READY]) == 1

    killed = SandboxExecResult(stdout="", stderr="", return_code=-1, error_type="timeout")
    sandbox = ScriptedSandbox(scripted={READY: [killed, killed]})
    server, task, sandbox, _ = multi_turn_server_with_workdirs(
        tmp_path / "slow", monkeypatch, {}, build_toml=BUILD_HEALTHCHECK, sandbox=sandbox
    )
    response = await server.run_request(request_for(task, rollout="r3"))
    assert response.failure is None, response.failure
    assert response.result.turns[1].failure_kind == "harbor:step_healthcheck_failed"
    assert len([e for e in sandbox.execs if e["command"] == READY]) == 2


# -- per-step artifacts (Harbor's _collect_step_artifacts, in shared mode too) ---------------------------------------


@dataclass
class ArtifactSandbox(ScriptedSandbox):
    """The agent's sandbox with files to collect: ``present`` maps a path to ``dir``/``file``, ``archives`` maps a
    folder the server archives to the bytes it hands back; anything else (the verifier's files) is the fake's own."""

    present: dict[str, str] = field(default_factory=dict)
    archives: dict[str, bytes] = field(default_factory=dict)
    archived: list[str] = field(default_factory=list)

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if command.startswith("if [ -d "):
            self.execs.append(
                {"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s, "user": user, "seen": None}
            )
            path = command.split("if [ -d ")[1].split(" ]")[0].strip("'")
            return SandboxExecResult(stdout=self.present.get(path, "none") + "\n", stderr="", return_code=0)
        if "tar -czf" in command:
            marker = " -C " if " -C " in command else "cd "
            self.archived.append(command.split(marker)[1].split(" ")[0].strip("'"))
        return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)

    async def download(self, remote_path, local_path):
        source = self.archived[-1] if self.archived else None
        if source in self.archives:
            Path(local_path).write_bytes(self.archives[source])
            return
        await super().download(remote_path, local_path)


def probes(sandbox: FakeSandbox) -> list[str]:
    """The artifact paths the server asked the sandbox about, in order: one probe per entry per collection pass
    (the transcript upload's own probe of /logs/agent is not one)."""
    asked = [
        e["command"].split("if [ -d ")[1].split(" ]")[0].strip("'")
        for e in sandbox.execs
        if "if [ -d " in e["command"]
    ]
    return [path for path in asked if path != "/logs/agent"]


def manifest_of(turn: HarborTurnResult) -> list[dict]:
    return json.loads((Path(turn.verifier_logs_dir) / "artifacts" / "manifest.json").read_text())


STEP_ARTIFACTS = """
[[steps]]
name = "plan"

[[steps]]
name = "build"
artifacts = [
  { source = "/app", destination = "app", exclude = [".venv", "__pycache__"] },
  { source = "/var/log/api", service = "api" },
]

[[verifier.collect]]
command = "cp /app/state.db /logs/artifacts/state.db"
timeout_sec = 10.0
"""


async def test_a_steps_artifacts_are_collected_with_the_tasks_after_its_agent_and_before_its_verifier(
    tmp_path, monkeypatch
):
    """Harbor's `_collect_step_artifacts` (0.22.0 multi_step.py): after the step's agent and before its verifier,
    the hooks, then `/logs/artifacts`, the task's artifacts and the step's own, staged per turn in shared mode too;
    nothing is stopped, and the pass is recorded in a manifest beside the staged trees."""
    sandbox = ArtifactSandbox(
        present={"/logs/artifacts": "dir", "/app/out.txt": "file", "/app": "dir"},
        archives={
            "/logs/artifacts": archive_of({"state.db": "db"}),
            "/app": archive_of({"out.txt": "done\n", "src/main.py": "print()\n"}),
        },
    )
    server, task, sandbox, _ = multi_turn_server(
        tmp_path,
        monkeypatch,
        STEP_ARTIFACTS,
        {"plan": "P", "build": "B"},
        head='artifacts = ["/app/out.txt"]\n',
        sandbox=sandbox,
    )

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    plan, build = response.result.turns
    assert (plan.reward, build.reward) == (1.0, 1.0)
    # Both turns ran on the one sandbox: nothing stopped it between the first collection pass and the second turn.
    assert len(server.server_client.seed_bodies()) == 2 and sandbox.execs[-1]["command"] != "stop"
    # The plan turn collects the convention entry and the task's; the build turn the step's two entries on top.
    assert probes(sandbox) == ["/logs/artifacts", "/app/out.txt", "/logs/artifacts", "/app/out.txt", "/app"]
    plan_dir, build_dir = (Path(turn.verifier_logs_dir) / "artifacts" for turn in (plan, build))
    assert (plan_dir / "00" / "logs" / "artifacts" / "state.db").read_text() == "db"
    assert (plan_dir / "01" / "app" / "out.txt").read_text() == "done\n" and not (plan_dir / "02").exists()
    assert (build_dir / "02" / "app" / "src" / "main.py").read_text() == "print()\n"
    assert (build_dir / "02" / "app" / "out.txt").read_text() == "done\n"
    # The step's exclude rules reach the archive command, unanchored as Harbor passes them.
    app_archive = next(
        e["command"] for e in sandbox.execs if "tar -czf" in e["command"] and " -C /app " in e["command"]
    )
    assert "--exclude=.venv --exclude=__pycache__" in app_archive
    assert [(e["index"], e["source"], e["origin"], e["status"]) for e in manifest_of(build)] == [
        (0, "/logs/artifacts", "convention", "ok"),
        (1, "/app/out.txt", "task", "ok"),
        (2, "/app", "step", "ok"),
        (3, "/var/log/api", "step", "skipped"),  # no Compose service `api` in this task
    ]
    assert manifest_of(build)[2]["destination"] == "artifacts/02/app"
    assert [e["origin"] for e in manifest_of(plan)] == ["convention", "task"]
    # A shared turn restores nothing, so the archives are gone once the pass ends; the staged trees stay.
    assert not list(plan_dir.glob("*.tar.gz")) and not list(build_dir.glob("*.tar.gz"))
    # Order within the build turn: the hook, the probes and the archives sit between the agent's close and test.sh;
    # the hook ran once per turn, in the collection pass, not again inside the shared verifier.
    commands = [e["command"] for e in sandbox.execs]
    hooks = [i for i, c in enumerate(commands) if c == "cp /app/state.db /logs/artifacts/state.db"]
    verifies = [i for i, c in enumerate(commands) if "timeout -s KILL" in c]
    app_probe = next(i for i, c in enumerate(commands) if "if [ -d /app ]" in c)
    assert len(hooks) == 2 and verifies[0] < hooks[1] < app_probe < verifies[1]


STEP_HOOK = "tar -C /app -czf /logs/artifacts/app.tgz ."
STEP_HOOKS = """
[[steps]]
name = "plan"

[[steps]]
name = "build"
[[steps.verifier.collect]]
command = "tar -C /app -czf /logs/artifacts/app.tgz ."
timeout_sec = 20.0

[[verifier.collect]]
command = "cp /app/state.db /logs/artifacts/state.db"
timeout_sec = 10.0
"""


async def test_a_steps_own_collect_hooks_run_after_the_tasks_in_its_collection_pass(tmp_path, monkeypatch):
    """Harbor's `_collect_hooks_for(step)` is the task's `[[verifier.collect]]` hooks then the step's own
    `[[steps.verifier.collect]]`, run per step before that step's entries are read. The step's hook used to be dropped
    at load (an "unknown key" warning), so a step whose hook materialises what its test.sh reads graded 0."""
    sandbox = ArtifactSandbox(present={"/logs/artifacts": "dir"}, archives={"/logs/artifacts": archive_of({"a": "1"})})
    server, task, sandbox, _ = multi_turn_server(
        tmp_path, monkeypatch, STEP_HOOKS, {"plan": "P", "build": "B"}, sandbox=sandbox
    )
    plain = multi_turn_server(
        tmp_path / "plain",
        monkeypatch,
        STEP_HOOKS.replace(
            '[[steps.verifier.collect]]\ncommand = "tar -C /app -czf /logs/artifacts/app.tgz ."\ntimeout_sec = 20.0\n',
            "",
        ),
        {"plan": "P", "build": "B"},
    )
    assert [hook.command for hook in task.turns[1].config.verifier.collect] == [STEP_HOOK]
    # The step's hook is budgeted with the task's: its timeout and the exec grace, for its turn alone.
    assert server._post_seed_budget_s(task) == plain[0]._post_seed_budget_s(plain[1]) + 20 + COLLECT_HOOK_GRACE_S

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert [turn.reward for turn in response.result.turns] == [1.0, 1.0]
    commands = [e["command"] for e in sandbox.execs]
    task_hooks = [i for i, c in enumerate(commands) if c == "cp /app/state.db /logs/artifacts/state.db"]
    step_hooks = [i for i, c in enumerate(commands) if c == STEP_HOOK]
    probes_at = [i for i, c in enumerate(commands) if "if [ -d /logs/artifacts ]" in c]
    verifies = [i for i, c in enumerate(commands) if "timeout -s KILL" in c]
    # Once per turn for the task's hook, once in all for the build step's, in Harbor's order within the build turn:
    # the task's hook, the step's, then the entries are read, then test.sh; nothing runs the step's hook again.
    assert len(task_hooks) == 2 and len(step_hooks) == 1
    assert verifies[0] < task_hooks[1] < step_hooks[0] < probes_at[1] < verifies[1]
    assert probes_at[0] < verifies[0] and probes_at[0] > task_hooks[0] and probes_at[0] < step_hooks[0]
    # The hook ran in the working directory as the task's hooks do, within its own timeout plus the grace.
    hook = sandbox.execs[step_hooks[0]]
    assert hook["cwd"] == "/app" and hook["timeout_s"] == 20 + COLLECT_HOOK_GRACE_S and hook["user"] is None


async def test_every_turn_collects_the_convention_folder_even_without_declared_artifacts(tmp_path, monkeypatch):
    """Harbor's pass always reads /logs/artifacts; a sandbox without the folder stages nothing and the manifest says
    so."""
    server, task, sandbox, _ = multi_turn_server(
        tmp_path, monkeypatch, '[[steps]]\nname = "a"\n[[steps]]\nname = "b"\n', {"a": "A", "b": "B"}
    )
    response = await server.run_request(request_for(task))
    assert response.failure is None, response.failure
    assert probes(sandbox) == ["/logs/artifacts", "/logs/artifacts"]
    for turn in response.result.turns:
        assert manifest_of(turn) == [
            {
                "index": 0,
                "source": "/logs/artifacts",
                "service": None,
                "origin": "convention",
                "destination": "artifacts/00/logs/artifacts",
                "type": None,
                "status": "absent",
            }
        ]
        assert not (Path(turn.verifier_logs_dir) / "artifacts" / "00").exists()


async def test_a_collection_the_sandbox_cannot_serve_masks_the_turn_as_an_outage(tmp_path, monkeypatch):
    sandbox = ArtifactSandbox(scripted={}, present={"/logs/artifacts": "dir"})
    sandbox.archives = {}  # the archive command is answered, the download then has nothing to hand back

    async def lost(remote_path, local_path):
        raise ConnectionError("sandbox gone")

    sandbox.download = lost
    server, task, sandbox, _ = multi_turn_server(
        tmp_path, monkeypatch, '[[steps]]\nname = "a"\n[[steps]]\nname = "b"\n', {"a": "A", "b": "B"}, sandbox=sandbox
    )
    response = await server.run_request(request_for(task))
    assert response.failure is None, response.failure
    (turn,) = response.result.turns
    assert turn.mask_sample and turn.failure_kind == "provider_unavailable" and "sandbox gone" in turn.failure_reason
    assert response.result.mask_sample and Path(turn.verifier_logs_dir).is_dir()


SLOPCODEBENCH_TASK = (
    Path(__file__).parents[3] / "tests" / "unit_tests" / "fixtures" / "harbor_slopcodebench" / "code_search"
)


def record_main_build(task_folder: Path, image: str) -> None:
    """Record ``image`` as the build prepare made of the task's ``environment/`` (its Dockerfile needs a build: the
    slopcodebench Dockerfiles set ``SHELL``), with the configuration the build would leave."""
    (task_folder.parent / BUILT_IMAGES_FILE).write_text(
        json.dumps({context_digest(task_folder / "environment"): {"image": image}})
    )
    write_image_config(task_folder.parent, image, Env=[f"PATH={IMAGE_PATH}"], WorkingDir="/app")


async def test_a_slopcodebench_task_runs_its_checkpoints_with_their_app_artifacts(tmp_path, monkeypatch):
    """`gabeorlanski/code_search` from the hub's `gabeorlanski/slopcodebench` (its real task.toml and Dockerfile):
    five checkpoints, each collecting `/app` under `app` with its excludes and gating two reward.json keys with
    `min_reward`. The loader used to refuse all 36 tasks of the dataset for the per-step artifacts."""
    folder = tmp_path / "datasets" / "ds"
    shutil.copytree(SLOPCODEBENCH_TASK, folder / "code_search")
    record_main_build(folder / "code_search", "registry.example/slopcodebench-code-search:abc")
    sandbox = ArtifactSandbox(
        present={"/app": "dir"},
        archives={"/app": archive_of({"code_search/cli.py": "import sys\n"})},
        verifier_files={"reward.json": '{"core_pass_rate": 1.0, "strict_pass_rate": 0.5}'},
    )
    server, _, sandbox, _ = make_server(tmp_path, monkeypatch, sandbox)
    task = load_task(folder / "code_search")
    server.config.tasksets["ds"].tasks["code_search"] = task.digest
    rewards_per_turn(server, sandbox, ['{"core_pass_rate": 1.0, "strict_pass_rate": 0.5}'] * 5, name="reward.json")

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    result = response.result
    assert [turn.name for turn in result.turns] == [f"checkpoint_{n}" for n in range(1, 6)]
    assert result.reward == 1.5 and result.verifier_rewards == {"core_pass_rate": 1.0, "strict_pass_rate": 0.5}
    assert result.stopped_after_turn is None and result.turns_without_verifier_result == 0
    for turn in result.turns:
        staged = Path(turn.verifier_logs_dir) / "artifacts" / "01" / "app" / "code_search" / "cli.py"
        assert staged.read_text() == "import sys\n"
        assert [(e["origin"], e["status"]) for e in manifest_of(turn)] == [("convention", "absent"), ("step", "ok")]
    app_archive = next(
        e["command"] for e in sandbox.execs if "tar -czf" in e["command"] and " -C /app " in e["command"]
    )
    assert all(f"--exclude={name}" in app_archive for name in (".venv", "node_modules", "target", ".git"))
    assert [
        turn["metadata"]["agent_timeout_sec"]
        for turn in [
            call[2]["json"].model_dump(mode="json", exclude_unset=True)
            for call in server.server_client.calls
            if call[1].endswith("/v1/responses")
        ]
    ] == ["7200.0"] * 5


# -- per-step separate verification (Harbor's resolve_step_verifier_mode and _run_separate_verifier per step) --------


def wire_verifier_sandboxes(server, monkeypatch, make=None) -> tuple[list[FakeSandbox], list]:
    """Hand out a fresh verifier sandbox per creation (``make`` builds one), one level below the real
    ``_create_verifier_sandbox``: that method runs as it is on the turn's view (image, environment, working
    directory and network policy), and the spec it builds is recorded where it reaches the provider, in the
    ``start`` that ``_start_with_pull_retries`` is handed; returns the sandboxes and the specs."""
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
    )
    verifiers: list[FakeSandbox] = []
    specs: list = []
    build = make if make is not None else FakeSandbox

    async def start_with_pull_retries(task, *, key, what, make, start):
        class Starter:
            async def start(self, spec):
                specs.append(spec)

        await start(Starter())
        verifier = build()
        verifiers.append(verifier)
        server._sandboxes[key] = verifier
        return verifier

    monkeypatch.setattr(server, "_start_with_pull_retries", start_with_pull_retries)
    return verifiers, specs


SEPARATE_STEPS_HEAD = 'artifacts = ["/app/out.txt"]\n'
SEPARATE_STEPS = """
[verifier.environment]
docker_image = "org/verifier:1"

[[steps]]
name = "plan"

[[steps]]
name = "build"
[steps.verifier]
user = "grader"
env = { STEP = "build" }

[[steps]]
name = "check"
"""


async def test_a_separate_mode_task_verifies_each_turn_in_its_own_sandbox_while_the_agents_keeps_running(
    tmp_path, monkeypatch
):
    """A multi-step task whose task-level verifier is separate: every turn's test.sh runs in a verifier sandbox of its
    own, created and stopped around that verification, fed the turn's collected artifacts and tests; the agent's
    sandbox is never stopped between turns, and on the last turn it is stopped after its own collection pass and
    before the verifier grades, as Harbor's `_run_step` stops the agent environment on the last step of a
    separate-mode task."""
    agent = ArtifactSandbox(
        present={"/app/out.txt": "file"}, archives={"/app": archive_of({"out.txt": "v1\n"})}, image_user="agent"
    )
    server, task, agent, created = multi_turn_server(
        tmp_path,
        monkeypatch,
        SEPARATE_STEPS,
        {"plan": "P", "build": "B", "check": "C"},
        head=SEPARATE_STEPS_HEAD,
        sandbox=agent,
        base=TASK_TOML.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n'),
    )
    (task.path / "tests").mkdir(exist_ok=True)
    (task.path / "tests" / "helpers.py").write_text("HELPER = 1\n")
    task = load_task(task.path)
    server.config.tasksets["ds"].tasks["hello"] = task.digest
    stopped_before_verify: list[bool] = []

    class Verifier(FakeSandbox):
        async def exec(self, command, **kwargs):
            if "timeout -s KILL" in command:
                stopped_before_verify.append(agent.stopped)
            return await super().exec(command, **kwargs)

    verifiers, specs = wire_verifier_sandboxes(server, monkeypatch, make=Verifier)
    agent_stopped_at_each_turn: list[bool] = []
    server.server_client.turn_hooks.append(lambda: agent_stopped_at_each_turn.append(agent.stopped))
    probes_before_stop: list[int] = []
    agent_stop = agent.stop

    async def stop():
        probes_before_stop.append(len(probes(agent)))
        await agent_stop()

    agent.stop = stop

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    result = response.result
    assert [turn.name for turn in result.turns] == ["plan", "build", "check"] and task.turns[
        1
    ].verifier_mode == "separate"
    assert [turn.verifier_mode for turn in result.turns] == ["separate"] * 3 and result.verifier_mode == "separate"
    assert [turn.reward for turn in result.turns] == [1.0, 1.0, 1.0]
    # One verifier sandbox per turn, from [verifier.environment], each stopped once its turn was graded; the agent's
    # sandbox was still running when every turn's agent session opened, and no verifier sandbox is left owned.
    assert len(verifiers) == 3 and all(verifier.stopped for verifier in verifiers)
    assert all(spec.image == "org/verifier:1" for spec in specs) and len(created) == 1
    assert agent_stopped_at_each_turn == [False, False, False]
    # The agent's sandbox was running for the first two verifications and stopped for the third, after the third
    # turn's own collection pass (two probes per pass: the convention folder and the task's artifact) and before its
    # test.sh ran in the verifier sandbox; the final cleanup has nothing left to stop.
    assert stopped_before_verify == [False, False, True] and probes_before_stop[:1] == [6]
    assert not any(key.endswith("/verifier") for key in server._sandboxes)
    # test.sh never ran in the agent's sandbox; each verifier got the task's tests/, the step's over them, the turn's
    # artifacts (the agent's /app/out.txt as the archive the agent packed) and the transcript.
    assert not any("timeout -s KILL" in e["command"] for e in agent.execs)
    # Nothing was graded in the agent's sandbox, so nothing is reset there before a turn: its /tests keeps what the
    # image shipped, as under Harbor 0.22.0's separate mode (Harbor main resets only after a shared verifier).
    assert not any(GRADING_RESET in e["command"] for e in agent.execs)
    for verifier in verifiers:
        assert len([e for e in verifier.execs if "timeout -s KILL" in e["command"]]) == 1
        assert uploaded_files(verifier, "helpers.py") == [b"HELPER = 1\n"]
        assert agent.archives["/app"] in verifier.uploaded_archives
        assert len(uploaded_files(verifier, "trajectory.json")) == 1
    tests_uploads = [
        c["command"] for c in verifiers[1].execs if "tar -xzf" in c["command"] and "-C /tests" in c["command"]
    ]
    assert len(tests_uploads) == 2  # the task's tests/, then steps/build/tests/
    # The build step's verifier settings over the task's: its user and env reach the run in the fresh sandbox.
    build_run = next(e for e in verifiers[1].execs if "timeout -s KILL" in e["command"])
    assert build_run["user"] == "grader" and build_run["env"] == {"STEP": "build"}
    # A step without its own user in another image keeps that image's default, as a single-step task's verifier does.
    assert next(e for e in verifiers[0].execs if "timeout -s KILL" in e["command"])["user"] is None
    # The turn's logs folder holds the staged artifacts and the manifest, as the separate path's does.
    staged = Path(result.turns[0].verifier_logs_dir) / "artifacts"
    assert (staged / "01" / "app" / "out.txt").read_text() == "v1\n" and (staged / "manifest.json").is_file()
    assert not list(staged.glob("*.tar.gz"))  # consumed by the restore into the verifier sandbox


MIXED_STEPS = """
[[steps]]
name = "plan"

[[steps]]
name = "build"
[steps.verifier]
environment_mode = "separate"
[steps.verifier.environment]
docker_image = "org/step-verifier:1"
memory_mb = 4096

[[steps]]
name = "check"
"""


async def test_steps_may_mix_verifier_modes_and_a_step_may_bring_its_own_verifier_image(tmp_path, monkeypatch):
    """Harbor's resolve_step_verifier_mode: a step's explicit mode (or its own [steps.verifier.environment]) wins over
    the task's; here a shared task with one separate step in an image of its own, graded between two shared turns.
    The last step is shared, so no turn stops the agent's sandbox (Harbor stops it before the verifier only on a
    separate last step)."""
    stopped_at_verify: list[bool] = []

    class Agent(FakeSandbox):
        async def exec(self, command, **kwargs):
            if "timeout -s KILL" in command:
                stopped_at_verify.append(self.stopped)
            return await super().exec(command, **kwargs)

    server, task, agent, _ = multi_turn_server(
        tmp_path, monkeypatch, MIXED_STEPS, {"plan": "P", "build": "B", "check": "C"}, sandbox=Agent()
    )
    verifiers, specs = wire_verifier_sandboxes(server, monkeypatch)
    resets_at_activation: list[int] = []
    server.server_client.turn_hooks.append(
        lambda: resets_at_activation.append(len([e for e in agent.execs if GRADING_RESET in e["command"]]))
    )

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    result = response.result
    assert [turn.verifier_mode for turn in task.turns] == ["shared", "separate", "shared"]
    assert [turn.verifier_mode for turn in result.turns] == ["shared", "separate", "shared"]
    assert result.verifier_mode == "mixed" and [turn.reward for turn in result.turns] == [1.0, 1.0, 1.0]
    # Two shared verifications in the agent's sandbox, one in the step's own image, sized by the step's block.
    assert len([e for e in agent.execs if "timeout -s KILL" in e["command"]]) == 2
    assert len(verifiers) == 1 and specs[0].image == "org/step-verifier:1" and specs[0].resources.memory_mib == 4096
    # An explicit step verifier image starts in its own WORKDIR (the step block names none) with no network policy.
    assert specs[0].workdir is None and "network_policy" not in specs[0].provider_options
    assert len([e for e in verifiers[0].execs if "timeout -s KILL" in e["command"]]) == 1
    # The third turn opened on the same, still running, agent sandbox after the separate verification, and its own
    # shared verification ran there with the sandbox still up.
    assert len(server.server_client.seed_bodies()) == 3 and verifiers[0].stopped
    assert stopped_at_verify == [False, False]
    # The grading files are reset in the agent's sandbox before a turn only when the previous turn was graded there:
    # before `build` (after `plan`'s shared verification, whose own reset before uploading its tests empties /tests
    # alone and removes only the grader's files from /logs/verifier, so it is not counted here), not before `check`
    # (after `build`'s separate verification, which left nothing there), as Harbor main resets only after a shared
    # verifier and 0.22.0's separate mode never touches the agent's /tests.
    assert resets_at_activation == [0, 1, 1]


async def test_a_step_may_opt_out_of_a_separate_mode_task_into_the_agents_sandbox(tmp_path, monkeypatch):
    steps = (
        '[verifier.environment]\ndocker_image = "org/verifier:1"\n'
        '[[steps]]\nname = "plan"\n[steps.verifier]\nenvironment_mode = "shared"\n[[steps]]\nname = "build"\n'
    )
    server, task, agent, _ = multi_turn_server(
        tmp_path,
        monkeypatch,
        steps,
        {"plan": "P", "build": "B"},
        base=TASK_TOML.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n'),
    )
    stopped_before_verify: list[bool] = []

    class Verifier(FakeSandbox):
        async def exec(self, command, **kwargs):
            if "timeout -s KILL" in command:
                stopped_before_verify.append(agent.stopped)
            return await super().exec(command, **kwargs)

    verifiers, _ = wire_verifier_sandboxes(server, monkeypatch, make=Verifier)
    response = await server.run_request(request_for(task))
    assert response.failure is None, response.failure
    assert [turn.verifier_mode for turn in response.result.turns] == ["shared", "separate"]
    assert len([e for e in agent.execs if "timeout -s KILL" in e["command"]]) == 1 and len(verifiers) == 1
    # The separate step is the last: the agent's sandbox was stopped before its verifier graded.
    assert stopped_before_verify == [True]


async def test_separate_verifier_for_artifacts_forces_the_turns_with_artifacts_into_the_task_image(
    tmp_path, monkeypatch
):
    """`separate_verifier_for_artifacts` applies per turn: a turn whose task or step declares artifacts grades in a
    fresh sandbox of the task's own image (overlay and all), the others stay in the agent's sandbox; the seed no
    longer refuses the task."""
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\nartifacts = ["/app/out.txt"]\n'
    server, task, agent, created = multi_turn_server(
        tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"}, separate_verifier_for_artifacts=True
    )
    verifiers, specs = wire_verifier_sandboxes(server, monkeypatch)

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert [turn.verifier_mode for turn in response.result.turns] == ["shared", "separate"]
    assert len(created) == 1 and len(verifiers) == 1 and specs[0].image == task.image
    # The task image keeps the task's working directory, as the agent's sandbox had it (no overlay here).
    assert specs[0].workdir == task.workdir == "/app" and "network_policy" not in specs[0].provider_options
    assert response.result.verifier_mode == "mixed"


async def test_a_separate_turn_whose_verifier_sandbox_fails_masks_that_turn_only_and_the_agent_keeps_running(
    tmp_path, monkeypatch
):
    server, task, agent, _ = multi_turn_server(
        tmp_path, monkeypatch, MIXED_STEPS, {"plan": "P", "build": "B", "check": "C"}
    )

    async def failing_create(task, *, episode_key=None):
        raise RuntimeError("provider down")

    monkeypatch.setattr(server, "_create_verifier_sandbox", failing_create)
    response = await server.run_request(request_for(task))
    assert response.failure is None, response.failure
    result = response.result
    assert [turn.name for turn in result.turns] == ["plan", "build"]
    assert result.turns[1].mask_sample and result.turns[1].failure_kind == "provider_unavailable"
    assert "provider down" in result.turns[1].failure_reason and result.mask_sample
    assert result.turns[1].verifier_mode == "separate"


async def test_a_task_shipping_harbors_seed_trajectory_runs_for_the_oracle_and_is_refused_for_any_other_agent(
    tmp_path, monkeypatch
):
    """Harbor 0.22.0's `_resolve_load_trajectory` loads `steps/<first step>/trajectory.json` into the agent's session for
    every agent but the oracle and nop ("the oracle runs the task's own solution ... task authors validate with it").
    No harness here can be seeded with it yet: the oracle runs the task as Harbor's does, and any other agent is
    refused at seed, terminally, naming the file, before any sandbox is paid for."""
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n'
    seed = '{"schema_version": "ATIF-v1.7", "steps": []}'
    server, task, _, created = multi_turn_server(tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"})
    (task.path / "steps" / "plan" / "trajectory.json").write_text(seed)
    task = load_task(task.path)
    assert task.seed_trajectory == Path("steps/plan/trajectory.json")
    server.config.tasksets["ds"].tasks["hello"] = task.digest

    response = await server.run_request(request_for(task))

    failure = response.failure
    assert failure is not None and failure.stage == "seed" and failure.terminal
    assert "ships Harbor's seed trajectory steps/plan/trajectory.json" in failure.failure_reason
    assert (
        "refused for agent 'agent'" in failure.failure_reason and "The oracle validates it" in failure.failure_reason
    )
    assert created == [] and not server.server_client.seed_bodies()

    server, task, _, created = multi_turn_server(
        tmp_path / "oracle",
        monkeypatch,
        steps,
        {"plan": "P", "build": "B"},
        agent_implementation="harbor_oracle_agent",
    )
    (task.path / "steps" / "plan" / "trajectory.json").write_text(seed)
    task = load_task(task.path)
    server.config.tasksets["ds"].tasks["hello"] = task.digest
    response = await server.run_request(request_for(task))
    assert response.failure is None, response.failure
    assert [turn.reward for turn in response.result.turns] == [1.0, 1.0] and len(created) == 1


async def test_a_seed_trajectory_is_not_refused_for_an_agent_block_the_server_cannot_read(
    tmp_path, monkeypatch, caplog
):
    """An agent instance with several implementations under it (or a client the merged config does not show) is not
    known to be any harness: `_seed_solution` places no solution for it and warns, and the seed trajectory check does
    the same rather than refusing an oracle wired that way as "agent 'agent'"."""
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n'
    server, task, _, created = multi_turn_server(
        tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"}, agent_implementation="harbor_oracle_agent"
    )
    (task.path / "steps" / "plan" / "trajectory.json").write_text('{"schema_version": "ATIF-v1.7", "steps": []}')
    task = load_task(task.path)
    server.config.tasksets["ds"].tasks["hello"] = task.digest
    block = OmegaConf.to_container(server.server_client.global_config_dict)
    block["agent"]["responses_api_agents"]["other_agent"] = dict(
        block["agent"]["responses_api_agents"]["harbor_oracle_agent"]
    )
    server.server_client.global_config_dict = OmegaConf.create(block)
    assert server._agent_implementation() is None

    with caplog.at_level("WARNING"):
        response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert [turn.reward for turn in response.result.turns] == [1.0, 1.0] and len(created) == 1
    assert "ships Harbor's seed trajectory steps/plan/trajectory.json" in caplog.text
    assert "no single responses_api_agents implementation" in caplog.text
    assert "the task runs without its prior context" in caplog.text


def test_the_budget_counts_a_verifier_sandbox_start_per_separate_turn(tmp_path, monkeypatch):
    server, task, _, _ = multi_turn_server(
        tmp_path, monkeypatch, MIXED_STEPS, {"plan": "P", "build": "B", "check": "C"}
    )
    shared_only = multi_turn_server(
        tmp_path / "other",
        monkeypatch,
        MIXED_STEPS.replace(
            'environment_mode = "separate"\n[steps.verifier.environment]\ndocker_image = "org/step-verifier:1"\nmemory_mb = 4096\n',
            "",
        ),
        {"plan": "P", "build": "B", "check": "C"},
    )
    assert server._post_seed_budget_s(task) == shared_only[0]._post_seed_budget_s(shared_only[1]) + 900
