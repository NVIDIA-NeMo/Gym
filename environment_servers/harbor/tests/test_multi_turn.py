# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor ``[[steps]]`` tasks run as turns: one agent session per turn on one sandbox, then the roll-up."""

import io
import json
import posixpath
import tarfile
from pathlib import Path

import pytest

from environment_servers.harbor.app import (
    NO_TURN_RESULT_KIND,
    HarborEpisodeRequest,
    HarborTurnResult,
    min_reward_failure,
    roll_up_turns,
)
from environment_servers.harbor.tests.conftest import TASK_TOML, FakeSandbox, agent_response, make_server
from nemo_gym.tasks.harbor.materialize import STEP_METADATA_KEY, materialize_task


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


def multi_turn_server(tmp_path, monkeypatch, steps_toml: str, steps: dict[str, str], *, head: str = "", **config):
    """``head`` holds top-level keys, which TOML only reads before the first table."""
    task_dir = tmp_path / "datasets" / "ds" / "hello"
    write_steps(task_dir, steps)
    return make_server(tmp_path, monkeypatch, toml=head + TASK_TOML + steps_toml, **config)


def rewards_per_turn(server, sandbox: FakeSandbox, rewards: list[str]) -> None:
    """Make the verifier of the n-th turn write ``rewards[n]``: each activation sets what its verify reads."""
    pending = list(rewards)

    def next_reward():
        sandbox.verifier_files = {"reward.txt": pending.pop(0)}

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


async def test_mean_averages_the_turns_that_produced_a_result(tmp_path, monkeypatch):
    steps = '[[steps]]\nname = "a"\n[[steps]]\nname = "b"\n'
    server, task, sandbox, _ = multi_turn_server(tmp_path, monkeypatch, steps, {"a": "A", "b": "B"})
    rewards_per_turn(server, sandbox, ["1", "0.5"])

    result = (await server.run_request(request_for(task))).result

    assert result.turn_reward_strategy == "mean"
    assert result.reward == 0.75 and result.verifier_rewards == {"reward": 0.75}


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


async def test_rows_whose_turns_differ_from_the_task_fail_terminally(tmp_path, monkeypatch):
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n'
    server, task, _, _ = multi_turn_server(tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"})
    request = request_for(task)
    request.task.task_input.turns.pop()

    response = await server.run_request(request)

    assert response.failure.terminal and "do not match the steps" in response.failure.failure_reason


async def test_a_single_turn_row_for_a_multi_step_task_is_refused_at_seed(tmp_path, monkeypatch):
    """Review (CI bot, round 34): a row with the task's digest and a top-level `responses_create_params` but no
    `turns` (a hand-built or stale-shaped row) passed the request schema for a task that has steps and took the
    single-turn path: one agent session and the task-level verifier, the steps never run. The row's turns are
    compared with the task's steps at seed, after the task resolves and before anything is provisioned, so both
    shapes are checked: no agent session is seeded and no sandbox is created."""
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n'
    server, task, _, created = multi_turn_server(tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"})
    row = request_for(task).model_dump(mode="json", exclude_unset=True)
    turns = row["task"]["task_input"].pop("turns")
    row["task"]["task_input"]["responses_create_params"] = turns[0]["responses_create_params"]

    response = await server.run_request(HarborEpisodeRequest.model_validate(row))

    assert response.result is None
    assert response.failure.stage == "seed" and response.failure.terminal
    assert "Row turns [] do not match the steps of task 'hello' ['plan', 'build']" in response.failure.failure_reason
    assert server.server_client.seed_bodies() == [] and created == []


async def test_a_step_verifiers_secret_template_passes_the_seed_and_fills_that_turns_verifier_alone(
    tmp_path, monkeypatch
):
    """Review (CI bot, round 34): the seed's preflight of each step's `[steps.verifier].env` resolved the templates
    without the server's `secrets`, which the task-level preflight and the turn's verifier itself consult, so a
    `${JUDGE_KEY}` only `secrets` supplies failed the seed terminally before any turn. Preflight and execution read
    the same sources; the value reaches that step's verifier process alone, never the seeds or the sandbox."""
    monkeypatch.delenv("JUDGE_KEY", raising=False)
    monkeypatch.setattr(
        "environment_servers.harbor.sandboxes.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
    )
    steps = (
        '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n[steps.verifier]\nenv = { API_KEY = "${JUDGE_KEY}" }\n'
    )
    server, task, sandbox, _ = multi_turn_server(
        tmp_path, monkeypatch, steps, {"plan": "P", "build": "B"}, secrets={"JUDGE_KEY": "synthetic-value"}
    )

    response = await server.run_request(request_for(task))

    assert response.failure is None, response.failure
    assert [turn.name for turn in response.result.turns] == ["plan", "build"]
    runs = [c for c in sandbox.execs if "test.sh" in c["command"] and "timeout" in c["command"]]
    assert [run["env"] for run in runs] == [{}, {"API_KEY": "synthetic-value"}]
    assert "synthetic-value" not in str(server.server_client.seed_bodies())
    assert "synthetic-value" not in str(server._sandbox_spec(task, "/app").env)
    # Without either source the seed still refuses the task, terminally, naming the variable.
    server.config.secrets = {}
    refused = await server.run_request(request_for(task, rollout="r2"))
    assert refused.failure.stage == "seed" and refused.failure.terminal
    assert "JUDGE_KEY" in refused.failure.failure_reason


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

    # plan: agent 60 (its own), verifier 120 (the task's); build: 300 and 30; check: 300 and 120.
    per_turn = (10 + 60 + 120 + 5) + (10 + 300 + 30 + 5) + (10 + 300 + 120 + 5)
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


async def test_a_forced_separate_verifier_is_refused_at_seed_for_a_multi_step_task(tmp_path, monkeypatch):
    """The loader admits multi-step tasks in shared mode only; `separate_verifier_for_artifacts` cannot force
    them into a mode their turns do not run, and the refusal costs no sandbox."""
    steps = '[[steps]]\nname = "plan"\n[[steps]]\nname = "build"\n'
    server, task, _, created = multi_turn_server(
        tmp_path,
        monkeypatch,
        steps,
        {"plan": "P", "build": "B"},
        head='artifacts = ["/app/out.txt"]\n',
        separate_verifier_for_artifacts=True,
    )

    response = await server.run_request(request_for(task))

    failure = response.failure
    assert failure.stage == "seed" and failure.terminal
    assert "multi-step task" in failure.failure_reason and "separate verification" in failure.failure_reason
    assert not created


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


def turn(name: str, rewards: dict[str, float] | None, *, masked: bool = False) -> HarborTurnResult:
    reward = (rewards or {}).get("reward", 0.0)
    return HarborTurnResult(
        name=name,
        reward=reward,
        verifier_rewards=rewards,
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
