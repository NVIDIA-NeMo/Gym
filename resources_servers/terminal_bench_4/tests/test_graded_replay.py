# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A resumed run's rollout that this server already graded is answered from its record instead of running again."""

import json
from types import SimpleNamespace

import pytest

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import NeMoGymResponse
from resources_servers.terminal_bench_4 import lifecycle
from resources_servers.terminal_bench_4.models import (
    AgentTermination,
    SandboxedVerifyRequest,
    TerminalBench4RunRequest,
)
from resources_servers.terminal_bench_4.tests.test_resources_sessions import BENCHMARK_DIR, OTHER_TASK, Harness


RUN = "a" * 32
AGENT_TIMINGS = {
    "agent_execution": {"started_at": "2026-11-02T10:00:00+00:00", "finished_at": "2026-11-02T10:05:00+00:00"}
}
VERIFIER_STARTED = "2026-11-02T10:06:00+00:00"


def _run_request(harness, *, run_id=RUN, task=None, attempt=0, rollout_id="rollout-1"):
    row = harness.row
    if task is not None:
        manifest = json.loads((BENCHMARK_DIR / "manifest.json").read_text())
        ref = next(entry["ref"] for entry in manifest["tasks"] if entry["name"] == task)
        row = row | {"task_name": f"terminal-bench/{task}", "task_ref": ref}
    return TerminalBench4RunRequest.model_validate(
        {
            "responses_create_params": {"input": "solve the task"},
            "task_name": row["task_name"],
            "task_ref": row["task_ref"],
            "dataset_ref": row["dataset_ref"],
            "rollout_id": rollout_id,
            "_ng_run_id": run_id,
            "_ng_task_index": 7,
            "_ng_rollout_index": 0,
            "_ng_attempt_index": attempt,
        }
    )


def _client(name):
    return SimpleNamespace(session={SESSION_ID_KEY: name})


async def _graded_episode(tmp_path, monkeypatch, *, termination="completed", verifier_started=VERIFIER_STARTED):
    """One legacy episode that the agent finished and the verifier graded 1.0, as a segment that was then cut."""
    harness = Harness(tmp_path, monkeypatch)

    async def finalize(session, *, grade):
        session.result["verifier"] = {"started_at": verifier_started}
        if grade:
            session.result["verifier_result"] = {"rewards": {"reward": 1.0}}
        await lifecycle.cleanup(session)

    monkeypatch.setattr(lifecycle, "finalize_session", finalize)
    request = _run_request(harness)
    seeded = await harness.server.seed_session(_client("client-a"), request)
    verified = await harness.server.verify(
        _client("client-a"),
        SandboxedVerifyRequest(
            session_id=seeded.session_id,
            responses_create_params=request.responses_create_params,
            response=NeMoGymResponse(
                id="resp-1",
                created_at=0,
                model="model",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            ),
            termination=AgentTermination(reason=termination),
            agent_started=True,
            agent_timings=AGENT_TIMINGS,
        ),
    )
    return harness, seeded.session_id, verified


async def test_a_graded_attempt_is_answered_to_the_resumed_run_without_running_again(tmp_path, monkeypatch):
    first, session_id, verified = await _graded_episode(tmp_path, monkeypatch)
    assert verified.reward == 1.0 and verified.evaluation_completed
    state = json.loads(next((tmp_path / "jobs").glob("*.json")).read_text())
    assert state["gym_identity"] == {"run_id": RUN, "task_index": 7, "rollout_index": 0, "attempt_index": 0}

    # The resources server restarts with the next segment; the resumed run sends the rollout again, from a new client
    # session and as its next attempt.
    resumed = Harness(tmp_path, monkeypatch)
    seeded = await resumed.server.seed_session(
        _client("client-b"), _run_request(resumed, attempt=1, rollout_id="rollout-2")
    )

    assert resumed.prepared == 0
    replayed = seeded.verified_response
    assert replayed is not None and replayed.reward == 1.0 and replayed.evaluation_completed
    assert replayed.model_dump()["replayed_from"] == {"session_id": session_id, "attempt_index": 0}
    assert seeded.session_id == session_id


@pytest.mark.parametrize(
    "change",
    [
        "killed",
        "interrupted",
        "exception",
        "verifier_before_agent_finished",
        "masked",
        "partial_record",
        "unreadable_index",
    ],
)
async def test_a_record_that_was_not_graded_normally_is_ignored_and_the_rollout_runs(tmp_path, monkeypatch, change):
    if change == "killed":
        await _graded_episode(tmp_path, monkeypatch, termination="killed")
    elif change == "verifier_before_agent_finished":
        await _graded_episode(tmp_path, monkeypatch, verifier_started="2026-11-02T10:04:00+00:00")
    else:
        await _graded_episode(tmp_path, monkeypatch)
    jobs = tmp_path / "jobs"
    record = next(jobs.glob("*.json"))
    state = json.loads(record.read_text())
    if change == "interrupted":
        state["result"]["interrupted"] = {"reason": "resources_shutdown"}
    elif change == "exception":
        state["result"]["exception_info"] = {"exception_type": "CancelledError"}
    elif change == "masked":
        state["verified_response"] |= {"evaluation_completed": False, "mask_sample": True}
    record.write_text(json.dumps(state))
    if change == "partial_record":
        record.write_text(record.read_text()[: len(record.read_text()) // 2])
    if change == "unreadable_index":
        next((jobs / "gym-rollouts" / RUN).iterdir()).write_text("not an identity")

    resumed = Harness(tmp_path, monkeypatch)
    seeded = await resumed.server.seed_session(
        _client("client-b"), _run_request(resumed, attempt=1, rollout_id="rollout-2")
    )

    assert seeded.verified_response is None
    assert resumed.prepared == 1


@pytest.mark.parametrize("other", ["run", "task"])
async def test_a_record_of_another_run_or_task_in_the_shared_directory_is_ignored(tmp_path, monkeypatch, other):
    await _graded_episode(tmp_path, monkeypatch)

    resumed = Harness(tmp_path, monkeypatch)
    request = (
        _run_request(resumed, run_id="b" * 32, rollout_id="rollout-2")
        if other == "run"
        else _run_request(resumed, task=OTHER_TASK, rollout_id="rollout-2")
    )
    seeded = await resumed.server.seed_session(_client("client-b"), request)

    assert seeded.verified_response is None
    assert resumed.prepared == 1


async def test_a_request_without_a_gym_identity_runs_as_before(tmp_path, monkeypatch):
    await _graded_episode(tmp_path, monkeypatch)

    resumed = Harness(tmp_path, monkeypatch)
    request = _run_request(resumed, rollout_id="rollout-2").model_dump()
    for key in ("_ng_run_id", "_ng_task_index", "_ng_rollout_index", "_ng_attempt_index"):
        request.pop(key)
    seeded = await resumed.server.seed_session(_client("client-b"), TerminalBench4RunRequest.model_validate(request))

    assert seeded.verified_response is None
    assert resumed.prepared == 1
