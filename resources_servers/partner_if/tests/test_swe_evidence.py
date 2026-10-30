# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SWE adapter: it seeds an Environment Server session with the row's Scale-SWE sandbox for opencode_agent,
and grades the actor's visible trajectory against the system prompt the row's contract records."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import ResourcesSeedSessionRequest, ReverifyMode
from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.partner_if.swe_app import SWEIFResourcesServer


JUDGE = "resources_servers.partner_if.grading_service.call_judge"
SCALE_SWE_VERIFY = "resources_servers.scale_swe.app.ScaleSWEResourcesServer.verify"
ANTI_CHEAT = "resources_servers.partner_if.swe_app.apply_anti_cheat_setup"
ACTOR_PROMPT = "You are a coding agent. End every reply with HELLO."
ADDITIONS = {"bash": "Keep each command short."}
REASONING = {"type": "reasoning", "id": "r", "summary": []}
TOOL_EXCHANGE = [
    {"type": "function_call", "id": "f", "call_id": "a", "name": "read", "arguments": "{}", "status": "completed"},
    {"type": "function_call_output", "call_id": "a", "output": "1: def f():"},
]
DESCRIPTOR = {"sandbox_id": "box", "workdir": "/workspace"}
HELLO = {
    "id": "m",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "HELLO", "annotations": []}],
}


def test_swe_evidence_preserves_actions_and_excludes_hidden_reasoning():
    from resources_servers.partner_if.swe_app import build_swe_evidence

    bundle = build_swe_evidence(
        context=[{"role": "user", "content": "Fix it"}],
        system_prompt="Read source with read.",
        output=[
            {"type": "reasoning", "summary": [{"text": "hidden"}]},
            {"type": "function_call", "call_id": "a", "name": "read", "arguments": '{"filePath":"/app/x.py"}'},
            {"type": "function_call_output", "call_id": "a", "output": "1: def f():"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done"}]},
        ],
    )
    assert [e.kind for e in bundle.trajectory] == ["tool_call", "tool_result", "assistant"]
    assert bundle.trajectory[0].arguments == {"filePath": "/app/x.py"}
    assert bundle.instructions == "Read source with read."
    assert bundle.answer == "Done"
    assert "hidden" not in bundle.model_dump_json()


def test_unknown_trajectory_kind_is_a_capture_error():
    from resources_servers.partner_if.swe_app import build_swe_evidence

    result = build_swe_evidence(context=[], system_prompt="Rules", output=[{"type": "unknown_new_kind"}])
    assert result.trajectory_error


def test_a_long_tool_output_keeps_its_start_and_end_and_says_how_much_is_left_out():
    from resources_servers.partner_if.swe_app import TOOL_OUTPUT_LIMIT, shorten_tool_output

    assert shorten_tool_output("x" * TOOL_OUTPUT_LIMIT) == "x" * TOOL_OUTPUT_LIMIT
    start, marker, end = shorten_tool_output("A" * 300 + "B" * 1000 + "C" * 300).split("\n")
    assert (start, end) == ("A" * (TOOL_OUTPUT_LIMIT // 2), "C" * (TOOL_OUTPUT_LIMIT // 2))
    assert marker == f"[… {1600 - TOOL_OUTPUT_LIMIT:,} characters omitted …]"


def test_swe_evidence_shortens_only_tool_outputs_and_answers_with_the_final_message():
    from resources_servers.partner_if.swe_app import TOOL_OUTPUT_LIMIT, build_swe_evidence

    # The SWE rubrics grade the agent's messages and tool calls, so those stay whole however long.
    long = "x" * (TOOL_OUTPUT_LIMIT * 4)
    bundle = build_swe_evidence(
        context=[{"role": "user", "content": "Fix it"}],
        system_prompt="Rules",
        output=[
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": long}]},
            {"type": "function_call", "call_id": "a", "name": "bash", "arguments": json.dumps({"command": long})},
            {"type": "function_call_output", "call_id": "a", "output": long},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done"}]},
        ],
    )
    message, call, result, final = bundle.trajectory
    assert message.text == long and call.arguments == {"command": long}
    start, marker, end = result.text.split("\n")
    assert start == end == "x" * (TOOL_OUTPUT_LIMIT // 2) and "characters omitted" in marker
    # Every message stays in the trajectory, so the answer need not repeat them.
    assert (final.text, bundle.answer) == ("Done", "Done")


def swe_server(**config):
    from resources_servers.partner_if.swe_app import SWEIFConfig

    config = SWEIFConfig(
        host="127.0.0.1",
        port=8001,
        name="swe",
        entrypoint="swe_app.py",
        sandbox_provider="sandbox",
        sandbox_config={},
        judge_model_server={"type": "responses_api_models", "name": "judge"},
        **config,
    )
    return SWEIFResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def opencode_response(output, *, status="completed", cutoff=False):
    """A response as opencode_agent reports it: it always sets a status, and a continuation that the output-token
    cap cut off also carries the reason."""
    from resources_servers.partner_if.tests.test_app import response

    return response().model_dump() | {
        "output": output,
        "status": "incomplete" if cutoff else status,
        "incomplete_details": {"reason": "max_output_tokens"} if cutoff else None,
    }


def swe_request(answer="HELLO", *, output=None, status="completed", cutoff=False, rubrics=1, **contract):
    from resources_servers.partner_if.swe_app import SWEIFVerifyRequest
    from resources_servers.partner_if.tests.test_app import response
    from resources_servers.partner_if.tests.test_shared_adapter import shared_request

    request = shared_request().model_dump()
    request.update(instance_id="fixture", image_url="fixture:latest", workdir="/workspace", task_name="swe-1")
    partner_if = request["verifier_metadata"]["partner_if"]
    partner_if.update(
        evidence_mode="live_agent", actor_system_prompt=ACTOR_PROMPT, tool_description_additions=ADDITIONS
    )
    partner_if.update(contract)
    partner_if["rubrics"] = partner_if["rubrics"][:rubrics]
    visible = response(answer).model_dump()["output"] if output is None else output
    request["response"] = opencode_response(visible, status=status, cutoff=cutoff)
    return SWEIFVerifyRequest.model_validate(request)


def scale_swe_outcome(body, reward=0):
    from resources_servers.scale_swe.app import ScaleSWEVerifyResponse

    return ScaleSWEVerifyResponse.model_validate(
        body.model_dump()
        | {
            "reward": reward,
            "evaluation_completed": True,
            "resolved": bool(reward),
            "patch_applied": True,
            "test_results": {},
            "test_output": "",
            "error": None,
            "eval_sandbox_start_time_taken": 0,
            "patch_verification_time_taken": 0,
        }
    )


def session(session_id="session-one"):
    """The verify request's session, as the cookie from its seed sets it."""
    return SimpleNamespace(session={SESSION_ID_KEY: session_id})


@pytest.mark.parametrize("completion, answer, expected", [(0, "HELLO", 1), (1, "GOODBYE", 0)])
async def test_if_points_do_not_depend_on_patch_correctness(completion, answer, expected):
    body = swe_request(answer)
    server = swe_server()
    assert server.config.judge_responses_create_params.max_output_tokens == 32768
    with patch(SCALE_SWE_VERIFY, AsyncMock(return_value=scale_swe_outcome(body, completion))):
        result = await server.verify(session(), body)
    assert result.reward == expected
    assert result.task_outcome["reward"] == completion
    assert result.grading_result.dimensions["if"].partial_score == expected
    assert result.model_dump()["task_name"] == "swe-1"


async def test_the_judge_reads_the_actors_prompt_and_tool_additions_from_the_contract():
    from resources_servers.partner_if.tests.test_app import response

    body = swe_request(rubrics=2)
    verdict = {"id": "semantic", "passed": True, "reason": "Greeting", "evidence": "HELLO"}
    judgment = {"items": [verdict | {"references": [{"source": "answer", "start_line": 1, "end_line": 1}]}]}
    with (
        patch(SCALE_SWE_VERIFY, AsyncMock(return_value=scale_swe_outcome(body))),
        patch(JUDGE, new_callable=AsyncMock, return_value=response(json.dumps(judgment))) as judge,
    ):
        result = await swe_server().verify(session(), body)
    evidence = judge.call_args.kwargs["json"]["input"][1]["content"]
    assert ACTOR_PROMPT in evidence and ADDITIONS["bash"] in evidence
    assert result.reward == 1 and not result.mask_sample


@pytest.mark.parametrize("output", [None, []], ids=["cut_off", "cut_off_empty"])
async def test_a_cut_off_trajectory_fails_every_item_without_the_judge(output):
    body = swe_request(output=output, cutoff=True)
    verify = AsyncMock(return_value=scale_swe_outcome(body))
    with patch(SCALE_SWE_VERIFY, verify), patch(JUDGE, new_callable=AsyncMock) as judge:
        result = await swe_server().verify(session(), body)
    judge.assert_not_called()
    assert (result.reward, result.mask_sample, result.failure_kind) == (0, False, "partner_if:truncated")
    assert [item.passed for item in result.if_result.items] == [False]
    # Scale-SWE still records its verdict and releases the actor's sandbox.
    verify.assert_awaited_once()
    assert result.task_outcome["evaluation_completed"] is True


@pytest.mark.parametrize(
    "status,output,row",
    [
        ("completed", [*TOOL_EXCHANGE, HELLO, REASONING], (1, False, None)),
        # A finished run with nothing visible is the model's failure.
        ("completed", [REASONING], (0, False, "partner_if:no_answer")),
        # opencode_agent reports a run its deadline stopped as incomplete, and a run that broke as failed.
        ("incomplete", [*TOOL_EXCHANGE, HELLO], (0, True, "partner_if:unscorable")),
        ("failed", [], (0, True, "partner_if:unscorable")),
    ],
    ids=["completed", "completed_without_visible_output", "deadline", "failed"],
)
async def test_a_run_is_graded_only_when_opencode_agent_reports_it_completed(status, output, row):
    body = swe_request(output=output, status=status)
    verify = AsyncMock(return_value=scale_swe_outcome(body))
    with patch(SCALE_SWE_VERIFY, verify), patch(JUDGE, new_callable=AsyncMock) as judge:
        result = await swe_server().verify(session(), body)
    judge.assert_not_called()
    assert (result.reward, result.mask_sample, result.failure_kind) == row
    verify.assert_awaited_once()


@pytest.mark.parametrize(
    "contract",
    [
        {"actor_system_prompt": None},
        {"actor_system_prompt": None, "tool_description_additions": {}, "evidence_mode": "tool_call_response"},
    ],
    ids=["no_actor_prompt", "not_a_live_agent"],
)
async def test_a_contract_the_judge_cannot_grade_is_refused_before_the_native_tests(contract):
    verify = AsyncMock()
    with patch(SCALE_SWE_VERIFY, verify), pytest.raises(HTTPException) as refused:
        await swe_server().verify(session(), swe_request(**contract))
    assert refused.value.status_code == 422
    verify.assert_not_awaited()


async def test_an_unsigned_contract_is_masked_after_scale_swe_releases_the_actors_sandbox():
    body = swe_request(audit_signoff=False)
    verify = AsyncMock(return_value=scale_swe_outcome(body))
    with patch(SCALE_SWE_VERIFY, verify), patch(JUDGE, new_callable=AsyncMock) as judge:
        result = await swe_server().verify(session(), body)
    judge.assert_not_called()
    verify.assert_awaited_once()
    assert result.mask_sample and result.failure_kind == "partner_if:unsigned_contract"
    assert result.task_outcome["reward"] == 0 and result.model_dump()["task_name"] == "swe-1"


async def test_reverify_cannot_run_the_swe_verification_again():
    assert await swe_server().get_reverify_mode() == ReverifyMode.UNSUPPORTED


async def test_the_verifier_fixture_grades_its_cases_without_a_judge_or_a_sandbox():
    from nemo_gym.verification import exercise_verifier_fixture
    from resources_servers.partner_if.swe_app import VERIFIER_FIXTURE

    with patch(JUDGE, new_callable=AsyncMock) as judge, patch(SCALE_SWE_VERIFY, AsyncMock()) as verify:
        results = await exercise_verifier_fixture(
            VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="stochastic"
        )
    assert {result.kind for result in results} == {"full_reward", "zero_reward", "malformed"}
    judge.assert_not_called()
    verify.assert_not_awaited()


def test_the_server_requires_one_worker():
    with pytest.raises(ValueError, match="num_workers=1"):
        swe_server(num_workers=2)


class FakeSandbox:
    """Answers every command; a command containing `failing` exits 1, and the first `failing_stops` stops fail."""

    def __init__(self, failing=None, failing_stops=0):
        self.commands, self.stops, self.failing, self.failing_stops = [], 0, failing, failing_stops

    async def exec(self, command, **options):
        self.commands.append(command)
        if self.failing and self.failing in command:
            return SandboxExecResult(stdout="", stderr="Permission denied", return_code=1)
        return SandboxExecResult(stdout="base\n" if "rev-parse HEAD" in command else "", stderr="", return_code=0)

    async def serialize(self):
        return dict(DESCRIPTOR)

    async def stop(self):
        if self.failing_stops:
            self.failing_stops -= 1
            raise RuntimeError("Provider unavailable")
        self.stops += 1


def seed_body(session_id="session-one", **task_data):
    return {
        "resources_session_id": session_id,
        "episode_id": {"rollout_id": "r", "attempt": 0},
        "task_id": {"taskset": "partner_if", "task_id": "fixture"},
        "task_data": {
            "instance_id": "fixture",
            "workdir": "/workspace",
            "image_url": "fixture:latest",
            "pre_commands": "configure-the-harness",
            "verifier_metadata": swe_request().verifier_metadata.model_dump(mode="json"),
        }
        | task_data,
    }


def started(*sandboxes):
    """Hand out these sandboxes, in order, as the ones the server starts."""
    return patch.object(SWEIFResourcesServer, "_create_sandbox", AsyncMock(side_effect=list(sandboxes)))


def test_a_seed_hands_the_prepared_sandbox_to_the_agent():
    sandbox = FakeSandbox()
    with started(sandbox), patch(ANTI_CHEAT, AsyncMock()) as anti_cheat:
        seeded = TestClient(swe_server().setup_webserver()).post("/seed_session", json=seed_body())
    assert seeded.status_code == 200
    assert seeded.json()["sandbox_access"] == {
        "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": DESCRIPTOR},
        "workdir": "/workspace",
    }
    # The row's pre_commands run first, in the task's directory, then Scale-SWE's own setup.
    assert sandbox.commands[0] == "cd /workspace && configure-the-harness"
    anti_cheat.assert_awaited_once()
    assert sandbox.stops == 0


def test_a_repeated_seed_gets_the_same_sandbox_and_starts_no_other():
    sandbox = FakeSandbox()
    with started(sandbox), patch(ANTI_CHEAT, AsyncMock()):
        client = TestClient(swe_server().setup_webserver(), raise_server_exceptions=False)
        first, second = (client.post("/seed_session", json=seed_body()) for _ in range(2))
        other_task = client.post("/seed_session", json=seed_body(instance_id="another"))
    assert first.status_code == second.status_code == 200 and first.json() == second.json()
    assert other_task.status_code == 409


@pytest.mark.parametrize(
    "failing,step", [("configure-the-harness", "pre_commands"), ("rev-parse HEAD", "base commit")], ids=["pre", "head"]
)
def test_a_seed_whose_setup_fails_stops_its_sandbox_and_fails(failing, step):
    # The row's pre_commands also set up the agent's harness, so a failure there is not the task the contract grades.
    sandbox = FakeSandbox(failing=failing)
    with started(sandbox), patch(ANTI_CHEAT, AsyncMock()), pytest.raises(RuntimeError, match=step):
        TestClient(swe_server().setup_webserver()).post("/seed_session", json=seed_body())
    assert sandbox.stops == 1


def test_a_seed_that_cannot_hand_over_its_sandbox_stops_it():
    sandbox = FakeSandbox()
    sandbox.serialize = AsyncMock(side_effect=RuntimeError("The provider cannot reconnect"))
    with started(sandbox), patch(ANTI_CHEAT, AsyncMock()):
        seeded = TestClient(swe_server().setup_webserver(), raise_server_exceptions=False).post(
            "/seed_session", json=seed_body()
        )
    assert seeded.status_code == 500 and sandbox.stops == 1


def test_a_seed_refuses_a_contract_without_the_actors_prompt_before_starting_a_sandbox():
    body = seed_body()
    del body["task_data"]["verifier_metadata"]["partner_if"]["actor_system_prompt"]
    with started() as create:
        seeded = TestClient(swe_server().setup_webserver()).post("/seed_session", json=body)
    assert seeded.status_code == 422
    create.assert_not_awaited()


def test_the_session_lifecycle_meets_gyms_close_contract():
    server = swe_server()
    with started(FakeSandbox(), FakeSandbox()), patch(ANTI_CHEAT, AsyncMock()):
        check_resources_session_contract(
            server.setup_webserver(),
            ResourcesSeedSessionRequest.model_validate(seed_body()),
            keeps_state=True,
            holds_state=lambda session_id: session_id in server._session_id_to_sandbox,
            # A verify after close is masked: the session's sandbox, and with it the patch, is gone.
            verify_body=swe_request().model_dump(mode="json"),
        )


@pytest.mark.parametrize("seed_first", [False, True])
def test_a_close_keeps_its_episode_so_another_episode_cannot_close_the_session_again(seed_first):
    close = {"resources_session_id": "session-one", "episode_id": {"rollout_id": "r", "attempt": 0}}
    other = close | {"episode_id": {"rollout_id": "other", "attempt": 0}}
    with started(FakeSandbox()), patch(ANTI_CHEAT, AsyncMock()):
        client = TestClient(swe_server().setup_webserver(), raise_server_exceptions=False)
        if seed_first:
            assert client.post("/seed_session", json=seed_body()).status_code == 200
        assert client.post("/close_session", json=close).status_code == 200
        assert client.post("/close_session", json=close).status_code == 200
        assert client.post("/close_session", json=other).status_code == 409
        assert client.post("/seed_session", json=seed_body()).status_code == 409


async def test_a_retired_session_keeps_its_seeds_episode():
    server = swe_server()
    sandbox = FakeSandbox()
    close = {"resources_session_id": "session-one", "episode_id": {"rollout_id": "r", "attempt": 0}}
    other = close | {"episode_id": {"rollout_id": "other", "attempt": 0}}
    with started(sandbox), patch(ANTI_CHEAT, AsyncMock()):
        client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
        assert client.post("/seed_session", json=seed_body()).status_code == 200
        await server.retire_session_state("session-one")
        assert sandbox.stops == 1
        assert client.post("/close_session", json=other).status_code == 409
        assert client.post("/close_session", json=close).status_code == 200
        assert client.post("/seed_session", json=seed_body()).status_code == 409


def test_a_sandbox_that_fails_to_stop_stays_for_a_retried_close():
    sandbox = FakeSandbox(failing_stops=1)
    close = {"resources_session_id": "session-one", "episode_id": {"rollout_id": "r", "attempt": 0}}
    with started(sandbox), patch(ANTI_CHEAT, AsyncMock()):
        client = TestClient(swe_server().setup_webserver(), raise_server_exceptions=False)
        assert client.post("/seed_session", json=seed_body()).status_code == 200
        assert client.post("/close_session", json=close).status_code == 500
        assert client.post("/close_session", json=close).status_code == 200
        # A closed session is not seeded again.
        assert client.post("/seed_session", json=seed_body()).status_code == 409
    assert sandbox.stops == 1


def test_shutdown_stops_the_sandboxes_of_unclosed_sessions():
    sandbox = FakeSandbox()
    with started(sandbox), patch(ANTI_CHEAT, AsyncMock()):
        with TestClient(swe_server().setup_webserver()) as client:
            assert client.post("/seed_session", json=seed_body()).status_code == 200
    assert sandbox.stops == 1
