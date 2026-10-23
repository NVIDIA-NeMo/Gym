# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.base_resources_server import ReverifyMode
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.sandbox.agent_tools import verify_agent_response


JUDGE = "resources_servers.partner_if.grading_service.call_judge"
SCALE_SWE_VERIFY = "resources_servers.scale_swe.app.ScaleSWEResourcesServer.verify"
REASONING = {"type": "reasoning", "id": "r", "summary": []}
TOOL_EXCHANGE = [
    {"type": "function_call", "id": "f", "call_id": "a", "name": "read", "arguments": "{}", "status": "completed"},
    {"type": "function_call_output", "call_id": "a", "output": "1: def f():"},
]
HELLO = {
    "id": "m",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "HELLO", "annotations": []}],
}


def test_swe_handoff_carries_working_directory():
    from resources_servers.partner_if.swe_app import SWEIFSeedResponse

    value = SWEIFSeedResponse(sandbox_handle="box", workdir="/app/repo")
    assert value.model_dump()["workdir"] == "/app/repo"


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


def swe_server():
    from nemo_gym.server_utils import ServerClient
    from resources_servers.partner_if.swe_app import SWEIFConfig, SWEIFResourcesServer

    config = SWEIFConfig(
        host="127.0.0.1",
        port=8001,
        name="swe",
        entrypoint="swe_app.py",
        sandbox_provider="sandbox",
        sandbox_config={},
        actor_system_prompt="Say HELLO.",
        judge_model_server={"type": "responses_api_models", "name": "judge"},
    )
    return SWEIFResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def opencode_response(output, *, cutoff=False):
    """OpenCode's two response shapes: an unset status, or the output-token cutoff."""
    from resources_servers.partner_if.tests.test_app import response

    return response().model_dump() | {
        "output": output,
        "status": "incomplete" if cutoff else None,
        "incomplete_details": {"reason": "max_output_tokens"} if cutoff else None,
    }


def swe_request(answer="HELLO", *, output=None, cutoff=False, **contract):
    from resources_servers.partner_if.swe_app import SWEIFVerifyRequest
    from resources_servers.partner_if.tests.test_app import response
    from resources_servers.partner_if.tests.test_shared_adapter import shared_request

    request = shared_request().model_dump()
    request.update(instance_id="fixture", image_url="fixture:latest", workdir="/workspace", task_name="swe-1")
    request["verifier_metadata"]["partner_if"].update(evidence_mode="live_agent", **contract)
    request["verifier_metadata"]["partner_if"]["rubrics"] = request["verifier_metadata"]["partner_if"]["rubrics"][:1]
    visible = response(answer).model_dump()["output"] if output is None else output
    request["response"] = opencode_response(visible, cutoff=cutoff)
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


@pytest.mark.parametrize("completion, answer, expected", [(0, "HELLO", 1), (1, "GOODBYE", 0)])
async def test_if_points_do_not_depend_on_patch_correctness(completion, answer, expected):
    body = swe_request(answer)
    server = swe_server()
    assert server.config.judge_responses_create_params.max_output_tokens == 32768
    with patch(SCALE_SWE_VERIFY, AsyncMock(return_value=scale_swe_outcome(body, completion))):
        result = await server.verify(SimpleNamespace(), body)
    assert result.reward == expected
    assert result.task_outcome["reward"] == completion
    assert result.grading_result.dimensions["if"].partial_score == expected
    assert result.model_dump()["task_name"] == "swe-1"


@pytest.mark.parametrize("output", [None, []], ids=["cut_off", "cut_off_empty"])
async def test_a_cut_off_trajectory_fails_every_item_without_the_judge(output):
    body = swe_request(output=output, cutoff=True)
    verify = AsyncMock(return_value=scale_swe_outcome(body))
    with patch(SCALE_SWE_VERIFY, verify), patch(JUDGE, new_callable=AsyncMock) as judge:
        result = await swe_server().verify(SimpleNamespace(), body)
    judge.assert_not_called()
    assert (result.reward, result.mask_sample, result.failure_kind) == (0, False, "partner_if:truncated")
    assert [item.passed for item in result.if_result.items] == [False]
    # Scale-SWE still records its verdict and releases the actor's sandbox.
    verify.assert_awaited_once()
    assert result.task_outcome["evaluation_completed"] is True


@pytest.mark.parametrize(
    "output,row",
    [
        ([*TOOL_EXCHANGE, HELLO, REASONING], (1, False, None)),
        ([], (0, True, "partner_if:unscorable")),
        # OpenCode sends no sign that a run finished, so a reasoning-only run cannot be told from an interrupted one.
        ([REASONING], (0, True, "partner_if:unscorable")),
        ([HELLO, *TOOL_EXCHANGE], (0, True, "partner_if:unscorable")),
    ],
    ids=["ends_in_a_message", "no_export", "reasoning_only", "interrupted_after_a_tool_call"],
)
async def test_an_opencode_run_is_graded_only_when_it_ends_in_an_assistant_message(output, row):
    body = swe_request(output=output)
    verify = AsyncMock(return_value=scale_swe_outcome(body))
    with patch(SCALE_SWE_VERIFY, verify), patch(JUDGE, new_callable=AsyncMock) as judge:
        result = await swe_server().verify(SimpleNamespace(), body)
    judge.assert_not_called()
    assert (result.reward, result.mask_sample, result.failure_kind) == row
    verify.assert_awaited_once()


@pytest.mark.parametrize("cutoff", [False, True])
async def test_opencodes_failure_preset_gets_its_unmasked_zero_only_for_a_cut_off_run(cutoff):
    # With execution_failure_reward_zero, OpenCode verifies a failed run with its output emptied and
    # requires an unmasked zero. Emptied, any run but a cut-off one is masked, so the preset fails it.
    body = swe_request(cutoff=cutoff)
    server = swe_server()

    async def post(**request):
        with patch(SCALE_SWE_VERIFY, AsyncMock(return_value=scale_swe_outcome(body))):
            result = await server.verify(SimpleNamespace(), type(body).model_validate(request["json"]))
        return SimpleNamespace(ok=True, read=AsyncMock(return_value=result.model_dump_json().encode()))

    verified = verify_agent_response(
        SimpleNamespace(post=post),
        ResourcesServerRef(type="resources_servers", name="swe"),
        body,
        body.response,
        {},
        force_zero_reward=True,
    )
    if not cutoff:
        with pytest.raises(ValueError, match="scores an empty response as unmasked zero"):
            await verified
        return
    row = await verified
    assert (row["reward"], row["mask_sample"], row["failure_kind"]) == (0, False, "agent_run_error")
    # The row keeps the run's own response.
    assert row["if_result"]["passed"] == 0 and row["response"]["output"]


async def test_an_unsigned_contract_is_masked_after_scale_swe_releases_the_actors_sandbox():
    body = swe_request(audit_signoff=False)
    verify = AsyncMock(return_value=scale_swe_outcome(body))
    with patch(SCALE_SWE_VERIFY, verify), patch(JUDGE, new_callable=AsyncMock) as judge:
        result = await swe_server().verify(SimpleNamespace(), body)
    judge.assert_not_called()
    verify.assert_awaited_once()
    assert result.mask_sample and result.failure_kind == "partner_if:unsigned_contract"
    assert result.task_outcome["reward"] == 0 and result.model_dump()["task_name"] == "swe-1"


async def test_reverify_cannot_run_the_swe_verification_again():
    assert await swe_server().get_reverify_mode() == ReverifyMode.UNSUPPORTED
