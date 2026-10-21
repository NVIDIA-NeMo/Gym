# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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
        complete=True,
    )
    assert [e.kind for e in bundle.trajectory] == ["tool_call", "tool_result", "assistant"]
    assert bundle.trajectory[0].arguments == {"filePath": "/app/x.py"}
    assert bundle.instructions == "Read source with read."
    assert bundle.answer == "Done"
    assert "hidden" not in bundle.model_dump_json()


def test_incomplete_swe_execution_cannot_be_graded_as_an_ordinary_omission():
    from resources_servers.partner_if.swe_app import build_swe_evidence

    result = build_swe_evidence(context=[], system_prompt="Rules", output=[], complete=False)
    assert result.trajectory_error


def test_unknown_trajectory_kind_is_a_capture_error():
    from resources_servers.partner_if.swe_app import build_swe_evidence

    result = build_swe_evidence(
        context=[], system_prompt="Rules", output=[{"type": "unknown_new_kind"}], complete=True
    )
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


def swe_request(answer="HELLO", **contract):
    from resources_servers.partner_if.swe_app import SWEIFVerifyRequest
    from resources_servers.partner_if.tests.test_app import response
    from resources_servers.partner_if.tests.test_shared_adapter import shared_request

    request = shared_request().model_dump()
    request.update(instance_id="fixture", image_url="fixture:latest", workdir="/workspace", task_name="swe-1")
    request["verifier_metadata"]["partner_if"].update(evidence_mode="live_agent", **contract)
    request["verifier_metadata"]["partner_if"]["rubrics"] = request["verifier_metadata"]["partner_if"]["rubrics"][:1]
    request["response"] = response(answer).model_dump()
    return SWEIFVerifyRequest.model_validate(request)


@pytest.mark.parametrize("completion, answer, expected", [(0, "HELLO", 1), (1, "GOODBYE", 0)])
async def test_if_points_do_not_depend_on_patch_correctness(completion, answer, expected):
    from resources_servers.scale_swe.app import ScaleSWEVerifyResponse

    body = swe_request(answer)
    server = swe_server()
    assert server.config.judge_responses_create_params.max_output_tokens == 32768
    outcome = ScaleSWEVerifyResponse.model_validate(
        body.model_dump()
        | {
            "reward": completion,
            "evaluation_completed": True,
            "resolved": bool(completion),
            "patch_applied": True,
            "test_results": {},
            "test_output": "",
            "error": None,
            "eval_sandbox_start_time_taken": 0,
            "patch_verification_time_taken": 0,
        }
    )
    with patch("resources_servers.scale_swe.app.ScaleSWEResourcesServer.verify", AsyncMock(return_value=outcome)):
        result = await server.verify(SimpleNamespace(), body)
    assert result.reward == expected
    assert result.task_outcome["reward"] == completion
    assert result.grading_result.dimensions["if"].partial_score == expected
    assert result.model_dump()["task_name"] == "swe-1"


async def test_an_unsigned_contract_is_masked_without_a_verification_sandbox():
    from nemo_gym.server_utils import SESSION_ID_KEY

    server = swe_server()
    actor = MagicMock(stop=AsyncMock())
    server._session_id_to_sandbox["s"] = actor
    server._session_id_to_base_commit["s"] = "abc"
    verify = AsyncMock(side_effect=AssertionError("No verification sandbox may start"))
    with patch("resources_servers.scale_swe.app.ScaleSWEResourcesServer.verify", verify):
        result = await server.verify(SimpleNamespace(session={SESSION_ID_KEY: "s"}), swe_request(audit_signoff=False))
    assert result.mask_sample and result.failure_kind == "partner_if:unsigned_contract"
    assert result.task_outcome is None and result.model_dump()["task_name"] == "swe-1"
    # The actor's sandbox is released as Scale-SWE's own verify releases it.
    actor.stop.assert_awaited_once()
    assert "s" not in server._session_id_to_sandbox and "s" not in server._session_id_to_base_commit


def test_process_local_swe_sessions_require_a_single_worker():
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
        num_workers=2,
    )
    with pytest.raises(ValueError, match="num_workers=1"):
        SWEIFResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
