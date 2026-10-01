# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.config_types import ModelServerRef
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.usersim.app import (
    PROBE_SCORERS,
    UserSimResourcesServer,
    UserSimResourcesServerConfig,
    _decode_provenance,
    _scorer_for_row,
    _validate_resolved_row,
)
from resources_servers.usersim.episode_contracts import (
    UserSimSimulationResult,
    UserSimVerificationInput,
    UserSimVerifyRequest,
)


REVISION = "a5f676bf6dc5a73914c8a0860f97c10dd2c214ee"  # pragma: allowlist secret


def _resolved_row(probe_type: str = "tool_calling", *, provenance_as_text: bool = True) -> dict:
    provenance = {"code_sha": REVISION, "bank_version": {}}
    return {
        "persona": {"first_name": "Avery"},
        "probe_type": probe_type,
        "probe_family": probe_type,
        "probe_variant": "default",
        "conversation_language": "English",
        "locale": "en_US",
        "trajectory_id": f"trajectory-{probe_type}",
        "usersim_config": {"max_turns": 2},
        "usersim_provenance": json.dumps(provenance) if provenance_as_text else provenance,
    }


def _server() -> UserSimResourcesServer:
    model = ModelServerRef(type="responses_api_models", name="support_model")
    return UserSimResourcesServer(
        config=UserSimResourcesServerConfig(
            host="127.0.0.1",
            port=12345,
            entrypoint="app.py",
            name="usersim_resources",
            probe_scorer_model=model,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def test_probe_scorer_model_is_required() -> None:
    with pytest.raises(ValueError, match="probe_scorer_model"):
        UserSimResourcesServerConfig(
            host="127.0.0.1",
            port=12345,
            entrypoint="app.py",
            name="usersim_resources",
        )


def _request(session_id: str = "session-0") -> SimpleNamespace:
    return SimpleNamespace(session={SESSION_ID_KEY: session_id})


def _seed_body(row: dict) -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id="resources-session-0",
        episode_id=EpisodeId(rollout_id="0-0", attempt=0),
        task_id=TaskId(taskset="usersim:example", task_id="0"),
        task_data={"resolved_row": row},
    )


def _native_result(probe_type: str = "tool_calling") -> UserSimSimulationResult:
    return UserSimSimulationResult.model_validate(
        {
            "trajectory_id": f"trajectory-{probe_type}",
            "conversation_messages": [
                {"role": "user", "content": "Please help."},
                {"role": "assistant", "content": "Here is the result."},
            ],
            "conversation_status": True,
            "simulation_outcome": {"status": "completed"},
        }
    )


@pytest.mark.parametrize("as_text", [True, False])
def test_resolved_row_accepts_json_and_legacy_mapping_provenance(as_text: bool) -> None:
    row = _resolved_row(provenance_as_text=as_text)
    _validate_resolved_row(row, expected_revision=REVISION)
    assert _decode_provenance(row["usersim_provenance"])["bank_version"] == {}


@pytest.mark.asyncio
async def test_verify_accepts_assistant_activation_failure_as_terminal_zero_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _server()
    server.session_id_to_seed = {}
    row = _resolved_row()
    request = _request()
    await server.seed_session(request, _seed_body(row))
    monkeypatch.setattr(server, "_evaluate", AsyncMock(side_effect=AssertionError("must not evaluate failed policy")))
    body = UserSimVerifyRequest(
        episode_id=EpisodeId(rollout_id="0-0", attempt=0),
        task_id=TaskId(taskset="usersim:example", task_id="0"),
        verification_input=UserSimVerificationInput(
            resolved_row=row,
            usersim_result={
                "trajectory_id": row["trajectory_id"],
                "conversation_messages": [{"role": "user", "content": "Teach me about local ecology."}],
                "conversation_status": False,
                "simulation_outcome": {
                    "status": "failed",
                    "failure_class": "assistant_activation_error",
                    "failure_attribution": "assistant_model",
                    "failure_detail": "Assistant Agent returned HTTP 500",
                },
            },
            invocations=[],
        ),
    )

    verified = await server.verify(request, body)

    assert verified.reward == 0.0
    assert verified.mask_sample is False
    assert verified.scenario_completed is False
    assert verified.reward_components["trajectory_evaluator_applied"] == 0.0
    assert verified.native_usersim_result is not None
    assert verified.native_usersim_result.simulation_outcome["failure_attribution"] == "assistant_model"


@pytest.mark.parametrize(
    ("provenance", "message"),
    [
        ("not json", "valid JSON"),
        (json.dumps({"code_sha": "0" * 40}), "does not match"),
        (None, "JSON object"),
    ],
)
def test_resolved_row_rejects_invalid_provenance(provenance: object, message: str) -> None:
    row = _resolved_row()
    row["usersim_provenance"] = provenance
    with pytest.raises(ValueError, match=message):
        _validate_resolved_row(row, expected_revision=REVISION)


@pytest.mark.asyncio
async def test_seed_preserves_prepared_row_without_resampling() -> None:
    server = _server()
    server.session_id_to_seed = {}
    row = _resolved_row()
    response = await server.seed_session(_request(), _seed_body(row))
    assert response.resolved_row == row
    assert server.session_id_to_seed["session-0"].resolved_row == row


@pytest.mark.asyncio
async def test_verify_rejects_mutated_prepared_row() -> None:
    server = _server()
    server.session_id_to_seed = {}
    row = _resolved_row()
    request = _request()
    await server.seed_session(request, _seed_body(row))
    changed = {**row, "trajectory_id": "different"}
    body = UserSimVerifyRequest(
        episode_id=EpisodeId(rollout_id="0-0", attempt=0),
        task_id=TaskId(taskset="usersim:example", task_id="0"),
        verification_input=UserSimVerificationInput(
            resolved_row=changed,
            usersim_result=_native_result(),
            invocations=[],
        ),
    )
    with pytest.raises(HTTPException, match="resolved row"):
        await server.verify(request, body)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("probe_type", "scorer"),
    [
        ("tool_calling", "tool_use"),
        ("safety_agentic", "safety_agentic"),
        ("financial_services", "financial_services"),
    ],
)
async def test_verify_uses_native_evaluator_and_explicit_probe_scorer(
    probe_type: str,
    scorer: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _server()
    server.session_id_to_seed = {}
    row = _resolved_row(probe_type)
    request = _request(probe_type)
    await server.seed_session(request, _seed_body(row))
    evaluation = {
        "envelope": {"axes": ["helpfulness", "accuracy", "coherence"], "scorers": [scorer]},
        "axes": {},
        "scorers": {scorer: {"status_proposal": True}},
        "skipped": False,
        "skipped_reason": None,
    }
    evaluate = AsyncMock(return_value=(evaluation, {"helpfulness": 1.0, "accuracy": 0.5, "coherence": 1.0}, 5 / 6))
    monkeypatch.setattr(server, "_evaluate", evaluate)
    body = UserSimVerifyRequest(
        episode_id=EpisodeId(rollout_id="0-0", attempt=0),
        task_id=TaskId(taskset="usersim:example", task_id="0"),
        verification_input=UserSimVerificationInput(
            resolved_row=row,
            usersim_result=_native_result(probe_type),
            invocations=[],
        ),
    )
    verification = await server.verify(request, body)
    assert PROBE_SCORERS[probe_type] == scorer
    assert verification.reward == pytest.approx(5 / 6)
    assert verification.scenario_completed is True
    assert verification.verifier_data["native_scorer_name"] == scorer
    evaluate.assert_awaited_once()


def test_health_scorer_only_applies_to_guarded_rows() -> None:
    row = _resolved_row("health_general_disclosure")
    assert _scorer_for_row(row) is None
    row["probe_variant"] = "guarded"
    assert _scorer_for_row(row) == "health_disclosure_concealment"


def test_resources_server_exposes_no_runtime_or_tool_routes() -> None:
    routes = {route.path for route in _server().setup_webserver().routes}
    assert "/seed_session" in routes
    assert "/verify" in routes
    assert "/close_session" in routes
    assert not any(path.startswith("/runtime") or path.startswith("/tools") for path in routes)
