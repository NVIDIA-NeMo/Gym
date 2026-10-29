# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import JUDGE_FAILED, SESSION_EXPIRED, SESSION_LOST, VERIFIER_ERROR, EpisodeId, TaskId
from nemo_gym.server_runtime import ResourcesSeedSessionRequest
from nemo_gym.server_runtime.testing.session_conformance import check_resources_session_contract
from resources_servers.nemo_user_sim import app as usersim_app
from resources_servers.nemo_user_sim.app import (
    PROBE_SCORERS,
    UserSimResourcesServer,
    UserSimResourcesServerConfig,
    _decode_provenance,
    _ResourcesModelFacade,
    _scorer_for_row,
    _validate_resolved_row,
)
from resources_servers.nemo_user_sim.episode_contracts import (
    UserSimSimulationResult,
    UserSimVerificationInput,
    UserSimVerifyRequest,
)


REVISION = "a5f676bf6dc5a73914c8a0860f97c10dd2c214ee"  # pragma: allowlist secret
PERSONAS_VERSION = "0.0.2"


def _resolved_row(probe_type: str = "tool_calling", *, provenance_as_text: bool = True) -> dict:
    provenance = {"code_sha": REVISION, "nemotron_personas_version": "synthetic", "bank_version": {}}
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
            name="nemo_user_sim_resources",
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
            name="nemo_user_sim_resources",
        )


def _request(session_id: str = "session-0") -> SimpleNamespace:
    return SimpleNamespace(session={SESSION_ID_KEY: session_id})


def _seed_body(row: dict, *, taskset: str = "nemo_user_sim:example") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id="resources-session-0",
        episode_id=EpisodeId(rollout_id="0-0", attempt=0),
        task_id=TaskId(taskset=taskset, task_id="0"),
        task_data={"resolved_row": row},
    )


def _usersim_result(probe_type: str = "tool_calling") -> UserSimSimulationResult:
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


class _ScriptedModelFacade:
    def __init__(self, content: str | Exception) -> None:
        self.content = content
        self.model_name = "scripted-judge"
        self.calls: list[dict[str, Any]] = []

    async def acompletion(self, messages: list[Any], **kwargs: Any) -> SimpleNamespace:
        self.calls.append({"messages": messages, **kwargs})
        if isinstance(self.content, Exception):
            raise self.content
        return SimpleNamespace(
            message=SimpleNamespace(content=self.content, reasoning_content=None, tool_calls=None),
            usage=None,
        )


def _judge_payload(score: int = 4) -> str:
    return json.dumps(
        {axis: {"score": score, "reasoning": "scripted"} for axis in ("helpfulness", "accuracy", "coherence")}
    )


def _verify_body(row: dict[str, Any]) -> UserSimVerifyRequest:
    return UserSimVerifyRequest(
        episode_id=EpisodeId(rollout_id="0-0", attempt=0),
        task_id=TaskId(taskset="nemo_user_sim:example", task_id="0"),
        verification_input=UserSimVerificationInput(
            resolved_row=row,
            usersim_result=_usersim_result(str(row["probe_type"])),
            invocations=[],
        ),
    )


async def _verify(
    server: UserSimResourcesServer,
    request: SimpleNamespace,
    row: dict[str, Any],
) -> Any:
    return await server.verify(request, _verify_body(row))


def _install_evaluator_scripts(
    monkeypatch: pytest.MonkeyPatch,
    facade: _ScriptedModelFacade,
    scorer: Callable[..., Any] | None = None,
) -> None:
    monkeypatch.setattr(usersim_app, "_ResourcesModelFacade", lambda *_args, **_kwargs: facade)
    if scorer is not None:
        import usersim.engine.evaluator.generator as generator

        monkeypatch.setattr(generator, "get_scorer", lambda _name: scorer)


@pytest.mark.parametrize("as_text", [True, False])
def test_resolved_row_accepts_json_and_legacy_mapping_provenance(as_text: bool) -> None:
    row = _resolved_row(provenance_as_text=as_text)
    _validate_resolved_row(row, expected_revision=REVISION, expected_personas_version=PERSONAS_VERSION)
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
        task_id=TaskId(taskset="nemo_user_sim:example", task_id="0"),
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
    assert verified.usersim_result is not None
    assert verified.usersim_result.simulation_outcome["failure_attribution"] == "assistant_model"
    assert not {"invocations", "resolved_row", "usersim_result"} & verified.verifier_data.keys()


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
        _validate_resolved_row(row, expected_revision=REVISION, expected_personas_version=PERSONAS_VERSION)


@pytest.mark.asyncio
async def test_seed_requires_pinned_personas_version_for_validation_tasks() -> None:
    server = _server()
    server.session_id_to_seed = {}
    row = _resolved_row(provenance_as_text=False)
    row["usersim_provenance"]["nemotron_personas_version"] = None

    with pytest.raises(HTTPException, match="Nemotron-Personas version") as error:
        await server.seed_session(
            _request("wrong-personas-version"),
            _seed_body(row, taskset="nemo_user_sim:validation"),
        )
    assert error.value.status_code == 422

    row["usersim_provenance"]["nemotron_personas_version"] = server.config.nemotron_personas_version
    response = await server.seed_session(
        _request("pinned-personas-version"),
        _seed_body(row, taskset="nemo_user_sim:validation"),
    )
    assert response.resources_session_id == "resources-session-0"


@pytest.mark.asyncio
async def test_seed_accepts_synthetic_provenance_independent_of_taskset_name() -> None:
    server = _server()
    server.session_id_to_seed = {}
    response = await server.seed_session(
        _request("synthetic-validation"),
        _seed_body(_resolved_row(), taskset="nemo_user_sim:validation"),
    )
    assert response.resources_session_id == "resources-session-0"


@pytest.mark.asyncio
async def test_seed_rejects_guarded_health_probe() -> None:
    row = _resolved_row("health_general_disclosure")
    row["probe_variant"] = "guarded"

    with pytest.raises(HTTPException, match="Guarded health probes are unsupported") as error:
        await _server().seed_session(_request("guarded"), _seed_body(row))

    assert error.value.status_code == 422


@pytest.mark.asyncio
async def test_seed_preserves_prepared_row_without_resampling() -> None:
    server = _server()
    server.session_id_to_seed = {}
    row = _resolved_row()
    response = await server.seed_session(_request(), _seed_body(row))
    assert response.resources_session_id == "resources-session-0"
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
        task_id=TaskId(taskset="nemo_user_sim:example", task_id="0"),
        verification_input=UserSimVerificationInput(
            resolved_row=changed,
            usersim_result=_usersim_result(),
            invocations=[],
        ),
    )
    with pytest.raises(HTTPException, match="resolved row"):
        await server.verify(request, body)


@pytest.mark.asyncio
async def test_verify_uses_real_evaluator_with_scripted_judge_and_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _server()
    server.session_id_to_seed = {}
    row = _resolved_row()
    request = _request()
    await server.seed_session(request, _seed_body(row))

    async def scorer(_trajectory: dict[str, Any], _models: dict[str, Any]) -> dict[str, Any]:
        return {"status_proposal": True, "scores": {}}

    facade = _ScriptedModelFacade(_judge_payload())
    _install_evaluator_scripts(monkeypatch, facade, scorer)

    verification = await _verify(server, request, row)

    assert PROBE_SCORERS["tool_calling"] == "tool_use"
    assert verification.reward == pytest.approx(0.8)
    assert verification.scenario_completed is True
    assert verification.verifier_data["scorer_name"] == "tool_use"
    assert verification.verifier_data["scorer_state"] == "ok"
    assert "invocations" not in verification.verifier_data
    assert len(facade.calls) == 1


@pytest.mark.asyncio
async def test_verify_budgets_for_reasoning_judge_output(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server()
    server.session_id_to_seed = {}
    row = _resolved_row("general_open_ended")
    request = _request("reasoning-judge")
    await server.seed_session(request, _seed_body(row))

    class BudgetAwareFacade(_ScriptedModelFacade):
        async def acompletion(self, messages: list[Any], **kwargs: Any) -> SimpleNamespace:
            self.calls.append({"messages": messages, **kwargs})
            content = _judge_payload() if kwargs["max_tokens"] >= 16_384 else '{"helpfulness":'
            return SimpleNamespace(
                message=SimpleNamespace(content=content, reasoning_content=None, tool_calls=None),
                usage=None,
            )

    facade = BudgetAwareFacade(_judge_payload())
    _install_evaluator_scripts(monkeypatch, facade)

    verification = await _verify(server, request, row)

    assert verification.mask_sample is False
    assert facade.calls[0]["max_tokens"] == 16_384


@pytest.mark.asyncio
async def test_verify_masks_swallowed_judge_failure_from_real_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _server()
    server.session_id_to_seed = {}
    row = _resolved_row("general_open_ended")
    request = _request("judge-failure")
    await server.seed_session(request, _seed_body(row))
    facade = _ScriptedModelFacade(RuntimeError("judge unavailable"))
    _install_evaluator_scripts(monkeypatch, facade)

    verification = await _verify(server, request, row)

    assert verification.reward == 0.0
    assert verification.mask_sample is True
    assert verification.failure_kind == JUDGE_FAILED
    assert verification.scenario_completed is True


@pytest.mark.asyncio
async def test_verify_masks_real_evaluator_scorer_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _server()
    server.session_id_to_seed = {}
    row = _resolved_row()
    request = _request("scorer-error")
    await server.seed_session(request, _seed_body(row))

    async def scorer(_trajectory: dict[str, Any], _models: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("scorer unavailable")

    _install_evaluator_scripts(monkeypatch, _ScriptedModelFacade(_judge_payload()), scorer)
    verification = await _verify(server, request, row)

    assert verification.mask_sample is True
    assert verification.failure_kind == VERIFIER_ERROR
    assert verification.verifier_data["scorer_state"] == "error"


@pytest.mark.asyncio
async def test_verify_treats_real_evaluator_scorer_skip_as_not_applicable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _server()
    server.session_id_to_seed = {}
    row = _resolved_row()
    request = _request("scorer-skip")
    await server.seed_session(request, _seed_body(row))

    async def scorer(_trajectory: dict[str, Any], _models: dict[str, Any]) -> dict[str, Any]:
        return {
            "status_proposal": True,
            "error": "no expected_identity on this trajectory — scorer skipped",
            "scores": {},
        }

    _install_evaluator_scripts(monkeypatch, _ScriptedModelFacade(_judge_payload()), scorer)
    verification = await _verify(server, request, row)

    assert verification.reward == pytest.approx(0.8)
    assert verification.mask_sample is False
    assert verification.scenario_completed is True
    assert verification.verifier_data["scorer_state"] == "not_applicable"
    assert verification.reward_components["scorer_applied"] == 0.0


@pytest.mark.asyncio
async def test_verify_masks_tri_state_scorer_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _server()
    server.session_id_to_seed = {}
    row = _resolved_row()
    request = _request("scorer-inconclusive")
    await server.seed_session(request, _seed_body(row))

    async def scorer(_trajectory: dict[str, Any], _models: dict[str, Any]) -> dict[str, Any]:
        return {"status_proposal": None, "scores": {}}

    _install_evaluator_scripts(monkeypatch, _ScriptedModelFacade(_judge_payload()), scorer)
    verification = await _verify(server, request, row)

    assert verification.reward == 0.0
    assert verification.mask_sample is True
    assert verification.failure_kind == "usersim:scorer_inconclusive"
    assert verification.scenario_completed is False


@pytest.mark.asyncio
async def test_model_facade_omits_unsupplied_json_schema_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server()
    server.server_client.post = AsyncMock(return_value=object())
    monkeypatch.setattr(usersim_app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(
        usersim_app,
        "get_response_json",
        AsyncMock(
            return_value={
                "id": "response-0",
                "created_at": 0,
                "object": "response",
                "model": "support-model",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "id": "message-0",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "{}", "annotations": []}],
                    }
                ],
            }
        ),
    )
    facade = _ResourcesModelFacade.__new__(_ResourcesModelFacade)
    facade.server = server
    facade.model = server.config.probe_scorer_model
    facade.model_name = "support-model"

    await facade.acompletion(
        [{"role": "user", "content": "score"}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "evaluation", "schema": {"type": "object"}},
        },
    )

    params = server.server_client.post.await_args.kwargs["json"]
    assert "strict" not in params.model_dump(exclude_none=True)["text"]["format"]

    await facade.acompletion(
        [{"role": "user", "content": "score"}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "evaluation", "schema": {"type": "object"}, "strict": False},
        },
    )
    params = server.server_client.post.await_args.kwargs["json"]
    assert params.model_dump(exclude_none=True)["text"]["format"]["strict"] is False

    await facade.acompletion(
        [{"role": "user", "content": "score"}],
        reasoning_effort="medium",
    )
    params = server.server_client.post.await_args.kwargs["json"]
    assert params.model_dump(exclude_none=True)["reasoning"] == {"effort": "medium"}


@pytest.mark.asyncio
async def test_verify_bounds_overall_evaluation_time(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server()
    server.config.evaluation_timeout_seconds = 0.01
    server.session_id_to_seed = {}
    row = _resolved_row("general_open_ended")
    request = _request("evaluation-timeout")
    await server.seed_session(request, _seed_body(row))

    class SlowFacade(_ScriptedModelFacade):
        async def acompletion(self, messages: list[Any], **kwargs: Any) -> SimpleNamespace:
            await usersim_app.asyncio.sleep(1)
            return await super().acompletion(messages, **kwargs)

    _install_evaluator_scripts(monkeypatch, SlowFacade(_judge_payload()))
    verification = await _verify(server, request, row)

    assert verification.mask_sample is True
    assert verification.failure_kind == VERIFIER_ERROR
    assert "overall timeout" in (verification.failure_reason or "")


@pytest.mark.asyncio
async def test_closed_and_unknown_sessions_verify_as_masked_lost_sessions() -> None:
    server = _server()
    server.session_id_to_seed = {}
    row = _resolved_row()
    request = _request("closed")
    body = _seed_body(row)
    await server.seed_session(request, body)
    await server.close_resources_session(
        request,
        SimpleNamespace(resources_session_id=body.resources_session_id, episode_id=body.episode_id),
    )

    for gone in (request, _request("unknown")):
        verification = await _verify(server, gone, row)
        assert (verification.reward, verification.mask_sample, verification.failure_kind) == (0.0, True, SESSION_LOST)


def test_resources_session_contract() -> None:
    server = _server()
    server.session_id_to_seed = {}
    server.resources_session_id_to_session_id = {}
    server.closed_resources_session_ids = {}

    check_resources_session_contract(
        server.setup_webserver(),
        _seed_body(_resolved_row()),
        keeps_state=True,
        holds_state=lambda resources_session_id: any(
            seeded.resources_session_id == resources_session_id for seeded in server.session_id_to_seed.values()
        ),
        verify_body=_verify_body(_resolved_row()).model_dump(mode="json"),
    )


@pytest.mark.asyncio
async def test_seed_retried_under_a_new_cookie_moves_the_record_and_close_without_it_releases() -> None:
    server = _server()
    server.session_id_to_seed = {}
    server.resources_session_id_to_session_id = {}
    server.closed_resources_session_ids = {}
    body = _seed_body(_resolved_row())

    await server.seed_session(_request("lost-reply"), body)
    retried = _request("retry")
    await server.seed_session(retried, body)

    assert set(server.session_id_to_seed) == {"retry"}
    assert server._seeded_episode(retried).resources_session_id == body.resources_session_id

    await server.close_resources_session(
        _request("no-cookie"),
        SimpleNamespace(resources_session_id=body.resources_session_id, episode_id=body.episode_id),
    )
    assert server.session_id_to_seed == {}
    assert server.resources_session_id_to_session_id == {}


@pytest.mark.asyncio
async def test_expired_session_is_removed_and_verifies_as_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 100.0
    monkeypatch.setattr(usersim_app, "monotonic", lambda: now)
    server = _server()
    server.config.session_ttl_seconds = 10.0
    server.session_id_to_seed = {}
    server.resources_session_id_to_session_id = {}
    request = _request("expired")
    await server.seed_session(request, _seed_body(_resolved_row()))
    now = 111.0

    verification = await _verify(server, request, _resolved_row())

    # The server dropped the seed because its lifetime ran out, which is an expiry, not a lost session.
    assert (verification.mask_sample, verification.failure_kind) == (True, SESSION_EXPIRED)
    assert "expired" not in server.session_id_to_seed
    assert server.resources_session_id_to_session_id == {}
    # The expiry record is bounded like the closed record.
    now = 122.0
    verification = await _verify(server, request, _resolved_row())
    assert verification.failure_kind == SESSION_LOST
    assert server.expired_session_ids == {}


@pytest.mark.asyncio
async def test_close_finds_the_session_by_id_before_the_cookie() -> None:
    server = _server()
    server.session_id_to_seed = {}
    server.resources_session_id_to_session_id = {}
    server.closed_resources_session_ids = {}
    first = _seed_body(_resolved_row())
    second = first.model_copy(update={"resources_session_id": "resources-session-1"})
    await server.seed_session(_request("cookie-a"), first)
    await server.seed_session(_request("cookie-b"), second)

    # The close names the first session but carries the second one's cookie.
    await server.close_resources_session(
        _request("cookie-b"),
        SimpleNamespace(resources_session_id=first.resources_session_id, episode_id=first.episode_id),
    )

    assert set(server.session_id_to_seed) == {"cookie-b"}
    assert server.resources_session_id_to_session_id == {"resources-session-1": "cookie-b"}
    assert first.resources_session_id in server.closed_resources_session_ids


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
