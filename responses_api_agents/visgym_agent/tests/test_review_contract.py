# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regressions for capture identity, failure scoring and shipped config composition."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from responses_api_agents.visgym_agent.app import TextActionAgent, TextActionAgentConfig, TextActionAgentRunRequest
from responses_api_agents.visgym_agent.tests.visual_context_contract import Episode, http


def episode(*, compact=False):
    return Episode(
        TextActionAgent, TextActionAgentConfig, TextActionAgentRunRequest, enabled=True if compact else None, turns=2
    )


def error_response(status, body):
    info = MagicMock(real_url="http://model/v1/responses")
    reply = MagicMock(ok=False)
    reply.content.read = AsyncMock(return_value=body)
    reply.raise_for_status.side_effect = aiohttp.ClientResponseError(
        request_info=info, history=(), status=status, message="model refusal"
    )
    return reply


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("owner", [None, "tq-sample-0042"])
@pytest.mark.parametrize("attempt", [0, 1])
async def test_capture_routes_keep_explicit_owner_and_index_fallback(compact, owner, attempt):
    run = episode(compact=compact)
    run.request = type(run.request).model_validate(
        run.request.model_dump(by_alias=True) | {"_ng_rollout_id": owner, "_ng_attempt_index": attempt}
    )
    await run.run()
    calls = [c for c in run.calls if c["url_path"].endswith(("/v1/responses", "/context/measure"))]
    expected = (owner or "7-2") + (f"-a{attempt}" if attempt else "")
    assert calls and all(c["url_path"].startswith(f"/ng-rollout/{expected}/training-token-capture/") for c in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [400, 429, 500, "json", "invalid"])
@pytest.mark.parametrize("first", [True, False])
async def test_model_infrastructure_failures_abort_without_verification(failure, first):
    run = episode()
    original = run.post

    async def post(**kwargs):
        if kwargs["url_path"].endswith("/v1/responses") and run.generated == (0 if first else 1):
            if failure == "invalid":
                return http({"id": "invalid"})
            if failure == "json":
                reply = http({})
                reply.json.side_effect = json.JSONDecodeError("bad reply", "", 0)
                return reply
            return error_response(failure, b'{"detail":"inference unavailable"}')
        return await original(**kwargs)

    run.agent.server_client.post.side_effect = post
    with pytest.raises((aiohttp.ClientResponseError, json.JSONDecodeError, ValidationError)):
        await run.agent.run(run.request)
    assert not any(c["url_path"] == "/verify" for c in run.calls)
    assert [c for c in run.calls if c["url_path"] == "/close"][-1]["json"]["discard_reward"]


@pytest.mark.asyncio
@pytest.mark.parametrize("overflow", [400, 413, "marker"])
@pytest.mark.parametrize("first", [True, False])
async def test_context_refusal_stops_without_publishing_an_unseen_turn(overflow, first):
    run = episode()
    original = run.post

    async def post(**kwargs):
        if kwargs["url_path"].endswith("/v1/responses") and run.generated == (0 if first else 1):
            if overflow == "marker":
                from responses_api_agents.visgym_agent.tests.visual_context_contract import response

                payload = response(99)
                payload.update(output=[], metadata={"nemo_gym_context_window_exceeded": "true"})
                return http(payload)
            return error_response(overflow, b'{"detail":"maximum context length exceeded"}')
        return await original(**kwargs)

    run.agent.server_client.post.side_effect = post
    if first:
        with pytest.raises((aiohttp.ClientResponseError, AssertionError)):
            await run.agent.run(run.request)
        assert not any(c["url_path"] == "/verify" for c in run.calls)
    else:
        result = await run.run()
        assert result.metadata["termination_reason"] == "context_length_exceeded"
        assert all(getattr(item, "role", None) != "user" for item in result.output)
        assert run.generated == 1
    closed = [c for c in run.calls if c["url_path"] == "/close"][-1]["json"]
    assert bool(closed.get("discard_reward")) == first


@pytest.mark.asyncio
async def test_instructions_replace_the_default_system_prompt():
    run = episode()
    run.request.responses_create_params.instructions = "Caller-supplied instructions"
    await run.run()
    assert run.requests[0]["instructions"] == "Caller-supplied instructions"
    assert not any(item.get("role") == "system" for item in run.requests[0]["input"])


def test_compaction_call_limit_is_checked_at_config_admission():
    cfg = episode(compact=True).agent.config.model_dump()
    cfg["max_steps"] = 21
    with pytest.raises(ValidationError, match="max_steps.*max_model_calls"):
        TextActionAgentConfig.model_validate(cfg)


def test_unmeasured_full_history_token_limit_is_rejected():
    cfg = episode().agent.config.model_dump()
    cfg["max_total_sequence_length"] = 51200
    with pytest.raises(ValidationError, match="max_total_sequence_length.*context_history"):
        TextActionAgentConfig.model_validate(cfg)


def test_overlay_targets_the_agent_bound_by_the_environment():
    root = Path(__file__).resolve().parents[3]
    base = OmegaConf.load(root / "environments/visgym/config.yaml")
    overlay = OmegaConf.load(Path(__file__).parents[1] / "configs/first_last_two.yaml")
    merged = OmegaConf.merge(base, overlay)
    names = [k for k, v in merged.items() if "responses_api_agents" in v]
    assert len(names) == 1
    options = merged[names[0]].responses_api_agents.visgym_agent
    assert options.context_history.enabled
    assert options.max_steps == options.context_history.max_model_calls == 20


@pytest.mark.parametrize("mode", [None, "fingerprint"])
def test_compaction_requires_source_history_before_the_first_model_call(mode):
    from responses_api_agents.visgym_agent.visual_context import visual_context_client

    run = episode(compact=True)
    run.agent.server_client.global_config_dict["token_id_capture"]["parent_resolution"] = mode
    with pytest.raises(ValueError, match="parent_resolution=source_history"):
        visual_context_client(
            run.agent,
            config=run.agent.config.context_history,
            logical_rollout_id="owner",
            initial_request=run.request.responses_create_params,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("skip", [False, True])
async def test_direct_response_keeps_internal_sampling_and_capture_skip(skip):
    from nemo_gym.core import ModelCallSampling, encode_model_sampling

    run = episode()
    sampling = {"temperature": 0.6, "top_p": 0.95, "max_output_tokens": 8192}
    run.request = type(run.request).model_validate(
        run.request.model_dump(by_alias=True)
        | {
            "_ng_rollout_id": "tq-row",
            "_ng_model_sampling": sampling,
            "_ng_skip_token_capture": skip,
        }
    )
    await run.run()
    paths = [c["url_path"] for c in run.calls if c["url_path"].endswith("/v1/responses")]
    segment = encode_model_sampling(ModelCallSampling(**sampling))
    assert paths and all(f"/ng-sampling/{segment}/" in path for path in paths)
    assert all(("/training-token-capture/" in path) == (not skip) for path in paths)


@pytest.mark.asyncio
async def test_invalid_capture_identity_is_rejected_before_creating_a_session():
    run = episode()
    run.request.capture_rollout_id = "invalid/id"
    with pytest.raises(ValueError, match="_ng_rollout_id"):
        await run.run()
    assert run.calls == []
