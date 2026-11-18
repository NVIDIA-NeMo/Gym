# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recorded-context configuration must not change ordinary text grading or recovery."""

from unittest.mock import AsyncMock, patch

import pytest
from omegaconf import OmegaConf

from benchmarks.partner_if.tests.test_sampling import resolved_config
from nemo_gym.client import ServerClient
from nemo_gym.config import BaseServerConfig, get_first_server_config_dict
from nemo_gym.server_runtime import ReverifyMode
from resources_servers.partner_if.app import PartnerIFResourcesServer, PartnerIFResourcesServerConfig, VerifyRequest
from resources_servers.partner_if.recorded_context_app import (
    RecordedContextResourcesServer,
    RecordedContextResourcesServerConfig,
)
from resources_servers.partner_if.tests.test_app import response


@pytest.fixture
def benchmark_config():
    return resolved_config("benchmarks/partner_if/config.yaml")


def settings(config, name):
    return OmegaConf.to_container(get_first_server_config_dict(config, name), resolve=True)


def test_resolved_benchmark_routes_recorded_context_to_separate_resources(benchmark_config):
    ordinary = settings(benchmark_config, "partner_if")
    assert ordinary["entrypoint"] == "app.py"
    assert "replay" not in ordinary
    assert not any(key.startswith(("ordering_", "replay_")) for key in ordinary)
    ordinary_config = PartnerIFResourcesServerConfig.model_validate(ordinary | {"name": "partner_if"})
    assert ordinary_config.accept_verification_input is False

    routes = settings(benchmark_config, "partner_if_dispatcher")["task_routes"]
    for task_id in ("pif_001", "pif_028"):
        environment = settings(benchmark_config, routes[task_id]["name"])
        assert environment["resources_server"]["name"] == "partner_if"
        text_agent = settings(benchmark_config, environment["agent_server"]["name"])
        assert text_agent["resources_server"]["name"] == "partner_if"

    replay_environment = settings(benchmark_config, routes["pif_029"]["name"])
    replay_agent_name = replay_environment["agent_server"]["name"]
    assert set(benchmark_config[replay_agent_name]["responses_api_agents"]) == {"gymnasium_agent"}
    replay_agent = settings(benchmark_config, replay_agent_name)
    replay_name = replay_agent["resources_server"]["name"]
    assert replay_name == "partner_if_recorded_context"
    replay = settings(benchmark_config, replay_name)
    assert replay["entrypoint"] == "recorded_context_app.py"
    replay_config = RecordedContextResourcesServerConfig.model_validate(replay | {"name": replay_name})
    assert replay_config.max_parent_turns == 6 and replay_config.replay.max_policy_calls == 8
    assert replay_config.REVERIFY_MODE == ReverifyMode.UNSUPPORTED
    assert RecordedContextResourcesServer.checkpoint_mode == "restart_only"


@pytest.mark.asyncio
async def test_ordinary_text_can_be_reverified_without_replay_receipts(benchmark_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ordinary = PartnerIFResourcesServerConfig.model_validate(
        settings(benchmark_config, "partner_if") | {"name": "partner_if"}
    )
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
        global_config_dict=benchmark_config,
    )
    resource = PartnerIFResourcesServer(config=ordinary, server_client=client)
    assert resource.config.accept_verification_input is False
    assert resource.checkpoint_mode_in_effect() == "stateless"
    # This is Gym's ordinary recovery of stateless grading, not recorded-context simulation.
    assert resource.checkpoint_verify == "replay"
    assert await resource.get_reverify_mode() == ReverifyMode.STATELESS
    assert not any(route.path.startswith("/replay/") for route in resource.setup_webserver().routes)

    contract = {
        "task_id": "ordinary-isolation",
        "origin": "synthetic",
        "source": "authored isolation regression",
        "rubric_version": "isolation-v1",
        "audit_signoff": True,
        "data_kind": "synthetic_smoke",
        "rubrics": [
            {
                "id": "contains-verified",
                "constraint": "The answer contains VERIFIED.",
                "verifier": {"id": "text_contains_v1", "params": {"source": "answer", "literal": "VERIFIED"}},
            }
        ],
    }
    body = VerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Say VERIFIED."}]},
            "verifier_metadata": {"partner_if": contract},
            "response": response("VERIFIED").model_dump(),
        }
    )
    original = body.model_dump(mode="json")
    with patch.object(
        ServerClient, "post", new_callable=AsyncMock, side_effect=AssertionError("No model calls")
    ) as post:
        first = await resource.verify(body)
        repeated = await resource.verify(body)
        missing_word = body.model_copy(update={"response": response("A different answer.")}, deep=True)
        failed = await resource.verify(missing_word)
        recovered = await resource.verify(body)
    assert [result.reward for result in (first, repeated, failed, recovered)] == [1, 1, 0, 1]
    assert all(
        result.evaluation_completed and not result.mask_sample for result in (first, repeated, failed, recovered)
    )
    assert first.if_result == repeated.if_result == recovered.if_result
    assert body.model_dump(mode="json") == original
    post.assert_not_called()
    state = [*vars(resource).values(), *(resource.__pydantic_private__ or {}).values()]
    assert not any(type(value).__name__ == "ReplayReceiptStore" for value in state)
    assert list(tmp_path.rglob("*")) == []
