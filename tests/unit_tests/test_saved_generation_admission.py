# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest

from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.rollout_recovery import IncompleteEvaluationError
from nemo_gym.rollout_store import RolloutStore
from tests.unit_tests.test_rollout_collection import FakeResponse, install_fake_server_client


def response(output=None):
    return NeMoGymResponse(
        id="resp_saved",
        object="response",
        created_at=0,
        model="controlled",
        output=[] if output is None else output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    ).model_dump(mode="json")


@pytest.mark.parametrize("slot", ["response", "_ng_failure_response", "_ng_failure_partial_response"])
def test_retained_generation_is_structural_not_placeholder_presence(slot):
    from nemo_gym.rollout_recovery import has_saved_generation

    assert has_saved_generation({slot: response()})
    for placeholder in (
        {},
        {"error": "failed"},
        {"output": []},
        {"status": "failed", "output": []},
        response() | {"status": "failed", "error": {"code": "server_error", "message": "diagnostic only"}},
    ):
        assert not has_saved_generation({slot: placeholder})


@pytest.mark.parametrize("empty_output", [False, True])
async def test_native_cleanup_partial_response_survives_collection(tmp_path, monkeypatch, empty_output):
    import httpx
    from omegaconf import OmegaConf

    from environment_servers.single_agent_turn.tests.test_app import _environment_server, _request, _Response

    environment, dependencies = _environment_server()
    if empty_output:
        dependencies.responses[2] = _Response(response())
    dependencies.responses = dependencies.responses[:3] + [
        TimeoutError("controlled lost agent cleanup reply"),
        _Response({"resources_session_id": "resource"}),
    ]
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    source = tmp_path / "source.jsonl"
    source.write_text(_request().task.model_dump_json() + "\n")
    output = artifacts / "rollouts.jsonl"
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(output),
        environment_server_routes={"test:source": "environment"},
        route_failures_to_sidecar=True,
        require_complete=False,
        disable_aggregation=True,
        disable_health_check=True,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=environment.setup_webserver()), base_url="http://environment"
    ) as http:

        async def post(**kwargs):
            result = await http.post("/run", json=kwargs["json"])
            assert result.status_code == 200
            return FakeResponse(result.status_code, result.json())

        client = install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
        client.global_config_dict = OmegaConf.create(
            {"environment": {"environment_servers": {"single_agent_turn": {}}}}
        )
        monkeypatch.setattr("nemo_gym.rollout_collection.get_global_config_dict", lambda: client.global_config_dict)
        with pytest.raises(IncompleteEvaluationError):
            await RolloutCollectionHelper().run_from_config(config)
        store = RolloutStore.read(output)
        assert store.pending() == []
        failure = store.failures()[0]
        assert "_ng_failure_partial_response" in failure
        assert has_actual_generation(failure)
        try:
            await RolloutCollectionHelper().run_from_config(config.model_copy(update={"resume_from_cache": True}))
        except IncompleteEvaluationError:
            pass
        assert client.post.await_count == 1
        assert len([call for call in dependencies.calls if call[1].endswith("/v1/responses")]) == 1


def has_actual_generation(row):
    from nemo_gym.rollout_recovery import has_saved_generation

    return has_saved_generation(row)
