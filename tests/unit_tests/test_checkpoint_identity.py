# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientConnectorError, ClientOSError, ServerDisconnectedError
from omegaconf import DictConfig
from pydantic import ValidationError

import nemo_gym.server_utils
from nemo_gym._checkpoint.settings import checkpoint_settings
from nemo_gym.episode_types import EpisodeId
from nemo_gym.rollout_correlation import current_episode_id, rollout_context
from nemo_gym.server_utils import BaseServerConfig, ServerClient


@pytest.mark.parametrize(
    ("rollout_id", "attempt"),
    [("r", 0), ("r", 1), ("r", 12), ("task-3-7", 2), ("x.y_z", 0), ("ends-a0", 0), ("ends-a0", 4)],
)
def test_capture_key_round_trips(rollout_id: str, attempt: int) -> None:
    episode_id = EpisodeId(rollout_id=rollout_id, attempt=attempt)

    assert EpisodeId.from_capture_key(episode_id.capture_key) == episode_id


def test_ambiguous_rollout_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="reserved attempt suffix"):
        EpisodeId(rollout_id="r-a1")


def test_current_episode_id_follows_rollout_context() -> None:
    assert current_episode_id() is None
    with rollout_context("r-a3"):
        assert current_episode_id() == EpisodeId(rollout_id="r", attempt=3)
    assert current_episode_id() is None


def test_checkpoint_settings_are_opt_in() -> None:
    assert checkpoint_settings(DictConfig({})) is None
    assert checkpoint_settings(DictConfig({"checkpoint": {"enabled": False}})) is None
    settings = checkpoint_settings(DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}}))
    assert settings is not None and settings.control_auth_token == "t"
    with pytest.raises(ValidationError, match="control_auth_token is required"):
        checkpoint_settings(DictConfig({"checkpoint": {"enabled": True}}))


def _client(monkeypatch: pytest.MonkeyPatch, extra: dict) -> tuple[ServerClient, AsyncMock]:
    config = {
        "policy": {"responses_api_models": {"vllm_model": {"host": "model", "port": 1}}},
        "env": {"resources_servers": {"example": {"host": "resources", "port": 2}}},
    } | extra
    client = ServerClient(
        head_server_config=BaseServerConfig(host="head", port=0),
        global_config_dict=DictConfig(config),
    )
    request_mock = AsyncMock(return_value="response")
    client_mock = MagicMock()
    client_mock.return_value.request = request_mock
    monkeypatch.setattr(nemo_gym.server_utils, "get_global_aiohttp_client", client_mock)
    return client, request_mock


async def _called_urls(client: ServerClient, request_mock: AsyncMock) -> list[str]:
    with rollout_context("r-a2"):
        await client.post(server_name="policy", url_path="/v1/chat/completions", json={})
        await client.post(server_name="env", url_path="/step", json={})
    return [call.kwargs["url"] for call in request_mock.await_args_list]


async def test_checkpointing_attributes_calls_without_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    client, request_mock = _client(monkeypatch, {"checkpoint": {"enabled": True, "control_auth_token": "t"}})

    assert await _called_urls(client, request_mock) == [
        "http://model:1/ng-rollout/r-a2/v1/chat/completions",
        "http://resources:2/ng-rollout/r-a2/step",
    ]


async def test_calls_stay_unprefixed_when_checkpointing_and_observability_are_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, request_mock = _client(monkeypatch, {})

    assert await _called_urls(client, request_mock) == [
        "http://model:1/v1/chat/completions",
        "http://resources:2/step",
    ]


def test_a_bad_checkpoint_block_fails_when_the_client_is_built_not_on_every_request() -> None:
    from omegaconf import OmegaConf

    from nemo_gym.server_utils import BaseServerConfig, ServerClient

    with pytest.raises(ValueError, match="control_auth_token is required"):
        ServerClient(
            head_server_config=BaseServerConfig(host="head", port=1),
            global_config_dict=OmegaConf.create({"checkpoint": {"enabled": True}}),
        )


def test_the_checkpoint_block_is_a_reserved_top_level_key() -> None:
    from nemo_gym.global_config import NEMO_GYM_RESERVED_TOP_LEVEL_KEYS

    assert "checkpoint" in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS


_CHECKPOINTING = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}


@pytest.mark.parametrize(
    "dropped",
    [
        ServerDisconnectedError(),
        ClientOSError(104, "Connection reset by peer"),
        ConnectionResetError("reset"),
    ],
)
async def test_a_run_whose_connection_dropped_is_not_sent_again_with_checkpointing(
    monkeypatch: pytest.MonkeyPatch, dropped: Exception
) -> None:
    # The server may have started the attempt, so a second send would run it twice and strand the first run's sessions.
    client, request_mock = _client(monkeypatch, _CHECKPOINTING)
    request_mock.side_effect = [dropped, "response"]
    monkeypatch.setattr(nemo_gym.server_utils.asyncio, "sleep", AsyncMock())

    with pytest.raises(type(dropped)):
        await client.post(server_name="env", url_path="/run", json={})

    assert request_mock.await_count == 1


async def test_a_run_whose_connection_was_never_made_is_retried_with_checkpointing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, request_mock = _client(monkeypatch, _CHECKPOINTING)
    refused = ClientConnectorError(MagicMock(), OSError(111, "Connection refused"))
    request_mock.side_effect = [refused, "response"]
    monkeypatch.setattr(nemo_gym.server_utils.asyncio, "sleep", AsyncMock())

    assert await client.post(server_name="env", url_path="/run", json={}) == "response"
    assert request_mock.await_count == 2


@pytest.mark.parametrize(("extra", "url_path"), [({}, "/run"), (_CHECKPOINTING, "/step")])
async def test_other_calls_are_still_sent_again_after_a_dropped_connection(
    monkeypatch: pytest.MonkeyPatch, extra: dict, url_path: str
) -> None:
    client, request_mock = _client(monkeypatch, extra)
    request_mock.side_effect = [ServerDisconnectedError(), "response"]
    monkeypatch.setattr(nemo_gym.server_utils.asyncio, "sleep", AsyncMock())

    assert await client.post(server_name="env", url_path=url_path, json={}) == "response"
    assert request_mock.await_count == 2
