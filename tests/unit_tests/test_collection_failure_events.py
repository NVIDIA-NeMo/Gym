# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collection's own failed `/run` dispatch is a CollectionFailure, whose explicit cause is the dispatch error."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiohttp import ClientResponseError
from multidict import CIMultiDict

from nemo_gym.global_config import AGENT_REF_KEY_NAME
from nemo_gym.rollout_collection import (
    AGENT_REQUEST_FAILED_FAILURE_CLASS,
    AGENT_RUN_ERROR_FAILURE_CLASS,
    NG_FAILURE_CLASS_KEY,
    RolloutCollectionHelper,
)
from nemo_gym.typed_exceptions import EVENTS_ROW_KEY, EnvServerFailure, remote_failure, safe_failure_header
from tests.unit_tests.test_rollout_collection import install_fake_server_client


ROW = {"responses_create_params": {"input": []}, AGENT_REF_KEY_NAME: {"name": "my_agent"}}


async def _dispatch(monkeypatch, error: BaseException):
    install_fake_server_client(monkeypatch, AsyncMock(side_effect=error))
    (future,) = RolloutCollectionHelper()._run_examples_with_metadata([dict(ROW)], route_failures_to_sidecar=True)
    return await future


async def test_a_dispatch_that_got_no_reply_is_a_collection_failure_caused_by_the_transport_error(monkeypatch) -> None:
    completed = await _dispatch(monkeypatch, TimeoutError("read timed out"))
    events, own = completed.failure_observations[EVENTS_ROW_KEY], completed.failure_observations["_ng_failure_id"]
    parent = events[own]
    assert (parent["type"], parent["failure_kind"]) == ("CollectionFailure", AGENT_REQUEST_FAILED_FAILURE_CLASS)
    cause = events[parent["cause_id"]]
    assert (cause["type"], cause["exception_type"]) == ("untyped", "TimeoutError")
    # The routing class, retry accounting and the row itself are unchanged.
    assert completed.result[NG_FAILURE_CLASS_KEY] == AGENT_REQUEST_FAILED_FAILURE_CLASS
    assert "reward" not in completed.result


async def test_a_gym_server_failure_stays_the_cause_with_its_own_kind_and_id_once(monkeypatch) -> None:
    try:
        raise EnvServerFailure("my_env:episode_failed", message="episode failed", operation="env.run")
    except EnvServerFailure as env_failure:
        headers = CIMultiDict(safe_failure_header(env_failure, server="my_environment_server"))
        env_id = env_failure.event_id
    info = SimpleNamespace(method="POST", url="http://env/run", real_url="http://env/run", headers=CIMultiDict())
    error = ClientResponseError(info, (), status=500, message="Internal Server Error", headers=headers)
    remote = remote_failure(headers)
    assert remote is not None
    error.__cause__ = remote  # as `raise_for_status` raises it: `raise e from remote_failure(e.headers)`

    completed = await _dispatch(monkeypatch, error)

    events, own = completed.failure_observations[EVENTS_ROW_KEY], completed.failure_observations["_ng_failure_id"]
    parent = events[own]
    assert (parent["type"], parent["failure_kind"]) == ("CollectionFailure", AGENT_RUN_ERROR_FAILURE_CLASS)
    transport = events[parent["cause_id"]]
    assert (transport["exception_type"], transport["http_status"]) == (
        "aiohttp.client_exceptions.ClientResponseError",
        500,
    )
    env = events[transport["cause_id"]]
    # The environment's own failure keeps its id and kind: the collector's kind does not replace it.
    assert env["event_id"] == env_id
    assert (env["type"], env["failure_kind"]) == ("EnvServerFailure", "my_env:episode_failed")
    assert sum(1 for e in events.values() if e["type"] == "EnvServerFailure") == 1
    assert completed.result[NG_FAILURE_CLASS_KEY] == AGENT_RUN_ERROR_FAILURE_CLASS
