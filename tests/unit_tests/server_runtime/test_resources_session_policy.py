# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Annotated
from unittest.mock import MagicMock

import pytest
from pydantic import BeforeValidator, TypeAdapter, ValidationError

from nemo_gym.client.server_client import ServerClient
from nemo_gym.core.episode_types import EpisodeId, TaskId
from nemo_gym.server_runtime.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSessionConflictError,
    SimpleResourcesServer,
    reject_partial_typed_seed,
)


class PolicyServer(SimpleResourcesServer):
    def has_session_state(self, session_id: str) -> bool:
        return False

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        pass

    async def release_session_state(self, session_id: str) -> None:
        pass

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        raise NotImplementedError


def _server() -> PolicyServer:
    return PolicyServer(
        config=BaseResourcesServerConfig(
            host="", port=0, entrypoint="", name="policy", session_close_retry_window_seconds=10
        ),
        server_client=MagicMock(spec=ServerClient),
    )


EPISODE = EpisodeId(rollout_id="rollout", attempt=0)
TASK = TaskId(taskset="tasks", task_id="task")
IDENTITY = (EPISODE, TASK)
SEED = ResourcesSeedSessionRequest(resources_session_id="session", episode_id=EPISODE, task_id=TASK, task_data={})
SEED_BODY = TypeAdapter(
    ResourcesSeedSessionRequest | Annotated[BaseSeedSessionRequest, BeforeValidator(reject_partial_typed_seed)]
)


@pytest.mark.parametrize("missing", ["resources_session_id", "episode_id", "task_id", "task_data"])
def test_partial_episode_seed_cannot_fall_back_to_agent_run_seed(missing: str) -> None:
    body = SEED.model_dump(mode="json")
    del body[missing]
    with pytest.raises(ValidationError):
        SEED_BODY.validate_python(body)


@pytest.mark.parametrize("field", ["episode_id", "task_id", "task_data"])
def test_agent_run_seed_keeps_dataset_fields_that_share_a_episode_seed_name(field: str) -> None:
    # A /run seed posts the whole dataset row, whose own columns may be named like the episode seed's fields.
    body = {"verifier_metadata": {"city": "Paris"}, field: "row-1"}
    assert isinstance(SEED_BODY.validate_python(body), BaseSeedSessionRequest)


def test_seed_union_accepts_both_complete_wire_formats() -> None:
    assert SEED_BODY.validate_python(SEED.model_dump(mode="json")) == SEED
    assert isinstance(SEED_BODY.validate_python({"verifier_metadata": {"city": "Paris"}}), BaseSeedSessionRequest)


def test_episode_seed_initializes_once_and_keeps_the_existing_task_on_retry() -> None:
    server = _server()
    assert server._check_episode_seed(SEED, identity=None, has_state=False, verified=False)
    # The same episode/task must reuse its state even if a retry supplies different task data.
    retry = SEED.model_copy(update={"task_data": {"city": "London"}})
    assert not server._check_episode_seed(retry, identity=IDENTITY, has_state=True, verified=False)


@pytest.mark.parametrize(
    ("identity", "has_state", "verified"),
    [
        (None, True, False),  # An episode seed cannot claim a live agent-run session.
        ((EpisodeId(rollout_id="other"), TASK), True, False),
        ((EPISODE, TaskId(taskset="tasks", task_id="other")), True, False),
        (None, False, True),  # Neither agent-run nor episode completed sessions can reopen.
        (IDENTITY, False, True),
    ],
)
def test_episode_seed_refuses_conflicting_or_verified_sessions(
    identity: tuple[EpisodeId, TaskId] | None, has_state: bool, verified: bool
) -> None:
    server = _server()
    with pytest.raises(ResourcesSessionConflictError) as error:
        server._check_episode_seed(SEED, identity=identity, has_state=has_state, verified=verified)
    assert error.value.status_code == 409


def test_agent_run_seed_deprecation_is_logged_once_and_names_the_migration(caplog) -> None:
    server = _server()
    with caplog.at_level(logging.WARNING):
        server._check_episode_seed(SEED, identity=None, has_state=False, verified=False)
        assert not caplog.records
        server._check_agent_run_seed("first", identity=None)
        server._check_agent_run_seed("second", identity=None)
    assert len(caplog.records) == 1
    assert "deprecated" in caplog.records[0].message
    assert "ResourcesSeedSessionRequest" in caplog.records[0].message


def test_agent_run_seed_cannot_replace_a_episode_session() -> None:
    server = _server()
    with pytest.raises(ResourcesSessionConflictError):
        server._check_agent_run_seed("session", identity=IDENTITY)


def test_close_checks_ownership_without_marking_a_failed_cleanup_closed() -> None:
    server = _server()
    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=EPISODE)
    server._check_session_close(close, identity=IDENTITY)
    # Resource cleanup happens after this check. If it raises, a seed retry still finds the open session.
    assert server._resources_session_fences.get("session") is None
    assert not server._check_episode_seed(SEED, identity=IDENTITY, has_state=True, verified=False)
    server._resources_session_fences.fence("session", EPISODE)
    server._check_session_close(close, identity=None)


@pytest.mark.parametrize("closed", [False, True])
def test_close_refuses_another_episode_before_and_after_release(closed: bool) -> None:
    server = _server()
    if closed:
        server._resources_session_fences.fence("session", EPISODE)
    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=EpisodeId(rollout_id="other"))
    with pytest.raises(ResourcesSessionConflictError):
        server._check_session_close(close, identity=None if closed else IDENTITY)


def test_close_of_unknown_session_fences_both_seed_protocols_until_expiry(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.server_runtime.base_resources_server.monotonic", lambda: clock[0])
    server = _server()
    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=EPISODE)
    server._check_session_close(close, identity=None)
    server._resources_session_fences.fence("session", EPISODE)
    with pytest.raises(ResourcesSessionConflictError):
        server._check_episode_seed(SEED, identity=None, has_state=False, verified=False)
    with pytest.raises(ResourcesSessionConflictError):
        server._check_agent_run_seed("session", identity=None)
    clock[0] += 10
    assert server._check_episode_seed(SEED, identity=None, has_state=False, verified=False)
    server._check_agent_run_seed("session", identity=None)
