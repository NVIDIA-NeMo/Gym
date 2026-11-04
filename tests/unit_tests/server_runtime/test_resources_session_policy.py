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

from typing import Annotated
from uuid import UUID

import pytest
from pydantic import BeforeValidator, TypeAdapter, ValidationError

from nemo_gym.core.episode import EpisodeId, TaskId
from nemo_gym.server_runtime.base_resources_server import (
    BaseSeedSessionRequest,
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSessionConflictError,
    ResourcesSessionFence,
    reject_typed_resources_seed,
)


EPISODE = EpisodeId(rollout_id="rollout", attempt=0)
TASK = TaskId(taskset="tasks", task_id="task")
IDENTITY = (EPISODE, TASK)
SEED = ResourcesSeedSessionRequest(resources_session_id="session", episode_id=EPISODE, task_id=TASK, task_data={})
SEED_BODY = TypeAdapter(
    ResourcesSeedSessionRequest | Annotated[BaseSeedSessionRequest, BeforeValidator(reject_typed_resources_seed)]
)


@pytest.mark.parametrize("field", ["resources_session_id", "episode_id", "task_id", "task_data"])
def test_partial_typed_seed_cannot_fall_back_to_legacy(field: str) -> None:
    body = SEED.model_dump(mode="json")
    with pytest.raises(ValidationError):
        SEED_BODY.validate_python({field: body[field]})


def test_seed_union_accepts_both_complete_wire_formats() -> None:
    assert SEED_BODY.validate_python(SEED.model_dump(mode="json")) == SEED
    assert isinstance(SEED_BODY.validate_python({"verifier_metadata": {"city": "Paris"}}), BaseSeedSessionRequest)


def test_typed_seed_initializes_once_and_keeps_the_existing_task_on_retry() -> None:
    fence = ResourcesSessionFence(window_seconds=10)
    assert fence.check_typed_seed(SEED, identity=None, has_state=False, verified=False)
    # The same episode/task must reuse its state even if a retry supplies different task data.
    retry = SEED.model_copy(update={"task_data": {"city": "London"}})
    assert not fence.check_typed_seed(retry, identity=IDENTITY, has_state=True, verified=False)


@pytest.mark.parametrize(
    ("identity", "has_state", "verified"),
    [
        (None, True, False),  # A typed seed cannot claim a live legacy session.
        ((EpisodeId(rollout_id="other"), TASK), True, False),
        ((EPISODE, TaskId(taskset="tasks", task_id="other")), True, False),
        (None, False, True),  # Neither legacy nor typed completed sessions can reopen.
        (IDENTITY, False, True),
    ],
)
def test_typed_seed_refuses_conflicting_or_verified_sessions(
    identity: tuple[EpisodeId, TaskId] | None, has_state: bool, verified: bool
) -> None:
    fence = ResourcesSessionFence(window_seconds=10)
    with pytest.raises(ResourcesSessionConflictError) as error:
        fence.check_typed_seed(SEED, identity=identity, has_state=has_state, verified=verified)
    assert error.value.status_code == 409


def test_legacy_seed_rotates_only_after_verification() -> None:
    fence = ResourcesSessionFence(window_seconds=10)
    assert fence.legacy_seed_session_id("session", identity=None, verified=False) == "session"
    new_id = fence.legacy_seed_session_id("session", identity=None, verified=True)
    assert new_id != "session"
    assert str(UUID(new_id)) == new_id
    # Rotation leaves the previous cookie available for the environment's verify-retry evidence.
    assert fence.closed_episode("session") is None


@pytest.mark.parametrize("verified", [False, True])
def test_legacy_seed_cannot_replace_a_typed_session(verified: bool) -> None:
    fence = ResourcesSessionFence(window_seconds=10)
    with pytest.raises(ResourcesSessionConflictError):
        fence.legacy_seed_session_id("session", identity=IDENTITY, verified=verified)


def test_close_checks_ownership_without_marking_a_failed_cleanup_closed() -> None:
    fence = ResourcesSessionFence(window_seconds=10)
    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=EPISODE)
    fence.check_close(close, identity=IDENTITY)
    # Resource cleanup happens after this check. If it raises, a seed retry still finds the open session.
    assert fence.closed_episode("session") is None
    assert not fence.check_typed_seed(SEED, identity=IDENTITY, has_state=True, verified=False)
    fence.mark_closed("session", EPISODE)
    fence.check_close(close, identity=None)


@pytest.mark.parametrize("closed", [False, True])
def test_close_refuses_another_episode_before_and_after_release(closed: bool) -> None:
    fence = ResourcesSessionFence(window_seconds=10)
    if closed:
        fence.mark_closed("session", EPISODE)
    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=EpisodeId(rollout_id="other"))
    with pytest.raises(ResourcesSessionConflictError):
        fence.check_close(close, identity=None if closed else IDENTITY)


def test_close_of_unknown_session_fences_both_seed_protocols_until_expiry(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("nemo_gym.server_runtime.base_resources_server.monotonic", lambda: clock[0])
    fence = ResourcesSessionFence(window_seconds=10)
    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=EPISODE)
    fence.check_close(close, identity=None)
    fence.mark_closed("session", EPISODE)
    with pytest.raises(ResourcesSessionConflictError):
        fence.check_typed_seed(SEED, identity=None, has_state=False, verified=False)
    with pytest.raises(ResourcesSessionConflictError):
        fence.legacy_seed_session_id("session", identity=None, verified=False)
    clock[0] += 10
    assert fence.check_typed_seed(SEED, identity=None, has_state=False, verified=False)
    assert fence.legacy_seed_session_id("session", identity=None, verified=False) == "session"
