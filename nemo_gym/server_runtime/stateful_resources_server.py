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

from abc import abstractmethod
from typing import Annotated, Any, ClassVar, Generic, TypeVar

from fastapi import Request
from pydantic import BeforeValidator, PrivateAttr, ValidationError

from nemo_gym.client.server_client import SESSION_ID_KEY
from nemo_gym.core.episode import EpisodeId, TaskId
from nemo_gym.server_runtime.base_resources_server import (
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    ClosedResourcesSessionRecords,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionFence,
    SimpleResourcesServer,
    reject_typed_resources_seed,
)


VerificationRecordT = TypeVar("VerificationRecordT")


class StatefulResourcesServer(SimpleResourcesServer, Generic[VerificationRecordT]):
    """Share seed, close, and checkpoint retirement for servers that keep per-session state.

    Implement ``has_session_state``, ``initialize_session_state``, and ``release_session_state`` for the
    environment's resources. This base owns episode/task binding, seed retries, cookie rotation, and close
    fencing. Verification remains environment-specific; ``_verified_sessions`` retains its minimal evidence
    alongside the released session's identity for the configured retry window.

    Set ``seed_request_model`` to validate a typed seed's ``task_data``. Override ``seed_session`` with a
    thin call to ``super()`` when the legacy HTTP request needs a more specific schema.
    """

    seed_request_model: ClassVar[type[BaseSeedSessionRequest]] = BaseSeedSessionRequest
    _session_episodes: dict[str, tuple[EpisodeId, TaskId]] = PrivateAttr(default_factory=dict)
    _verified_sessions: ClosedResourcesSessionRecords[tuple[tuple[EpisodeId, TaskId] | None, VerificationRecordT]] = (
        PrivateAttr()
    )
    _fence: ResourcesSessionFence = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._verified_sessions = ClosedResourcesSessionRecords(self.config.session_close_retry_window_seconds)
        self._fence = ResourcesSessionFence(self.config.session_close_retry_window_seconds)

    @abstractmethod
    def has_session_state(self, session_id: str) -> bool:
        """Whether the environment holds live state for this session."""

    @abstractmethod
    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        """Create state from validated task data, replacing a previous legacy run if needed.

        A typed retry with an existing binding never calls this hook. Publish live state only after construction
        succeeds. Keep partial resources separately, reachable by ``release_session_state``, if cleanup fails.
        """

    @abstractmethod
    async def release_session_state(self, session_id: str) -> None:
        """Release live and partially created resources; succeed if none remain.

        Raise on cleanup failure and retain what a later call needs to retry. This base removes the active
        identity and records a close only after the hook succeeds.
        """

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest
        | Annotated[BaseSeedSessionRequest, BeforeValidator(reject_typed_resources_seed)],
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        """Initialize or reuse a typed session, or start a deprecated cookie-based legacy run."""
        if not isinstance(body, ResourcesSeedSessionRequest):
            session_id = request.session[SESSION_ID_KEY]
            async with self._fence.locked(session_id):
                verified = self._verified_sessions.get(session_id)
                session_id = self._fence.legacy_seed_session_id(
                    session_id,
                    identity=self._session_episodes.get(session_id) or (verified[0] if verified is not None else None),
                    verified=verified is not None,
                )
                await self.initialize_session_state(session_id, body)
                request.session[SESSION_ID_KEY] = session_id
            return BaseSeedSessionResponse()

        session_id = body.resources_session_id
        async with self._fence.locked(session_id):
            if self._fence.check_typed_seed(
                body,
                identity=self._session_episodes.get(session_id),
                has_state=self.has_session_state(session_id),
                verified=session_id in self._verified_sessions,
            ):
                try:
                    task = self.seed_request_model.model_validate(body.task_data)
                except ValidationError as error:
                    raise ResourcesSeedRejectedError(
                        f"task_data is not a valid {self.seed_request_model.__name__}: {error}"
                    ) from error
                await self.initialize_session_state(session_id, task)
                self._session_episodes[session_id] = (body.episode_id, body.task_id)
            # Bind only after initialization succeeds. MCP auto-exposure signs this same session ID.
            request.session[SESSION_ID_KEY] = session_id
        return ResourcesSeedSessionResponse(resources_session_id=session_id)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Release by ID, including a close without a cookie, then fence late requests."""
        session_id = body.resources_session_id
        async with self._fence.locked(session_id):
            verified = self._verified_sessions.get(session_id)
            identity = self._session_episodes.get(session_id) or (verified[0] if verified is not None else None)
            self._fence.check_close(body, identity=identity)
            await self.release_session_state(session_id)
            self._session_episodes.pop(session_id, None)
            self._fence.mark_closed(session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def retire_session_state(self, session_id: str) -> None:
        """Release a retired attempt, retaining its identity if cleanup fails."""
        async with self._fence.locked(session_id):
            identity = self._session_episodes.get(session_id)
            await self.release_session_state(session_id)
            self._session_episodes.pop(session_id, None)
            if identity is not None:
                self._fence.mark_closed(session_id, identity[0])
