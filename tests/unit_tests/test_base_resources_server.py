# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import time
from typing import ClassVar, Optional
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request, Response

from nemo_gym.base_resources_server import (
    RESERVED_MCP_TOOL_NAMES,
    BaseCloseSessionRequest,
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyResponse,
    ReverifyMode,
    SessionCloseReason,
    SessionCloseStatus,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient


def _resources_server() -> SimpleResourcesServer:
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")

    class TestSimpleResourcesServer(SimpleResourcesServer):
        async def verify(self, body):
            pass

    return TestSimpleResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestBaseVerifyResponse:
    def test_failure_reason_defaults_none_and_round_trips(self) -> None:
        response = BaseVerifyResponse(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
            response=NeMoGymResponse.model_construct(id="resp-1", output=[]),
            reward=0.0,
        )
        assert response.failure_reason is None
        assert response.model_dump()["failure_reason"] is None

        rescued = response.model_copy(update={"failure_reason": "judge response unparseable after 3 attempts"})
        assert rescued.model_dump()["failure_reason"] == "judge response unparseable after 3 attempts"
        assert rescued.reward == 0.0


class TestBaseMultiRewardVerifyResponse:
    def test_reward_components_round_trip(self) -> None:
        response = BaseMultiRewardVerifyResponse(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hi"),
            response=NeMoGymResponse.model_construct(id="resp-1", output=[]),
            reward=2.0,
            reward_components={"correctness": 1.0, "format": 1.0},
        )
        dumped = response.model_dump()
        assert dumped["reward_components"] == {"correctness": 1.0, "format": 1.0}
        assert dumped["reward"] == 2.0


class TestBaseResourcesServer:
    def test_sanity(self) -> None:
        _resources_server().setup_webserver()

    def test_reverify_mode(self) -> None:
        assert asyncio.run(_resources_server().get_reverify_mode()) == ReverifyMode.UNKNOWN


class _RecordingServer(SimpleResourcesServer):
    """A server that records what the base actually asked it to do.

    Pydantic models reject stray attributes, so the logs live on the class and each
    instance clears them.
    """

    seeded: ClassVar[list] = []
    released: ClassVar[list] = []
    release_error: ClassVar[Optional[Exception]] = None
    release_delay_s: ClassVar[float] = 0.0

    async def verify(self, body):  # pragma: no cover - never called in these tests
        pass

    async def seed_session(self, body):
        type(self).seeded.append(body)
        return BaseSeedSessionResponse()

    async def release_session(self, session_id, reason):
        type(self).released.append((session_id, reason))
        if type(self).release_delay_s:
            await asyncio.sleep(type(self).release_delay_s)
        if type(self).release_error is not None:
            raise type(self).release_error


def _server(**config) -> _RecordingServer:
    _RecordingServer.seeded = []
    _RecordingServer.released = []
    _RecordingServer.release_error = None
    _RecordingServer.release_delay_s = 0.0
    return _RecordingServer(
        config=BaseResourcesServerConfig(host="", port=0, entrypoint="", name="", **config),
        server_client=MagicMock(spec=ServerClient),
    )


def _request(session_id: Optional[str] = "cookie-session") -> Request:
    session = {SESSION_ID_KEY: session_id} if session_id else {}
    return Request(scope={"type": "http", "session": session})


class TestSeedDeduplication:
    """The transport may repeat a seed it is not sure was delivered."""

    def test_a_repeat_of_the_same_seed_does_not_allocate_twice(self) -> None:
        server = _server()
        body = BaseSeedSessionRequest(_ng_session_id="s1")

        first = asyncio.run(server._seed_session_endpoint(_request(), body))
        second = asyncio.run(server._seed_session_endpoint(_request(), BaseSeedSessionRequest(_ng_session_id="s1")))

        assert len(server.seeded) == 1, "the second transmission must replay, not allocate"
        assert second is first

    def test_the_same_id_with_a_different_payload_is_a_conflict(self) -> None:
        server = _server()
        asyncio.run(server._seed_session_endpoint(_request(), BaseSeedSessionRequest(_ng_session_id="s1")))

        class _Different(BaseSeedSessionRequest):
            task: str = "other"

        with pytest.raises(HTTPException) as caught:
            asyncio.run(server._seed_session_endpoint(_request(), _Different(_ng_session_id="s1")))

        assert caught.value.status_code == 409

    def test_a_seed_arriving_after_close_cannot_resurrect_the_session(self) -> None:
        """A delayed retransmission must not recreate what was already torn down."""
        server = _server()
        request = _request("s1")
        asyncio.run(server._seed_session_endpoint(request, BaseSeedSessionRequest(_ng_session_id="s1")))
        asyncio.run(server._close_session_endpoint(request, BaseCloseSessionRequest(_ng_session_id="s1"), Response()))

        with pytest.raises(HTTPException) as caught:
            asyncio.run(server._seed_session_endpoint(request, BaseSeedSessionRequest(_ng_session_id="s1")))

        assert caught.value.status_code == 409
        assert len(server.seeded) == 1


class TestCloseContract:
    def _seed(self, server, session_id="s1", token=None):
        body = BaseSeedSessionRequest(_ng_session_id=session_id, _ng_session_close_token=token)
        asyncio.run(server._seed_session_endpoint(_request(session_id), body))

    def test_releasing_a_live_session_reports_closed(self) -> None:
        server = _server()
        self._seed(server)
        response = Response()

        result = asyncio.run(
            server._close_session_endpoint(_request("s1"), BaseCloseSessionRequest(_ng_session_id="s1"), response)
        )

        assert (result.status, result.released) == (SessionCloseStatus.CLOSED, True)
        assert server.released == [("s1", SessionCloseReason.COMPLETED)]

    def test_an_unknown_id_and_an_already_closed_one_are_indistinguishable(self) -> None:
        """Otherwise the endpoint tells a caller whether another rollout's session exists."""
        server = _server()
        self._seed(server)
        asyncio.run(
            server._close_session_endpoint(_request("s1"), BaseCloseSessionRequest(_ng_session_id="s1"), Response())
        )

        closed_again = asyncio.run(
            server._close_session_endpoint(_request("s1"), BaseCloseSessionRequest(_ng_session_id="s1"), Response())
        )
        never_existed = asyncio.run(
            server._close_session_endpoint(_request(None), BaseCloseSessionRequest(_ng_session_id="nope"), Response())
        )

        assert closed_again.model_dump() == never_existed.model_dump()
        assert closed_again.status is SessionCloseStatus.ALREADY_CLOSED

    def test_a_cookie_and_id_that_disagree_release_neither(self) -> None:
        server = _server()
        self._seed(server)

        with pytest.raises(HTTPException) as caught:
            asyncio.run(
                server._close_session_endpoint(
                    _request("cookie-session"), BaseCloseSessionRequest(_ng_session_id="s1"), Response()
                )
            )

        assert caught.value.status_code == 409
        assert server.released == []

    def test_a_failed_release_reports_it_and_stays_open(self) -> None:
        server = _server()
        self._seed(server)

        type(server).release_error = RuntimeError("provider unreachable")
        response = Response()

        result = asyncio.run(
            server._close_session_endpoint(_request("s1"), BaseCloseSessionRequest(_ng_session_id="s1"), response)
        )

        assert (result.status, result.released) == (SessionCloseStatus.RELEASE_FAILED, False)
        assert response.status_code == 503
        assert not server._sessions["s1"].is_closed, "a failed release must remain retryable"

    def test_concurrent_closes_release_once(self) -> None:
        server = _server()
        self._seed(server)
        type(server).release_delay_s = 0.02

        async def _run():
            return await asyncio.gather(
                *(
                    server._close_session_endpoint(
                        _request("s1"), BaseCloseSessionRequest(_ng_session_id="s1"), Response()
                    )
                    for _ in range(10)
                )
            )

        results = asyncio.run(_run())

        assert server.released == [("s1", SessionCloseReason.COMPLETED)], "cleanup must not run once per caller"
        assert all(r.released for r in results)


class TestCloseCapability:
    """Without a cookie the capability is the only thing authorizing release."""

    def _seed_with_token(self, server, token="secret-token"):
        asyncio.run(
            server._seed_session_endpoint(
                _request("s1"), BaseSeedSessionRequest(_ng_session_id="s1", _ng_session_close_token=token)
            )
        )

    def test_a_matching_capability_releases_without_a_cookie(self) -> None:
        server = _server()
        self._seed_with_token(server)

        result = asyncio.run(
            server._close_session_endpoint(
                _request(None),
                BaseCloseSessionRequest(_ng_session_id="s1", _ng_session_close_token="secret-token"),
                Response(),
            )
        )

        assert result.status is SessionCloseStatus.CLOSED

    def test_a_wrong_capability_does_not_release(self) -> None:
        server = _server()
        self._seed_with_token(server)

        with pytest.raises(HTTPException) as caught:
            asyncio.run(
                server._close_session_endpoint(
                    _request(None),
                    BaseCloseSessionRequest(_ng_session_id="s1", _ng_session_close_token="guess"),
                    Response(),
                )
            )

        assert caught.value.status_code == 403
        assert server.released == []

    def test_a_public_session_id_alone_cannot_close_someone_elses_session(self) -> None:
        server = _server()
        self._seed_with_token(server)

        with pytest.raises(HTTPException) as caught:
            asyncio.run(
                server._close_session_endpoint(
                    _request(None), BaseCloseSessionRequest(_ng_session_id="s1"), Response()
                )
            )

        assert caught.value.status_code == 403

    def test_the_token_is_never_stored_or_returned(self) -> None:
        server = _server()
        self._seed_with_token(server, token="plaintext-secret")

        record = server._sessions["s1"]

        assert record.close_token_digest is not None
        assert "plaintext-secret" not in repr(record)


class TestExpiry:
    def _seed(self, server, session_id="s1"):
        asyncio.run(
            server._seed_session_endpoint(_request(session_id), BaseSeedSessionRequest(_ng_session_id=session_id))
        )

    def test_an_idle_session_is_reclaimed(self) -> None:
        server = _server(session_ttl_s=10.0)
        self._seed(server)
        record = server._sessions["s1"]
        record.last_seen -= 20.0

        asyncio.run(server._sweep_once(time.monotonic(), ttl=10.0))

        assert server.released == [("s1", SessionCloseReason.EXPIRED)]

    def test_a_busy_session_is_not_reclaimed_however_long_it_has_been_quiet(self) -> None:
        """A rollout can sit in a model call for minutes; that is not abandonment."""
        server = _server(session_ttl_s=10.0)
        self._seed(server)
        record = server._sessions["s1"]
        record.last_seen -= 999.0
        record.active_handlers = 1

        asyncio.run(server._sweep_once(time.monotonic(), ttl=10.0))

        assert server.released == []

    def test_the_absolute_lifetime_reclaims_even_a_busy_session(self) -> None:
        """The bound exists for a handler that has hung, so activity must not defeat it."""
        server = _server(session_ttl_s=10.0, session_max_lifetime_s=60.0)
        self._seed(server)
        record = server._sessions["s1"]
        record.active_handlers = 1
        record.created_at -= 120.0

        asyncio.run(server._sweep_once(time.monotonic(), ttl=10.0))

        assert server.released == [("s1", SessionCloseReason.EXPIRED)]

    def test_a_tombstone_is_dropped_once_a_delayed_seed_can_no_longer_arrive(self) -> None:
        server = _server(session_ttl_s=10.0, session_tombstone_s=30.0)
        self._seed(server)
        asyncio.run(
            server._close_session_endpoint(_request("s1"), BaseCloseSessionRequest(_ng_session_id="s1"), Response())
        )
        server._sessions["s1"].closed_at -= 60.0

        asyncio.run(server._sweep_once(time.monotonic(), ttl=10.0))

        assert "s1" not in server._sessions


class TestActiveSession:
    def test_a_handler_suppresses_idle_expiry_while_it_runs(self) -> None:
        server = _server(session_ttl_s=10.0)
        asyncio.run(server._seed_session_endpoint(_request("s1"), BaseSeedSessionRequest(_ng_session_id="s1")))

        async def _run():
            async with server.active_session("s1"):
                assert server._sessions["s1"].active_handlers == 1
            assert server._sessions["s1"].active_handlers == 0

        asyncio.run(_run())

    def test_a_handler_cannot_start_once_release_has_begun(self) -> None:
        """Otherwise close can tear a resource down while verification is still reading it."""
        server = _server()
        asyncio.run(server._seed_session_endpoint(_request("s1"), BaseSeedSessionRequest(_ng_session_id="s1")))
        asyncio.run(
            server._close_session_endpoint(_request("s1"), BaseCloseSessionRequest(_ng_session_id="s1"), Response())
        )

        async def _run():
            async with server.active_session("s1"):
                pass

        with pytest.raises(HTTPException) as caught:
            asyncio.run(_run())

        assert caught.value.status_code == 409


class TestHookNaming:
    def test_the_base_does_not_define_close_session(self) -> None:
        """`GymnasiumServer`, TALES and OpenAir already own that name with another signature."""
        assert not hasattr(SimpleResourcesServer, "close_session")

    def test_an_existing_close_session_can_bridge_into_the_hook(self) -> None:
        bridged = []

        class _Legacy(SimpleResourcesServer):
            async def verify(self, body):  # pragma: no cover
                pass

            async def close_session(self, session_id):
                bridged.append(session_id)

            async def release_session(self, session_id, reason):
                await self.close_session(session_id)

        server = _Legacy(
            config=BaseResourcesServerConfig(host="", port=0, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )
        asyncio.run(server._seed_session_endpoint(_request("s1"), BaseSeedSessionRequest(_ng_session_id="s1")))
        asyncio.run(
            server._close_session_endpoint(_request("s1"), BaseCloseSessionRequest(_ng_session_id="s1"), Response())
        )

        assert bridged == ["s1"]


class TestRouteRegistration:
    def test_lifecycle_routes_are_registered(self) -> None:
        routes = {route.path for route in _server().setup_webserver().routes}
        assert {"/seed_session", "/close_session"} <= routes

    def test_release_session_is_never_offered_to_the_model(self) -> None:
        """A policy that could call it could end its own episode's resources."""
        server = _server()
        server.setup_webserver()
        assert "close_session" in RESERVED_MCP_TOOL_NAMES
        assert "release_session" not in {tool.name for tool in server.mcp_tools([], None) or []}

    def test_the_sweeper_is_off_unless_a_ttl_is_configured(self) -> None:
        assert BaseResourcesServerConfig(host="", port=0, entrypoint="", name="").session_ttl_s is None
