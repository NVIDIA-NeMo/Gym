# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Where a failure happened, on real failure rows from past runs.

Each fixture is a real sidecar row, trimmed to the fields the rule reads and scrubbed of paths, hosts and ids.
"""

import asyncio

import pytest

from nemo_gym.failure_kinds import (
    INFRA,
    MODEL,
    failure_cause_of,
    failure_cause_of_text,
)
from nemo_gym.sandbox.providers.base import SandboxOperationTimeoutError


class TestTimeouts:
    """The agent's own time budget is the model's; an infra call's timeout is infra."""

    async def test_the_agents_own_asyncio_timeout_is_the_models(self) -> None:
        with pytest.raises(TimeoutError) as caught:
            async with asyncio.timeout(0.01):
                await asyncio.sleep(1)
        assert isinstance(caught.value.__context__, asyncio.CancelledError)
        assert failure_cause_of(caught.value) == MODEL

    def test_a_timeout_raised_from_none_is_no_evidence(self) -> None:
        try:
            try:
                raise ConnectionRefusedError("hidden on purpose")
            except ConnectionRefusedError:
                raise TimeoutError("agent budget") from None
        except TimeoutError as error:
            assert failure_cause_of(error) == MODEL

    def test_a_named_sandbox_timeout_is_infra(self) -> None:
        assert failure_cause_of(SandboxOperationTimeoutError("write_file")) == INFRA

    def test_the_opensandbox_write_file_timeout_as_relayed_text_is_infra(self) -> None:
        # The body a legacy_agent relay carried, before the error had a name.
        body = (
            '"SandboxOperationTimeoutError(\\"Timed out during OpenSandbox write_file(<path>) after 300s; '
            "sandbox_id='<uuid>'\\\")\""
        )
        assert failure_cause_of_text(body) == INFRA

    async def test_an_aiohttp_total_timeout_is_infra(self) -> None:
        aiohttp = pytest.importorskip("aiohttp")
        from aiohttp import web

        async def slow(request):
            await asyncio.sleep(5)
            return web.Response(text="late")

        app = web.Application()
        app.router.add_get("/", slow)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                with pytest.raises(TimeoutError) as caught:
                    async with session.get(f"http://127.0.0.1:{port}/", timeout=aiohttp.ClientTimeout(total=0.2)):
                        pass
        finally:
            await runner.cleanup()
        assert failure_cause_of(caught.value) == INFRA

    @pytest.mark.parametrize(
        ("text", "cause"),
        [
            # Ray could not carry the task's exception back.
            ("UnserializableException: Failed to deserialize exception.", INFRA),
            ("ray.exceptions.OutOfMemoryError: Task was killed due to the node running low on memory.", INFRA),
            ("ray.exceptions.WorkerCrashedError: The worker died unexpectedly", INFRA),
            # A RayTaskError takes the cause of what it wraps; a config ImportError stays final.
            ("RayTaskError(ImportError): cannot import name 'DEFAULT_TOOLS' from 'stirrup.tools'", MODEL),
            ("RayTaskError(ClientConnectorError): Cannot connect to host", INFRA),
        ],
    )
    def test_ray_failures(self, text: str, cause: str) -> None:
        assert failure_cause_of_text(text) == cause


class TestEnvironmentServerPath:
    """The server records the cause from the live error; terminal=False is a retry."""

    @pytest.mark.parametrize("cause", [MODEL, INFRA, None])
    def test_the_cause_travels_in_the_header_and_the_failure_record_is_unchanged(self, cause) -> None:
        """The versioned failure record is unchanged: the cause of a handled failure travels in the header."""
        from unittest.mock import MagicMock

        from fastapi.testclient import TestClient

        from nemo_gym.base_environment_server import (
            BaseEnvironmentServer,
            BaseEnvironmentServerConfig,
            HandledEpisodeError,
        )
        from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeFailure
        from nemo_gym.failure_kinds import FAILURE_CAUSE_HEADER
        from nemo_gym.rollout_collection import NG_FAILURE_CAUSE_KEY, _episode_record
        from nemo_gym.server_utils import ServerClient

        class Server(BaseEnvironmentServer[BaseEpisodeRequest, BaseEpisodeResponse]):
            request_model = BaseEpisodeRequest
            response_model = BaseEpisodeResponse

            async def run(self, request, cleanup):
                failure = EpisodeFailure(failure_reason="500, message='Internal Server Error'", terminal=False)
                raise HandledEpisodeError(failure, cause=cause)

            async def aggregate_metrics(self, body):
                raise NotImplementedError

        config = BaseEnvironmentServerConfig(
            name="env",
            host="127.0.0.1",
            port=1,
            entrypoint="app.py",
            default_episode_timeout_seconds=1,
            cleanup_timeout_seconds=0.01,
        )
        app = Server(config=config, server_client=MagicMock(spec=ServerClient)).setup_webserver()
        task = {"task_id": {"taskset": "t:default", "task_id": "a"}, "task_input": {}}
        reply = TestClient(app).post("/run", json={"episode_id": {"rollout_id": "r", "attempt": 1}, "task": task})

        body = reply.json()
        # The failure record carries only its version-1 fields.
        assert set(body["failure"]) == {"failure_reason", "terminal"}
        assert reply.headers.get(FAILURE_CAUSE_HEADER) == cause
        if cause is not None:
            body[NG_FAILURE_CAUSE_KEY] = cause  # what the collector adds from the header
        # A 500 with no cause in the header is infra by its text; a cause the server read wins over the text.
        assert _episode_record(body)[NG_FAILURE_CAUSE_KEY] == (cause or INFRA)

    def test_episode_failure_cause_reads_the_live_error(self) -> None:
        from types import SimpleNamespace

        from aiohttp import ClientPayloadError, ClientResponseError
        from multidict import CIMultiDict

        from nemo_gym.failure_kinds import FAILURE_CAUSE_HEADER, episode_failure_cause

        assert episode_failure_cause(TimeoutError(), terminal=False) == INFRA
        assert episode_failure_cause(ClientPayloadError("400, message='x'"), terminal=True) == INFRA
        named_model = ClientResponseError(
            request_info=SimpleNamespace(real_url="http://agent/run"),
            history=(),
            status=500,
            headers=CIMultiDict({FAILURE_CAUSE_HEADER: MODEL}),
        )
        assert episode_failure_cause(named_model, terminal=False) == MODEL
        assert episode_failure_cause(ValueError("bad seed row"), terminal=True) == MODEL
