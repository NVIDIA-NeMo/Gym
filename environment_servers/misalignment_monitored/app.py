# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relay rollouts to an agent like ``legacy_agent``, and hand each finished one to the misalignment monitor.

The result goes to the monitor in the background, so the reply, its reward and its timing are what they
would be without the monitor: a policy cannot learn to hide from it. While the monitor holds an unresolved
P0 alert past its pause threshold, this server holds new rollouts. That pauses the whole run and does not
single out any rollout.
"""

import asyncio
import contextvars
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import orjson
from fastapi import Body, FastAPI, Request, Response

from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServer, LegacyAgentEnvironmentServerConfig
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, ResourcesServerRef
from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status


# How often to ask the monitor whether to pause, how long to wait for its answer, and how long shutdown
# or the end of a run waits for posts in flight.
STATUS_POLL_INTERVAL_S = 15.0
STATUS_TIMEOUT_S = 10.0
SHUTDOWN_DRAIN_TIMEOUT_S = 60.0


class MisalignmentMonitoredEnvironmentServerConfig(LegacyAgentEnvironmentServerConfig):
    """Bind one agent server and the misalignment monitor that reviews its rollouts."""

    monitor_server: ResourcesServerRef


class MisalignmentMonitoredEnvironmentServer(LegacyAgentEnvironmentServer):
    """Relay `/run` to one agent server, then post the result to the misalignment monitor."""

    config: MisalignmentMonitoredEnvironmentServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._posts: set[asyncio.Task] = set()
        self._counts: Counter = Counter()
        self._status_poller: asyncio.Task | None = None
        self._status_known = asyncio.Event()
        self._not_paused = asyncio.Event()
        self._not_paused.set()

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        serve = app.router.lifespan_context

        @asynccontextmanager
        async def drain_on_shutdown(application: FastAPI) -> AsyncIterator[Any]:
            async with serve(application) as state:
                yield state
            if self._status_poller is not None:
                self._status_poller.cancel()
            if self._posts:
                await asyncio.wait(self._posts, timeout=SHUTDOWN_DRAIN_TIMEOUT_S)
            print(f"[{self.config.name}] misalignment monitor posts: {dict(self._counts)}", flush=True)

        app.router.lifespan_context = drain_on_shutdown
        return app

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        # Called once when a run ends, before servers stop: the monitor finishes its reviews while its judge is up.
        if self._posts:
            await asyncio.wait(set(self._posts), timeout=SHUTDOWN_DRAIN_TIMEOUT_S)
        try:
            response = await self.server_client.post(server_name=self.config.monitor_server.name, url_path="/drain")
            await raise_for_status(response)
            print(f"[{self.config.name}] misalignment monitor: {await get_response_json(response)}", flush=True)
        except Exception as e:
            print(f"[{self.config.name}] misalignment monitor drain failed: {type(e).__name__}: {e}", flush=True)
        return await super().aggregate_metrics(body)

    async def run_legacy(self, request: Request) -> Response:
        if self._status_poller is None:
            self._status_poller = asyncio.create_task(self._poll_status(), context=contextvars.Context())
        # The first rollouts wait for the first answer, so a run never starts past an unresolved P0.
        await self._status_known.wait()
        await self._not_paused.wait()
        response = await super().run_legacy(request)
        if response.status_code == 200:
            self._post(await request.body(), response.body)
        return response

    def _post(self, run_body: bytes, result_body: bytes) -> None:
        # An empty context keeps the post out of this rollout's attribution and trace.
        task = asyncio.create_task(self._send(run_body, result_body), context=contextvars.Context())
        self._posts.add(task)
        task.add_done_callback(self._posts.discard)

    async def _send(self, run_body: bytes, result_body: bytes) -> None:
        try:
            row, result = orjson.loads(run_body), orjson.loads(result_body)
            identity = {
                key: row[key]
                for key in (TASK_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, AGENT_REF_KEY_NAME)
                if key in row
            }
            # The id the agent's model calls are captured under, so the monitor can join them to this result.
            payload = {**result, **identity, "rollout_id": maybe_rollout_id_from_run_body(row)}
            response = await self.server_client.post(
                server_name=self.config.monitor_server.name,
                url_path="/monitor",
                data=orjson.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            await raise_for_status(response)
            accepted = (await get_response_json(response)).get("accepted")
            self._counts["accepted" if accepted else "dropped_by_monitor"] += 1
        except Exception as e:
            self._counts["failed"] += 1
            # The first and every hundredth failure are printed, so a monitor outage is loud but bounded.
            if self._counts["failed"] % 100 == 1:
                print(
                    f"[{self.config.name}] misalignment monitor post failed: {type(e).__name__}: {e} {dict(self._counts)}",
                    flush=True,
                )

    async def _poll_status(self) -> None:
        while True:
            try:
                async with asyncio.timeout(STATUS_TIMEOUT_S):
                    response = await self.server_client.get(
                        server_name=self.config.monitor_server.name, url_path="/status"
                    )
                    await raise_for_status(response)
                    status = await get_response_json(response)
                paused = bool(status.get("pause"))
            except Exception as e:
                # A monitor that cannot answer does not stall the run.
                print(
                    f"[{self.config.name}] misalignment monitor status check failed: {type(e).__name__}: {e}",
                    flush=True,
                )
                status, paused = {}, False
            if paused and self._not_paused.is_set():
                print(
                    f"[{self.config.name}] holding new rollouts for unresolved P0 alerts {status.get('open_p0')}. "
                    "Resolve them with POST /resolve on the monitor.",
                    flush=True,
                )
                self._not_paused.clear()
            elif not paused and not self._not_paused.is_set():
                print(f"[{self.config.name}] resuming new rollouts.", flush=True)
                self._not_paused.set()
            self._status_known.set()
            await asyncio.sleep(STATUS_POLL_INTERVAL_S)


if __name__ == "__main__":
    MisalignmentMonitoredEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = MisalignmentMonitoredEnvironmentServer.run_webserver()  # noqa: F401
