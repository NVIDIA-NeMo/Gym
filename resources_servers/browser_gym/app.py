# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
from fastapi import FastAPI
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import SimpleResourcesServer
from resources_servers.browser_gym.browser_pool import BrowserPool
from resources_servers.browser_gym.schemas import (
    BrowserGymResourcesServerConfig,
    CUACloseRequest,
    CUACloseResponse,
    CUADumpLocalStorageRequest,
    CUADumpLocalStorageResponse,
    CUASeedSessionRequest,
    CUASeedSessionResponse,
    CUAStepRequest,
    CUAStepResponse,
    CUAVerifyRequest,
    CUAVerifyResponse,
)
from resources_servers.browser_gym.setup_playwright import ensure_playwright


logger = logging.getLogger(__name__)


class BrowserGymResourcesServer(SimpleResourcesServer):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: BrowserGymResourcesServerConfig
    browser_pool: BrowserPool = Field(default=None, exclude=True)
    _verify_session: Optional[aiohttp.ClientSession] = None

    def model_post_init(self, __context):
        super().model_post_init(__context)
        ensure_playwright()
        self.browser_pool = BrowserPool(
            max_concurrent=self.config.max_concurrent_browsers,
            pool_size=self.config.browser_pool_size,
            default_viewport_width=self.config.default_viewport_width,
            default_viewport_height=self.config.default_viewport_height,
            session_ttl_seconds=self.config.session_ttl_seconds,
            reaper_interval_seconds=self.config.session_reaper_interval_seconds,
        )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/step")(self.step)
        app.post("/dump_local_storage")(self.dump_local_storage)
        app.post("/close")(self.close)

        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def _lifespan_with_shutdown(app):
            self.browser_pool.start_reaper()
            async with parent_lifespan(app) as maybe_state:
                yield maybe_state
            logger.info("Server shutting down — closing all browser sessions")
            await self.browser_pool.shutdown()
            if self._verify_session and not self._verify_session.closed:
                await self._verify_session.close()

        app.router.lifespan_context = _lifespan_with_shutdown

        return app

    async def seed_session(self, body: CUASeedSessionRequest) -> CUASeedSessionResponse:
        env_id = str(uuid.uuid4())
        screenshot = await self.browser_pool.create_session(
            env_id=env_id,
            start_url=body.start_url,
            viewport_width=body.viewport_width,
            viewport_height=body.viewport_height,
        )
        return CUASeedSessionResponse(env_id=env_id, screenshot=screenshot)

    async def step(self, body: CUAStepRequest) -> CUAStepResponse:
        try:
            screenshot, current_url, error = await self.browser_pool.execute_action(body.env_id, body.action)
            return CUAStepResponse(screenshot=screenshot, current_url=current_url, error=error)
        except (TimeoutError, asyncio.TimeoutError):
            logger.error(
                "Browser stuck for env_id=%s action=%s — returning empty screenshot",
                body.env_id,
                body.action.action_type,
            )
            return CUAStepResponse(
                screenshot="", current_url="error:browser_stuck", error="Browser stuck — screenshot timed out"
            )

    async def dump_local_storage(self, body: CUADumpLocalStorageRequest) -> CUADumpLocalStorageResponse:
        try:
            ls_dump = await self.browser_pool.dump_local_storage(body.env_id)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("dump_local_storage timed out for env_id=%s — returning empty", body.env_id)
            ls_dump = ""
        except Exception as e:
            logger.warning("dump_local_storage failed for env_id=%s: %s", body.env_id, e)
            ls_dump = ""
        return CUADumpLocalStorageResponse(local_storage_dump=ls_dump)

    def _get_verify_session(self) -> aiohttp.ClientSession:
        if self._verify_session is None or self._verify_session.closed:
            self._verify_session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    limit=self.config.verify_connector_limit,
                    limit_per_host=self.config.verify_connector_limit_per_host,
                ),
                timeout=aiohttp.ClientTimeout(total=self.config.verify_timeout_seconds),
            )
        return self._verify_session

    async def verify(self, body: CUAVerifyRequest) -> CUAVerifyResponse:
        vm = body.verifier_metadata or {}
        gym_url = vm.get("gym_url", "")
        task_id = vm.get("task_id", "")

        local_storage_dump = ""
        if body.response and body.response.local_storage_dump:
            local_storage_dump = body.response.local_storage_dump

        if not gym_url or not task_id:
            logger.warning("Missing gym_url or task_id in verifier_metadata, returning reward=0.0")
            return CUAVerifyResponse(
                **body.model_dump(), reward=0.0, verification_result={"error": "missing gym_url or task_id"}
            )

        verify_url = f"{gym_url.rstrip('/')}/api/v1/get_actual_state"

        model_response = ""
        if body.response and body.response.trajectory and body.response.trajectory.final_message:
            model_response = body.response.trajectory.final_message

        try:
            session = self._get_verify_session()
            max_verify_retries = 3
            transient_codes = {502, 503, 504}
            resp_text = ""
            resp_status = 0

            for attempt in range(1, max_verify_retries + 1):
                form_data = aiohttp.FormData()
                form_data.add_field("taskId", task_id)
                form_data.add_field(
                    "localStorageDump",
                    local_storage_dump,
                    filename="localStorageDump.json",
                    content_type="application/json",
                )
                if model_response:
                    form_data.add_field("modelResponse", model_response)

                async with session.post(verify_url, data=form_data) as resp:
                    resp_status = resp.status
                    if resp.status == 200:
                        result = await resp.json()
                        break

                    resp_text = await resp.text()
                    if resp.status in transient_codes and attempt < max_verify_retries:
                        wait = 1.0 * (2 ** (attempt - 1))
                        logger.warning(
                            "Verification API returned %d (attempt %d/%d) — retrying in %.0fs",
                            resp.status,
                            attempt,
                            max_verify_retries,
                            wait,
                        )
                        await asyncio.sleep(wait)
                        continue

                    logger.warning(f"Verification API returned {resp.status}: {resp_text}")
                    return CUAVerifyResponse(
                        **body.model_dump(),
                        reward=0.0,
                        verification_result={"error": resp_text, "status_code": resp.status},
                    )
            else:
                logger.warning(f"Verification API returned {resp_status}: {resp_text}")
                return CUAVerifyResponse(
                    **body.model_dump(),
                    reward=0.0,
                    verification_result={"error": resp_text, "status_code": resp_status},
                )

            assertions = result.get("assertions", [])
            if not assertions:
                logger.warning("Verification returned no assertions for task_id=%s", task_id)
                return CUAVerifyResponse(**body.model_dump(), reward=0.0, verification_result=result)

            all_passed = all(a.get("result") == "pass" for a in assertions)
            reward = 1.0 if all_passed else 0.0

            if not all_passed:
                failed = [a for a in assertions if a.get("result") != "pass"]
                logger.info(
                    "Verification task_id=%s reward=%.1f — %d/%d assertions failed: %s",
                    task_id,
                    reward,
                    len(failed),
                    len(assertions),
                    failed,
                )

            return CUAVerifyResponse(**body.model_dump(), reward=reward, verification_result=result)

        except Exception as e:
            logger.error("Verification failed for task_id=%s: %s: %s", task_id, type(e).__name__, e)
            return CUAVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                verification_result={"error": f"{type(e).__name__}: {e}"},
            )

    async def close(self, body: CUACloseRequest) -> CUACloseResponse:
        success = await self.browser_pool.close_session(body.env_id)
        if success:
            return CUACloseResponse(message="Session closed", success=True)
        return CUACloseResponse(message="Session not found (already closed)", success=True)


if __name__ == "__main__":
    BrowserGymResourcesServer.run_webserver()
