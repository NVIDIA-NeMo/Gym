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

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class BrowserbaseDOMRuntime:
    session: Any
    stagehand_session_id: Optional[str]
    start_url: Optional[str] = None


class BrowserbaseDOMSessionManager:
    def __init__(
        self,
        *,
        browserbase_api_key: str,
        project_id: str,
        model_api_key: str,
        stagehand_model: str,
        proxies: bool = False,
        advanced_stealth: bool = False,
        stagehand_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not browserbase_api_key:
            raise ValueError("Missing Browserbase API key.")
        if not project_id:
            raise ValueError("Missing Browserbase project ID.")
        if not model_api_key:
            raise ValueError("Missing Stagehand model API key.")

        self.browserbase_api_key = browserbase_api_key
        self.project_id = project_id
        self.model_api_key = model_api_key
        self.stagehand_model = stagehand_model
        self.proxies = proxies
        self.advanced_stealth = advanced_stealth
        self._stagehand_factory = stagehand_factory
        self._client: Any = None
        self._client_lock = asyncio.Lock()

    def _get_stagehand_factory(self) -> Callable[..., Any]:
        if self._stagehand_factory is not None:
            return self._stagehand_factory

        try:
            from stagehand import AsyncStagehand
        except ImportError as exc:  # pragma: no cover - exercised in environments without browser deps
            raise ImportError(
                "Browserbase DOM environments require the `stagehand` package. "
                "Install it in the target resources server environment."
            ) from exc

        self._stagehand_factory = AsyncStagehand
        return AsyncStagehand

    async def _get_client(self) -> Any:
        async with self._client_lock:
            if self._client is None:
                stagehand_factory = self._get_stagehand_factory()
                self._client = stagehand_factory(
                    browserbase_api_key=self.browserbase_api_key,
                    browserbase_project_id=self.project_id,
                    model_api_key=self.model_api_key,
                )
        return self._client

    def _build_browserbase_params(self) -> Optional[Dict[str, Any]]:
        browserbase_params: Dict[str, Any] = {}
        if self.proxies:
            browserbase_params["proxies"] = self.proxies
        if self.advanced_stealth:
            browserbase_params["browserSettings"] = {"advancedStealth": True}
        return browserbase_params or None

    async def create_runtime(self, start_url: Optional[str] = None) -> BrowserbaseDOMRuntime:
        client = await self._get_client()
        session = await client.sessions.start(
            model_name=self.stagehand_model,
            browserbase_session_create_params=self._build_browserbase_params(),
        )

        runtime = BrowserbaseDOMRuntime(
            session=session,
            stagehand_session_id=getattr(session, "id", None),
            start_url=start_url,
        )
        if start_url:
            try:
                await session.navigate(url=start_url)
            except Exception:
                await self.cleanup_runtime(runtime)
                raise
        return runtime

    async def cleanup_runtime(self, runtime: Optional[BrowserbaseDOMRuntime]) -> None:
        if runtime is None or runtime.session is None:
            return

        try:
            await runtime.session.end()
        except Exception:
            pass

    async def close(self) -> None:
        if self._client is None:
            return

        try:
            await self._client.close()
        except Exception:
            pass
        finally:
            self._client = None

    async def navigate(self, runtime: BrowserbaseDOMRuntime, url: str) -> str:
        try:
            await runtime.session.navigate(url=url)
            return f"Navigated to {url}"
        except Exception as exc:
            return f"Error navigating to {url}: {exc}"

    async def observe(self, runtime: BrowserbaseDOMRuntime, instruction: str) -> str:
        try:
            response = await runtime.session.observe(instruction=instruction)
            actions = [
                {
                    "description": action.description,
                    "selector": action.selector,
                    "method": action.method,
                }
                for action in response.data.result
            ]
            if not actions:
                return "No matching elements found"
            return json.dumps(actions, indent=2)
        except Exception as exc:
            return f"Error observing page: {exc}"

    async def act(self, runtime: BrowserbaseDOMRuntime, instruction: str) -> str:
        try:
            response = await runtime.session.act(input=instruction)
            result = response.data.result
            status = "Success" if result.success else "Failed"
            return f"{status}: {result.message}"
        except Exception as exc:
            return f"Error executing action: {exc}"

    async def extract(self, runtime: BrowserbaseDOMRuntime, instruction: str, schema_json: str) -> str:
        try:
            schema = json.loads(schema_json)
        except json.JSONDecodeError as exc:
            return f"Error parsing schema JSON: {exc}"

        try:
            response = await runtime.session.extract(
                instruction=instruction,
                schema=schema,
            )
            return json.dumps(response.data.result, indent=2)
        except Exception as exc:
            return f"Error extracting data: {exc}"
