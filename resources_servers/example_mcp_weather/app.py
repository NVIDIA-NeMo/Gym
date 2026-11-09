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

from typing import Any, Optional, cast

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSessionLostError,
    SimpleResourcesServer,
    lost_session_verification,
)


def _weather_sentence(city: str) -> str:
    return f"The weather in {city} is sunny and 72 F."


def _extract_assistant_text(body: BaseVerifyRequest) -> str:
    texts: list[str] = []
    for output_item in body.response.output:
        if getattr(output_item, "type", None) != "message" or getattr(output_item, "role", None) != "assistant":
            continue
        content = getattr(output_item, "content", None)
        if isinstance(content, list):
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    texts.append(text)
        elif isinstance(content, str):
            texts.append(content)
    return "\n".join(texts).strip()


class ExampleMCPWeatherResourcesServerConfig(BaseResourcesServerConfig):
    pass


class ExampleMCPWeatherSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")

    # Task-specific ground truth travels in verifier_metadata (per Gym convention), e.g.
    # {"expected_city": "Paris"}.
    verifier_metadata: Optional[dict[str, Any]] = None


class ExampleMCPWeatherGetWeatherRequest(BaseModel):
    city: str


class ExampleMCPWeatherGetWeatherResponse(BaseModel):
    weather: str


class ExampleMCPWeatherVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: Optional[dict[str, Any]] = None


class ExampleMCPWeatherVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    expected_weather: str
    tool_call_seen: bool
    final_response_mentions_weather: bool


class ExampleMCPWeatherResourcesServer(SimpleResourcesServer):
    config: ExampleMCPWeatherResourcesServerConfig
    # The weather calls of each live session, keyed by the session cookie's SESSION_ID_KEY. Every tool call reaches
    # the same key: over HTTP the cookie carries it, and over MCP the signed session token minted at seed time does.
    session_id_to_state: dict[str, dict[str, Any]] = Field(default_factory=dict)
    seed_request_model = ExampleMCPWeatherSeedSessionRequest

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/get_weather")(self.get_weather)
        return app

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self.session_id_to_state

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        """Start an empty weather-call history for this task."""
        self.session_id_to_state[session_id] = {"weather_calls": []}

    async def release_session_state(self, session_id: str) -> None:
        """Drop the live weather-call history."""
        self.session_id_to_state.pop(session_id, None)

    async def get_weather(
        self,
        request: Request,
        body: ExampleMCPWeatherGetWeatherRequest,
    ) -> ExampleMCPWeatherGetWeatherResponse:
        """Get a deterministic weather report for a city."""
        # No await between this lookup and the append, so verify's snapshot and release cannot interleave with a call.
        # A handler that awaits must hold self.session_lock(session_id).
        state = self.session_id_to_state.get(request.session[SESSION_ID_KEY])
        if state is None:
            # Never seeded, or already closed. The signed MCP session token stays valid after a close, so the call
            # is refused with a 404 rather than recreating state that nothing would release.
            raise ResourcesSessionLostError("No open example_mcp_weather session for this request")
        weather = _weather_sentence(body.city)
        state["weather_calls"].append({"city": body.city, "weather": weather})
        return ExampleMCPWeatherGetWeatherResponse(weather=weather)

    async def verify(
        self,
        request: Request,
        body: ExampleMCPWeatherVerifyRequest,
    ) -> ExampleMCPWeatherVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        async with self.verifying(session_id) as verification:
            if verification.is_live:
                state = self.session_id_to_state[session_id]
                # Only the unique cities are needed on a retry; do not retain the complete tool-call history.
                called_cities = frozenset(str(call.get("city", "")).casefold() for call in state["weather_calls"])
                verification.retain(called_cities)
            elif not verification.is_lost:
                called_cities = cast(frozenset[str], verification.retained)

            expected_city_value = (body.verifier_metadata or {}).get("expected_city", "Paris")
            expected_weather = _weather_sentence(expected_city_value)
            expected_city = expected_city_value.casefold()
            final_text = _extract_assistant_text(body)
            final_response_mentions_weather = expected_weather.casefold() in final_text.casefold()

            if verification.is_lost:
                # Without live state or retained evidence, tool use cannot be reconstructed from the answer.
                return ExampleMCPWeatherVerifyResponse(
                    **(body.model_dump() | lost_session_verification("No open example_mcp_weather session to verify")),
                    expected_weather=expected_weather,
                    tool_call_seen=False,
                    final_response_mentions_weather=final_response_mentions_weather,
                )

            tool_call_seen = expected_city in called_cities
            return ExampleMCPWeatherVerifyResponse(
                **body.model_dump(),
                reward=float(tool_call_seen and final_response_mentions_weather),
                expected_weather=expected_weather,
                tool_call_seen=tool_call_seen,
                final_response_mentions_weather=final_response_mentions_weather,
            )


if __name__ == "__main__":
    ExampleMCPWeatherResourcesServer.run_webserver()
