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

from typing import Annotated, Any, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionLostError,
    SimpleResourcesServer,
    lost_session_verification,
    reject_typed_resources_seed,
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


class ExampleMCPWeatherSeedSessionResponse(BaseSeedSessionResponse):
    # When expose_tools_over_mcp is on, the response JSON also carries an "mcp" key
    # (server name, /mcp URL path, per-rollout session-token header) added at startup.
    # A typed seed returns ResourcesSeedSessionResponse instead, which carries the same metadata in resources_tools.
    pass


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

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest
        | Annotated[ExampleMCPWeatherSeedSessionRequest, BeforeValidator(reject_typed_resources_seed)],
    ) -> ResourcesSeedSessionResponse | ExampleMCPWeatherSeedSessionResponse:
        response = await super().seed_session(body, request=request)
        return (
            response if isinstance(response, ResourcesSeedSessionResponse) else ExampleMCPWeatherSeedSessionResponse()
        )

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
        state = self.session_id_to_state.get(session_id)
        verified = self._verified_sessions.get(session_id)
        if self._fence.closed_episode(session_id) is not None:
            state, verified = None, None
        expected_city_value = (body.verifier_metadata or {}).get("expected_city", "Paris")
        expected_weather = _weather_sentence(expected_city_value)
        expected_city = expected_city_value.casefold()
        final_text = _extract_assistant_text(body)
        final_response_mentions_weather = expected_weather.casefold() in final_text.casefold()

        if state is None and verified is None:
            # The session's weather calls are gone (never seeded, closed, or the server restarted), so this server
            # cannot tell whether the tool was used. Mask the sample rather than score the policy 0.
            return ExampleMCPWeatherVerifyResponse(
                **(body.model_dump() | lost_session_verification("No open example_mcp_weather session to verify")),
                expected_weather=expected_weather,
                tool_call_seen=False,
                final_response_mentions_weather=final_response_mentions_weather,
            )

        # Match the city case-insensitively. The weather sentence is derived deterministically from the
        # city, so a city match is sufficient; comparing the sentence exactly would spuriously reject a
        # correct call that used different casing (e.g. get_weather("PARIS")).
        if state is not None:
            called_cities = frozenset(str(call.get("city", "")).casefold() for call in state["weather_calls"])
            self.session_id_to_state.pop(session_id)
            self._verified_sessions.remember(session_id, (self._session_episodes.pop(session_id, None), called_cities))
        else:
            called_cities = verified[1]
        tool_call_seen = expected_city in called_cities
        reward = float(tool_call_seen and final_response_mentions_weather)

        return ExampleMCPWeatherVerifyResponse(
            **body.model_dump(),
            reward=reward,
            expected_weather=expected_weather,
            tool_call_seen=tool_call_seen,
            final_response_mentions_weather=final_response_mentions_weather,
        )


if __name__ == "__main__":
    ExampleMCPWeatherResourcesServer.run_webserver()
