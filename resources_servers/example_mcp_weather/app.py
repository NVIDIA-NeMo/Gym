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

from typing import Any, Optional

from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    MCPResourcesServer,
    MCPServerMetadata,
    gym_tool,
)
from nemo_gym.server_utils import SESSION_ID_KEY


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
    mcp: MCPServerMetadata


class ExampleMCPWeatherVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: Optional[dict[str, Any]] = None


class ExampleMCPWeatherVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    expected_weather: str
    tool_call_seen: bool
    final_response_mentions_weather: bool


class ExampleMCPWeatherResourcesServer(MCPResourcesServer):
    config: ExampleMCPWeatherResourcesServerConfig
    session_id_to_state: dict[str, dict[str, Any]] = Field(default_factory=dict)

    async def seed_session(
        self,
        request: Request,
        body: ExampleMCPWeatherSeedSessionRequest,
    ) -> ExampleMCPWeatherSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        expected_city = (body.verifier_metadata or {}).get("expected_city", "Paris")
        self.session_id_to_state[session_id] = {
            "expected_city": expected_city,
            "weather_calls": [],
        }
        return ExampleMCPWeatherSeedSessionResponse(mcp=self.build_mcp_session_metadata(request))

    @gym_tool
    def get_weather(self, session_id: str, city: str) -> str:
        """Get a deterministic weather report for a city."""
        # session_id is injected by the base class (from the per-rollout MCP token); it is hidden from
        # the tool's MCP input schema, so the model only sees `city`.
        state = self.session_id_to_state.setdefault(session_id, {"weather_calls": []})
        weather = _weather_sentence(city)
        state["weather_calls"].append({"city": city, "weather": weather})
        return weather

    async def verify(
        self,
        request: Request,
        body: ExampleMCPWeatherVerifyRequest,
    ) -> ExampleMCPWeatherVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        state = self.session_id_to_state.get(session_id, {"weather_calls": []})
        expected_city_value = (body.verifier_metadata or {}).get("expected_city", "Paris")
        expected_weather = _weather_sentence(expected_city_value)
        expected_city = expected_city_value.casefold()

        # Match the city case-insensitively. The weather sentence is derived deterministically from the
        # city, so a city match is sufficient; comparing the sentence exactly would spuriously reject a
        # correct call that used different casing (e.g. get_weather("PARIS")).
        tool_call_seen = any(str(call.get("city", "")).casefold() == expected_city for call in state["weather_calls"])
        final_text = _extract_assistant_text(body)
        final_response_mentions_weather = expected_weather.casefold() in final_text.casefold()
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
