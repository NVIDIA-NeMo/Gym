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

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFence,
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
    # The episode and task each Environment Server session was seeded for, keyed by resources_session_id.
    _session_episodes: dict[str, tuple[EpisodeId, TaskId]] = PrivateAttr(default_factory=dict)
    _fence: ResourcesSessionFence = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._fence = ResourcesSessionFence(self.config.session_close_retry_window_seconds)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/get_weather")(self.get_weather)
        return app

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest | ExampleMCPWeatherSeedSessionRequest,
    ) -> ResourcesSeedSessionResponse | ExampleMCPWeatherSeedSessionResponse:
        if not isinstance(body, ResourcesSeedSessionRequest):
            # An Agent's /run seeds with the row's own fields, under the session its cookie names.
            self.session_id_to_state[request.session[SESSION_ID_KEY]] = {"weather_calls": []}
            return ExampleMCPWeatherSeedSessionResponse()

        # An Environment Server names the session itself. Seeding the same ID again for the same episode must
        # return the open session unchanged, so a retried seed keeps the weather calls already recorded.
        session_id = body.resources_session_id
        async with self._fence.locked(session_id):
            self._fence.raise_if_closed(session_id)
            identity = self._session_episodes.get(session_id)
            if identity is None:
                # task_data holds the row's own fields, the body an Agent's /run seed sends. Input that fails
                # validation gets a final 422: retrying the episode would send the same input again.
                try:
                    ExampleMCPWeatherSeedSessionRequest.model_validate(body.task_data)
                except ValidationError as error:
                    raise ResourcesSeedRejectedError(
                        f"task_data is not an example_mcp_weather task: {error}"
                    ) from error
                self._session_episodes[session_id] = (body.episode_id, body.task_id)
                self.session_id_to_state[session_id] = {"weather_calls": []}
            elif identity != (body.episode_id, body.task_id):
                raise ResourcesSessionConflictError("resources_session_id is already bound to another episode or task")
            # Bind the cookie to the session. MCP auto-exposure signs this same ID into the session token it adds to
            # the reply as resources_tools, so tool calls over HTTP and over MCP both find the state stored here.
            request.session[SESSION_ID_KEY] = session_id
        return ResourcesSeedSessionResponse(resources_session_id=session_id)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        # The body names the session, so a close that carries no cookie (the seed's reply was lost) still releases it.
        session_id = body.resources_session_id
        async with self._fence.locked(session_id):
            closed_episode = self._fence.closed_episode(session_id)
            if closed_episode is not None and closed_episode != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
            identity = self._session_episodes.get(session_id)
            if identity is not None and identity[0] != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the seeded resources session")
            # Releasing in-memory state cannot fail. A server that holds an external resource releases it here and
            # lets a failure raise before mark_closed, so the caller can retry the close.
            self.session_id_to_state.pop(session_id, None)
            self._session_episodes.pop(session_id, None)
            # Closing an unknown or already-closed session also succeeds, and the fence refuses a later seed of it.
            self._fence.mark_closed(session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

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
        state = self.session_id_to_state.get(request.session[SESSION_ID_KEY])
        expected_city_value = (body.verifier_metadata or {}).get("expected_city", "Paris")
        expected_weather = _weather_sentence(expected_city_value)
        expected_city = expected_city_value.casefold()
        final_text = _extract_assistant_text(body)
        final_response_mentions_weather = expected_weather.casefold() in final_text.casefold()

        if state is None:
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
        tool_call_seen = any(str(call.get("city", "")).casefold() == expected_city for call in state["weather_calls"])
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
