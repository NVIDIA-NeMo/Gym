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
"""In-tree agent-server for wrapping any LangChain `deepagents` graph as a native Gym `responses_api_agents` server.

Generic base for wrapping any LangChain `deepagents` graph as a native Gym `responses_api_agents` server
(model calls go through Gym's own `model_server`, on-policy, instead of the agent bringing its own model).
Concrete instances subclass `DeepAgentsAgent` and implement `build_agent(model)` — see
`reasoning_search_agent.py` for the one used by this repo's reasoning_gym/tavily_search comparison.

See `responses_langchain_bridge.py` for the LangChain <-> Gym Responses API bridge (`GymResponsesChatModel`
and the message-conversion helpers) and why it can't just be `langchain_openai.ChatOpenAI`.

The compiled deepagents graph is built once, in `__init__`, and stored as `self.agent` — not per-request,
and named to match how `deepagents`' own docs/examples name the object `create_deep_agent()` returns
(`agent = create_deep_agent(...)`), rather than `langgraph_agent`'s `self.graph`.
"""

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

from fastapi import Body, Request, Response
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field

from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.langchain_deepagents_agent.responses_langchain_bridge import (
    GymResponsesChatModel,
    _request_context,
    to_langchain,
    to_responses,
)
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


class DeepAgentsAgentConfig(SimpleAgentConfig):
    """No tool-specific fields — those belong on a concrete subclass's config (see reasoning_search_agent.py).

    `max_steps` (inherited from SimpleAgentConfig) is unused: deepagents runs its own internal tool loop
    and answers in one call, the same as remote_agent's `max_steps: 1` in
    examples/langchain_deepagent/configs/config_reasoning_gym.yaml.
    """


class DeepAgentsAgent(SimpleAgent):
    """Inherits SimpleAgent.run() unchanged (seed_session -> self-POST /v1/responses -> verify). Overrides
    responses() with everything true of any deepagents-based agent wired into Gym.

    Does not build a TrajectoryRecord (per-tool-call/model-call observability) — see README.md's "Known
    limitation" section. That's deliberately deferred, not an oversight."""

    config: DeepAgentsAgentConfig
    agent: Any = Field(default=None, exclude=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = self.build_agent(GymResponsesChatModel(agent=self))

    @abstractmethod
    def build_agent(self, model: BaseChatModel) -> Any:
        """Return a compiled deepagents graph, e.g. create_deep_agent(model=model, tools=[...], ...).
        Called once, from __init__ — not per-request."""
        raise NotImplementedError

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        # Different servers have different session-middleware keys (get_session_middleware_key() is
        # f"{class_name}___{server_name}"), so a cookie from the resources server's /seed_session does not
        # automatically ride through to /verify unless re-set explicitly here.
        for key, value in request.cookies.items():
            response.set_cookie(key, value)

        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None

        if isinstance(body.input, str):
            input_items = [NeMoGymEasyInputMessage(role="user", content=body.input)]
        else:
            input_items = body.input
        input_messages = to_langchain(input_items)

        model_url_path = self.url_path_for_request("/v1/responses", request)
        token = _request_context.set(
            {"rollout_id": rollout_id, "cookies": request.cookies, "model_url_path": model_url_path}
        )
        try:
            final_state = await self.agent.ainvoke({"messages": input_messages})
        finally:
            _request_context.reset(token)

        new_messages = final_state["messages"][len(input_messages) :]
        return NeMoGymResponse.model_validate(to_responses(new_messages, self.config.model_server.name))
