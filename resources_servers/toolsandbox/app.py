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
"""ToolSandbox resources server (native gym multi-turn).

ToolSandbox is a multi-turn, tool-using benchmark. This server ports the
vendored apple/ToolSandbox (``tool_sandbox/`` next to this file) onto gym's
aviary-style env pattern so the conversation is driven *natively* — the
agent-under-test is the gym policy model driven by an agent harness, while the
**user simulator** and the **Python execution environment** run inside this
server. ``verify()`` is pure scoring.

Mapping of ToolSandbox's ``Scenario.play()`` loop:

* ``seed_session(task_idx)`` deep-copies the scenario's starting
  ``ExecutionContext``, runs the ``SYSTEM -> EXECUTION_ENVIRONMENT`` preamble
  (loads the REPL tool imports), and returns the agent-facing conversation head
  (system prompt + first user turn) plus the scenario's tools.
* ``/step`` appends the agent's output to the sandbox message log — replicating
  the vendored ``OpenAIAPIAgent`` message-append logic without calling a model
  — then runs the ``USER`` / ``EXECUTION_ENVIRONMENT`` roles until control
  returns to the agent, the conversation ends, or the per-scenario message cap
  is hit. It returns the new agent-facing messages as observations.
* ``/close`` scores the final context via ``Evaluation.evaluate`` and caches the
  milestone ``similarity`` reward (the harness closes before it verifies).
* ``/verify`` returns that cached reward. Continuous, in ``[0, 1]``.

The vendored ``ExecutionContext`` uses a process-global ``contextvars`` var as
the ambient "current context" that every role reads. To keep concurrent
episodes isolated, each ``seed_session`` / ``step`` / ``close`` runs its role
work inside its own ``asyncio.Task`` (which snapshots contextvars at creation)
and re-captures ``get_current_context()`` afterwards — because the execution
environment deep-copies and swaps the bound context object during parallel
tool-call handling.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from fastapi import FastAPI, Request
from openai import NOT_GIVEN
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.responses import FunctionToolParam
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymFunctionCallOutput
from resources_servers.toolsandbox.schemas import (
    ToolSandboxCloseRequest,
    ToolSandboxCloseResponse,
    ToolSandboxResourcesServerConfig,
    ToolSandboxSeedSessionRequest,
    ToolSandboxSeedSessionResponse,
    ToolSandboxStepRequest,
    ToolSandboxStepResponse,
    ToolSandboxVerifyRequest,
    ToolSandboxVerifyResponse,
)


# The vendored tool_sandbox package lives next to this file and uses top-level
# ``tool_sandbox.`` imports, so make its parent dir importable.
_VENDOR_DIR = os.path.dirname(os.path.abspath(__file__))
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

from tool_sandbox.common.execution_context import (  # noqa: E402
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
    get_current_context,
    set_current_context,
)
from tool_sandbox.common.message_conversion import (  # noqa: E402
    Message,
    openai_tool_call_to_python_code,
    sanitize_tool_call_id,
)
from tool_sandbox.common.scenario import Scenario  # noqa: E402
from tool_sandbox.common.tool_conversion import convert_to_openai_tools  # noqa: E402
from tool_sandbox.common.tool_discovery import ToolBackend  # noqa: E402
from tool_sandbox.roles.base_role import BaseRole  # noqa: E402
from tool_sandbox.roles.execution_environment import ExecutionEnvironment  # noqa: E402
from tool_sandbox.roles.openai_api import OpenAIRoleConfig, _sampling_kwargs  # noqa: E402
from tool_sandbox.roles.openai_api_user import OpenAIAPIUser  # noqa: E402
from tool_sandbox.scenarios import named_scenarios  # noqa: E402


# Smoke-test subset — same names upstream used with the ``--test_mode`` flag.
_TEST_SCENARIO_NAMES = [
    "send_message_with_contact_content_cellular_off_multiple_user_turn",
    "send_message_with_contact_content_cellular_off_multiple_user_turn_10_distraction_tools",
    "send_message_with_contact_content_cellular_off_3_distraction_tools_arg_description_scrambled",
]

logger = logging.getLogger(__name__)


def _agent_visible_tools(ctx: ExecutionContext) -> Dict[str, Any]:
    """The scenario tools the agent-under-test is allowed to call."""
    return {
        name: tool
        for name, tool in ctx.get_available_tools(scrambling_allowed=True).items()
        if RoleType.AGENT in getattr(tool, "visible_to", (RoleType.AGENT,))
    }


class _ServerClientUser(OpenAIAPIUser):
    """User simulator that talks to a gym model server instead of AsyncOpenAI.

    Reuses the vendored :class:`OpenAIAPIUser.respond` logic verbatim (reading
    the ambient context, building tools, appending its reply) and only replaces
    the model call with a ``server_client`` POST to the model server's
    OpenAI-compatible ``/v1/chat/completions`` endpoint.
    """

    def __init__(
        self,
        server_client: Any,
        server_name: str,
        model_name: Optional[str],
        sampling: Dict[str, Any],
    ) -> None:
        # Deliberately skip OpenAIAPIUser.__init__ (no AsyncOpenAI client).
        self._server_client = server_client
        self._server_name = server_name
        self.model_name = model_name or ""
        self._sampling = sampling

    async def teardown(self) -> None:  # no client to close
        pass

    async def _model_inference(self, openai_messages, openai_tools) -> ChatCompletion:
        sampling = dict(self._sampling)
        # vLLM reasoning toggle rides in extra_body; flatten it into the body
        # since we POST raw JSON rather than going through the OpenAI SDK.
        extra_body = sampling.pop("extra_body", None)
        req: Dict[str, Any] = {"messages": openai_messages, **sampling}
        if self.model_name:
            req["model"] = self.model_name
        if openai_tools is not NOT_GIVEN and openai_tools:
            req["tools"] = list(openai_tools)
        if extra_body:
            req.update(extra_body)
        resp = await self._server_client.post(
            server_name=self._server_name,
            url_path="/v1/chat/completions",
            json=req,
        )
        resp.raise_for_status()
        return ChatCompletion.model_validate(await resp.json())


class ToolSandboxResourcesServer(SimpleResourcesServer):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: ToolSandboxResourcesServerConfig

    # env_id -> mutable episode state (context, evaluation, roles, cap info).
    envs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # env_id -> scoring details (reward + milestone/minefield/turn breakdown),
    # cached at /close time. `reward` is the combined milestone similarity.
    scoring: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Deterministic (name, Scenario) list; built once, indexed by task_idx.
    _scenarios: Optional[List[Tuple[str, Scenario]]] = PrivateAttr(default=None)

    # ------------------------------------------------------------------
    # wiring
    # ------------------------------------------------------------------

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/step")(self.step)
        app.post("/close")(self.close)
        return app

    @property
    def scenario_index(self) -> List[Tuple[str, Scenario]]:
        """Sorted list of ``(name, Scenario)``; ``task_idx`` indexes into it."""
        if self._scenarios is None:
            backend = ToolBackend[self.config.preferred_tool_backend.upper()]
            all_scenarios = named_scenarios(preferred_tool_backend=backend)
            if self.config.test_mode:
                names = [n for n in _TEST_SCENARIO_NAMES if n in all_scenarios]
            elif self.config.scenarios:
                names = [n for n in self.config.scenarios if n in all_scenarios]
            else:
                names = sorted(all_scenarios.keys())
            self._scenarios = [(n, all_scenarios[n]) for n in sorted(names)]
            logger.info("ToolSandbox: %d scenarios available", len(self._scenarios))
        return self._scenarios

    def _make_roles(self) -> Dict[RoleType, BaseRole]:
        sampling = _sampling_kwargs(
            OpenAIRoleConfig(
                base_url="",
                model=self.config.user_model_name or "",
                temperature=self.config.user_temperature,
                top_p=self.config.user_top_p,
                max_tokens=self.config.user_max_tokens,
                enable_thinking=self.config.user_enable_thinking,
            )
        )
        user = _ServerClientUser(
            server_client=self.server_client,
            server_name=self.config.user_model_server.name,
            model_name=self.config.user_model_name,
            sampling=sampling,
        )
        return {
            RoleType.USER: user,
            RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
        }

    # ------------------------------------------------------------------
    # seed_session
    # ------------------------------------------------------------------

    async def seed_session(
        self, request: Request, body: ToolSandboxSeedSessionRequest
    ) -> ToolSandboxSeedSessionResponse:
        index = self.scenario_index
        if not 0 <= body.task_idx < len(index):
            raise ValueError(f"task_idx={body.task_idx} out of range [0, {len(index)})")
        name, scenario = index[body.task_idx]
        env_id = str(uuid.uuid4())
        ctx = copy.deepcopy(scenario.starting_context)
        max_messages = self.config.max_messages or scenario.max_messages

        # Isolated context so concurrent seeds never share the ambient var.
        ctx, baseline, obs, tools = await asyncio.create_task(self._seed_task(ctx))

        self.envs[env_id] = {
            "ctx": ctx,
            "evaluation": scenario.evaluation,
            "max_messages": max_messages,
            "baseline": baseline,
            "roles": self._make_roles(),
            "name": name,
            "categories": list(getattr(scenario, "categories", []) or []),
        }
        logger.info("ToolSandbox seed env=%s task_idx=%d scenario=%s", env_id, body.task_idx, name)
        return ToolSandboxSeedSessionResponse(env_id=env_id, scenario=name, obs=obs, tools=tools)

    async def _seed_task(
        self, ctx: ExecutionContext
    ) -> Tuple[ExecutionContext, int, List[NeMoGymEasyInputMessage], List[FunctionToolParam]]:
        set_current_context(ctx)
        # Preamble: run every SYSTEM -> EXECUTION_ENVIRONMENT message so the REPL
        # namespace is populated with the tool imports (adds no new messages).
        db = ctx.get_database(
            DatabaseNamespace.SANDBOX,
            drop_sandbox_message_index=False,
            get_all_history_snapshots=True,
        )
        exec_role = ExecutionEnvironment()
        max_idx = ctx.max_sandbox_message_index
        for i in range(max_idx + 1):
            if db["recipient"][i] == RoleType.EXECUTION_ENVIRONMENT and db["sender"][i] == RoleType.SYSTEM:
                await exec_role.respond(ending_index=i)
        ctx = get_current_context()
        baseline = ctx.max_sandbox_message_index
        obs = self._seed_obs(ctx)
        tools = self._seed_tools(ctx)
        return ctx, baseline, obs, tools

    @staticmethod
    def _seed_obs(ctx: ExecutionContext) -> List[NeMoGymEasyInputMessage]:
        """Agent-facing conversation head: system prompt + initial user turn."""
        db = (
            ctx.get_database(
                DatabaseNamespace.SANDBOX,
                get_all_history_snapshots=True,
                drop_sandbox_message_index=False,
            )
            .filter(pl.col("recipient") == RoleType.AGENT)
            .sort("sandbox_message_index")
        )
        obs: List[NeMoGymEasyInputMessage] = []
        for row in db.to_dicts():
            if row["sender"] == RoleType.SYSTEM:
                role = "system"
            elif row["sender"] == RoleType.USER:
                role = "user"
            else:
                continue
            obs.append(NeMoGymEasyInputMessage(role=role, content=row["content"]))
        return obs

    def _seed_tools(self, ctx: ExecutionContext) -> List[FunctionToolParam]:
        tools: List[FunctionToolParam] = []
        for spec in convert_to_openai_tools(_agent_visible_tools(ctx)):
            fn = spec["function"]
            params = fn.get("parameters") or {"type": "object", "properties": {}}
            if isinstance(params, dict) and not self.config.tool_schema_additional_properties:
                params.setdefault("additionalProperties", False)
            tools.append(
                FunctionToolParam(
                    type="function",
                    name=fn["name"],
                    description=fn.get("description") or "",
                    parameters=params,
                    strict=False,
                )
            )
        return tools

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    async def step(self, request: Request, body: ToolSandboxStepRequest) -> ToolSandboxStepResponse:
        st = self.envs.get(body.env_id)
        if st is None:
            raise KeyError(f"Unknown env_id {body.env_id!r}")
        obs, done = await asyncio.create_task(self._advance(body))
        return ToolSandboxStepResponse(obs=obs, reward=0.0, done=done)

    async def _advance(self, body: ToolSandboxStepRequest) -> Tuple[List[Any], bool]:
        st = self.envs[body.env_id]
        set_current_context(st["ctx"])

        # 1. Append the agent's messages (or return an error obs on bad input).
        error_obs = self._append_agent_messages(body)
        if error_obs is not None:
            st["ctx"] = get_current_context()
            return error_obs, False

        # 2. Advance internal roles until it's the agent's turn again / done.
        since_index = get_current_context().max_sandbox_message_index
        done = await self._role_loop(st)
        st["ctx"] = get_current_context()

        # 3. Collect the new agent-facing messages as observations.
        obs = self._collect_agent_obs(st["ctx"], since_index)
        return obs, done

    def _append_agent_messages(self, body: ToolSandboxStepRequest) -> Optional[List[NeMoGymFunctionCallOutput]]:
        """Replicate OpenAIAPIAgent's message append without calling a model.

        Returns ``None`` on success, or a list of error observations (leaving
        the context unchanged) when a tool call can't be converted.
        """
        ctx = get_current_context()
        if body.function_calls:
            available = set(_agent_visible_tools(ctx))
            messages: List[Message] = []
            for fc in body.function_calls:
                call_id = fc.call_id or "call"
                try:
                    exec_name = ctx.get_execution_facing_tool_name(fc.name)
                    tool_call = ChatCompletionMessageToolCall(
                        id=call_id,
                        type="function",
                        function=Function(name=fc.name, arguments=fc.arguments or "{}"),
                    )
                    code = openai_tool_call_to_python_code(tool_call, available, execution_facing_tool_name=exec_name)
                except (KeyError, ValueError, TypeError) as exc:
                    logger.info("ToolSandbox bad tool call %r: %s", fc.name, exc)
                    return [
                        NeMoGymFunctionCallOutput(
                            call_id=f.call_id or "call",
                            output=f"Error: invalid tool call ({type(exc).__name__}: {exc})",
                        )
                        for f in body.function_calls
                    ]
                messages.append(
                    Message(
                        sender=RoleType.AGENT,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=code,
                        openai_tool_call_id=sanitize_tool_call_id(call_id),
                        openai_function_name=fc.name,
                    )
                )
            BaseRole.add_messages(messages)
        else:
            # No tool calls => a natural-language reply to the user simulator.
            BaseRole.add_messages(
                [
                    Message(
                        sender=RoleType.AGENT,
                        recipient=RoleType.USER,
                        content=body.text or "",
                    )
                ]
            )
        return None

    async def _role_loop(self, st: Dict[str, Any]) -> bool:
        """Run USER / EXECUTION_ENVIRONMENT roles until the agent's turn / done.

        Returns ``True`` when the episode is over (conversation ended, message
        cap reached, or a role produced no messages), else ``False``.
        """
        cap = st["max_messages"] + st["baseline"]
        roles = st["roles"]
        while True:
            db = get_current_context().get_database(DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False)
            if not db["conversation_active"][-1]:
                return True
            if db["sandbox_message_index"][-1] >= cap:
                return True
            recipient = db["recipient"][-1]
            if recipient == RoleType.AGENT:
                return False
            if recipient not in roles:
                # SYSTEM or unexpected recipient: nothing internal to run.
                return False
            index_before = db["sandbox_message_index"][-1]
            await roles[recipient].respond()
            db_after = get_current_context().get_database(DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False)
            if db_after["sandbox_message_index"][-1] == index_before:
                logger.warning(
                    "ToolSandbox role %s produced no messages; ending episode",
                    recipient,
                )
                return True

    @staticmethod
    def _collect_agent_obs(ctx: ExecutionContext, since_index: int) -> List[Any]:
        """Messages addressed to the agent produced after ``since_index``."""
        db = (
            ctx.get_database(
                DatabaseNamespace.SANDBOX,
                get_all_history_snapshots=True,
                drop_sandbox_message_index=False,
            )
            .filter((pl.col("sandbox_message_index") > since_index) & (pl.col("recipient") == RoleType.AGENT))
            .sort("sandbox_message_index")
        )
        obs: List[Any] = []
        for row in db.to_dicts():
            if row["sender"] == RoleType.EXECUTION_ENVIRONMENT:
                obs.append(
                    NeMoGymFunctionCallOutput(
                        call_id=row["openai_tool_call_id"] or "call",
                        output=row["content"] or "",
                    )
                )
            elif row["sender"] == RoleType.USER:
                obs.append(NeMoGymEasyInputMessage(role="user", content=row["content"] or ""))
        return obs

    # ------------------------------------------------------------------
    # close / verify
    # ------------------------------------------------------------------

    async def close(self, request: Request, body: ToolSandboxCloseRequest) -> ToolSandboxCloseResponse:
        st = self.envs.pop(body.env_id, None)
        if st is not None:
            details = await self._score(st)
            self.scoring[body.env_id] = details
            logger.info(
                "ToolSandbox close env=%s scenario=%s reward=%.4f "
                "milestone=%.4f minefield=%.4f turn_count=%d exception=%s",
                body.env_id,
                details["scenario"],
                details["reward"],
                details["milestone_similarity"],
                details["minefield_similarity"],
                details["turn_count"],
                details["exception_type"],
            )
        return ToolSandboxCloseResponse(message="Success", success=True)

    async def _score(self, st: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the full milestone/minefield breakdown for a finished episode.

        Mirrors the fields the legacy BYOB CLI wrapper surfaced in
        ``scoring_details`` so downstream analysis is unchanged: ``similarity``
        (the reward), ``milestone_similarity``, ``minefield_similarity``,
        ``turn_count``, ``categories``, and ``exception_type`` (``None`` unless
        scoring raised, matching the CLI's per-scenario exception capture).
        """
        details: Dict[str, Any] = {
            "scenario": st["name"],
            "similarity": 0.0,
            "milestone_similarity": 0.0,
            "minefield_similarity": 0.0,
            "turn_count": 0,
            "categories": [str(c) for c in st["categories"]],
            "exception_type": None,
        }
        try:
            result = await asyncio.create_task(self._evaluate(st))
            details.update(
                similarity=float(result.similarity),
                milestone_similarity=float(result.milestone_similarity),
                minefield_similarity=float(result.minefield_similarity),
                turn_count=int(result.turn_count),
            )
        except Exception as exc:  # noqa: BLE001 -- scoring failures => reward 0
            logger.warning("ToolSandbox evaluate failed for %s: %s", st["name"], exc)
            details["exception_type"] = type(exc).__name__
        details["reward"] = details["similarity"]
        return details

    @staticmethod
    async def _evaluate(st: Dict[str, Any]) -> Any:
        set_current_context(st["ctx"])
        return st["evaluation"].evaluate(
            execution_context=get_current_context(),
            max_turn_count=st["max_messages"],
        )

    async def verify(self, body: ToolSandboxVerifyRequest) -> ToolSandboxVerifyResponse:
        details = self.scoring.pop(body.response.env_id, {"reward": 0.0})
        reward = float(details.get("reward", 0.0))
        # Surface the scoring breakdown as top-level response fields.
        scoring_fields = {k: v for k, v in details.items() if k != "reward"}
        return ToolSandboxVerifyResponse(**(body.model_dump() | scoring_fields), reward=reward)


if __name__ == "__main__":
    ToolSandboxResourcesServer.run_webserver()
