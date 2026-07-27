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
import asyncio
import atexit
import logging
import os
import shutil
import sys
import tempfile
from asyncio import Semaphore
from time import time
from typing import Any, Optional
from uuid import uuid4

import model_tools  # noqa: F401  # fail-fast if hermes-agent isn't installed  # pyright: ignore[reportMissingImports]
from fastapi import Request
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


def _trajectory_to_output_items(messages, n_input):
    output_items = []
    for item in messages[n_input:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "") or ""
        if isinstance(content, list):
            content = "".join(c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "") for c in content)
        if role == "assistant":
            output_items.append(
                NeMoGymResponseOutputMessageForTraining(
                    id=f"msg-{len(output_items)}",
                    content=[NeMoGymResponseOutputText(type="output_text", text=content, annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                    prompt_token_ids=item.get("prompt_token_ids") or [],
                    generation_token_ids=item.get("generation_token_ids") or [],
                    generation_log_probs=item.get("generation_log_probs") or [],
                    routed_experts=item.get("routed_experts"),
                )
            )
            for tc in item.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if not fn:
                    continue
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=fn.get("arguments", ""),
                        call_id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        type="function_call",
                        id=tc.get("id"),
                        status="completed",
                    )
                )
        elif role == "tool":
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=item.get("tool_call_id", ""),
                    output=content,
                    status="completed",
                )
            )
    return output_items


LOG = logging.getLogger(__name__)


# if ray close sys.stderr mid-request, write to the original fd
class _SafeStderrHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = sys.__stderr__
            if stream is None:
                return
            stream.write(msg + "\n")
            stream.flush()
        except Exception:
            pass


if not LOG.handlers:
    LOG.addHandler(_SafeStderrHandler(level=logging.WARNING))


def _split_input_to_user_and_history(input_items) -> tuple[str, list[dict], Optional[str]]:
    items = list(input_items)
    system_message: Optional[str] = None
    if items:
        first = items[0]
        first_role = getattr(first, "role", None) or (first.get("role") if isinstance(first, dict) else None)
        first_content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
        if first_role == "system":
            if isinstance(first_content, list):
                first_content = "".join(
                    (p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in first_content
                )
            system_message = first_content or ""
            items = items[1:]

    user_message = ""
    history: list[dict] = []
    for idx, item in enumerate(items):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        if isinstance(content, list):
            content = "".join((p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content)
        content = content or ""
        if idx == len(items) - 1 and role == "user":
            user_message = content
        else:
            history.append({"role": role, "content": content})
    return user_message, history, system_message


class HermesAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    concurrency: int = 32
    max_turns: int = 90
    enabled_toolsets: Optional[list[str]] = None
    disabled_toolsets: Optional[list[str]] = None
    temperature: float | None = None
    terminal_backend: str = "local"
    terminal_timeout: int = 180
    system_prompt: Optional[str] = None
    compression_enabled: bool = True
    compression_threshold: float = 0.85
    delegation_max_iterations: int = 50
    checkpoints_enabled: bool = False


class HermesAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class HermesAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class HermesAgent(SimpleResponsesAPIAgent):
    config: HermesAgentConfig
    sem: Semaphore = None
    # Set of agents currently running run_conversation, plus a flag tracking whether the single
    # shared SIGTERM dispatcher has been installed on the event loop. See _ensure_sigterm_handler.
    active_agents: set = None
    sigterm_installed: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _ensure_sigterm_handler(self) -> None:
        """Install exactly one SIGTERM handler on the event loop that interrupts *every* in-flight
        agent. Registering a fresh per-call handler is unsafe under concurrency: add_signal_handler
        replaces the previous handler, so concurrent responses() calls clobber each other and the
        first to finish removes the only remaining handler — leaving later SIGTERMs unhandled and
        their trajectories lost. A single dispatcher over `active_agents` avoids that race."""
        if self.sigterm_installed:
            return
        import signal

        def _dispatch():
            for ag in list(self.active_agents):
                if hasattr(ag, "interrupt"):
                    ag.interrupt("timeout")

        try:
            asyncio.get_event_loop().add_signal_handler(signal.SIGTERM, _dispatch)
            self.sigterm_installed = True
        except (NotImplementedError, OSError):
            pass  # not supported on this platform (e.g. Windows, non-main thread)

    def _build_config(self) -> str:
        import yaml

        config: dict[str, Any] = {
            "model": str(self.config.model_server.name),
            "provider": "auto",
            "toolsets": ["hermes-cli"],
            "agent": {"max_turns": self.config.max_turns},
            "memory": {
                "memory_enabled": False,
                "user_profile_enabled": False,
            },
            "compression": {
                "enabled": self.config.compression_enabled,
                "threshold": self.config.compression_threshold,
            },
            "terminal": {
                "backend": self.config.terminal_backend,
                "timeout": self.config.terminal_timeout,
            },
            "delegation": {
                "max_iterations": self.config.delegation_max_iterations,
            },
            "checkpoints": {
                "enabled": self.config.checkpoints_enabled,
            },
        }
        return yaml.dump(config, default_flow_style=False)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        self.active_agents = set()
        # hermes-agent reads these from env (cli.py / batch_runner.py); env vars are
        # process-global, so multiple HermesAgent instances in one process share them
        os.environ["TERMINAL_ENV"] = self.config.terminal_backend
        os.environ["TERMINAL_TIMEOUT"] = str(self.config.terminal_timeout)

        # Build config.yaml with config parameters
        hermes_home = tempfile.mkdtemp(prefix="hermes_agent_")
        atexit.register(shutil.rmtree, hermes_home, True)
        with open(os.path.join(hermes_home, "config.yaml"), "w") as _f:
            _f.write(self._build_config())
        os.environ["HERMES_HOME"] = hermes_home

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        from run_agent import AIAgent  # from hermes-agent on path  # pyright: ignore[reportMissingImports]

        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, history, input_system = _split_input_to_user_and_history(body.input)
        system_message = self.config.system_prompt or input_system

        # A prefixed self-call carries the rollout id into the model-server base URL.
        rollout_id = request.path_params.get("rollout_id") if request is not None else None
        base_url = self.resolve_model_base_url(self.config.model_server.name, rollout_id)
        model_name = str(self.config.model_server.name)

        agent = AIAgent(
            base_url=base_url,
            api_key="gym",  # pragma: allowlist secret
            model=model_name,
            use_streaming=False,
            temperature=self.config.temperature,
            insert_reasoning=True,
            max_iterations=self.config.max_turns,
            enabled_toolsets=self.config.enabled_toolsets,
            disabled_toolsets=self.config.disabled_toolsets,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            persist_session=False,
            save_trajectories=False,
        )
        _original_build_api_kwargs = agent._build_api_kwargs

        def _patched_build_api_kwargs(api_messages):
            kw = _original_build_api_kwargs(api_messages)
            ctk = kw.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})
            ctk.setdefault("enable_thinking", True)
            ctk["truncate_history_thinking"] = False
            return kw

        agent._build_api_kwargs = _patched_build_api_kwargs

        # Interrupt the agent cleanly on SIGTERM so run_conversation returns with partial messages
        # instead of being killed mid-turn (which would leave response.json unwritten). A single
        # shared dispatcher interrupts every in-flight agent; we just register this one in the set.
        self._ensure_sigterm_handler()
        self.active_agents.add(agent)

        try:
            result = await asyncio.to_thread(
                agent.run_conversation,
                user_message,
                system_message,
                history,
            )
        finally:
            self.active_agents.discard(agent)

        messages = result.get("messages") or []
        # aiagent omits system from returned messages
        n_input = len(history) + 1

        output_items = _trajectory_to_output_items(messages, n_input)

        has_assistant_message = any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        )
        if not has_assistant_message:
            LOG.warning(
                "Hermes agent ended without an assistant message. Padding empty assistant message. This should not happen often, investigate: error=%r",
                result.get("error"),
            )
            last_valid = next(
                (
                    m
                    for m in reversed(messages)
                    if isinstance(m, dict) and m.get("role") == "assistant" and m.get("generation_token_ids")
                ),
                None,
            )
            pti = last_valid["prompt_token_ids"] if last_valid else [0]
            gti = last_valid["generation_token_ids"] if last_valid else [0]
            glp = (last_valid.get("generation_log_probs") if last_valid else None) or [0.0]
            routed_experts = last_valid.get("routed_experts") if last_valid else None
            output_items.append(
                NeMoGymResponseOutputMessageForTraining(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text=result.get("error") or "", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                    prompt_token_ids=pti,
                    generation_token_ids=gti,
                    generation_log_probs=glp,
                    routed_experts=routed_experts,
                )
            )

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
            output=output_items,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=0,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=0,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=0,
            ),
        )

    async def run(self, request: Request, body: HermesAgentRunRequest) -> HermesAgentVerifyResponse:
        async with self.sem:
            cookies = request.cookies

            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = seed_resp.cookies

            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_resp_json},
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            gym_resp = NeMoGymResponse.model_validate(agent_resp_json)
            turns = sum(
                1
                for item in gym_resp.output
                if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            )
            last = gym_resp.output[-1] if gym_resp.output else None
            naturally = getattr(last, "type", None) == "message" and getattr(last, "role", None) == "assistant"

            return HermesAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    HermesAgent.run_webserver()
