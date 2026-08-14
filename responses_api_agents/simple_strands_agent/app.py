# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import shutil
import signal
from asyncio import Semaphore
from contextlib import suppress
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.simple_strands_agent.setup_ssa import ensure_ssa


LOG = logging.getLogger(__name__)


class SimpleStrandsAgentError(RuntimeError):
    pass


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_text(item) for item in content)
    if not isinstance(content, dict):
        return str(content)
    if "text" in content:
        return str(content["text"] or "")
    if "json" in content:
        return json.dumps(content["json"], ensure_ascii=False)
    return ""


def _extract_instruction(body_input) -> tuple[str, Optional[str]]:
    user_message = ""
    system_messages = []
    for item in body_input:
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        text = _content_text(content)
        if role in {"system", "developer"}:
            system_messages.append(text)
        elif role == "user" and text:
            user_message = text
    return user_message, "\n\n".join(system_messages) or None


def _tool_output(block: dict[str, Any]) -> str:
    content = block.get("content") or []
    return "\n".join(text for item in content if (text := _content_text(item)))


def trajectory_to_output_items(messages: list[dict[str, Any]]) -> list[Any]:
    output_items: list[Any] = []
    item_index = 0
    for message in messages:
        role = message.get("role")
        content = message.get("content") or []
        if role == "assistant":
            for block in content:
                if "reasoningContent" in block:
                    reasoning = block["reasoningContent"].get("reasoningText") or {}
                    text = reasoning.get("text") or ""
                    if text:
                        output_items.append(
                            NeMoGymResponseReasoningItem(
                                id=f"reasoning-{item_index}",
                                summary=[NeMoGymSummary(text=text, type="summary_text")],
                                encrypted_content=reasoning.get("signature"),
                            )
                        )
                        item_index += 1
                elif "text" in block and block["text"]:
                    output_items.append(
                        NeMoGymResponseOutputMessage(
                            id=f"message-{item_index}",
                            content=[NeMoGymResponseOutputText(text=block["text"], annotations=[])],
                        )
                    )
                    item_index += 1
                elif "toolUse" in block:
                    tool = block["toolUse"]
                    call_id = str(tool.get("toolUseId") or f"call-{item_index}")
                    output_items.append(
                        NeMoGymResponseFunctionToolCall(
                            id=call_id,
                            call_id=call_id,
                            name=str(tool.get("name") or ""),
                            arguments=json.dumps(tool.get("input") or {}, ensure_ascii=False),
                            status="completed",
                        )
                    )
                    item_index += 1
        elif role == "user":
            for block in content:
                tool_result = block.get("toolResult")
                if not tool_result:
                    continue
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        call_id=str(tool_result.get("toolUseId") or ""),
                        output=_tool_output(tool_result),
                        status="completed" if tool_result.get("status") != "error" else "incomplete",
                    )
                )
    return output_items


class SimpleStrandsAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    model: Optional[str] = None
    concurrency: int = 8
    max_turns: int = 30
    max_output_tokens: int = 4096
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    timeout: int = 900
    model_timeout: int = 600
    shell_timeout: int = 120
    conversation_window: int = 300
    tools: list[str] = Field(default_factory=lambda: ["bash", "str_replace_editor"])
    prompt_tag: str = "swe_generic_v2"
    native_user_prompt: bool = False
    system_prompt: Optional[str] = None
    workspace_root: str = "outputs/simple_strands_agent/workspaces"
    keep_workspaces: bool = False
    ssa_source_root: Optional[str] = None
    ssa_python: Optional[str] = None


class SimpleStrandsAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SimpleStrandsAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class SimpleStrandsAgent(SimpleResponsesAPIAgent):
    config: SimpleStrandsAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _ssa_python: Path = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        self._ssa_python = ensure_ssa(self.config.ssa_source_root, self.config.ssa_python)

    def _workspace(self) -> Path:
        root = Path(self.config.workspace_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        work_dir = root / f"ssa-{uuid4().hex[:8]}"
        work_dir.mkdir(parents=True)
        return work_dir

    async def _run_ssa(self, payload: dict[str, Any]) -> dict[str, Any]:
        work_dir = Path(payload["work_dir"])
        request_path = work_dir / "request.json"
        result_path = work_dir / "result.json"
        request_path.write_text(json.dumps(payload))
        runner = Path(__file__).with_name("ssa_runner.py")
        process = await asyncio.create_subprocess_exec(
            str(self._ssa_python),
            str(runner),
            str(request_path),
            str(result_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.config.timeout)
        except asyncio.TimeoutError as error:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.communicate(), timeout=10)
            except asyncio.TimeoutError:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.communicate()
            raise SimpleStrandsAgentError(f"SSA timed out after {self.config.timeout}s") from error

        result = json.loads(result_path.read_text()) if result_path.is_file() else {}
        if process.returncode != 0 or result.get("error"):
            detail = result.get("error") or stderr.decode(errors="replace")[-1000:]
            if stdout:
                LOG.debug("SSA stdout: %s", stdout.decode(errors="replace")[-1000:])
            raise SimpleStrandsAgentError(f"SSA failed: {detail}")
        return result

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]
        instruction, input_system = _extract_instruction(body.input)
        system_parts = [self.config.system_prompt, body.instructions, input_system]
        system_prompt = "\n\n".join(part for part in system_parts if part) or None
        rollout_id = request.path_params.get("rollout_id") if request is not None else None
        work_dir = self._workspace()
        model_name = self.config.model or str(body.model or self.config.model_server.name)
        reasoning_effort = (body.reasoning or {}).get("effort") or self.config.reasoning_effort
        payload = {
            "instruction": instruction,
            "system_prompt": system_prompt,
            "model": model_name,
            "model_base_url": self.resolve_model_base_url(self.config.model_server.name, rollout_id),
            "work_dir": str(work_dir),
            "output_dir": str(work_dir / "output"),
            "rollout_id": rollout_id or uuid4().hex,
            "max_turns": self.config.max_turns,
            "max_output_tokens": body.max_output_tokens or self.config.max_output_tokens,
            "temperature": body.temperature if body.temperature is not None else self.config.temperature,
            "reasoning_effort": reasoning_effort,
            "model_timeout": self.config.model_timeout,
            "shell_timeout": self.config.shell_timeout,
            "conversation_window": self.config.conversation_window,
            "tools": self.config.tools,
            "prompt_tag": self.config.prompt_tag,
            "native_user_prompt": self.config.native_user_prompt,
        }
        try:
            result = await self._run_ssa(payload)
        finally:
            if not self.config.keep_workspaces:
                shutil.rmtree(work_dir, ignore_errors=True)

        output_items = trajectory_to_output_items(result.get("messages") or [])
        if not any(getattr(item, "type", None) == "message" for item in output_items):
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"message-{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                )
            )
        usage = result.get("usage") or {}
        input_tokens = int(usage.get("inputTokens") or 0)
        output_tokens = int(usage.get("outputTokens") or 0)
        cached_tokens = int(usage.get("cacheReadInputTokens") or 0)
        reasoning_tokens = int(usage.get("reasoningTokens") or 0)
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
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached_tokens),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=reasoning_tokens),
                total_tokens=input_tokens + output_tokens,
            ),
        )

    async def run(self, request: Request, body: SimpleStrandsAgentRunRequest) -> SimpleStrandsAgentVerifyResponse:
        async with self.sem:
            cookies = request.cookies
            seed_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_response)
            cookies = seed_response.cookies
            agent_response = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_response)
            cookies = agent_response.cookies
            response_json = await get_response_json(agent_response)
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": response_json},
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            verify_json = await get_response_json(verify_response)
            response = NeMoGymResponse.model_validate(response_json)
            turns = sum(1 for item in response.output if getattr(item, "type", None) == "message")
            last = response.output[-1] if response.output else None
            return SimpleStrandsAgentVerifyResponse.model_validate(
                verify_json
                | {
                    "turns_used": turns,
                    "finished_naturally": getattr(last, "type", None) == "message",
                }
            )


if __name__ == "__main__":
    SimpleStrandsAgent.run_webserver()
