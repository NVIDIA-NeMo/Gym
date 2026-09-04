# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import tempfile
from asyncio import Semaphore
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import Body, Request
from nemo_fabric import Fabric, FabricConfig, RunRequest
from pydantic import ConfigDict, Field, PositiveInt

from nemo_gym.base_resources_server import NEMO_GYM_MCP_METADATA_KEY, BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import SKILLS_REF_KEY_NAME, get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


LOG = logging.getLogger(__name__)
_MODEL_API_KEY_ENV = "NEMO_GYM_FABRIC_MODEL_API_KEY"  # pragma: allowlist secret


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            text = part.get("text")
        else:
            text = getattr(part, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _extract_request_input(body_input: Any) -> tuple[Any, Optional[str]]:
    if isinstance(body_input, str):
        return body_input, None

    messages: list[dict[str, Any]] = []
    instruction_parts: list[str] = []
    for item in body_input:
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        text = _content_text(content)
        if role in {"system", "developer"}:
            if text:
                instruction_parts.append(text)
            continue
        message = item.model_dump(mode="json", exclude_none=True) if hasattr(item, "model_dump") else dict(item)
        messages.append(message)

    if len(messages) == 1 and messages[0].get("role") == "user":
        request_input: Any = _content_text(messages[0].get("content"))
    else:
        request_input = messages
    return request_input, "\n\n".join(instruction_parts) or None


def _fabric_event_items(events: list[Any], response_text: str) -> list[Any]:
    items: list[Any] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "commandExecution":
            call_id = str(event.get("id") or f"call_{uuid4().hex}")
            arguments = {"command": event.get("command")}
            if event.get("cwd"):
                arguments["cwd"] = event["cwd"]
            items.append(
                NeMoGymResponseFunctionToolCall(
                    id=f"fc_{uuid4().hex}",
                    call_id=call_id,
                    name="shell",
                    arguments=json.dumps(arguments, ensure_ascii=False, sort_keys=True),
                    status="completed",
                )
            )
            items.append(
                NeMoGymResponseFunctionCallOutput(
                    id=f"fco_{uuid4().hex}",
                    call_id=call_id,
                    output=_content_text(event.get("aggregatedOutput")),
                    status="completed",
                    type="function_call_output",
                )
            )
        elif isinstance(event.get("message"), dict) and isinstance(event["message"].get("content"), list):
            for block in event["message"]["content"]:
                if not isinstance(block, dict):
                    continue
                if block.get("name") and "input" in block:
                    call_id = str(block.get("id") or f"call_{uuid4().hex}")
                    items.append(
                        NeMoGymResponseFunctionToolCall(
                            id=f"fc_{uuid4().hex}",
                            call_id=call_id,
                            name=str(block["name"]),
                            arguments=json.dumps(block.get("input") or {}, ensure_ascii=False, sort_keys=True),
                            status="completed",
                        )
                    )
                elif block.get("tool_use_id"):
                    items.append(
                        NeMoGymResponseFunctionCallOutput(
                            id=f"fco_{uuid4().hex}",
                            call_id=str(block["tool_use_id"]),
                            output=_content_text(block.get("content")),
                            status="completed",
                            type="function_call_output",
                        )
                    )
                else:
                    text = _content_text(block.get("text"))
                    if text and text.strip() != response_text.strip():
                        items.append(
                            NeMoGymResponseOutputMessage(
                                id=f"msg_{uuid4().hex}",
                                content=[NeMoGymResponseOutputText(text=text, annotations=[])],
                                role="assistant",
                                status="completed",
                                type="message",
                            )
                        )
        elif etype == "agentMessage":
            text = _content_text(event.get("text"))
            if text and text.strip() != response_text.strip():
                items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{uuid4().hex}",
                        content=[NeMoGymResponseOutputText(text=text, annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )
    return items


def _fabric_output_items(output: dict[str, Any], response_text: str) -> list[Any]:
    messages = output.get("messages")
    items: list[Any] = []
    if not (isinstance(messages, list) and messages):
        events = output.get("events")
        if isinstance(events, list) and events:
            items.extend(_fabric_event_items(events, response_text))
    if isinstance(messages, list) and messages:
        pending: list[str] = []
        for message in messages[:-1]:
            if not isinstance(message, dict):
                continue
            role = message.get("role") or message.get("type")
            if role in {"human", "user", "system", "developer"}:
                continue
            if role == "tool":
                call_id = pending.pop(0) if pending else str(message.get("tool_call_id") or message.get("id") or "")
                items.append(
                    NeMoGymResponseFunctionCallOutput(
                        id=f"fco_{uuid4().hex}",
                        call_id=call_id,
                        output=_content_text(message.get("content")),
                        status="completed",
                        type="function_call_output",
                    )
                )
                continue
            text = _content_text(message.get("content"))
            if text:
                items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{uuid4().hex}",
                        content=[NeMoGymResponseOutputText(text=text, annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                call_id = str(tool_call.get("id") or tool_call.get("call_id") or f"call_{uuid4().hex}")
                pending.append(call_id)
                arguments = tool_call.get("args", tool_call.get("arguments", function.get("arguments")))
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)
                items.append(
                    NeMoGymResponseFunctionToolCall(
                        id=f"fc_{uuid4().hex}",
                        call_id=call_id,
                        name=str(tool_call.get("name") or function.get("name") or ""),
                        arguments=arguments,
                        status="completed",
                    )
                )
    items.append(
        NeMoGymResponseOutputMessage(
            id=f"msg_{uuid4().hex}",
            content=[NeMoGymResponseOutputText(text=response_text, annotations=[])],
            role="assistant",
            status="completed",
            type="message",
        )
    )
    return items


def _skill_paths(skills_root: Optional[str]) -> list[str]:
    if not skills_root:
        return []
    root = Path(skills_root).resolve()
    if not root.is_dir():
        raise ValueError(f"skills_ref path is not a directory: {root}")
    if (root / "SKILL.md").is_file():
        return [str(root)]
    paths = [str(path) for path in sorted(root.iterdir()) if path.is_dir() and (path / "SKILL.md").is_file()]
    if not paths:
        raise ValueError(f"skills_ref path contains no SKILL.md directories: {root}")
    return paths


def _mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_mapping"):
        value = value.to_mapping()
    return dict(value) if isinstance(value, dict) else {}


def _usage_value(usage: dict[str, Any], *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return 0


def _normalized_usage(result: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    usage = _mapping(result.get("usage"))
    if not usage:
        output_usage = _mapping(output.get("usage"))
        usage = _mapping(output_usage.get("total")) or output_usage
    details = _deep_merge(_mapping(usage.get("extensions")), _mapping(usage.get("metadata")))
    return _deep_merge(details, usage)


def _turns_used(output: dict[str, Any]) -> int:
    usage = _mapping(output.get("usage"))
    for value in (
        output.get("turns_used"),
        output.get("num_turns"),
        output.get("api_calls"),
        usage.get("api_calls"),
    ):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    messages = output.get("messages")
    if isinstance(messages, list):
        turns = sum(
            1 for message in messages if isinstance(message, dict) and message.get("role") in {"ai", "assistant"}
        )
        if turns:
            return turns
    return 1


class NeMoFabricAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    adapter_id: str
    model_provider: str = "openai-compatible"
    model: str
    model_api_key: str = "local"  # pragma: allowlist secret
    concurrency: PositiveInt = 32
    timeout: PositiveInt = 600
    system_prompt: Optional[str] = None
    cwd: Optional[str] = None
    harness_settings: dict[str, Any] = Field(default_factory=dict)
    fabric_config: dict[str, Any] = Field(default_factory=dict)


class NeMoFabricAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class NeMoFabricAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    fabric_result: dict[str, Any] = Field(default_factory=dict)


class NeMoFabricAgent(SimpleResponsesAPIAgent):
    config: NeMoFabricAgentConfig
    sem: Semaphore | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)

    def _resources_server_base_url(self) -> str:
        config = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.resources_server.name,
        )
        return self.server_client._build_server_base_url(config)

    def _rollout_mcp_servers(self, seed_response: dict[str, Any]) -> dict[str, Any]:
        metadata = seed_response.get(NEMO_GYM_MCP_METADATA_KEY)
        if not isinstance(metadata, dict):
            return {}

        name = str(metadata.get("server_name") or self.config.resources_server.name)
        url_path = str(metadata.get("url_path") or "/mcp")
        server: dict[str, Any] = {
            "transport": "streamable-http",
            "url": f"{self._resources_server_base_url().rstrip('/')}/{url_path.lstrip('/')}",
        }
        headers = metadata.get("headers")
        if isinstance(headers, dict) and headers:
            server["custom_headers"] = {str(key): str(value) for key, value in headers.items()}
        else:
            LOG.warning("MCP seed metadata for %r has no session headers", name)
        return {name: server}

    def _fabric_config(
        self,
        *,
        model_base_url: str,
        workspace: Path,
        system_prompt: Optional[str],
        mcp_servers: dict[str, Any],
        skills: list[str],
    ) -> FabricConfig:
        authored = deepcopy(self.config.fabric_config)
        generated: dict[str, Any] = {
            "metadata": {"name": self.config.name or "nemo-gym-fabric-agent"},
            "harness": {
                "adapter_id": self.config.adapter_id,
                "settings": deepcopy(self.config.harness_settings),
            },
            "runtime": {"timeout_seconds": self.config.timeout},
            "environment": {
                "provider": "local",
                "workspace": str(workspace),
                "env": {_MODEL_API_KEY_ENV: self.config.model_api_key},
            },
            "models": {
                "default": {
                    "provider": self.config.model_provider,
                    "model": self.config.model,
                    "api_key_env": _MODEL_API_KEY_ENV,
                    "base_url": (
                        model_base_url.removesuffix("/v1")
                        if self.config.model_provider == "anthropic"
                        else model_base_url
                    ),
                }
            },
        }
        merged = _deep_merge(authored, generated)

        authored_instruction = ((authored.get("instructions") or {}).get("system") or {}).get("content")
        instruction_parts = [part for part in (authored_instruction, system_prompt) if isinstance(part, str) and part]
        if instruction_parts:
            instructions = deepcopy(merged.get("instructions") or {})
            system = deepcopy(instructions.get("system") or {})
            system.update({"content": "\n\n".join(instruction_parts), "mode": "replace"})
            instructions["system"] = system
            merged["instructions"] = instructions

        static_servers = (authored.get("mcp") or {}).get("servers") or {}
        all_servers = _deep_merge(static_servers, mcp_servers)
        if all_servers:
            mcp = deepcopy(merged.get("mcp") or {})
            mcp["servers"] = all_servers
            merged["mcp"] = mcp

        static_skills = list((authored.get("skills") or {}).get("paths") or [])
        all_skills = list(dict.fromkeys([*static_skills, *skills]))
        if all_skills:
            skill_config = deepcopy(merged.get("skills") or {})
            skill_config["paths"] = all_skills
            merged["skills"] = skill_config

        return FabricConfig.from_mapping(merged)

    async def _create_response(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        mcp_servers: Optional[dict[str, Any]] = None,
        skills_path: Optional[str] = None,
        rollout_id: Optional[str] = None,
    ) -> tuple[NeMoGymResponse, dict[str, Any]]:
        request_input, input_system = _extract_request_input(body.input)
        system_parts = [part for part in (self.config.system_prompt, body.instructions, input_system) if part]
        system_prompt = "\n\n".join(system_parts) or None
        model_base_url = self.resolve_model_base_url(self.config.model_server.name, rollout_id)

        configured_workspace = Path(self.config.cwd).resolve() if self.config.cwd else None
        if configured_workspace is not None and not configured_workspace.is_dir():
            raise ValueError(f"configured cwd is not a directory: {configured_workspace}")
        workspace_context = (
            tempfile.TemporaryDirectory(prefix="nemo_gym_fabric_")
            if configured_workspace is None
            else nullcontext(str(configured_workspace))
        )
        with workspace_context as workspace:
            fabric_config = self._fabric_config(
                model_base_url=model_base_url,
                workspace=Path(workspace),
                system_prompt=system_prompt,
                mcp_servers=mcp_servers or {},
                skills=_skill_paths(skills_path),
            )
            result = await Fabric().run(
                fabric_config,
                base_dir=workspace,
                request=RunRequest(
                    input=request_input,
                    request_id=rollout_id or f"request-{uuid4().hex}",
                    context={"nemo_gym_rollout_id": rollout_id} if rollout_id else {},
                ),
            )

        result_mapping = result.to_mapping()
        if result.status != "succeeded":
            error = result.error.message if result.error is not None else "unknown Fabric failure"
            error_mapping = _mapping(result_mapping.get("error"))
            details = {k: v for k, v in error_mapping.items() if k != "message" and v not in (None, {}, [])}
            if details:
                error = f"{error} ({json.dumps(details, default=str, sort_keys=True)})"
            raise RuntimeError(f"NeMo Fabric invocation failed: {error}")

        output = _mapping(result.output)
        usage = _normalized_usage(result_mapping, output)
        input_tokens = _usage_value(usage, "input_tokens", "prompt_tokens", "inputTokens")
        output_tokens = _usage_value(usage, "output_tokens", "completion_tokens", "outputTokens")
        cached_tokens = _usage_value(usage, "cached_input_tokens", "cache_read_input_tokens", "cachedInputTokens")
        cache_write_tokens = _usage_value(
            usage, "cache_write_input_tokens", "cache_creation_input_tokens", "cacheWriteInputTokens"
        )
        reasoning_tokens = _usage_value(usage, "reasoning_tokens", "reasoning_output_tokens", "reasoningOutputTokens")
        total_tokens = _usage_value(usage, "total_tokens", "totalTokens") or input_tokens + output_tokens
        response_text = output.get("response", output.get("output"))
        if not isinstance(response_text, str):
            response_text = (
                "" if response_text is None else json.dumps(response_text, ensure_ascii=False, sort_keys=True)
            )
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=str(output.get("model") or self.config.model),
            object="response",
            output=_fabric_output_items(output, response_text),
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(
                    cache_write_tokens=cache_write_tokens,
                    cached_tokens=cached_tokens,
                ),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=reasoning_tokens),
                total_tokens=total_tokens,
            ),
        )
        return response, json.loads(json.dumps(result_mapping, default=str))

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        rollout_id = request.path_params.get("rollout_id")
        response, _ = await self._create_response(body, rollout_id=rollout_id)
        return response

    async def run(self, request: Request, body: NeMoFabricAgentRunRequest) -> NeMoFabricAgentVerifyResponse:
        if self.sem is None:  # pragma: no cover
            raise RuntimeError("NeMo Fabric agent concurrency control is not initialized")
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
            seed_json = await get_response_json(seed_response)

            skills_path = ((body.model_extra or {}).get(SKILLS_REF_KEY_NAME) or {}).get("path")
            agent_response, fabric_result = await self._create_response(
                body.responses_create_params,
                mcp_servers=self._rollout_mcp_servers(seed_json),
                skills_path=skills_path,
                rollout_id=self.rollout_id_from_run(body),
            )
            agent_json = agent_response.model_dump(mode="json")

            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_json},
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            verify_json = await get_response_json(verify_response)
            fabric_output = _mapping(fabric_result.get("output"))
            finished_naturally = not fabric_output.get("failed", False)
            if "completed" in fabric_output:
                finished_naturally = bool(fabric_output["completed"])
            return NeMoFabricAgentVerifyResponse.model_validate(
                verify_json
                | {
                    "turns_used": _turns_used(fabric_output),
                    "finished_naturally": finished_naturally,
                    "fabric_result": fabric_result,
                }
            )


if __name__ == "__main__":
    NeMoFabricAgent.run_webserver()
