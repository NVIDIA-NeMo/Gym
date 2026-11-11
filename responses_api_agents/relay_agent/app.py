# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runs a task on a vendor-hosted agent through Relay and verifies the files that the run delivers.

The vendor runs the agent loop, web search and code sandbox on its own servers. This agent only sends the prompt,
waits for the background run and collects the output files, as the WANDR paper runs did.

Relay is WANDR's harness for hosted endpoints (https://github.com/perplexityai/wandr, ``agents/relay``,
Apache-2.0). It is installed at the pinned commit in requirements.txt and imported, not copied. Only the
required-files paragraph of the prompt is restated here, from ``agents/relay/agent.py`` (``_required_files_instruction``)
at commit ccb0baeb96f1c77a48e47f92122c57479ee99700, because that module imports Harbor.
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path, PurePosixPath
from time import time
from typing import Any, Callable, Literal, Optional, Sequence

from fastapi import Request
from pydantic import ConfigDict, Field, PrivateAttr
from relay.core import ProducedFile, Relay, RelayEvent, WorkspaceSnapshot
from relay.providers.openai.endpoint import OpenAIResponsesEndpoint

from nemo_gym.agents import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, ResourcesServerRef
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse


logger = logging.getLogger(__name__)
LOGGED_EVENT_FIELDS = ("response_id", "status", "restart", "restart_reason", "restart_message", "count", "error")


class RelayAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    # OpenAI Responses model name at base_url, e.g. openai/openai/gpt-5.5 on the NVIDIA inference API.
    model: str
    base_url: str = "https://inference-api.nvidia.com/v1"
    api_key: Optional[str] = None
    # Relay's OpenAI request fields: reasoning_effort, reasoning_summary, search_context_size, max_tool_calls,
    # max_output_tokens. Upstream configs/wandr.yaml sets reasoning_effort high.
    request: dict[str, Any] = Field(default_factory=lambda: {"reasoning_effort": "high"})
    # sandbox: the model writes the files in its code interpreter container, and Relay downloads them.
    delivery_channel: Literal["sandbox", "stdout", "output"] = "sandbox"
    # Upstream: max_full_restarts 2 and override_timeout_sec 57600 (16 h) per task.
    max_full_restarts: int = Field(default=2, ge=0)
    full_restart_initial_delay_s: float = Field(default=5.0, ge=0.0)
    timeout_s: float = Field(default=57600.0, gt=0.0)
    max_concurrent_runs: int = Field(default=24, ge=1)
    # When set, each run writes its prompt, final text, usage and raw vendor output here as one JSON file.
    trace_dir: Optional[str] = None


class RelayAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    required_output_files: list[str] = Field(default_factory=list)


class RelayAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class RelayAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def required_files_instruction(paths: Sequence[str]) -> Optional[str]:
    """Relay's required-files paragraph (agents/relay/agent.py at ccb0baeb), so the prompt matches the paper runs."""
    if not paths:
        return None
    files = "\n".join(f"- `{path}`" for path in paths)
    return (
        "Required final files. The run is only complete if all of these exact "
        "relative paths are delivered with valid task output. Do not create "
        "empty placeholder files for required outputs; an empty required file "
        f"is not a completed deliverable:\n{files}"
    )


def last_user_text(params: NeMoGymResponseCreateParamsNonStreaming) -> str:
    """The task instruction: the last user message. System messages are for tool-calling agents and are dropped."""
    if isinstance(params.input, str):
        return params.input
    for item in reversed(params.input):
        role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
        if role != "user":
            continue
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        if isinstance(content, str):
            return content
        parts = [part.get("text") if isinstance(part, dict) else getattr(part, "text", None) for part in content]
        return "".join(part for part in parts if part)
    raise ValueError("responses_create_params.input has no user message")


def lifecycle_logger(task: str) -> Callable[[RelayEvent], None]:
    """Relay observer that logs run status (submitted, polls, restarts, files) but not the vendor's tool traffic."""

    def observe(event: RelayEvent) -> None:
        if event.semantic_type != "lifecycle":
            return
        fields = {key: event.metadata[key] for key in LOGGED_EVENT_FIELDS if event.metadata.get(key) is not None}
        # Printed, not logged: the server process has no INFO handler, and this is the only view of a run of hours.
        print(f"[relay {task}] {event.name} {event.message or ''} {fields}", flush=True)

    return observe


class CollectedFiles:
    """Relay's workspace: empty, like /workspace in the WANDR task image, and it keeps the delivered files."""

    root = PurePosixPath("/workspace")

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    async def snapshot(self) -> WorkspaceSnapshot:
        return WorkspaceSnapshot(root=self.root, tree="")

    async def materialize(self, files: Sequence[ProducedFile]) -> None:
        for file in files:
            self.files[file.path.as_posix()] = file.content.decode("utf-8", errors="replace")


class RelayAgent(SimpleResponsesAPIAgent):
    config: RelayAgentConfig
    _semaphore: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_runs)

    def make_endpoint(self) -> OpenAIResponsesEndpoint:
        return OpenAIResponsesEndpoint(
            model_name=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            request=self.config.request,
            delivery_channel=self.config.delivery_channel,
        )

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError("relay_agent runs whole tasks through /run only")

    async def run(self, request: Request, body: RelayAgentRunRequest) -> RelayAgentVerifyResponse:
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=request.cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies

        workspace = CollectedFiles()
        relay = Relay(
            endpoint=self.make_endpoint(),
            workspace=workspace,
            observer=lifecycle_logger(str((body.model_extra or {}).get("task_name", "task"))),
            extra_instruction=required_files_instruction(body.required_output_files),
            required_file_paths=body.required_output_files,
            max_full_restarts=self.config.max_full_restarts,
            full_restart_initial_delay_sec=self.config.full_restart_initial_delay_s,
            full_restart_max_delay_sec=max(60.0, self.config.full_restart_initial_delay_s),
        )
        instruction = last_user_text(body.responses_create_params)
        started = time()
        result, error = None, None
        async with self._semaphore:
            try:
                result = await asyncio.wait_for(relay.run(instruction), timeout=self.config.timeout_s)
            except Exception as exc:  # The run boundary: a failed run is scored on whatever files it delivered.
                error = f"{type(exc).__name__}: {exc}"
                logger.warning("Relay run failed: %s", error)
        endpoint_result = result.endpoint_result if result is not None else None
        text = endpoint_result.text if endpoint_result is not None else ""

        response = NeMoGymResponse(
            id=(endpoint_result.response_id if endpoint_result else None) or f"relay_{uuid.uuid4().hex}",
            created_at=int(started),
            model=self.config.model,
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid.uuid4().hex}", content=[NeMoGymResponseOutputText(text=text)]
                )
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        relay_fields = {
            "relay_error": error,
            "relay_duration_s": time() - started,
            "relay_usage": endpoint_result.usage if endpoint_result else {},
            "relay_cost_usd": endpoint_result.cost_usd if endpoint_result else None,
        }
        if self.config.trace_dir:
            self._write_trace(body, response.id, endpoint_result, workspace, relay_fields)

        verify_request = RelayAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": response.model_dump(), "output_files": workspace.files}
        )
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return RelayAgentVerifyResponse.model_validate(await get_response_json(verify_response) | relay_fields)

    def _write_trace(
        self,
        body: RelayAgentRunRequest,
        response_id: str,
        endpoint_result: Any,
        workspace: CollectedFiles,
        relay_fields: dict[str, Any],
    ) -> None:
        trace = {
            "task_name": (body.model_extra or {}).get("task_name"),
            "response_id": response_id,
            "text": endpoint_result.text if endpoint_result else "",
            "raw": endpoint_result.raw if endpoint_result else {},
            "files": {name: len(text) for name, text in workspace.files.items()},
            **relay_fields,
        }
        directory = Path(self.config.trace_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{trace['task_name'] or 'task'}_{uuid.uuid4().hex[:12]}.json").write_text(
            json.dumps(trace, default=str)
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    RelayAgent.run_webserver()
