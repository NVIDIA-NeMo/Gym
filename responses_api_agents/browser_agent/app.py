# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
import uuid
from typing import Any, Dict, Optional

from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.browser_gym.schemas import (
    CUADumpLocalStorageResponse,
    CUANeMoGymResponse,
    CUASeedSessionResponse,
    CUAStep,
    CUAStepResponse,
    CUATrajectory,
    CUAVerifyRequest,
    CUAVerifyResponse,
)
from responses_api_agents.browser_agent.adapters import AdapterFactory
from responses_api_agents.browser_agent.adapters.base import BaseCUAAdapter
from responses_api_agents.browser_agent.trajectory_writer import save_debug_trajectory


logger = logging.getLogger(__name__)


class BrowserAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: Optional[ModelServerRef] = None

    cua_adapter_type: str = "openai"
    cua_model: str = "computer-use-preview"
    cua_is_opus: bool = False
    cua_effort_level: str = "high"
    cua_max_tokens: int = 4096
    cua_turns_to_keep: int = 8
    cua_screenshot_turn_limit: int = 8
    cua_max_conversation_turns: int = 8
    max_steps: int = 50
    viewport_width: int = 1280
    viewport_height: int = 720

    cua_debug_trajectories: bool = False
    cua_debug_output_dir: str = "/tmp/cua_debug_trajectories"


class BrowserAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )
    verifier_metadata: Optional[dict] = None


def _build_nemo_response(
    trajectory: CUATrajectory,
    env_id: str,
    local_storage_dump: Optional[str],
    model_name: str,
    usage: Optional[Any] = None,
) -> CUANeMoGymResponse:
    """Construct a CUANeMoGymResponse with all required Response fields."""
    response_id = f"cua_{uuid.uuid4().hex[:24]}"

    final_text = trajectory.final_message or "Task completed"
    output_message = NeMoGymResponseOutputMessage(
        id=f"msg_{uuid.uuid4().hex[:16]}",
        content=[NeMoGymResponseOutputText(annotations=[], text=final_text)],
        role="assistant",
        status="completed",
        type="message",
    )

    return CUANeMoGymResponse(
        id=response_id,
        created_at=int(time.time()),
        model=model_name,
        object="response",
        output=[output_message],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=usage,
        env_id=env_id,
        trajectory=trajectory,
        local_storage_dump=local_storage_dump,
    )


class BrowserAgent(SimpleResponsesAPIAgent):
    config: BrowserAgentConfig

    def _create_adapter(
        self, viewport_width: Optional[int] = None, viewport_height: Optional[int] = None
    ) -> BaseCUAAdapter:
        """Create a CUA adapter for any provider.

        All API calls are routed through the model server (stateless proxy).
        The adapter owns context management and action mapping; the model
        server handles authentication, retries, and provider API transport.
        """
        if not self._uses_model_server():
            raise ValueError(
                f"model_server must be configured for adapter '{self.config.cua_adapter_type}'. "
                "All provider API calls must route through a model server."
            )

        adapter_type = self.config.cua_adapter_type
        kwargs = {
            "model": self.config.cua_model,
            "viewport_width": viewport_width or self.config.viewport_width,
            "viewport_height": viewport_height or self.config.viewport_height,
        }

        if adapter_type == "openai":
            kwargs["api_caller"] = self._make_openai_model_server_caller()

        elif adapter_type in ("anthropic_sonnet", "anthropic_opus"):
            kwargs["is_opus"] = self.config.cua_is_opus
            kwargs["max_tokens"] = self.config.cua_max_tokens
            kwargs["turns_to_keep"] = self.config.cua_turns_to_keep
            kwargs["screenshot_turn_limit"] = self.config.cua_screenshot_turn_limit
            kwargs["effort_level"] = self.config.cua_effort_level
            kwargs["api_caller"] = self._make_anthropic_model_server_caller()

        elif adapter_type == "gemini":
            kwargs["max_conversation_turns"] = self.config.cua_max_conversation_turns
            kwargs["api_caller"] = self._make_gemini_model_server_caller()

        return AdapterFactory.create(adapter_type, **kwargs)

    def _make_openai_model_server_caller(self):
        """Create an async callable that routes OpenAI API calls through the model server."""

        async def caller(api_params: Dict[str, Any]):
            resp = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=api_params,
            )
            if not resp.ok:
                err_body = await resp.content.read()
                logger.error("OpenAI model server error (status=%s): %s", resp.status, err_body)
            await raise_for_status(resp)
            return await get_response_json(resp)

        return caller

    def _make_anthropic_model_server_caller(self):
        """Create an async callable that routes Anthropic API calls through the model server."""

        async def caller(api_params: Dict[str, Any]):
            proxy_body = {
                "messages": api_params["messages"],
                "system": api_params.get("system", ""),
                "tools": api_params.get("tools", []),
                "betas": api_params.get("betas", []),
                "max_tokens": api_params.get("max_tokens", 4096),
            }
            if api_params.get("output_config"):
                proxy_body["output_config"] = api_params["output_config"]

            resp = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=proxy_body,
            )
            if not resp.ok:
                err_body = await resp.content.read()
                logger.error("Anthropic model server error (status=%s): %s", resp.status, err_body)
            await raise_for_status(resp)
            return await get_response_json(resp)

        return caller

    def _make_gemini_model_server_caller(self):
        """Create an async callable that routes Gemini API calls through the model server."""

        async def caller(api_params: Dict[str, Any]):
            proxy_body = {
                "contents": api_params["contents"],
                "config": api_params.get("config", {}),
            }

            resp = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=proxy_body,
            )
            if not resp.ok:
                err_body = await resp.content.read()
                logger.error("Gemini model server error (status=%s): %s", resp.status, err_body)
            await raise_for_status(resp)
            return await get_response_json(resp)

        return caller

    def _uses_model_server(self) -> bool:
        return self.config.model_server is not None

    # ──────────────────────────────────────────────────────────────
    # Unified CUA loop -- all providers go through their adapter
    # ──────────────────────────────────────────────────────────────

    async def _responses_via_adapter(
        self, task_prompt: str, env_id: str, screenshot_b64: str, viewport_w: int = 1280, viewport_h: int = 720
    ) -> tuple[CUATrajectory, Optional[str], Optional[Any]]:
        """Run the CUA loop through the provider adapter.

        All providers (OpenAI, Anthropic, Gemini) follow the same path:
        adapter handles context management and action mapping, routing
        API calls through the model server.
        """
        adapter = self._create_adapter(viewport_width=viewport_w, viewport_height=viewport_h)

        cumulative_input_tokens = 0
        cumulative_output_tokens = 0

        try:
            adapter_resp = await adapter.initialize(task_prompt, screenshot_b64)

            if adapter_resp.usage:
                cumulative_input_tokens += adapter_resp.usage.input_tokens
                cumulative_output_tokens += adapter_resp.usage.output_tokens

            trajectory = CUATrajectory(steps=[], task_prompt=task_prompt, initial_screenshot=screenshot_b64)
            step_count = 0
            step_data = None

            while not adapter_resp.done and step_count < self.config.max_steps:
                for action in adapter_resp.actions:
                    try:
                        step_resp_raw = await self.server_client.post(
                            server_name=self.config.resources_server.name,
                            url_path="/step",
                            json={"env_id": env_id, "action": action.model_dump()},
                        )
                        await raise_for_status(step_resp_raw)
                        step_data = CUAStepResponse.model_validate(await get_response_json(step_resp_raw))

                        trajectory.steps.append(
                            CUAStep(
                                action=action,
                                screenshot_before=screenshot_b64,
                                screenshot_after=step_data.screenshot,
                                current_url=step_data.current_url,
                                raw_provider_response=adapter_resp.raw_response,
                            )
                        )
                        screenshot_b64 = step_data.screenshot
                    except Exception as step_err:
                        logger.warning(
                            "Step action failed (action=%s): %s — continuing with current screenshot",
                            action.action_type,
                            step_err,
                        )

                step_count += 1
                if adapter_resp.done:
                    break
                last_url = step_data.current_url if step_data else "about:blank"
                try:
                    adapter_resp = await adapter.step(screenshot_b64, action_result=last_url)
                    if adapter_resp.usage:
                        cumulative_input_tokens += adapter_resp.usage.input_tokens
                        cumulative_output_tokens += adapter_resp.usage.output_tokens
                except Exception as adapter_err:
                    logger.error("Adapter step failed: %s — ending loop", adapter_err)
                    break

            if adapter_resp.message:
                trajectory.final_message = adapter_resp.message

            ls_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/dump_local_storage",
                json={"env_id": env_id},
            )
            await raise_for_status(ls_resp)
            ls_data = CUADumpLocalStorageResponse.model_validate(await get_response_json(ls_resp))

            adapter_usage = None
            if cumulative_input_tokens > 0 or cumulative_output_tokens > 0:
                adapter_usage = NeMoGymResponseUsage(
                    input_tokens=cumulative_input_tokens,
                    output_tokens=cumulative_output_tokens,
                    total_tokens=cumulative_input_tokens + cumulative_output_tokens,
                    input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                    output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                )

            return trajectory, ls_data.local_storage_dump, adapter_usage
        finally:
            adapter.reset()

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    async def responses(self, body: BrowserAgentRunRequest) -> CUANeMoGymResponse:
        body = body.model_copy(deep=True)
        rcp = body.responses_create_params

        task_prompt = ""
        if isinstance(rcp.input, str):
            task_prompt = rcp.input
        elif isinstance(rcp.input, list):
            for msg in rcp.input:
                if hasattr(msg, "role") and msg.role == "user":
                    content = msg.content if hasattr(msg, "content") else ""
                    if isinstance(content, str):
                        task_prompt = content
                    elif isinstance(content, list):
                        for part in content:
                            if hasattr(part, "text"):
                                task_prompt = part.text
                                break
                            elif isinstance(part, dict) and part.get("type") == "input_text":
                                task_prompt = part.get("text", "")
                                break
                    break

        verifier_metadata = body.verifier_metadata or {}
        start_url = verifier_metadata.get("start_url", "about:blank")
        vm_viewport = verifier_metadata.get("viewport", {})
        viewport_w = vm_viewport.get("width", self.config.viewport_width)
        viewport_h = vm_viewport.get("height", self.config.viewport_height)

        env_id = None
        usage = None

        try:
            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json={"start_url": start_url, "viewport_width": viewport_w, "viewport_height": viewport_h},
            )
            await raise_for_status(seed_resp)
            seed_data = CUASeedSessionResponse.model_validate(await get_response_json(seed_resp))
            env_id = seed_data.env_id
            screenshot_b64 = seed_data.screenshot

            trajectory, local_storage_dump, usage = await self._responses_via_adapter(
                task_prompt, env_id, screenshot_b64, viewport_w, viewport_h
            )

        finally:
            if env_id:
                try:
                    await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/close",
                        json={"env_id": env_id},
                    )
                except Exception as e:
                    logger.warning(f"Error closing browser session: {e}")

        return _build_nemo_response(trajectory, env_id, local_storage_dump, self.config.cua_model, usage)

    async def run(self, body: BrowserAgentRunRequest) -> CUAVerifyResponse:
        try:
            response = await self.responses(body)

            verify_request = CUAVerifyRequest.model_validate(body.model_dump() | {"response": response})
            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
            )
            await raise_for_status(verify_resp)
            result = CUAVerifyResponse.model_validate(await get_response_json(verify_resp))

            if self.config.cua_debug_trajectories:
                try:
                    save_debug_trajectory(
                        output_dir=self.config.cua_debug_output_dir,
                        env_id=response.env_id,
                        trajectory=response.trajectory,
                        reward=result.reward,
                        local_storage_dump=response.local_storage_dump,
                        adapter_type=self.config.cua_adapter_type,
                        model_name=self.config.cua_model,
                        verifier_metadata=body.verifier_metadata,
                        verification_result=getattr(result, "verification_result", None),
                    )
                except Exception as e:
                    logger.warning(f"Failed to save debug trajectory: {e}")

            return result
        except Exception:
            logger.exception("Error in run")
            raise


if __name__ == "__main__":
    BrowserAgent.run_webserver()
