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
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymComputerCallOutput,
    NeMoGymComputerToolCall,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from resources_servers.browser_gym.schemas import (
    BrowserAction,
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
    cua_api_key: str = ""
    cua_org: Optional[str] = None
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


# ──────────────────────────────────────────────────────────────────────
# OpenAI action → BrowserAction mapping (shared with the adapter)
# ──────────────────────────────────────────────────────────────────────


def _get_coord(action: Dict[str, Any]) -> Optional[List[int]]:
    """Extract coordinates -- handles both x/y and coordinate formats."""
    if "x" in action and "y" in action:
        return [int(action["x"]), int(action["y"])]
    if "coordinate" in action:
        return action["coordinate"]
    return None


def _normalize_drag_path(path: Any) -> Optional[List[List[int]]]:
    """Convert drag path (list of {"x":..,"y":..} dicts) to list of [x,y] arrays."""
    if not path or not isinstance(path, list):
        return None
    result = []
    for point in path:
        if isinstance(point, dict) and "x" in point and "y" in point:
            result.append([int(point["x"]), int(point["y"])])
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            result.append([int(point[0]), int(point[1])])
    return result if result else None


def _map_openai_action(action: Dict[str, Any]) -> Optional[BrowserAction]:
    """Map an OpenAI computer_call action dict to a unified BrowserAction."""
    action_type = action.get("type", "")
    coord = _get_coord(action)

    if action_type == "click":
        return BrowserAction(action_type="click", coordinate=coord, button=action.get("button", "left"))
    elif action_type == "double_click":
        return BrowserAction(action_type="double_click", coordinate=coord)
    elif action_type == "triple_click":
        return BrowserAction(action_type="triple_click", coordinate=coord)
    elif action_type == "drag":
        normalized_path = _normalize_drag_path(action.get("path"))
        start, end = None, None
        if normalized_path and len(normalized_path) >= 2:
            start, end = normalized_path[0], normalized_path[-1]
        else:
            start = _get_coord(action) or (
                [int(action["start_x"]), int(action["start_y"])]
                if "start_x" in action and "start_y" in action
                else action.get("start_coordinate")
            )
            end = (
                [int(action["destination_x"]), int(action["destination_y"])]
                if "destination_x" in action and "destination_y" in action
                else action.get("destination_coordinate") or action.get("target_coordinate")
            )
        return BrowserAction(action_type="drag", start_coordinate=start, end_coordinate=end, path=normalized_path)
    elif action_type == "keypress":
        keys = action.get("keys", [])
        if not keys and "key" in action:
            keys = [action["key"]]
        return BrowserAction(action_type="keypress", keys=keys)
    elif action_type == "type":
        return BrowserAction(action_type="type", text=action.get("text", ""))
    elif action_type == "scroll":
        return BrowserAction(
            action_type="scroll", coordinate=coord, scroll_x=action.get("scroll_x"), scroll_y=action.get("scroll_y")
        )
    elif action_type in ("move", "mouse_move", "hover"):
        return BrowserAction(action_type="hover", coordinate=coord)
    elif action_type == "screenshot":
        return BrowserAction(action_type="screenshot")
    elif action_type == "wait":
        duration = action.get("ms") or action.get("duration") or 1000
        return BrowserAction(action_type="wait", duration=int(duration))
    elif action_type == "goto":
        return BrowserAction(action_type="goto", url=action.get("url"))
    elif action_type == "new_tab":
        return BrowserAction(action_type="new_tab", url=action.get("url"))
    elif action_type == "close_tab":
        return BrowserAction(action_type="close_tab")
    elif action_type == "switch_tab":
        return BrowserAction(action_type="switch_tab", tab_index=action.get("tab_index"))
    elif action_type == "zoom":
        return BrowserAction(action_type="zoom", region=action.get("region"))
    elif action_type == "list_tabs":
        return BrowserAction(action_type="screenshot")
    else:
        logger.warning(f"Unknown OpenAI action type: {action_type}")
        return None


class BrowserAgent(SimpleResponsesAPIAgent):
    config: BrowserAgentConfig

    def _create_adapter(
        self, viewport_width: Optional[int] = None, viewport_height: Optional[int] = None
    ) -> BaseCUAAdapter:
        """Create a CUA adapter.

        For Anthropic with a model server configured, injects an api_caller that
        routes API calls through the model server (stateless proxy) while the
        adapter still owns all context management.
        """
        adapter_type = self.config.cua_adapter_type
        kwargs = {
            "api_key": self.config.cua_api_key,
            "model": self.config.cua_model,
            "viewport_width": viewport_width or self.config.viewport_width,
            "viewport_height": viewport_height or self.config.viewport_height,
        }

        if adapter_type in ("anthropic_sonnet", "anthropic_opus"):
            kwargs["is_opus"] = self.config.cua_is_opus
            kwargs["max_tokens"] = self.config.cua_max_tokens
            kwargs["turns_to_keep"] = self.config.cua_turns_to_keep
            kwargs["screenshot_turn_limit"] = self.config.cua_screenshot_turn_limit
            kwargs["effort_level"] = self.config.cua_effort_level

            if self._uses_model_server():
                kwargs["api_caller"] = self._make_anthropic_model_server_caller()
                kwargs["api_key"] = ""

        elif adapter_type == "gemini":
            kwargs["max_conversation_turns"] = self.config.cua_max_conversation_turns

        return AdapterFactory.create(adapter_type, **kwargs)

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

    def _uses_model_server(self) -> bool:
        return self.config.model_server is not None

    # ──────────────────────────────────────────────────────────────
    # Model-server path (OpenAI only) -- context managed server-side
    # ──────────────────────────────────────────────────────────────

    async def _responses_via_model_server(
        self, task_prompt: str, env_id: str, screenshot_b64: str, viewport_w: int, viewport_h: int
    ) -> tuple[CUATrajectory, Optional[str], Optional[Any]]:
        """Run the CUA loop through the NeMo-Gym model server."""
        trajectory = CUATrajectory(steps=[], task_prompt=task_prompt, initial_screenshot=screenshot_b64)
        cumulative_usage = None

        cua_tools = [
            {
                "type": "computer_use_preview",
                "display_width": viewport_w,
                "display_height": viewport_h,
                "environment": "browser",
            }
        ]

        initial_input = [
            NeMoGymEasyInputMessage(
                role="user",
                content=[
                    {"type": "input_text", "text": task_prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}", "detail": "auto"},
                ],
            )
        ]

        request_params = NeMoGymResponseCreateParamsNonStreaming(
            input=initial_input,
            tools=cua_tools,
            truncation="auto",
            reasoning={"summary": "auto"},
        )

        model_resp_raw = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=request_params,
        )
        if not model_resp_raw.ok:
            err_body = await model_resp_raw.content.read()
            logger.error("Model server error on initial CUA request (status=%s): %s", model_resp_raw.status, err_body)
        await raise_for_status(model_resp_raw)
        model_response = NeMoGymResponse.model_validate(await get_response_json(model_resp_raw))

        if model_response.usage:
            cumulative_usage = model_response.usage

        step_count = 0

        while step_count < self.config.max_steps:
            computer_calls: List[NeMoGymComputerToolCall] = []
            final_message: Optional[str] = None

            for item in model_response.output:
                if isinstance(item, NeMoGymComputerToolCall):
                    computer_calls.append(item)
                elif isinstance(item, NeMoGymResponseOutputMessage):
                    for block in item.content:
                        if hasattr(block, "text"):
                            final_message = block.text

            if not computer_calls:
                if final_message:
                    trajectory.final_message = final_message
                break

            followup_items = []

            for call in computer_calls:
                action_dict = call.action if isinstance(call.action, dict) else call.action
                browser_action = _map_openai_action(action_dict)
                if not browser_action:
                    continue

                step_resp_raw = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path="/step",
                    json={"env_id": env_id, "action": browser_action.model_dump()},
                )
                await raise_for_status(step_resp_raw)
                step_data = CUAStepResponse.model_validate(await get_response_json(step_resp_raw))

                trajectory.steps.append(
                    CUAStep(
                        action=browser_action,
                        screenshot_before=screenshot_b64,
                        screenshot_after=step_data.screenshot,
                        current_url=step_data.current_url,
                        raw_provider_response=model_response.model_dump(),
                    )
                )
                screenshot_b64 = step_data.screenshot

                followup_items.append(
                    NeMoGymComputerCallOutput(
                        type="computer_call_output",
                        call_id=call.call_id,
                        output={
                            "type": "computer_screenshot",
                            "image_url": f"data:image/png;base64,{screenshot_b64}",
                        },
                    )
                )

            step_count += 1

            request_params = NeMoGymResponseCreateParamsNonStreaming(
                input=followup_items,
                tools=cua_tools,
                truncation="auto",
                reasoning={"summary": "auto"},
                previous_response_id=model_response.id,
            )

            model_resp_raw = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=request_params,
            )
            if not model_resp_raw.ok:
                err_body = await model_resp_raw.content.read()
                logger.error(
                    "Model server error on follow-up CUA request step=%d (status=%s): %s",
                    step_count,
                    model_resp_raw.status,
                    err_body,
                )
            await raise_for_status(model_resp_raw)
            model_response = NeMoGymResponse.model_validate(await get_response_json(model_resp_raw))

            if model_response.usage and cumulative_usage:
                cumulative_usage.input_tokens += model_response.usage.input_tokens
                cumulative_usage.output_tokens += model_response.usage.output_tokens
                cumulative_usage.total_tokens += model_response.usage.total_tokens

        ls_resp = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/dump_local_storage",
            json={"env_id": env_id},
        )
        await raise_for_status(ls_resp)
        ls_data = CUADumpLocalStorageResponse.model_validate(await get_response_json(ls_resp))

        return trajectory, ls_data.local_storage_dump, cumulative_usage

    # ──────────────────────────────────────────────────────────────
    # Adapter path (Anthropic, Gemini) -- context managed agent-side
    # Anthropic with model server: adapter manages context, API call
    # routes through model server via injected api_caller.
    # ──────────────────────────────────────────────────────────────

    async def _responses_via_adapter(
        self, task_prompt: str, env_id: str, screenshot_b64: str, viewport_w: int = 1280, viewport_h: int = 720
    ) -> tuple[CUATrajectory, Optional[str]]:
        """Run the CUA loop through a direct-API adapter (Anthropic/Gemini)."""
        adapter = self._create_adapter(viewport_width=viewport_w, viewport_height=viewport_h)

        try:
            adapter_resp = await adapter.initialize(task_prompt, screenshot_b64)

            trajectory = CUATrajectory(steps=[], task_prompt=task_prompt, initial_screenshot=screenshot_b64)
            step_count = 0

            while not adapter_resp.done and step_count < self.config.max_steps:
                for action in adapter_resp.actions:
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

                step_count += 1
                if adapter_resp.done:
                    break
                adapter_resp = await adapter.step(screenshot_b64)

            if adapter_resp.message:
                trajectory.final_message = adapter_resp.message

            ls_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/dump_local_storage",
                json={"env_id": env_id},
            )
            await raise_for_status(ls_resp)
            ls_data = CUADumpLocalStorageResponse.model_validate(await get_response_json(ls_resp))

            return trajectory, ls_data.local_storage_dump
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

            is_anthropic = self.config.cua_adapter_type in ("anthropic_sonnet", "anthropic_opus")

            if self._uses_model_server() and not is_anthropic:
                trajectory, local_storage_dump, usage = await self._responses_via_model_server(
                    task_prompt, env_id, screenshot_b64, viewport_w, viewport_h
                )
            else:
                trajectory, local_storage_dump = await self._responses_via_adapter(
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
