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
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Request, Response
from pydantic import ConfigDict, Field, model_validator

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputMessageForTraining,
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
from responses_api_agents.browser_agent.trajectory_writer import (
    append_debug_step,
    finalize_debug_trajectory,
    init_debug_trajectory,
)


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
    max_steps: int = 250
    run_timeout_seconds: float = 7200.0
    viewport_width: int = 1280
    viewport_height: int = 720

    cua_debug_trajectories: bool = False
    cua_debug_output_dir: str = str(PARENT_DIR / "results" / "cua_debug_trajectories")

    @model_validator(mode="after")
    def _resolve_debug_output_dir(self) -> "BrowserAgentConfig":
        p = Path(self.cua_debug_output_dir)
        if not p.is_absolute():
            self.cua_debug_output_dir = str(PARENT_DIR / p)
        return self


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
    """Construct a CUANeMoGymResponse with all required Response fields.

    When trajectory steps contain RL token ID data (prompt_token_ids,
    generation_token_ids, generation_log_probs), the output message uses
    NeMoGymResponseOutputMessageForTraining with concatenated token arrays
    across all steps.
    """
    response_id = f"cua_{uuid.uuid4().hex[:24]}"
    final_text = trajectory.final_message or "Task completed"

    all_prompt_ids: list[int] = []
    all_gen_ids: list[int] = []
    all_gen_logprobs: list[float] = []
    for step in trajectory.steps:
        all_prompt_ids.extend(step.prompt_token_ids)
        all_gen_ids.extend(step.generation_token_ids)
        all_gen_logprobs.extend(step.generation_log_probs)

    msg_kwargs = {
        "id": f"msg_{uuid.uuid4().hex[:16]}",
        "content": [NeMoGymResponseOutputText(annotations=[], text=final_text)],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }

    if all_gen_ids:
        output_message = NeMoGymResponseOutputMessageForTraining(
            **msg_kwargs,
            prompt_token_ids=all_prompt_ids,
            generation_token_ids=all_gen_ids,
            generation_log_probs=all_gen_logprobs,
        )
    else:
        output_message = NeMoGymResponseOutputMessage(**msg_kwargs)

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
        self,
        cookie_jar: Dict[str, Any],
        viewport_width: Optional[int] = None,
        viewport_height: Optional[int] = None,
    ) -> BaseCUAAdapter:
        """Create a CUA adapter for any provider.

        All API calls are routed through the model server (stateless proxy).
        The adapter owns context management and action mapping; the model
        server handles authentication, retries, and provider API transport.

        cookie_jar is a mutable ``{"cookies": ...}`` dict shared with the
        model-server callers so that session cookies propagate across calls.
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
            kwargs["api_caller"] = self._make_openai_model_server_caller(cookie_jar)

        elif adapter_type in ("anthropic_sonnet", "anthropic_opus"):
            kwargs["is_opus"] = self.config.cua_is_opus
            kwargs["max_tokens"] = self.config.cua_max_tokens
            kwargs["turns_to_keep"] = self.config.cua_turns_to_keep
            kwargs["screenshot_turn_limit"] = self.config.cua_screenshot_turn_limit
            kwargs["effort_level"] = self.config.cua_effort_level
            kwargs["api_caller"] = self._make_anthropic_model_server_caller(cookie_jar)

        elif adapter_type == "gemini":
            kwargs["max_conversation_turns"] = self.config.cua_max_conversation_turns
            kwargs["api_caller"] = self._make_gemini_model_server_caller(cookie_jar)

        return AdapterFactory.create(adapter_type, **kwargs)

    def _make_openai_model_server_caller(self, cookie_jar: Dict[str, Any]):
        """Create an async callable that routes OpenAI API calls through the model server.

        The cookie_jar is a mutable dict with a "cookies" key that is read and
        updated after every call so that session affinity is maintained.
        """

        async def caller(api_params: Dict[str, Any]):
            resp = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=api_params,
                cookies=cookie_jar["cookies"],
            )
            cookie_jar["cookies"] = resp.cookies
            if not resp.ok:
                err_body = await resp.content.read()
                logger.error("OpenAI model server error (status=%s): %s", resp.status, err_body)
            await raise_for_status(resp)
            return await get_response_json(resp)

        return caller

    def _make_anthropic_model_server_caller(self, cookie_jar: Dict[str, Any]):
        """Create an async callable that routes Anthropic API calls through the model server.

        The cookie_jar is a mutable dict with a "cookies" key that is read and
        updated after every call so that session affinity is maintained.
        """

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
                cookies=cookie_jar["cookies"],
            )
            cookie_jar["cookies"] = resp.cookies
            if not resp.ok:
                err_body = await resp.content.read()
                logger.error("Anthropic model server error (status=%s): %s", resp.status, err_body)
            await raise_for_status(resp)
            return await get_response_json(resp)

        return caller

    def _make_gemini_model_server_caller(self, cookie_jar: Dict[str, Any]):
        """Create an async callable that routes Gemini API calls through the model server.

        The cookie_jar is a mutable dict with a "cookies" key that is read and
        updated after every call so that session affinity is maintained.
        """

        async def caller(api_params: Dict[str, Any]):
            proxy_body = {
                "contents": api_params["contents"],
                "config": api_params.get("config", {}),
            }

            resp = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=proxy_body,
                cookies=cookie_jar["cookies"],
            )
            cookie_jar["cookies"] = resp.cookies
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
        self,
        task_prompt: str,
        env_id: str,
        screenshot_b64: str,
        cookie_jar: Dict[str, Any],
        viewport_w: int = 1280,
        viewport_h: int = 720,
    ) -> tuple[CUATrajectory, Optional[str], Optional[Any], Optional[Path]]:
        """Run the CUA loop through the provider adapter.

        All providers (OpenAI, Anthropic, Gemini) follow the same path:
        adapter handles context management and action mapping, routing
        API calls through the model server.

        cookie_jar is a mutable ``{"cookies": ...}`` dict that is read and
        updated after every downstream ``server_client.post`` call so that
        session cookies propagate correctly.

        Returns (trajectory, local_storage_dump, usage, debug_rollout_dir).
        debug_rollout_dir is non-None only when debug trajectories are enabled
        and the directory was successfully created.
        """
        adapter = self._create_adapter(cookie_jar, viewport_width=viewport_w, viewport_height=viewport_h)
        debug_enabled = self.config.cua_debug_trajectories

        cumulative_input_tokens = 0
        cumulative_output_tokens = 0
        debug_rollout_dir = None

        try:
            loop_start = time.time()
            trajectory = CUATrajectory(steps=[], task_prompt=task_prompt, initial_screenshot=screenshot_b64)

            try:
                adapter_resp = await adapter.initialize(task_prompt, screenshot_b64)
            except Exception as init_err:
                logger.error("[CUA %s] Adapter initialize failed: %s — returning empty trajectory", env_id, init_err)
                return trajectory, "", None, None

            logger.info(
                "[CUA %s] initialize done — actions=%d done=%s",
                env_id,
                len(adapter_resp.actions),
                adapter_resp.done,
            )

            if adapter_resp.usage:
                cumulative_input_tokens += adapter_resp.usage.input_tokens
                cumulative_output_tokens += adapter_resp.usage.output_tokens

            if debug_enabled:
                try:
                    debug_rollout_dir = init_debug_trajectory(
                        output_dir=self.config.cua_debug_output_dir,
                        env_id=env_id,
                        initial_screenshot=screenshot_b64,
                        task_prompt=task_prompt,
                        adapter_type=self.config.cua_adapter_type,
                        model_name=self.config.cua_model,
                    )
                except Exception as e:
                    logger.warning("[CUA %s] failed to init debug trajectory: %s", env_id, e)

            step_count = 0
            action_index = 0
            step_data = None
            consecutive_failures = 0
            max_consecutive_failures = 3
            browser_crashed = False

            while not adapter_resp.done and step_count < self.config.max_steps:
                if browser_crashed:
                    break
                elapsed = time.time() - loop_start
                if elapsed >= self.config.run_timeout_seconds:
                    logger.warning(
                        "[CUA %s] hard timeout reached (%.0fs >= %.0fs) after %d steps — ending loop",
                        env_id,
                        elapsed,
                        self.config.run_timeout_seconds,
                        step_count,
                    )
                    break
                last_action_error = None
                for action_i, action in enumerate(adapter_resp.actions):
                    try:
                        step_resp_raw = await self.server_client.post(
                            server_name=self.config.resources_server.name,
                            url_path="/step",
                            json={"env_id": env_id, "action": action.model_dump()},
                            cookies=cookie_jar["cookies"],
                        )
                        cookie_jar["cookies"] = step_resp_raw.cookies
                        await raise_for_status(step_resp_raw)
                        step_data = CUAStepResponse.model_validate(await get_response_json(step_resp_raw))

                        if step_data.error:
                            last_action_error = step_data.error
                            logger.warning(
                                "[CUA %s] action %s returned error: %s",
                                env_id,
                                action.action_type,
                                step_data.error,
                            )

                        if not step_data.screenshot or step_data.current_url == "error:browser_stuck":
                            logger.error(
                                "[CUA %s] browser stuck (empty screenshot returned) — closing session",
                                env_id,
                            )
                            browser_crashed = True
                            break

                        consecutive_failures = 0
                        action_index += 1
                        is_first_action = action_i == 0
                        trajectory.steps.append(
                            CUAStep(
                                action=action,
                                screenshot_before=screenshot_b64,
                                screenshot_after=step_data.screenshot,
                                current_url=step_data.current_url,
                                raw_provider_response=adapter_resp.raw_response if is_first_action else None,
                                prompt_token_ids=adapter_resp.prompt_token_ids if is_first_action else [],
                                generation_token_ids=adapter_resp.generation_token_ids if is_first_action else [],
                                generation_log_probs=adapter_resp.generation_log_probs if is_first_action else [],
                            )
                        )

                        if debug_enabled and debug_rollout_dir:
                            try:
                                append_debug_step(
                                    rollout_dir=debug_rollout_dir,
                                    step_idx=action_index,
                                    action=action,
                                    screenshot_after=step_data.screenshot,
                                    current_url=step_data.current_url,
                                    raw_provider_response=adapter_resp.raw_response,
                                )
                            except Exception as e:
                                logger.warning("[CUA %s] failed to append debug step %d: %s", env_id, action_index, e)

                        screenshot_b64 = step_data.screenshot
                    except Exception as step_err:
                        consecutive_failures += 1
                        logger.warning(
                            "[CUA %s] action %s failed (%d/%d consecutive): %s",
                            env_id,
                            action.action_type,
                            consecutive_failures,
                            max_consecutive_failures,
                            step_err,
                        )
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error(
                                "[CUA %s] %d consecutive action failures — browser appears crashed, ending loop",
                                env_id,
                                consecutive_failures,
                            )
                            browser_crashed = True
                            break

                current_url = step_data.current_url if step_data else "about:blank"
                step_count += 1
                total_elapsed = time.time() - loop_start

                logger.info(
                    "[CUA %s] step %d/%d (total %.1fs) — actions=%d url=%s",
                    env_id,
                    step_count,
                    self.config.max_steps,
                    total_elapsed,
                    len(adapter_resp.actions),
                    current_url[:80] if current_url else "?",
                )

                if adapter_resp.done or browser_crashed:
                    break
                last_url = current_url
                try:
                    adapter_resp = await adapter.step(
                        screenshot_b64, action_result=last_url, action_error=last_action_error
                    )
                    if adapter_resp.usage:
                        cumulative_input_tokens += adapter_resp.usage.input_tokens
                        cumulative_output_tokens += adapter_resp.usage.output_tokens
                except Exception as adapter_err:
                    logger.error("[CUA %s] Adapter step failed: %s — ending loop", env_id, adapter_err)
                    break

            total_elapsed = time.time() - loop_start
            logger.info(
                "[CUA %s] loop finished — steps=%d total_time=%.1fs input_tokens=%d output_tokens=%d",
                env_id,
                step_count,
                total_elapsed,
                cumulative_input_tokens,
                cumulative_output_tokens,
            )

            if adapter_resp.message:
                trajectory.final_message = adapter_resp.message

            local_storage_dump = ""
            if browser_crashed:
                logger.warning("[CUA %s] skipping dump_local_storage — browser crashed", env_id)
            else:
                try:
                    ls_resp = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/dump_local_storage",
                        json={"env_id": env_id},
                        cookies=cookie_jar["cookies"],
                    )
                    cookie_jar["cookies"] = ls_resp.cookies
                    await raise_for_status(ls_resp)
                    ls_data = CUADumpLocalStorageResponse.model_validate(await get_response_json(ls_resp))
                    local_storage_dump = ls_data.local_storage_dump
                except Exception as ls_err:
                    logger.warning("[CUA %s] dump_local_storage failed: %s", env_id, ls_err)

            adapter_usage = None
            if cumulative_input_tokens > 0 or cumulative_output_tokens > 0:
                adapter_usage = NeMoGymResponseUsage(
                    input_tokens=cumulative_input_tokens,
                    output_tokens=cumulative_output_tokens,
                    total_tokens=cumulative_input_tokens + cumulative_output_tokens,
                    input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                    output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                )

            return trajectory, local_storage_dump, adapter_usage, debug_rollout_dir
        finally:
            adapter.reset()

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    async def _do_responses(
        self, body: BrowserAgentRunRequest, cookie_jar: Dict[str, Any]
    ) -> tuple[CUANeMoGymResponse, Optional[Path]]:
        """Core CUA loop shared by both ``responses()`` and ``run()``.

        Accepts a mutable *cookie_jar* (``{"cookies": ...}``) that is read
        and updated after every downstream call.

        Returns (response, debug_rollout_dir).
        """
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
                cookies=cookie_jar["cookies"],
            )
            cookie_jar["cookies"] = seed_resp.cookies
            await raise_for_status(seed_resp)
            seed_data = CUASeedSessionResponse.model_validate(await get_response_json(seed_resp))
            env_id = seed_data.env_id
            screenshot_b64 = seed_data.screenshot

            trajectory, local_storage_dump, usage, debug_rollout_dir = await self._responses_via_adapter(
                task_prompt, env_id, screenshot_b64, cookie_jar, viewport_w, viewport_h
            )

        finally:
            if env_id:
                try:
                    await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/close",
                        json={"env_id": env_id},
                        cookies=cookie_jar["cookies"],
                    )
                except Exception as e:
                    logger.warning(f"Error closing browser session: {e}")

        nemo_resp = _build_nemo_response(trajectory, env_id, local_storage_dump, self.config.cua_model, usage)
        return nemo_resp, debug_rollout_dir

    async def responses(
        self, request: Request, response: Response, body: BrowserAgentRunRequest
    ) -> CUANeMoGymResponse:
        cookie_jar: Dict[str, Any] = {"cookies": request.cookies}

        result, _debug_dir = await self._do_responses(body, cookie_jar)

        for k, v in cookie_jar["cookies"].items():
            response.set_cookie(k, v)
        return result

    async def run(self, request: Request, body: BrowserAgentRunRequest) -> CUAVerifyResponse:
        cookie_jar: Dict[str, Any] = {"cookies": request.cookies}
        try:
            cua_response, debug_rollout_dir = await self._do_responses(body, cookie_jar)

            verify_request = CUAVerifyRequest.model_validate(body.model_dump() | {"response": cua_response})
            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
                cookies=cookie_jar["cookies"],
            )
            cookie_jar["cookies"] = verify_resp.cookies
            await raise_for_status(verify_resp)
            result = CUAVerifyResponse.model_validate(await get_response_json(verify_resp))

            if self.config.cua_debug_trajectories and debug_rollout_dir:
                try:
                    finalize_debug_trajectory(
                        rollout_dir=debug_rollout_dir,
                        env_id=cua_response.env_id,
                        trajectory=cua_response.trajectory,
                        reward=result.reward,
                        local_storage_dump=cua_response.local_storage_dump,
                        adapter_type=self.config.cua_adapter_type,
                        model_name=self.config.cua_model,
                        verifier_metadata=body.verifier_metadata,
                        verification_result=getattr(result, "verification_result", None),
                    )
                except Exception as e:
                    logger.warning(f"Failed to finalize debug trajectory: {e}")

            return result
        except Exception:
            logger.exception("Error in run")
            raise


if __name__ == "__main__":
    BrowserAgent.run_webserver()
