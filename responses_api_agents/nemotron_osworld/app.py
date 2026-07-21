# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Nemotron-Omni OSWorld agent — the validated Omni-Nano-v3 harness over the OSWorld
resources server.

``run()`` wires the three servers together: ``/seed_session`` (allocate + official task
setup) -> self ``/v1/responses`` (the agent loop) -> ``/verify`` (official evaluator +
release), threading the session cookie through every call and forwarding the faithful
OSWorld ``action_history`` to the verifier.

``responses()`` is the host-side observe -> act loop, driven by the REFERENCE agent —
``mm_agents.nvidia.nemotron_agent.NemotronAgent`` imported from the pinned ``osworld``
fork (prompt template, message/history serialization, free-text ``## Action/## Code``
parsing, 0-1 fractional coordinate projection, 5-retry parse recovery, and the internal
force-``FAIL`` at the step cap are all the vendor-validated code, not a port). Per step it
fetches ``/screenshot`` from the resources server, calls ``agent.predict`` in a worker
thread (the reference ``call_llm`` is synchronous httpx with its own strict
``finish_reason == "stop"`` retry contract — pointed at the gym MODEL SERVER's
``/v1/chat/completions`` via ``VLLM_API_ENDPOINT``), and executes the returned pyautogui
snippets via ``/execute`` with the exact ``PythonController`` wrapping.
"""

import asyncio
import base64
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import Request, Response
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, get_server_url, raise_for_status


logger = logging.getLogger("nemo_gym.osworld.nemotron_agent")

# Mirrors desktop_env.controllers.python.PythonController's pkgs_prefix + command shape:
# actions run as `python -c "<prefix + action>"` with shell=False in the guest.
_PYAUTOGUI_PKGS_PREFIX = "import pyautogui; import time; pyautogui.FAILSAFE = False; {command}"
_SPECIAL_ACTIONS = ("WAIT", "FAIL", "DONE")


class NemotronOSWorldAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    # The model id sent in the chat-completions payload (the model server's served name).
    model_name: str = "vllm_local"

    # Reference-run knobs (Omni-Nano-v3 47-48% runs; see the fork's reference launch scripts).
    max_steps: int = 100
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 4096
    max_image_history_length: int = 3
    thinking: bool = True
    coordinate_type: str = "relative"
    client_password: str = "password"
    screen_width: int = 1920
    screen_height: int = 1080
    # The reference runner's env.step pause (sleep_after_execution); WAIT sleeps the same.
    sleep_after_execution_s: float = 5.0
    # Debug aid: when set, save each step's observed screenshot + the parsed actions under
    # <dir>/<task_id8>/ so failing trajectories can be replayed visually. OFF for real gates.
    debug_trajectory_dir: str = ""


class NemotronOSWorldRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class NemotronOSWorldVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class NemotronOSWorldVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def _extract_instruction(response_input) -> str:
    """Pull the task instruction text out of the incoming Responses-API input."""
    if isinstance(response_input, str):
        return response_input
    for message in response_input:
        role = getattr(message, "role", None) or (message.get("role") if isinstance(message, dict) else None)
        if role != "user":
            continue
        content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for part in content:
                text = getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else None)
                if text:
                    texts.append(text)
            if texts:
                return "\n".join(texts)
    return ""


def _message_to_text(message: Any) -> str:
    """Flatten the reference agent's returned message (or error string) into rollout text."""
    if isinstance(message, dict):
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        content = message.get("content") or ""
        if reasoning:
            return f"<think>{reasoning}</think>\n{content}"
        return str(content)
    return str(message)


class NemotronOSWorldAgent(SimpleResponsesAPIAgent):
    config: NemotronOSWorldAgentConfig
    # Faithful OSWorld action history per rollout session; run() forwards it to /verify
    # (the evaluator inspects the LAST entry for the infeasible/FAIL contract). Keyed by
    # the session id as seen INSIDE the cookie-threaded requests, so run() retrieves it
    # through the cookie-scoped /action_history endpoint rather than its own session.
    session_id_to_action_history: Dict[str, List[str]] = Field(default_factory=dict)

    def setup_webserver(self):
        app = super().setup_webserver()
        app.post("/action_history")(self.action_history)
        return app

    async def action_history(self, request: Request) -> Dict[str, List[str]]:
        session_id = request.session[SESSION_ID_KEY]
        return {"action_history": self.session_id_to_action_history.pop(session_id, [])}

    def _ensure_reference_llm_endpoint(self) -> None:
        """Point the reference agent's own LLM client at the gym model server.

        ``NemotronAgent.call_llm`` reads ``VLLM_API_ENDPOINT``/``VLLM_API_KEY`` and speaks
        chat-completions with its own strict retry contract — keeping it unmodified is the
        faithfulness guarantee, so we configure it via its environment seam. Process-global
        but constant (one model server per agent process).
        """
        if not os.environ.get("VLLM_API_ENDPOINT"):
            base_url = get_server_url(self.config.model_server.name)
            os.environ["VLLM_API_ENDPOINT"] = f"{base_url}/v1/chat/completions"
        os.environ.setdefault("VLLM_API_KEY", "EMPTY")

    def _make_reference_agent(self) -> Any:
        """Build the reference NemotronAgent from the pinned fork (lazy import: the fork's
        dependency set is heavy and Linux-only, and unit tests stub this seam)."""
        from mm_agents.nvidia.nemotron_agent import NemotronAgent  # noqa: PLC0415

        return NemotronAgent(
            model=self.config.model_name,
            max_steps=self.config.max_steps,
            max_image_history_length=self.config.max_image_history_length,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            screen_size=(self.config.screen_width, self.config.screen_height),
            coordinate_type=self.config.coordinate_type,
            password=self.config.client_password,
            thinking=self.config.thinking,
        )

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        instruction = _extract_instruction(body.input)
        session_id = request.session[SESSION_ID_KEY]
        resources_name = self.config.resources_server.name
        resources_cookies = request.cookies

        self._ensure_reference_llm_endpoint()
        agent = self._make_reference_agent()
        agent.reset(None)

        action_history: List[str] = []
        output_items: List[dict] = []
        terminal = False

        debug_dir = None
        if self.config.debug_trajectory_dir:
            debug_dir = os.path.join(self.config.debug_trajectory_dir, session_id[:8])
            os.makedirs(debug_dir, exist_ok=True)

        for step_idx in range(self.config.max_steps):
            # OBSERVE — fetch the current screenshot from the resources server.
            shot_resp = await self.server_client.post(
                server_name=resources_name, url_path="/screenshot", cookies=resources_cookies
            )
            await raise_for_status(shot_resp)
            resources_cookies = shot_resp.cookies
            shot_json = await get_response_json(shot_resp)
            obs = {"screenshot": base64.b64decode(shot_json["image_base64"])}

            # THINK — the reference agent builds messages, calls the model server's
            # chat-completions endpoint (its own sync client, in a worker thread), and
            # parses the free-text `## Action/## Code` contract (incl. retry recovery
            # and the internal force-FAIL at the step cap).
            message, actions, _cot = await asyncio.to_thread(agent.predict, instruction, obs, step_idx=step_idx)
            if debug_dir is not None:
                with open(os.path.join(debug_dir, f"step_{step_idx:03d}.png"), "wb") as f:
                    f.write(obs["screenshot"])
                with open(os.path.join(debug_dir, "trace.jsonl"), "a") as f:
                    f.write(json.dumps({"step": step_idx, "actions": actions, "message": _message_to_text(message)[:4000]}) + "\n")
            output_items.append(
                {
                    "id": f"msg_{step_idx + 1}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": _message_to_text(message), "annotations": []}],
                }
            )

            # ACT — execute each returned action with the reference runner's semantics.
            for action in actions:
                action_history.append(action)
                if action == "WAIT":
                    await asyncio.sleep(self.config.sleep_after_execution_s)
                    continue
                if action in ("FAIL", "DONE"):
                    terminal = True
                    break
                command = ["python", "-c", _PYAUTOGUI_PKGS_PREFIX.format(command=action)]
                exec_resp = await self.server_client.post(
                    server_name=resources_name,
                    url_path="/execute",
                    json={"command": command, "shell": False},
                    cookies=resources_cookies,
                )
                # Tool errors are non-fatal; keep the rollout going (just refresh cookies).
                resources_cookies = exec_resp.cookies
                await asyncio.sleep(self.config.sleep_after_execution_s)
            if terminal:
                break

        self.session_id_to_action_history[session_id] = action_history

        for key, value in resources_cookies.items():
            response.set_cookie(key, value)

        return NeMoGymResponse.model_validate(
            {
                "id": f"resp_osworld_{session_id[:16]}",
                "created_at": time.time(),
                "model": self.config.model_name,
                "object": "response",
                "output": output_items,
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            }
        )

    async def run(self, request: Request, body: NemotronOSWorldRunRequest) -> NemotronOSWorldVerifyResponse:
        """One rollout. NEVER raises: a raised exception propagates through gym's rollout
        TaskGroup and kills the ENTIRE run (observed live: one seed 500 after sandbox-pool
        starvation crashed a 24-task gate). Infrastructure failures instead return a
        zero-reward row marked with ``verify_error`` so a post-run pass can strip and
        re-run exactly those tasks."""
        resources_cookie_holder: Dict[str, str] = {}
        try:
            return await self._run_rollout(request, body, resources_cookie_holder)
        except Exception as e:  # noqa: BLE001 - one bad rollout must not kill the run
            logger.warning("OSWorld rollout failed (%r); emitting marked zero-reward row", e)
            # BaseVerifyRequest requires a `response`; the rollout died before producing one.
            placeholder_response = {
                "id": "resp_rollout_infra_failure",
                "created_at": time.time(),
                "model": self.config.model_name,
                "object": "response",
                "output": [],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            }
            if resources_cookie_holder:
                # Best-effort verify so the resources server releases the session's sandbox
                # (otherwise it leaks until process exit / record TTL). Its score is NOT
                # used: the rollout is incomplete, and the marked row gets re-run anyway.
                try:
                    release_request = NemotronOSWorldVerifyRequest.model_validate(
                        body.model_dump() | {"response": placeholder_response, "action_history": []}
                    )
                    await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/verify",
                        json=release_request.model_dump(),
                        cookies=dict(resources_cookie_holder),
                    )
                except Exception as release_error:  # noqa: BLE001 - release is best-effort
                    logger.warning("post-failure sandbox release also failed: %r", release_error)
            return NemotronOSWorldVerifyResponse.model_validate(
                body.model_dump()
                | {
                    "response": placeholder_response,
                    "reward": 0.0,
                    "verify_error": f"rollout_infra_failure: {e!r}"[:400],
                }
            )

    async def _run_rollout(
        self, request: Request, body: NemotronOSWorldRunRequest, resources_cookie_holder: Dict[str, str]
    ) -> NemotronOSWorldVerifyResponse:
        cookies = request.cookies

        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies
        resources_cookie_holder.update(cookies)

        response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(response)
        cookies = response.cookies
        resources_cookie_holder.update(cookies)

        history_response = await self.server_client.post(
            server_name=self.config.name, url_path="/action_history", cookies=cookies
        )
        await raise_for_status(history_response)
        cookies = history_response.cookies
        action_history: Optional[List[str]] = (await get_response_json(history_response)).get("action_history")

        verify_request = NemotronOSWorldVerifyRequest.model_validate(
            body.model_dump()
            | {
                "response": await get_response_json(response),
                "action_history": action_history or [],
            }
        )

        # Merge every cookie seen this rollout (seed's resources session cookie included):
        # if an intermediate hop's response dropped the resources session cookie, verify
        # would otherwise land on a fresh session and score 0 with `no_seeded_sandbox`.
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies={**resources_cookie_holder, **dict(cookies)},
        )
        await raise_for_status(verify_response)
        return NemotronOSWorldVerifyResponse.model_validate(await get_response_json(verify_response))


if __name__ == "__main__":
    NemotronOSWorldAgent.run_webserver()
