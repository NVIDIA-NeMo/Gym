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
"""Agent that orchestrates skill evaluations against real NeMo Gym infrastructure.

Flow of /run:
  1. seed_session → skill_workspace (scoped tmpdir with SKILL.md / scripts / fixtures)
  2. build input: prepend SKILL.md when with_skill=True; inject tool schemas if missing
  3. model ↔ tool loop with dispatch to skill_workspace /run_bash and /read_file
  4. /verify → skill_judge, forwarding captured tool_calls via verifier_metadata
  5. /close the workspace (always runs, even on failure)
"""

import json
import logging
from typing import Any, List, Optional

from fastapi import Request, Response
from pydantic import ValidationError

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
)
from nemo_gym.base_responses_api_agent import Body, SimpleResponsesAPIAgent
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.skill_eval_agent.schemas import (
    SkillEvalAgentConfig,
    SkillEvalAgentRunRequest,
    SkillEvalAgentVerifyResponse,
)


logger = logging.getLogger(__name__)

_SNIPPET_CHARS = 2000
_SKILL_SYSTEM_PREFIX = "You have access to the following skill. Follow its guidance when applicable.\n\n"

_TOOLS_SCHEMA: List[dict] = [
    {
        "type": "function",
        "name": "run_bash",
        "description": "Execute a bash command inside the sandbox workspace. The workspace is seeded with SKILL.md, scripts/, references/, and scenario fixtures.",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Bash command to execute."},
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Optional timeout override (clamped by the workspace server).",
                },
            },
            "required": ["cmd"],
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read a UTF-8 text file from the workspace (output capped by the workspace server).",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative path."},
            },
            "required": ["path"],
        },
    },
]


def _tool_call_log_entry(name: str, arguments: str, payload: dict) -> dict:
    """Compact log entry that matches skill_judge.ToolCallLogEntry fields."""
    stdout = str(payload.get("stdout") or "")
    stderr = str(payload.get("stderr") or "")
    entry: dict[str, Any] = {
        "name": name,
        "arguments": arguments,
        "stdout_snippet": stdout[:_SNIPPET_CHARS],
        "stderr_snippet": stderr[:_SNIPPET_CHARS],
        "truncated": bool(payload.get("truncated", False)),
    }
    if "exit_code" in payload:
        entry["exit_code"] = payload["exit_code"]
    elif name == "read_file":
        content = str(payload.get("content") or "")
        entry["stdout_snippet"] = content[:_SNIPPET_CHARS]
    return entry


class SkillEvalAgent(SimpleResponsesAPIAgent):
    config: SkillEvalAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Pure model proxy.

        The skill-eval tool loop is orchestrated inside /run, which needs the
        env_id from seed_session. /v1/responses without that context can only
        forward to the model server.
        """
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
        )
        await raise_for_status(model_response)
        return NeMoGymResponse.model_validate(await get_response_json(model_response))

    async def run(
        self,
        request: Request,
        body: SkillEvalAgentRunRequest = Body(),
    ) -> SkillEvalAgentVerifyResponse:
        metadata = dict(body.verifier_metadata or {})
        skill_path = metadata.get("skill_path")
        scenario_id = metadata.get("scenario_id")
        if skill_path is None or scenario_id is None:
            raise RuntimeError("verifier_metadata must include skill_path and scenario_id")
        files = list(metadata.get("files") or [])
        with_skill = bool(metadata.get("with_skill", False))
        # with_references defaults to `with_skill` for back-compat: earlier input
        # JSONLs didn't carry the flag, and historical behavior was "references
        # always on." New input JSONLs should stamp this explicitly per cell.
        with_references = bool(metadata.get("with_references", with_skill))
        with_scripts = bool(metadata.get("with_scripts", with_skill))
        skill_md = metadata.get("skill_md") or ""

        seed_resp = await self.server_client.post(
            server_name=self.config.workspace_server.name,
            url_path="/seed_session",
            json={
                "skill_path": skill_path,
                "scenario_id": scenario_id,
                "files": files,
                "with_references": with_references,
                "with_scripts": with_scripts,
            },
        )
        await raise_for_status(seed_resp)
        env_id = (await get_response_json(seed_resp))["env_id"]

        try:
            params = body.responses_create_params.model_copy(deep=True)
            if isinstance(params.input, str):
                params.input = [NeMoGymEasyInputMessage(role="user", content=params.input)]
            else:
                params.input = list(params.input or [])
            if with_skill and skill_md:
                params.input.insert(
                    0,
                    NeMoGymEasyInputMessage(role="system", content=f"{_SKILL_SYSTEM_PREFIX}{skill_md}"),
                )
            if self.config.inject_tools and not params.tools:
                params.tools = _TOOLS_SCHEMA

            model_response, tool_calls = await self._rollout_loop(params, env_id)

            verify_metadata = dict(metadata)
            verify_metadata["tool_calls"] = tool_calls
            verify_body = {
                "responses_create_params": params.model_dump(mode="json"),
                "response": model_response.model_dump(mode="json"),
                "verifier_metadata": verify_metadata,
            }
            judge_resp = await self.server_client.post(
                server_name=self.config.judge_server.name,
                url_path="/verify",
                json=verify_body,
            )
            await raise_for_status(judge_resp)
            return SkillEvalAgentVerifyResponse.model_validate(await get_response_json(judge_resp))
        finally:
            try:
                await self.server_client.post(
                    server_name=self.config.workspace_server.name,
                    url_path="/close",
                    json={"env_id": env_id},
                )
            except Exception:
                logger.exception("skill_workspace close failed for env_id=%s", env_id)

    async def _rollout_loop(
        self,
        params: NeMoGymResponseCreateParamsNonStreaming,
        env_id: str,
    ) -> tuple[NeMoGymResponse, list[dict]]:
        new_outputs: list = []
        tool_calls: list[dict] = []
        model_response: Optional[NeMoGymResponse] = None
        step = 0
        max_steps = self.config.max_steps

        while True:
            step += 1
            new_body = params.model_copy(update={"input": list(params.input) + new_outputs})

            raw = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
            )
            await raise_for_status(raw)
            raw_json = await get_response_json(raw)
            try:
                model_response = NeMoGymResponse.model_validate(raw_json)
            except ValidationError as e:
                raise RuntimeError(f"Received invalid response from model server: {json.dumps(raw_json)}") from e

            output = model_response.output
            new_outputs.extend(output)

            if model_response.incomplete_details:
                break

            fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            msgs: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and getattr(o, "role", None) == "assistant"
            ]
            if not fn_calls and msgs:
                break

            for call in fn_calls:
                tool_output, log_entry = await self._dispatch_tool_call(env_id, call)
                tool_calls.append(log_entry)
                new_outputs.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=call.call_id,
                        output=tool_output,
                    )
                )

            if max_steps and step >= max_steps:
                break

        assert model_response is not None
        model_response.output = new_outputs
        return model_response, tool_calls

    async def _dispatch_tool_call(
        self,
        env_id: str,
        call: NeMoGymResponseFunctionToolCall,
    ) -> tuple[str, dict]:
        name = call.name
        try:
            args = json.loads(call.arguments) if call.arguments else {}
        except json.JSONDecodeError as e:
            err = f"invalid JSON in tool arguments: {e}"
            return json.dumps({"error": err}), {
                "name": name,
                "arguments": call.arguments or "",
                "stderr_snippet": err[:_SNIPPET_CHARS],
            }

        if name == "run_bash":
            payload = {"env_id": env_id, "cmd": str(args.get("cmd", ""))}
            if "timeout_seconds" in args:
                payload["timeout_seconds"] = args["timeout_seconds"]
            url_path = "/run_bash"
        elif name == "read_file":
            payload = {"env_id": env_id, "path": str(args.get("path", ""))}
            url_path = "/read_file"
        else:
            err = f"unknown tool: {name}"
            return json.dumps({"error": err}), {
                "name": name,
                "arguments": json.dumps(args),
                "stderr_snippet": err[:_SNIPPET_CHARS],
            }

        try:
            resp = await self.server_client.post(
                server_name=self.config.workspace_server.name,
                url_path=url_path,
                json=payload,
            )
            if not resp.ok:
                body_text = (await resp.content.read()).decode(errors="replace")
                err = f"workspace {url_path} returned {resp.status}: {body_text}"
                return json.dumps({"error": err}), {
                    "name": name,
                    "arguments": json.dumps(args),
                    "stderr_snippet": err[:_SNIPPET_CHARS],
                }
            payload_out = await get_response_json(resp)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            return json.dumps({"error": err}), {
                "name": name,
                "arguments": json.dumps(args),
                "stderr_snippet": err[:_SNIPPET_CHARS],
            }

        return json.dumps(payload_out), _tool_call_log_entry(name, json.dumps(args), payload_out)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        response = await self.server_client.post(
            server_name=self.config.judge_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    SkillEvalAgent.run_webserver()
