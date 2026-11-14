# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One direct vision request per Harbor task; no policy tools or tool loop."""

import base64
import json
import mimetypes
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import httpx
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import Agent, ContentPart, FinalMetrics, ImageSource, Metrics, Step, Trajectory


class VialsVisionAgent(BaseAgent):
    SUPPORTS_ATIF = True

    def __init__(
        self,
        *args,
        max_output_tokens: int = 32768,
        temperature: float = 1.0,
        top_p: float = 1.0,
        chat_template_kwargs: dict | None = None,
        request_timeout_sec: float = 14400,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if max_output_tokens < 1 or request_timeout_sec <= 0:
            raise ValueError("Token limit and request timeout must be positive")
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.request_timeout_sec = request_timeout_sec

    @staticmethod
    def name() -> str:
        return "vials-vision"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        if self.mcp_servers:
            raise ValueError("VIALS no-tool agent does not accept MCP servers")

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        base_url = self._get_env("VIALS_POLICY_BASE_URL")
        if not base_url or not self.model_name:
            raise ValueError("Set VIALS_POLICY_BASE_URL and harbor_agent.model_name")
        with TemporaryDirectory(prefix="vials-vision-") as directory:
            local = Path(directory)
            await environment.download_file("/task/vials_input.json", local / "input.json")
            payload = json.loads((local / "input.json").read_text())
            names = payload["images"]
            if not names or len(names) != len(set(names)):
                raise ValueError("VIALS requires nonempty, uniquely named image panels")
            content = [{"type": "text", "text": payload["user"]}]
            parts = [ContentPart(type="text", text=payload["user"])]
            for index, name in enumerate(names):
                if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
                    raise ValueError("Image names must be task-local filenames")
                image = local / f"panel-{index}"
                await environment.download_file(f"/task/{name}", image)
                media_type = mimetypes.guess_type(name)[0] or "image/png"
                url = f"data:{media_type};base64,{base64.b64encode(image.read_bytes()).decode('ascii')}"
                content.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
                parts.append(ContentPart(type="image", source=ImageSource(media_type=media_type, path=url)))
            messages = []
            steps = []
            if payload.get("system"):
                messages.append({"role": "system", "content": payload["system"]})
                steps.append(Step(step_id=1, source="system", message=payload["system"]))
            messages.append({"role": "user", "content": content})
            steps.append(Step(step_id=len(steps) + 1, source="user", message=parts))
            body = {
                "model": self.model_name,
                "messages": messages,
                "tool_choice": "none",
                "max_completion_tokens": self.max_output_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": False,
            }
            # vLLM rejects tools=[]; omit the field entirely. No tools are registered.
            if self.chat_template_kwargs:
                body["chat_template_kwargs"] = self.chat_template_kwargs
            headers = {"Authorization": f"Bearer {self._get_env('VIALS_POLICY_API_KEY') or 'EMPTY'}"}
            async with httpx.AsyncClient(timeout=self.request_timeout_sec) as client:
                response = await client.post(base_url.rstrip("/") + "/chat/completions", json=body, headers=headers)
                response.raise_for_status()
                result = response.json()
            choice = result["choices"][0]
            message = choice["message"]
            if message.get("tool_calls") or message.get("function_call"):
                raise ValueError("Policy returned a tool call in a no-tool evaluation")
            answer = message.get("content") or ""
            if not isinstance(answer, str):
                raise ValueError("Policy answer must be text")
            usage = result.get("usage") or {}
            context.n_input_tokens = usage.get("prompt_tokens", 0)
            context.n_output_tokens = usage.get("completion_tokens", 0)
            context.n_cache_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            context.metadata = {
                "finish_reason": choice.get("finish_reason"),
                "max_output_tokens": self.max_output_tokens,
            }
            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    source="agent",
                    message=answer,
                    reasoning_content=message.get("reasoning_content"),
                    llm_call_count=1,
                    metrics=Metrics(prompt_tokens=context.n_input_tokens, completion_tokens=context.n_output_tokens),
                )
            )
            trajectory = Trajectory(
                session_id=self.session_id or uuid4().hex,
                agent=Agent(name=self.name(), version=self.version(), model_name=self.model_name),
                steps=steps,
                final_metrics=FinalMetrics(
                    total_prompt_tokens=context.n_input_tokens,
                    total_completion_tokens=context.n_output_tokens,
                    total_cached_tokens=context.n_cache_tokens,
                    total_steps=len(steps),
                ),
                extra=context.metadata,
            )
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            (self.logs_dir / "trajectory.json").write_text(trajectory.model_dump_json(indent=2))
            (self.logs_dir / "policy-response.json").write_text(json.dumps(result) + "\n")
            (local / "answer.txt").write_text(answer)
            # Harness I/O, not a tool exposed to or selected by the policy.
            await environment.upload_file(local / "answer.txt", "/task/answer.txt")
