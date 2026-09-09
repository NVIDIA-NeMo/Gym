# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import logging
import os
import shlex
import shutil
import signal
import sys
import tempfile
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext

from nemo_gym.agents.config import AgentHarnessConfig
from nemo_gym.agents.terminus_2_llm import NemoGymLLM
from nemo_gym.agents.terminus_2_output import trajectory_to_responses
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)


LOG = logging.getLogger(__name__)


def _message_text(item: Any) -> str:
    content = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "") for part in content)


def _extract_instruction(body_input: Any) -> tuple[str, Optional[str]]:
    if isinstance(body_input, str):
        return body_input, None

    messages: list[tuple[str, str]] = []
    for item in body_input or []:
        role = getattr(item, "role", None) if not isinstance(item, dict) else item.get("role")
        text = _message_text(item)
        if role and text:
            messages.append((role, text))

    conversation = [(role, text) for role, text in messages if role in {"user", "assistant"}]
    if len(conversation) <= 1 and (not conversation or conversation[0][0] == "user"):
        user_message = conversation[0][1] if conversation else ""
        system_text = "\n\n".join(text for role, text in messages if role in {"system", "developer"})
        return user_message, system_text or None
    return "\n\n".join(f"{role.title()}: {text}" for role, text in messages), None


class LocalEnvironment:
    def __init__(self, cwd: Path, command_timeout_sec: float) -> None:
        self.cwd = cwd.resolve()
        self.command_timeout_sec = command_timeout_sec

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | float | None = None,
    ) -> ExecResult:
        run_cwd = Path(cwd).resolve() if cwd else self.cwd
        if not run_cwd.is_dir():
            raise FileNotFoundError(f"Terminus-2 working directory does not exist: {run_cwd}")

        process_env = os.environ.copy()
        if env:
            process_env.update({str(key): str(value) for key, value in env.items()})

        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-lc",
            command,
            cwd=str(run_cwd),
            env=process_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        timeout = self.command_timeout_sec if timeout_sec is None else float(timeout_sec)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.CancelledError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.communicate()
            raise
        except asyncio.TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = await proc.communicate()
            message = stderr.decode(errors="replace")
            message += f"\nCommand timed out after {timeout:g} seconds"
            return ExecResult(stdout=stdout.decode(errors="replace"), stderr=message.strip(), return_code=124)

        return ExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=int(proc.returncode or 0),
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path).resolve()
        target = Path(target_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if source != target:
            await asyncio.to_thread(shutil.copy2, source, target)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self.upload_file(source_path, str(target_path))


class LocalTmuxSession(TmuxSession):
    @property
    def _tmux_start_session(self) -> str:
        if sys.platform != "darwin":
            return super()._tmux_start_session

        session_name = shlex.quote(self._session_name)
        login_shell = shlex.quote("bash --login")
        pipe_command = shlex.quote(f"cat > {shlex.quote(str(self._logging_path))}")
        return (
            "export TERM=xterm-256color && "
            "export SHELL=/bin/bash && "
            f"tmux new-session -x {self._pane_width} -y {self._pane_height} -d "
            f"-s {session_name} {login_shell} && tmux pipe-pane -t {session_name} {pipe_command}"
        )


class StandaloneTerminus2(Terminus2):
    async def _handle_llm_interaction(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        result = await super()._handle_llm_interaction(*args, **kwargs)
        if result[1]:
            self._consecutive_completion_claims = getattr(self, "_consecutive_completion_claims", 0) + 1
        else:
            self._consecutive_completion_claims = 0
        return result

    @property
    def finished_naturally(self) -> bool:
        return getattr(self, "_consecutive_completion_claims", 0) >= 2

    async def run(self, instruction: str, environment: LocalEnvironment, context: AgentContext) -> None:
        self.context_length_exceeded = False
        try:
            await super().run(instruction, environment, context)
        except Exception as exc:
            self.logger.info("Agent error: %s: %s. Returning completed turns.", type(exc).__name__, exc)
        finally:
            self._attach_routed_experts_to_trajectory()
            llm = getattr(self, "_llm", None)
            if isinstance(llm, NemoGymLLM):
                self.context_length_exceeded = llm.context_length_exceeded

    def _attach_routed_experts_to_trajectory(self) -> None:
        llm = getattr(self, "_llm", None)
        if not isinstance(llm, NemoGymLLM):
            return

        modified = False
        for step in getattr(self, "_trajectory_steps", []):
            if getattr(step, "source", None) != "agent" or step.metrics is None:
                continue
            routed_experts = llm.pop_routed_experts_for_rollout_details(
                step.metrics.prompt_token_ids,
                step.metrics.completion_token_ids,
                step.metrics.logprobs,
            )
            if routed_experts is None:
                continue
            extra = step.metrics.extra or {}
            extra["routed_experts"] = routed_experts
            step.metrics.extra = extra
            modified = True

        if modified:
            self._dump_trajectory()

    async def setup(self, environment: LocalEnvironment) -> None:
        self._consecutive_completion_claims = 0
        self._standalone_session_name = f"terminus-2-{uuid4().hex[:12]}"
        recording_path = self.logs_dir / "recording.cast" if self._record_terminal_session else None
        self._session = LocalTmuxSession(
            session_name=self._standalone_session_name,
            environment=environment,
            logging_path=self.logs_dir / "terminus_2.pane",
            local_asciinema_recording_path=recording_path,
            remote_asciinema_recording_path=recording_path,
            pane_width=self._tmux_pane_width,
            pane_height=self._tmux_pane_height,
        )
        await self._session.start()

    async def close(self, environment: LocalEnvironment) -> None:
        if self._session is not None:
            try:
                await self._session.stop()
            except Exception as exc:
                self.logger.warning("Could not stop Terminus-2 recording cleanly: %s", exc)
        session_name = getattr(self, "_standalone_session_name", None)
        if session_name:
            await environment.exec(
                f"tmux kill-session -t {shlex.quote(session_name)}",
                timeout_sec=30,
            )


class Terminus2Harness:
    def __init__(self, config: AgentHarnessConfig):
        self.config = config

    def _logs_dir(self) -> Path:
        if not self.config.settings.get("keep_logs", False):
            return Path(tempfile.mkdtemp(prefix="nemo-gym-terminus-2-"))
        root = Path(self.config.settings.get("logs_root", "outputs/terminus_2_agent/runs")).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        path = root / f"run-{int(time())}-{uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _model_name(self, body: NeMoGymResponseCreateParamsNonStreaming) -> str:
        return self.config.model.model or body.model or "model"

    def _build_agent(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        logs_dir: Path,
        api_base: str,
    ) -> StandaloneTerminus2:
        model_settings = self.config.model.settings
        settings = self.config.settings
        model_info = model_settings.get(
            "model_info",
            {
                "max_input_tokens": 262144,
                "max_output_tokens": 81920,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
        )
        temperature = body.temperature if body.temperature is not None else model_settings.get("temperature", 0.7)
        model_name = self._model_name(body)
        llm = NemoGymLLM(
            model_name=model_name,
            api_base=api_base,
            api_key=self.config.model.api_key,
            collect_rollout_details=model_settings.get("collect_rollout_details", False),
            model_info=model_info,
            responses_create_params=body.model_dump(exclude_none=True),
            timeout_sec=model_settings.get("timeout_seconds", 2400),
        )
        return StandaloneTerminus2(
            logs_dir=logs_dir,
            model_name=model_name,
            max_turns=self.config.max_turns,
            parser_name=settings.get("parser_name", "json"),
            api_base=api_base,
            temperature=temperature,
            reasoning_effort=model_settings.get("reasoning_effort"),
            collect_rollout_details=model_settings.get("collect_rollout_details", False),
            enable_summarize=settings.get("enable_summarize", True),
            proactive_summarization_threshold=settings.get("proactive_summarization_threshold", 8000),
            max_thinking_tokens=model_settings.get("max_thinking_tokens"),
            model_info=model_info,
            trajectory_config=settings.get("trajectory_config", {"raw_content": False}),
            tmux_pane_width=settings.get("tmux_pane_width", 160),
            tmux_pane_height=settings.get("tmux_pane_height", 40),
            store_all_messages=settings.get("store_all_messages", False),
            record_terminal_session=settings.get("record_terminal_session", False),
            interleaved_thinking=settings.get("interleaved_thinking", False),
            llm=llm,
        )

    async def _run_terminus(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        instruction: str,
        api_base: str,
    ) -> tuple[dict[str, Any], AgentContext, dict[str, bool], bool, bool]:
        logs_dir = self._logs_dir()
        temporary_workspace: Optional[Path] = None
        try:
            if self.config.workspace:
                workspace = self.config.workspace.expanduser().resolve()
                if not workspace.is_dir():
                    raise FileNotFoundError(f"Terminus-2 workspace does not exist: {workspace}")
            else:
                workspace = temporary_workspace = Path(tempfile.mkdtemp(prefix="nemo-gym-terminus-2-workspace-"))

            environment = LocalEnvironment(workspace, self.config.settings.get("command_timeout_seconds", 1800))
            context = AgentContext()
            agent = self._build_agent(body, logs_dir, api_base)
            timed_out = False
            try:
                await agent.setup(environment)
                try:
                    await asyncio.wait_for(
                        agent.run(instruction, environment, context),
                        timeout=self.config.timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    timed_out = True
                    LOG.warning("Terminus-2 timed out after %gs", self.config.timeout_seconds)
            finally:
                await agent.close(environment)

            trajectory_path = logs_dir / "trajectory.json"
            trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.exists() else {"steps": []}
            flags = {"context_length_exceeded": agent.context_length_exceeded}
            return trajectory, context, flags, timed_out, agent.finished_naturally
        finally:
            if not self.config.settings.get("keep_logs", False):
                shutil.rmtree(logs_dir, ignore_errors=True)
            if temporary_workspace is not None:
                shutil.rmtree(temporary_workspace, ignore_errors=True)

    async def run(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_base_url: Optional[str] = None,
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if body.temperature is None:
            body.temperature = self.config.model.settings.get("temperature", 0.7)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_instruction, input_system = _extract_instruction(body.input)
        instruction_parts = [part for part in (self.config.system_prompt, input_system, user_instruction) if part]
        instruction = "\n\n".join(instruction_parts)

        api_base = model_base_url or self.config.model.base_url
        if not api_base:
            raise ValueError("Terminus2Harness requires a model base URL")
        trajectory, context, flags, timed_out, finished_naturally = await self._run_terminus(
            body, instruction, api_base
        )
        output_items = trajectory_to_responses(trajectory)
        if not output_items:
            output_items = [
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                ).model_dump()
            ]

        final_metrics = trajectory.get("final_metrics", {})
        input_tokens = final_metrics.get("total_prompt_tokens", 0)
        output_tokens = final_metrics.get("total_completion_tokens", 0)
        cached_tokens = final_metrics.get("total_cached_tokens", 0)
        if input_tokens == 0 and output_tokens == 0:
            input_tokens = context.n_input_tokens or 0
            output_tokens = context.n_output_tokens or 0
            cached_tokens = context.n_cache_tokens or 0

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self._model_name(body),
            object="response",
            output=output_items,
            parallel_tool_calls=False,
            temperature=body.temperature,
            top_p=body.top_p,
            tool_choice=body.tool_choice,
            tools=body.tools,
            background=False,
            reasoning={"effort": None, "generate_summary": None, "summary": None},
            service_tier="default",
            status="completed",
            text={"format": {"type": "text"}, "verbosity": "medium"},
            top_logprobs=0,
            truncation="disabled",
            store=True,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached_tokens),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            ),
            metadata={
                "terminus_2": json.dumps(
                    {
                        "context": context.model_dump(mode="json"),
                        "agent_error_flags": flags,
                        "timed_out": timed_out,
                        "finished_naturally": finished_naturally,
                    }
                )
            },
        )
