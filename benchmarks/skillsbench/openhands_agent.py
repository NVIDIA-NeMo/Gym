# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the OpenHands SDK on the Slurm driver with task tools in OpenSandbox.

The model server has a private Slurm address. Running OpenHands inside the
remote task image would make its model requests originate in AWS, so this
adapter keeps the OpenHands agent beside the model and forwards only terminal
actions to the task's Harbor environment.
"""

import asyncio
import json
import shlex
import threading
import uuid
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext as HarborAgentContext
from harbor.models.trajectories import (
    Agent as AtifAgent,
)
from harbor.models.trajectories import (
    FinalMetrics,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.models.trajectories import (
    Observation as AtifObservation,
)
from openhands.sdk import LLM, Agent, Tool
from openhands.sdk.context import AgentContext
from openhands.sdk.conversation import LocalConversation
from openhands.sdk.skills import load_skills_from_dir
from openhands.sdk.tool import ToolExecutor, register_tool
from openhands.tools.terminal.definition import TerminalAction, TerminalObservation, TerminalTool
from pydantic import SecretStr


_EXECUTORS: dict[str, "OpenSandboxTerminalExecutor"] = {}


class _RunControl:
    def __init__(self) -> None:
        self.cancelled = threading.Event()
        self.sandbox_discarded = threading.Event()
        self.conversation: LocalConversation | None = None
        self.lock = threading.Lock()
        self.active_commands: dict[str, Future[Any]] = {}
        # Neither wrapper exit nor process-group exit proves that descendants
        # which detached into other sessions have stopped.
        self.command_started = False


class OpenSandboxTerminalExecutor(ToolExecutor):
    """Forward an OpenHands terminal action to one Harbor task sandbox."""

    def __init__(self, environment: BaseEnvironment, loop: asyncio.AbstractEventLoop, cwd: str, control: _RunControl):
        self.environment = environment
        self.loop = loop
        self.cwd = cwd
        self.control = control

    def __call__(self, action: TerminalAction, conversation: LocalConversation | None = None) -> TerminalObservation:
        if self.control.cancelled.is_set():
            return TerminalObservation.from_text(
                "The agent time limit expired; terminal commands are disabled.",
                is_error=True,
                command=action.command,
                exit_code=1,
            )
        if action.is_input:
            return TerminalObservation.from_text(
                "Interactive key input is unavailable. Start long-running processes in the background.",
                is_error=True,
                command=action.command,
                exit_code=1,
            )
        if not action.command:
            return TerminalObservation.from_text("", command="", exit_code=0)

        command_id = uuid.uuid4().hex
        marker = f"__OPENHANDS_CWD_{command_id}__"
        action_command = (
            f'{action.command}\n__openhands_status=$?\nprintf \'\\n{marker}%s\\n\' "$PWD"\nexit "$__openhands_status"'
        )
        # Keep the shell session independent of the provider's exec wrapper so
        # background services survive between actions. This is not a guarantee
        # that all descendants remain in this process group.
        inner_command = f"bash -ic {shlex.quote(action_command)}"
        command = f"setsid -w bash -c {shlex.quote(inner_command)}"
        timeout = min(600, max(1, int(action.timeout or 180)))
        future: Future[Any] | None = None
        try:
            with self.control.lock:
                if self.control.cancelled.is_set():
                    return TerminalObservation.from_text(
                        "The agent time limit expired; terminal commands are disabled.",
                        is_error=True,
                        command=action.command,
                        exit_code=1,
                    )
                future = asyncio.run_coroutine_threadsafe(
                    self.environment.exec(command, cwd=self.cwd, timeout_sec=timeout), self.loop
                )
                self.control.command_started = True
                self.control.active_commands[command_id] = future
            result = future.result(timeout=timeout + 30)
        except FutureTimeoutError:
            # A provider can exceed its own command deadline. Do not leave
            # that command running while the SDK starts another action.
            self.control.cancelled.set()
            if future is not None and future.done():
                # The provider raised its own timeout. Its client-side error
                # does not prove the remote process has exited.
                stop = asyncio.run_coroutine_threadsafe(self.environment.stop(delete=True), self.loop)
                stop.result(timeout=30)
                raise RuntimeError("Sandbox command timed out without confirmed termination")
            cleanup = asyncio.run_coroutine_threadsafe(
                SkillsBenchOpenHands._terminate_active_commands(self.environment, self.control), self.loop
            )
            try:
                cleanup.result(timeout=45)
            except BaseException:
                # No verifier may inspect a sandbox with an unconfirmed
                # command still running.
                stop = asyncio.run_coroutine_threadsafe(self.environment.stop(delete=True), self.loop)
                stop.result(timeout=30)
                raise
            return TerminalObservation.from_text(
                "Sandbox command exceeded its deadline.",
                is_error=True,
                command=action.command,
                exit_code=1,
            )
        except FutureCancelledError:
            # Cleanup cancels client waits only after sandbox deletion succeeds.
            # Provider-side cancellation is not itself proof of remote exit.
            if not self.control.sandbox_discarded.is_set():
                stop = asyncio.run_coroutine_threadsafe(self.environment.stop(delete=True), self.loop)
                stop.result(timeout=30)
                self.control.sandbox_discarded.set()
            return TerminalObservation.from_text(
                "The task sandbox was discarded after the deadline.",
                is_error=True,
                command=action.command,
                exit_code=1,
            )
        except Exception as exc:
            if future is not None:
                # An exec transport error can arrive after the remote command
                # started. Discard the sandbox rather than grade its later work.
                stop = asyncio.run_coroutine_threadsafe(self.environment.stop(delete=True), self.loop)
                stop.result(timeout=30)
            return TerminalObservation.from_text(
                f"Sandbox command failed: {type(exc).__name__}: {exc}",
                is_error=True,
                command=action.command,
                exit_code=1,
            )
        finally:
            with self.control.lock:
                if future is not None and future.done():
                    self.control.active_commands.pop(command_id, None)

        output = result.stdout or ""
        if marker in output:
            output, reported_cwd = output.rsplit(marker, 1)
            next_cwd = reported_cwd.strip().splitlines()[0]
            if next_cwd.startswith("/"):
                self.cwd = next_cwd
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if len(output) > 50_000:
            output = output[:20_000] + "\n[output truncated]\n" + output[-20_000:]
        return TerminalObservation.from_text(
            output,
            is_error=result.return_code != 0,
            command=action.command,
            exit_code=result.return_code,
        )


class SkillsBenchTerminalTool(TerminalTool):
    """OpenHands terminal tool whose executor is bound to a task sandbox."""

    name = "terminal"

    @classmethod
    def create(cls, conv_state: Any, instance_id: str) -> list["SkillsBenchTerminalTool"]:
        description = (
            "Run a shell command in the task sandbox. Each call starts a fresh shell; "
            "only the current directory persists. Exported variables, activated "
            "virtual environments, aliases and shell functions do not persist. "
            "Put dependent shell steps in one command or run a script. "
            "Interactive key input is unavailable."
        )
        # OpenHands tool definitions are frozen; a shallow copy preserves the executor.
        return [
            tool.model_copy(update={"description": description})
            for tool in super().create(conv_state, executor=_EXECUTORS[instance_id])
        ]


register_tool("SkillsBenchTerminalTool", SkillsBenchTerminalTool)


class SkillsBenchOpenHands(BaseAgent):
    """OpenHands SDK harness for the official SkillsBench tasks."""

    capabilities = AgentCapabilities(atif=True)

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        api_base: str | None = None,
        skill_mode: str = "with-skill",
        max_turns: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.95,
        model_info: dict[str, int] | None = None,
        responses_create_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        if not model_name or not api_base:
            raise ValueError("OpenHands requires a model name and the Gym model server URL")
        if skill_mode not in {"with-skill", "no-skill"}:
            raise ValueError(f"Invalid SkillsBench skill mode: {skill_mode!r}")
        self.api_base = api_base
        self.skill_mode = skill_mode
        self.max_turns = max_turns
        self.temperature = (responses_create_params or {}).get("temperature", temperature)
        self.top_p = (responses_create_params or {}).get("top_p", top_p)
        self.model_info = dict(model_info or {})
        limit = (responses_create_params or {}).get("max_output_tokens")
        if limit is not None:
            self.model_info["max_output_tokens"] = limit

    @staticmethod
    def name() -> str:
        return "skillsbench-openhands-sdk"

    def version(self) -> str:
        from openhands.sdk import __version__

        return __version__

    async def setup(self, environment: BaseEnvironment) -> None:
        # Docker tasks can bake skills into their images too. Enforce the
        # condition at the agent boundary, independent of the Harbor provider.
        if self.skill_mode != "no-skill":
            return
        result = await environment.exec(
            "rm -rf /root/.agents/skills /root/.codex/skills "
            "/root/.claude/skills /root/.gemini/skills "
            "/root/.opencode/skills /root/.opencode/skill "
            "/root/.goose/skills /root/.factory/skills "
            "/root/.openhands/skills",
            timeout_sec=300,
        )
        if result.return_code != 0:
            raise RuntimeError(f"Failed to remove preinstalled skills: {result.stderr or result.stdout}")

    @staticmethod
    async def _terminate_active_commands(environment: BaseEnvironment, control: _RunControl) -> None:
        with control.lock:
            command_started = control.command_started
            active = list(control.active_commands.values())
        if not command_started:
            return
        # The adapters expose no sandbox-wide descendant ownership contract.
        # Killing a process group misses setsid children even while the wrapper
        # is active. Only sandbox deletion establishes the required boundary.
        await environment.stop(delete=True)
        control.sandbox_discarded.set()
        # A provider may keep waiting on an exec after deletion. Release those
        # client waits only after deletion is confirmed, then run() joins the SDK.
        for future in active:
            future.cancel()

    async def run(self, instruction: str, environment: BaseEnvironment, context: HarborAgentContext) -> None:
        cwd_result = await environment.exec("pwd", timeout_sec=30)
        if cwd_result.return_code != 0:
            raise RuntimeError(f"Could not determine task working directory: {cwd_result.stderr}")
        cwd = cwd_result.stdout.strip().splitlines()[-1]
        loop = asyncio.get_running_loop()
        instance_id = uuid.uuid4().hex
        control = _RunControl()
        _EXECUTORS[instance_id] = OpenSandboxTerminalExecutor(environment, loop, cwd, control)
        worker = asyncio.create_task(
            asyncio.to_thread(self._run_openhands, instruction, environment, context, instance_id, cwd, control)
        )
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Harbor starts verification after this coroutine exits. Wait until
            # the SDK thread and any in-flight sandbox command have stopped.
            control.cancelled.set()
            try:
                await SkillsBenchOpenHands._terminate_active_commands(environment, control)
            except BaseException:
                # Fail closed: Harbor must not verify a sandbox that might
                # still contain a command running past the agent deadline.
                await SkillsBenchOpenHands._terminate_active_commands(environment, control)
                raise
            finally:
                if control.conversation is not None:
                    await asyncio.to_thread(control.conversation.pause)
                await asyncio.shield(worker)
            raise
        finally:
            _EXECUTORS.pop(instance_id, None)

    def _run_openhands(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: HarborAgentContext,
        instance_id: str,
        cwd: str,
        control: _RunControl,
    ) -> None:
        model_name = self.model_name or ""
        model = model_name if model_name.startswith("openai/") else f"openai/{model_name}"
        llm = LLM(
            model=model,
            base_url=self.api_base,
            api_key=SecretStr("dummy"),
            temperature=self.temperature,
            top_p=self.top_p,
            max_input_tokens=self.model_info.get("max_input_tokens", 131072),
            max_output_tokens=self.model_info.get("max_output_tokens", 32768),
        )

        skills = []
        if self.skill_mode == "with-skill":
            skills_dir = environment.environment_dir / "skills"
            _, _, agent_skills = load_skills_from_dir(skills_dir, strict=False)
            skills = [
                skill.model_copy(
                    update={
                        "source": f"/root/.agents/skills/{Path(skill.source).parent.name}/SKILL.md",
                        "content": skill.content
                        + f"\n\nThis skill's files are in /root/.agents/skills/{Path(skill.source).parent.name}/.",
                    }
                )
                for skill in agent_skills.values()
            ]
            if not skills:
                raise RuntimeError(f"OpenHands found no SkillsBench skills in {skills_dir}")

        agent = Agent(
            llm=llm,
            tools=[Tool(name="SkillsBenchTerminalTool", params={"instance_id": instance_id})],
            agent_context=AgentContext(
                skills=skills,
                load_user_skills=False,
                load_project_skills=False,
                load_public_skills=False,
            ),
        )
        workspace = self.logs_dir / "openhands-workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        conversation = LocalConversation(
            agent=agent,
            workspace=workspace,
            persistence_dir=self.logs_dir / "openhands-state",
            profile_store_dir=self.logs_dir / "openhands-profiles",
            max_iteration_per_run=self.max_turns,
            visualizer=None,
            delete_on_close=False,
        )
        control.conversation = conversation
        try:
            remote_instruction = (
                f"{instruction}\n\nYour terminal runs in the task sandbox at {cwd}. "
                "Skills, when available, are installed under /root/.agents/skills. "
                "Each terminal call starts a fresh shell; only the current directory persists. "
                "Chain commands that depend on exported variables or virtualenv activation. "
                "Use the terminal to inspect and edit task files there."
            )
            conversation.send_message(remote_instruction)
            try:
                if not control.cancelled.is_set():
                    conversation.run()
            finally:
                self._write_trace(conversation, context)
        finally:
            conversation.close()
            control.conversation = None

    @staticmethod
    def _text(parts: list[dict[str, Any]] | None) -> str:
        return "".join(part.get("text", "") for part in parts or [] if part.get("type") == "text")

    def _write_trace(self, conversation: LocalConversation, context: HarborAgentContext) -> None:
        events = [event.model_dump(mode="json") for event in conversation.state.events]
        events_path = self.logs_dir / "openhands-events.jsonl"
        with events_path.open("w") as output:
            for event in events:
                output.write(json.dumps(event) + "\n")

        steps: list[Step] = []
        pending: dict[str, Step] = {}
        for event in events:
            kind = event.get("kind")
            timestamp = event.get("timestamp")
            if kind == "SystemPromptEvent":
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        timestamp=timestamp,
                        source="system",
                        message=event["system_prompt"]["text"]
                        + "\n"
                        + (event.get("dynamic_context") or {}).get("text", ""),
                    )
                )
            elif kind == "MessageEvent":
                message = event.get("llm_message") or {}
                role = message.get("role")
                if role in {"user", "assistant"}:
                    steps.append(
                        Step(
                            step_id=len(steps) + 1,
                            timestamp=timestamp,
                            source="user" if role == "user" else "agent",
                            message=self._text(message.get("content")),
                        )
                    )
            elif kind == "ActionEvent":
                call_id = event.get("tool_call_id") or event["id"]
                action = event.get("action") or {}
                message = self._text(event.get("thought"))
                if action.get("kind") == "FinishAction" and action.get("message"):
                    message = action["message"]
                step = Step(
                    step_id=len(steps) + 1,
                    timestamp=timestamp,
                    source="agent",
                    message=message,
                    tool_calls=[
                        ToolCall(
                            tool_call_id=call_id,
                            function_name=event.get("tool_name") or "unknown",
                            arguments={key: value for key, value in action.items() if key != "kind"},
                        )
                    ],
                    reasoning_content=event.get("reasoning_content") or None,
                )
                steps.append(step)
                pending[call_id] = step
            elif kind == "ObservationEvent":
                call_id = event.get("tool_call_id")
                step = pending.pop(call_id, None)
                if step is not None:
                    observation = event.get("observation") or {}
                    step.observation = AtifObservation(
                        results=[
                            ObservationResult(
                                source_call_id=call_id,
                                content=self._text(observation.get("content")),
                            )
                        ]
                    )

        metrics = conversation.state.stats.get_combined_metrics().get_snapshot()
        usage = metrics.accumulated_token_usage
        if usage:
            context.n_input_tokens = usage.prompt_tokens
            context.n_cache_tokens = usage.cache_read_tokens
            context.n_output_tokens = usage.completion_tokens
        context.cost_usd = metrics.accumulated_cost
        context.metadata = {
            "harness": "openhands-sdk",
            "openhands_version": self.version(),
            "skill_mode": self.skill_mode,
            "events_path": str(events_path),
        }
        if steps:
            trajectory = Trajectory(
                session_id=str(conversation.state.id),
                agent=AtifAgent(name=self.name(), version=self.version(), model_name=self.model_name),
                steps=steps,
                final_metrics=FinalMetrics(
                    total_prompt_tokens=context.n_input_tokens,
                    total_completion_tokens=context.n_output_tokens,
                    total_cached_tokens=context.n_cache_tokens,
                    total_cost_usd=context.cost_usd,
                    total_steps=len(steps),
                ),
            )
            (self.logs_dir / "trajectory.json").write_text(trajectory.model_dump_json(exclude_none=True))
