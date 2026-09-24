# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mini-SWE execution on a caller-owned sandbox with an injected model callback."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from shlex import quote
from time import monotonic, time
from typing import Any
from uuid import uuid4

import yaml
from aiohttp import ClientResponseError
from minisweagent.config import builtin_config_dir
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
)
from pydantic import BaseModel, Field, TypeAdapter

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputItem,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
    TrajectoryRecord,
    TrajectoryTurn,
)
from nemo_gym.sandbox import AsyncSandbox


MINI_CONFIG = yaml.safe_load((builtin_config_dir / "mini.yaml").read_text())
LOGGER = logging.getLogger(__name__)


def responses_input(messages):
    """Replay native Responses items and associate observations with their calls."""
    items = []
    for message in messages:
        if message["role"] == "tool":
            items.append(
                {"type": "function_call_output", "call_id": message["tool_call_id"], "output": message["content"]}
            )
        elif "response_output" in message.get("extra", {}):
            items.extend(message["extra"]["response_output"])
        else:
            items.append({"role": message["role"], "content": message.get("content", "")})
    return items


class MiniSWEConfig(BaseModel):
    step_limit: int = Field(default=0, ge=0)
    step_timeout_sec: int = Field(default=600, gt=0)


class HarnessOutcome(BaseModel):
    reason: str
    exit_code: int | None = None
    detail: str | None = None
    artifacts: list[str] = Field(default_factory=list)


class HarnessContext(BaseModel):
    session_id: str
    task_id: str | None = None
    rollout_id: str | None = None
    instruction: str
    user: str | int | None = None
    workdir: str | None = None
    setup_timeout_sec: float = Field(default=360, gt=0)
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    skills_dir: str | None = None


class MiniSWEHarness:
    """Execute only: the caller provisions, grades, and destroys the sandbox."""

    def __init__(
        self,
        *,
        sandbox: AsyncSandbox,
        context: HarnessContext,
        config: MiniSWEConfig,
        params: NeMoGymResponseCreateParamsNonStreaming,
        query: Callable[[dict], Awaitable[NeMoGymResponse]],
        model_name: str,
        directory: Path,
        observability_enabled: bool = False,
    ) -> None:
        self.sandbox = sandbox
        self.context = context
        self.config = config
        self.params = params
        self.query = query
        self.model_name = model_name
        self.directory = directory
        self.observability_enabled = observability_enabled
        self.extra_instruction = ""
        self.result = None
        self.remote_directory = f"/tmp/nemo-gym-miniswe-{uuid4().hex}"

    async def setup(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        await self._install_runner()
        result = await self.sandbox.exec("command -v setsid", user=self.context.user, cwd=self.context.workdir)
        if result.return_code:
            raise RuntimeError("mini-SWE requires setsid for process cleanup")
        if self.context.skills_dir:
            self.extra_instruction += (
                f"\nTask skills are in {self.context.skills_dir}. Read the relevant SKILL.md files.\n"
            )
        if self.context.mcp_servers:
            (self.directory / "mcp.json").write_text(json.dumps(self.context.mcp_servers))
            remote = f"/tmp/{self.context.session_id}-mcp"
            command = f"python3 -m venv {remote} && {remote}/bin/pip -q install mcp==1.29.0 httpx-aiohttp==0.2.0"
            result = await self.sandbox.exec(
                command, user=self.context.user, cwd=self.context.workdir, timeout_s=self.context.setup_timeout_sec
            )
            if result.return_code:
                raise RuntimeError(f"Task MCP client setup failed: {result.stderr}")
            await self.sandbox.upload(Path(__file__).with_name("mcp_client.py"), remote + "/client.py")
            await self.sandbox.upload(self.directory / "mcp.json", remote + "/servers.json")
            cli = f"{remote}/bin/python {remote}/client.py"
            daemon = (
                f"echo $$ >> /tmp/{self.context.session_id}.pids; "
                f"echo $$ >> {self.remote_directory}/processes; exec {cli} serve"
            )
            started = await self.sandbox.exec(
                "bash -c "
                + quote(
                    f"setsid --fork bash -c {quote(daemon)} > {remote}/server.log 2>&1 < /dev/null; "
                    f"for i in $(seq 1 60); do [ -S {remote}/server.sock ] && exit 0; sleep 1; done; "
                    f"cat {remote}/server.log; exit 1"
                ),
                user=self.context.user,
                cwd=self.context.workdir,
                timeout_s=65,
            )
            if started.return_code:
                raise RuntimeError(f"Task MCP session setup failed: {started.stdout}")
            listed = await self.sandbox.exec(
                cli + " list", user=self.context.user, cwd=self.context.workdir, timeout_s=60
            )
            if listed.return_code:
                raise RuntimeError(f"Task MCP discovery failed: {listed.stderr}")
            self.extra_instruction += (
                f"\nTask MCP tools (JSON schemas): {listed.stdout}\n"
                f"Call with: {cli} call SERVER TOOL 'JSON_ARGUMENTS'.\n"
            )

    async def _install_runner(self) -> None:
        remote = self.remote_directory
        # Bootstrap for the sandbox's architecture, independently of the Gym host.
        script = (
            "import io,pathlib,platform,tarfile,urllib.request; "
            "arch={'x86_64':'x86_64','aarch64':'aarch64'}[platform.machine()]; "
            "url=f'https://github.com/astral-sh/uv/releases/download/0.10.12/uv-{arch}-unknown-linux-musl.tar.gz'; "
            "archive=tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url,timeout=120).read()),mode='r:gz'); "
            "member=next(m for m in archive if m.name.endswith('/uv')); "
            f"target=pathlib.Path('{remote}/uv'); "
            "target.write_bytes(archive.extractfile(member).read()); target.chmod(0o755)"
        )
        result = await self.sandbox.exec(
            f"mkdir -p {remote} && python3 -c {quote(script)} && "
            f"{remote}/uv venv {remote}/venv --python 3.13 && "
            f"{remote}/uv pip install --python {remote}/venv/bin/python mini-swe-agent==2.4.6",
            user=self.context.user,
            cwd=self.context.workdir,
            timeout_s=self.context.setup_timeout_sec,
        )
        if result.return_code:
            raise RuntimeError(f"mini-SWE runner installation failed: {result.stdout}\n{result.stderr}")
        await self.sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), remote + "/runner.py")

    async def _upload_json(self, name: str, payload: dict) -> None:
        local = self.directory / name
        local.write_text(json.dumps(payload))
        remote = f"{self.remote_directory}/{name}"
        await self.sandbox.upload(local, remote + ".tmp")
        result = await self.sandbox.exec(f"mv {quote(remote + '.tmp')} {quote(remote)}", user=self.context.user)
        local.unlink()
        if result.return_code:
            raise RuntimeError(f"mini-SWE relay write failed: {result.stderr}")

    async def _start_runner(self, budget: float) -> None:
        await self._upload_json(
            "input.json",
            {
                "instruction": self.context.instruction + self.extra_instruction,
                "workdir": self.context.workdir,
                "step_limit": self.config.step_limit,
                "step_timeout_sec": min(budget, self.config.step_timeout_sec),
                "process_registry": f"/tmp/{self.context.session_id}.pids",
            },
        )
        remote = self.remote_directory
        command = (
            f"echo $$ > {remote}/runner.pid; echo $$ >> {remote}/processes; "
            f"echo $$ >> /tmp/{self.context.session_id}.pids; "
            f"exec {remote}/venv/bin/python {remote}/runner.py {remote}"
        )
        result = await self.sandbox.exec(
            f"setsid --fork bash -c {quote(command)} > {remote}/runner.log 2>&1 < /dev/null",
            user=self.context.user,
            cwd=self.context.workdir,
            timeout_s=30,
        )
        if result.return_code:
            raise RuntimeError(f"mini-SWE runner launch failed: {result.stdout}\n{result.stderr}")

    async def _download_json(self, name: str) -> dict:
        local = self.directory / name
        await self.sandbox.download(f"{self.remote_directory}/{name}", local)
        try:
            return json.loads(local.read_text())
        finally:
            local.unlink(missing_ok=True)

    async def _next_event(self, index: int) -> dict:
        remote = self.remote_directory
        result = await self.sandbox.exec(
            f"if [ -f {remote}/output.json ]; then echo output; "
            f"elif [ -f {remote}/request-{index}.json ]; then echo request; "
            f"elif [ -f {remote}/runner.pid ] && ! kill -0 $(cat {remote}/runner.pid) 2>/dev/null; then "
            "echo exited; else echo waiting; fi",
            user=self.context.user,
            timeout_s=30,
        )
        if result.return_code:
            raise RuntimeError(f"mini-SWE relay read failed: {result.stderr}")
        kind = result.stdout.strip()
        if kind == "output":
            return {"kind": "output", "payload": await self._download_json("output.json")}
        if kind == "request":
            return await self._download_json(f"request-{index}.json")
        return {"kind": kind}

    async def close(self) -> None:
        """Stop runner and tool process groups before resources begins verification."""
        remote = self.remote_directory
        result = await self.sandbox.exec(
            f"if [ -f {remote}/processes ]; then "
            f"while read pid; do kill -TERM -- -$pid 2>/dev/null || true; done < {remote}/processes; "
            "sleep 0.2; "
            f"while read pid; do kill -KILL -- -$pid 2>/dev/null || true; done < {remote}/processes; fi",
            user=self.context.user,
            timeout_s=10,
        )
        if result.return_code:
            raise RuntimeError(f"mini-SWE process cleanup failed: {result.stderr}")

    async def _read_result(self) -> dict:
        remote = self.remote_directory
        result = await self.sandbox.exec(
            f"if [ -f {remote}/output.json ]; then echo output; "
            f"elif [ -f {remote}/trajectory.json ]; then echo trajectory; fi",
            user=self.context.user,
            timeout_s=30,
        )
        if result.return_code:
            raise RuntimeError(f"mini-SWE result read failed: {result.stderr}")
        if result.stdout.strip() == "output":
            return await self._download_json("output.json")
        if result.stdout.strip() == "trajectory":
            return {"mini_swe_trajectory": await self._download_json("trajectory.json")}
        return {}

    async def execute(self, budget: float) -> tuple[NeMoGymResponse, HarnessOutcome, dict]:
        responses = []
        output_items = []
        invocation = observations = trajectory = conversation_adapter = None
        if self.observability_enabled:
            invocation = AgentInvocation(invocation_id=self.context.session_id)
            observations = AgentObservationBundle(source="miniswe", records=[invocation])
            trajectory = TrajectoryRecord(
                task_id=self.context.task_id or self.context.session_id,
                rollout_id=self.context.rollout_id or self.context.session_id,
            )
            conversation_adapter = TypeAdapter(list[NeMoGymResponseInputItem])
            execution_started = monotonic()
        step_count = 0

        async def query(messages):
            params = self.params.model_dump(exclude_none=True)
            params["input"] = responses_input(messages)
            # mini-SWE executes bash calls; task MCP tools are discovered in setup
            # and made available through the task-local CLI described in the prompt.
            params["tools"] = [{"type": "function", **BASH_TOOL["function"], "strict": False}]
            if self.observability_enabled:
                invocation.conversation = conversation_adapter.validate_python(params["input"])
            response = await self.query(params)
            responses.append(response)
            output_items.extend(response.output)
            if self.observability_enabled:
                invocation.conversation.extend(response.output)
                turn_model_calls = []
                if response.id:
                    reference = ModelCallRef(
                        model_ref=ModelServerRef(type="responses_api_models", name=self.model_name),
                        response_id=response.id,
                    )
                    invocation.model_calls.append(reference)
                    turn_model_calls.append(reference)
                else:
                    trajectory.gaps.append(
                        ObservationGap(
                            code="model_call_reference_unavailable",
                            invocation_id=invocation.invocation_id,
                            detail=f"turn:{len(trajectory.turns) + 1}",
                        )
                    )
                # Record the decision before parsing: rejected model output is still
                # an observed decision, and must not disappear during recovery.
                trajectory.turns.append(
                    TrajectoryTurn(
                        invocation_id=invocation.invocation_id,
                        task_id=trajectory.task_id,
                        rollout_id=trajectory.rollout_id,
                        turn_no=len(trajectory.turns) + 1,
                        timestamp=time(),
                        question=params["input"],
                        answer=[item.model_dump(mode="json") for item in response.output],
                        reasoning_content=[
                            item.model_dump(mode="json") for item in response.output if item.type == "reasoning"
                        ]
                        or None,
                        step_count=step_count,
                        model_calls=turn_model_calls,
                    )
                )
            return {"response": response.model_dump(mode="json", exclude_none=True)}

        active_tool = None
        active_started = None
        result = {}
        termination = HarnessOutcome(reason="completed")
        try:
            async with asyncio.timeout(budget):
                await self._start_runner(budget)
                index = 0
                while True:
                    event = await self._next_event(index)
                    kind = event["kind"]
                    if kind == "waiting":
                        await asyncio.sleep(0.05)
                        continue
                    if kind == "exited":
                        logs = await self.sandbox.exec(f"cat {self.remote_directory}/runner.log", timeout_s=30)
                        raise RuntimeError(f"mini-SWE runner exited without a result: {logs.stdout}")
                    if kind == "output":
                        result = event["payload"]
                        termination = HarnessOutcome.model_validate(result["termination"])
                        break
                    reply = {}
                    if kind == "query":
                        try:
                            reply = await query(event["messages"])
                        except Exception as error:
                            detail = getattr(error, "response_content", b"").decode(errors="replace")
                            overflow = (
                                isinstance(error, ClientResponseError)
                                and error.status == 400
                                and (
                                    "context_length_exceeded" in detail
                                    or "context length" in detail.lower()
                                    or "maximum model length" in detail.lower()
                                    or ("max_tokens" in detail and "too large" in detail.lower())
                                )
                            )
                            reply = {
                                "error": {
                                    "type": "ContextWindowExceeded" if overflow else type(error).__name__,
                                    "detail": detail or str(error),
                                }
                            }
                    elif kind == "tool_started":
                        if self.observability_enabled:
                            active_tool = ToolCallObservation(
                                invocation_id=invocation.invocation_id,
                                tool_call_id=event["action"]["tool_call_id"],
                                tool_name="bash",
                                started_at=event["started_at"],
                                timing_source="executor",
                                status="incomplete",
                            )
                            active_started = monotonic()
                            observations.records.append(active_tool)
                    elif kind == "tool_finished":
                        item = NeMoGymFunctionCallOutput.model_validate(responses_input([event["message"]])[0])
                        output_items.append(item)
                        if self.observability_enabled:
                            invocation.conversation.append(item)
                            active_tool.status = event["status"]
                            active_tool.error_type = event["error_type"]
                            active_tool.duration_ms = event["duration_ms"]
                            if event["completed_at"] >= active_tool.started_at:
                                active_tool.completed_at = event["completed_at"]
                            else:
                                observations.gaps.append(
                                    ObservationGap(
                                        code="tool_clock_moved_backwards",
                                        invocation_id=invocation.invocation_id,
                                    )
                                )
                            active_tool = None
                            step_count += 1
                            if trajectory.turns:
                                trajectory.turns[-1].step_count = step_count
                    else:
                        raise RuntimeError(f"Unknown mini-SWE relay event: {kind}")
                    await self._upload_json(f"response-{index}.json", reply)
                    index += 1
        except asyncio.CancelledError:
            termination = HarnessOutcome(reason="cancelled")
        except TimeoutError:
            termination = HarnessOutcome(reason="timeout")
        except Exception as exc:
            termination = HarnessOutcome(reason="infrastructure_error", detail=f"{type(exc).__name__}: {exc}")
        finally:
            try:
                await self.close()
            except Exception as error:
                LOGGER.exception("Failed to stop mini-SWE processes; resources must quiesce the sandbox")
                termination = HarnessOutcome(reason="infrastructure_error", detail=f"Process cleanup failed: {error}")
            if active_tool is not None:
                active_tool.status = (
                    "cancelled"
                    if termination.reason == "cancelled"
                    else "timeout"
                    if termination.reason == "timeout"
                    else "failed"
                )
                active_tool.duration_ms = (monotonic() - active_started) * 1000
                active_tool.error_type = termination.reason
            if not result:
                try:
                    result = await self._read_result()
                except Exception:
                    # Preserve already captured responses when the sandbox transport fails.
                    LOGGER.exception("Unable to retrieve the partial mini-SWE trajectory")
        native_trajectory = result.get("mini_swe_trajectory")
        if native_trajectory is not None:
            (self.directory / "trajectory.json").write_text(json.dumps(native_trajectory, indent=2))
        response = NeMoGymResponse(
            id="resp_" + uuid4().hex,
            created_at=int(time()),
            model=self.model_name,
            object="response",
            output=output_items,
            tool_choice=self.params.tool_choice,
            tools=self.params.tools,
            parallel_tool_calls=self.params.parallel_tool_calls,
            # An empty or partial sum is not a measured rollout total.
            usage=NeMoGymResponseUsage.sum_from_list([r.usage for r in responses])
            if responses and all(r.usage is not None for r in responses)
            else None,
        )
        extra = {key: result[key] for key in ("mini_swe_trajectory", "harness_version", "runtime") if key in result}
        if self.observability_enabled:
            invocation.status = (
                "completed"
                if termination.reason == "completed"
                else "failed"
                if termination.reason == "infrastructure_error"
                else "incomplete"
            )
            invocation.duration_ms = (monotonic() - execution_started) * 1000
            extra["ng_agent_observations"] = observations.model_dump(mode="json")
            extra["ng_trajectory"] = trajectory.model_dump(mode="json")
        termination.artifacts = [str(self.directory / "trajectory.json")] if native_trajectory is not None else []
        self.result = (response, termination, extra)

        return self.result
