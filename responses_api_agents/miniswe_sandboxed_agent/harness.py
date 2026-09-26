# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run mini-SWE in a borrowed sandbox against Gym's model-server URL."""

import asyncio
import json
import logging
from pathlib import Path
from shlex import quote
from time import monotonic, time
from typing import Any
from uuid import uuid4

from aiohttp import ClientTimeout
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
from nemo_gym.server_utils import request


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
        model_base_url: str,
        model_name: str,
        directory: Path,
        observability_enabled: bool = False,
    ) -> None:
        self.sandbox = sandbox
        self.context = context
        self.config = config
        self.params = params
        self.model_base_url = model_base_url
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
        result = await self.sandbox.exec(
            f"mkdir -p {remote} && python3 -c 'import platform; print(platform.machine())'",
            user=self.context.user,
            cwd=self.context.workdir,
            timeout_s=self.context.setup_timeout_sec,
        )
        if result.return_code:
            raise RuntimeError(f"mini-SWE bootstrap probe failed: {result.stdout}\n{result.stderr}")
        arch = result.stdout.strip()
        if arch not in {"x86_64", "aarch64"}:
            raise RuntimeError(f"Unsupported mini-SWE sandbox architecture: {arch!r}")
        # Minimal task images may lack CA certificates. Fetch uv with Gym's TLS
        # transport, selecting the sandbox's architecture rather than the host's.
        url = f"https://github.com/astral-sh/uv/releases/download/0.10.12/uv-{arch}-unknown-linux-musl.tar.gz"
        archive_path = self.directory / "uv.tar.gz"
        try:
            async with await request("GET", url, timeout=ClientTimeout(total=120)) as response:
                response.raise_for_status()
                archive_path.write_bytes(await response.read())
            await self.sandbox.upload(archive_path, remote + "/uv.tar.gz")
        finally:
            archive_path.unlink(missing_ok=True)
        script = (
            "import pathlib,tarfile; "
            f"archive=tarfile.open('{remote}/uv.tar.gz',mode='r:gz'); "
            "member=next(m for m in archive if m.name.endswith('/uv')); "
            f"target=pathlib.Path('{remote}/uv'); "
            "target.write_bytes(archive.extractfile(member).read()); target.chmod(0o755)"
        )
        result = await self.sandbox.exec(
            f"python3 -c {quote(script)} && "
            f"{remote}/uv venv {remote}/venv --python 3.13 && "
            f"{remote}/uv pip install --python {remote}/venv/bin/python mini-swe-agent==2.4.6",
            user=self.context.user,
            cwd=self.context.workdir,
            timeout_s=self.context.setup_timeout_sec,
        )
        if result.return_code:
            raise RuntimeError(f"mini-SWE runner installation failed: {result.stdout}\n{result.stderr}")
        await self.sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), remote + "/runner.py")

    async def close(self) -> None:
        """Stop the agent and its shell process groups before verification."""
        registry = f"{self.remote_directory}/processes"
        result = await self.sandbox.exec(
            f"if [ -f {quote(registry)} ]; then "
            f"while read pid; do kill -TERM -- -$pid 2>/dev/null || true; done < {quote(registry)}; "
            "sleep 0.2; "
            f"while read pid; do kill -KILL -- -$pid 2>/dev/null || true; done < {quote(registry)}; fi",
            user=self.context.user,
            timeout_s=10,
        )
        if result.return_code:
            raise RuntimeError(f"mini-SWE process cleanup failed: {result.stderr}")

    async def _download_artifact(self, name: str) -> dict:
        local = self.directory / name
        await self.sandbox.download(f"{self.remote_directory}/{name}", local)
        return json.loads(local.read_text())

    async def execute(self, budget: float) -> tuple[NeMoGymResponse, HarnessOutcome, dict]:
        """Run one sandbox command and collect its persisted native artifacts."""
        payload = {
            "instruction": self.context.instruction + self.extra_instruction,
            "workdir": self.context.workdir,
            "step_limit": self.config.step_limit,
            "step_timeout_sec": min(budget, self.config.step_timeout_sec),
            "model_timeout_sec": max(1, budget),
            "model_base_url": self.model_base_url,
            "model_params": self.params.model_dump(exclude_none=True),
            "session_id": self.context.session_id,
            "process_registry": f"/tmp/{self.context.session_id}.pids",
        }
        local_config = self.directory / "input.json"
        local_config.write_text(json.dumps(payload))
        remote = self.remote_directory
        command = (
            f"echo $$ >> {quote(remote + '/processes')}; "
            f"echo $$ >> {quote(payload['process_registry'])}; "
            f"exec {quote(remote + '/venv/bin/python')} {quote(remote + '/runner.py')} {quote(remote)}"
        )
        termination = HarnessOutcome(reason="completed")
        started = monotonic()
        run_result = None
        try:
            await self.sandbox.upload(local_config, f"{remote}/input.json")
            # --wait keeps this single exec open until mini-SWE exits. There is
            # no host-side request loop or sandbox polling between model calls.
            run_result = await self.sandbox.exec(
                f"setsid --fork --wait bash -c {quote(command)}",
                user=self.context.user,
                cwd=self.context.workdir,
                timeout_s=budget,
            )
            if run_result.error_type:
                termination = HarnessOutcome(reason="infrastructure_error", detail=run_result.error_type)
            elif run_result.return_code:
                termination = HarnessOutcome(
                    reason="infrastructure_error",
                    detail=f"mini-SWE command exited {run_result.return_code}: {run_result.stderr}",
                )
        except asyncio.CancelledError:
            termination = HarnessOutcome(reason="cancelled")
        except TimeoutError:
            termination = HarnessOutcome(reason="timeout")
        except Exception as error:
            termination = HarnessOutcome(reason="infrastructure_error", detail=f"{type(error).__name__}: {error}")
        finally:
            try:
                await self.close()
            except Exception as error:
                LOGGER.exception("Failed to stop mini-SWE processes; resources must quiesce the sandbox")
                termination = HarnessOutcome(reason="infrastructure_error", detail=f"Process cleanup failed: {error}")

        try:
            result = await self._download_artifact("result.json")
        except Exception:
            LOGGER.exception("Unable to retrieve the mini-SWE run artifact")
            result = {}
        try:
            native_trajectory = await self._download_artifact("trajectory.json")
        except Exception:
            native_trajectory = None
        if native_trajectory is not None:
            (self.directory / "trajectory.json").write_text(json.dumps(native_trajectory, indent=2))
        if termination.reason == "completed":
            if result.get("termination"):
                termination = HarnessOutcome.model_validate(result["termination"])
            else:
                termination = HarnessOutcome(reason="infrastructure_error", detail="mini-SWE produced no result")

        history = result.get("model_history", [])
        tool_history = result.get("tool_history", [])
        responses = []
        for entry in history:
            try:
                responses.append(NeMoGymResponse.model_validate(entry["response"]))
            except Exception:
                LOGGER.exception("Unable to project a malformed mini-SWE model response")
                termination = HarnessOutcome(reason="infrastructure_error", detail="Invalid model response")
                break
        history = history[: len(responses)]
        output_items = []
        for index, model_response in enumerate(responses, start=1):
            output_items.extend(model_response.output)
            for tool in tool_history:
                if tool["model_index"] == index and tool.get("message") is not None:
                    output_items.append(
                        NeMoGymFunctionCallOutput.model_validate(responses_input([tool["message"]])[0])
                    )
        response = NeMoGymResponse(
            id="resp_" + uuid4().hex,
            created_at=int(time()),
            model=self.model_name,
            object="response",
            output=output_items,
            tool_choice=self.params.tool_choice,
            tools=self.params.tools,
            parallel_tool_calls=self.params.parallel_tool_calls,
            usage=NeMoGymResponseUsage.sum_from_list([r.usage for r in responses])
            if responses and all(r.usage is not None for r in responses)
            else None,
        )
        extra = {key: result[key] for key in ("harness_version", "runtime") if key in result}
        if native_trajectory is not None:
            extra["mini_swe_trajectory"] = native_trajectory
            termination.artifacts = [str(self.directory / "trajectory.json")]
        if self.observability_enabled:
            invocation = AgentInvocation(invocation_id=self.context.session_id)
            observations = AgentObservationBundle(source="miniswe", records=[invocation])
            trajectory = TrajectoryRecord(
                task_id=self.context.task_id or self.context.session_id,
                rollout_id=self.context.rollout_id or self.context.session_id,
            )
            adapter = TypeAdapter(list[NeMoGymResponseInputItem])
            for index, entry in enumerate(history, start=1):
                model_response = responses[index - 1]
                request_items = entry["request"]["input"]
                invocation.conversation = adapter.validate_python(request_items)
                invocation.conversation.extend(model_response.output)
                reference = None
                if model_response.id:
                    reference = ModelCallRef(
                        model_ref=ModelServerRef(type="responses_api_models", name=self.model_name),
                        response_id=model_response.id,
                    )
                    invocation.model_calls.append(reference)
                else:
                    trajectory.gaps.append(
                        ObservationGap(
                            code="model_call_reference_unavailable",
                            invocation_id=invocation.invocation_id,
                            detail=f"turn:{index}",
                        )
                    )
                trajectory.turns.append(
                    TrajectoryTurn(
                        invocation_id=invocation.invocation_id,
                        task_id=trajectory.task_id,
                        rollout_id=trajectory.rollout_id,
                        turn_no=index,
                        timestamp=entry["timestamp"],
                        question=request_items,
                        answer=[item.model_dump(mode="json") for item in model_response.output],
                        reasoning_content=[
                            item.model_dump(mode="json") for item in model_response.output if item.type == "reasoning"
                        ]
                        or None,
                        step_count=sum(
                            1
                            for tool in tool_history
                            if tool["model_index"] <= index and tool.get("message") is not None
                        ),
                        model_calls=[reference] if reference else [],
                    )
                )
            for tool in tool_history:
                status = tool.get("status", "incomplete")
                if status == "incomplete":
                    status = termination.reason if termination.reason in {"cancelled", "timeout"} else "failed"
                if tool.get("duration_ms") is None:
                    trajectory.gaps.append(
                        ObservationGap(
                            code="tool_timing_unavailable",
                            invocation_id=invocation.invocation_id,
                            detail=tool["tool_call_id"],
                        )
                    )
                observations.records.append(
                    ToolCallObservation(
                        invocation_id=invocation.invocation_id,
                        tool_call_id=tool["tool_call_id"],
                        tool_name="bash",
                        started_at=tool["started_at"],
                        completed_at=tool.get("completed_at"),
                        duration_ms=tool.get("duration_ms"),
                        timing_source="executor",
                        status=status,
                        error_type=tool.get("error_type")
                        or (termination.reason if tool.get("status") == "incomplete" else None),
                    )
                )
                if tool.get("message") is not None and tool["model_index"] == len(history):
                    invocation.conversation.append(
                        NeMoGymFunctionCallOutput.model_validate(responses_input([tool["message"]])[0])
                    )
            invocation.status = (
                "completed"
                if termination.reason == "completed"
                else "failed"
                if termination.reason == "infrastructure_error"
                else "incomplete"
            )
            invocation.duration_ms = (monotonic() - started) * 1000
            extra["ng_agent_observations"] = observations.model_dump(mode="json")
            extra["ng_trajectory"] = trajectory.model_dump(mode="json")
        self.result = (response, termination, extra)
        return self.result
