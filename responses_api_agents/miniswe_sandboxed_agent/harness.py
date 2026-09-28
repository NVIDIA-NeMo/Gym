# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run mini-SWE in a borrowed sandbox against Gym's model-server URL."""

import asyncio
import base64
import binascii
import json
import logging
from pathlib import Path
from shlex import quote
from time import monotonic, time
from typing import Any
from uuid import uuid4

from minisweagent.models.utils.actions_toolcall_response import BASH_TOOL_RESPONSE_API
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
from responses_api_agents.miniswe_sandboxed_agent.bootstrap import bootstrap_assets


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


def original_response_id(response_id: str) -> str:
    """Undo LiteLLM's routing envelope so refs join Gym's captured response IDs."""
    if response_id.startswith("resp_"):
        try:
            decoded = base64.b64decode(response_id[5:], validate=True).decode()
            if decoded.startswith("litellm:custom_llm_provider:") and ";response_id:" in decoded:
                return decoded.split(";response_id:", 1)[1]
        except (ValueError, UnicodeDecodeError, binascii.Error):
            pass
    return response_id


def project_native_trajectory(trajectory: dict) -> tuple[list[dict], list[dict]]:
    """Project saved native decisions and observations without inventing timings."""
    history, tools, conversation = [], [], []
    for message in trajectory.get("messages", []):
        extra = message.get("extra", {})
        response = message if message.get("object") == "response" else extra.get("response")
        if isinstance(response, dict) and response.get("object") == "response":
            response = dict(response, id=original_response_id(response.get("id", "")))
            history.append(
                {
                    "request": {"input": list(conversation)},
                    "response": {k: v for k, v in response.items() if k != "extra"},
                    "timestamp": extra.get("timestamp", response.get("created_at", 0)),
                }
            )
            # Rejected responses are persisted as evidence, but native mini-SWE
            # does not replay them in the next request.
            if message.get("object") == "response":
                conversation.extend(response.get("output", []))
                for action in extra.get("actions", []):
                    tools.append(
                        {"tool_call_id": action["tool_call_id"], "model_index": len(history), "status": "incomplete"}
                    )
        if message.get("type") == "function_call_output":
            observation = {k: v for k, v in message.items() if k != "extra"}
            conversation.append(observation)
            tool = next((t for t in reversed(tools) if t["tool_call_id"] == message["call_id"]), None)
            if tool is not None:
                error = extra.get("exception_type")
                tool.update(
                    message={"role": "tool", "tool_call_id": message["call_id"], "content": message["output"]},
                    completed_at=extra.get("timestamp"),
                    status="timeout"
                    if error == "TimeoutExpired"
                    else "failed"
                    if extra.get("returncode")
                    else "completed",
                    error_type=error,
                )
        elif message.get("role") in {"system", "user", "assistant"}:
            conversation.append({k: v for k, v in message.items() if k != "extra"})
    return history, tools


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
        await self._install_miniswe()
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
            command = (
                f"{self.remote_directory}/uv --no-config venv {remote} --python {self.remote_directory}/python/bin/python3 && "
                f"{self.remote_directory}/uv --no-config pip install --python {remote}/bin/python mcp==1.29.0 httpx-aiohttp==0.2.0"
            )
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

    async def _install_miniswe(self) -> None:
        remote = self.remote_directory
        result = await self.sandbox.exec(
            f"mkdir -p {remote} && uname -m && "
            "if ls /lib/ld-musl-*.so.1 >/dev/null 2>&1; then echo musl; else echo gnu; fi",
            user=self.context.user,
            cwd=self.context.workdir,
            timeout_s=self.context.setup_timeout_sec,
        )
        if result.return_code:
            raise RuntimeError(f"mini-SWE bootstrap probe failed: {result.stdout}\n{result.stderr}")
        arch, libc = result.stdout.strip().splitlines()
        uv, python = await bootstrap_assets(arch, libc)
        await self.sandbox.upload(uv, remote + "/uv.tar.gz")
        await self.sandbox.upload(python, remote + "/python.tar.gz")
        # Extract as the task user: uploads may be root-owned and non-root task
        # users must own both the executable and the new Python environment.
        result = await self.sandbox.exec(
            f"tar -xzf {remote}/uv.tar.gz -C {remote} --strip-components=1 "
            f"uv-{arch}-unknown-linux-musl/uv && "
            f"tar -xzf {remote}/python.tar.gz -C {remote} && "
            f"UV_PYTHON_DOWNLOADS=never {remote}/uv --no-config venv {remote}/venv --python {remote}/python/bin/python3 && "
            f"{remote}/uv --no-config pip install --python {remote}/venv/bin/python mini-swe-agent==2.4.6",
            user=self.context.user,
            cwd=self.context.workdir,
            timeout_s=self.context.setup_timeout_sec,
        )
        if result.return_code:
            raise RuntimeError(f"mini-SWE installation failed: {result.stdout}\n{result.stderr}")

    async def close(self) -> None:
        """Stop the agent and its shell process groups before verification."""
        cleanup = asyncio.create_task(self._close())
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        cleanup.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _close(self) -> None:
        # Native LocalEnvironment gives each command its own process group.
        # An inherited per-run marker also finds those groups after their parent
        # exits, including children orphaned by a cancelled sandbox exec.
        script = """
import os, signal, time
from pathlib import Path
marker = MARKER
pids = set()
for entry in (Path('/proc').iterdir() if Path('/proc').exists() else []):
    if not entry.name.isdigit():
        continue
    try:
        if marker in (entry / 'environ').read_bytes().split(b'\\0'):
            pids.add(int(entry.name))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
for sig in (signal.SIGTERM, signal.SIGKILL):
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
    time.sleep(0.2)
""".replace("MARKER", repr(f"MSWEA_GLOBAL_CONFIG_DIR={self.remote_directory}/config".encode()))
        registry = f"{self.remote_directory}/processes"
        result = await self.sandbox.exec(
            f"{quote(self.remote_directory + '/venv/bin/python')} -c {quote(script)} && "
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
        remote = self.remote_directory
        model_kwargs = self.params.model_dump(exclude_none=True)
        for key in ("input", "model", "tools", "stream"):
            model_kwargs.pop(key, None)
        model_kwargs.update(
            api_base=self.model_base_url,
            api_key="dummy_key",
            timeout=max(1, budget),
            max_retries=0,
            extra_headers={"x-session-id": self.context.session_id},
            extra_body={"tools": [{**BASH_TOOL_RESPONSE_API, "strict": False}]},
        )
        if "max_output_tokens" in model_kwargs:
            # LiteLLM's OpenAI adapter raises caps below 16, even for Gym URLs.
            # Pass the requested cap unchanged and let the model server validate it.
            model_kwargs["extra_body"]["max_output_tokens"] = model_kwargs.pop("max_output_tokens")
        payload = {
            "run": {"task": self.context.instruction + self.extra_instruction},
            "agent": {
                "agent_class": "default",
                "step_limit": self.config.step_limit,
                "cost_limit": 0,
            },
            "environment": {
                "environment_class": "local",
                "cwd": self.context.workdir or "",
                "timeout": max(1, int(min(budget, self.config.step_timeout_sec))),
            },
            "model": {
                "model_class": "litellm_response",
                "model_name": "openai/" + self.model_name,
                "cost_tracking": "ignore_errors",
                "model_kwargs": model_kwargs,
            },
        }
        local_config = self.directory / "config.yaml"
        local_config.write_text(json.dumps(payload))
        command = (
            f"echo $$ >> {quote(remote + '/processes')}; "
            f"echo $$ >> {quote('/tmp/' + self.context.session_id + '.pids')}; "
            "export MSWEA_CONFIGURED=true MSWEA_SILENT_STARTUP=true LITELLM_LOCAL_MODEL_COST_MAP=true "
            "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1; "
            f"export MSWEA_GLOBAL_CONFIG_DIR={quote(remote + '/config')}; "
            f"exec {quote(remote + '/venv/bin/python')} -m minisweagent.run.mini "
            f"-c mini.yaml -c {quote(remote + '/config.yaml')} -o {quote(remote + '/trajectory.json')} "
            f"> {quote(remote + '/agent.log')} 2>&1"
        )
        termination = HarnessOutcome(reason="completed")
        started = monotonic()
        run_result = None
        try:
            await self.sandbox.upload(local_config, f"{remote}/config.yaml")
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
            except asyncio.CancelledError:
                termination = HarnessOutcome(reason="cancelled")
            except Exception as error:
                LOGGER.exception("Failed to stop mini-SWE processes; resources must quiesce the sandbox")
                termination = HarnessOutcome(reason="infrastructure_error", detail=f"Process cleanup failed: {error}")

        try:
            await self.sandbox.download(f"{remote}/agent.log", self.directory / "agent.log")
        except Exception:
            LOGGER.warning("Unable to retrieve mini-SWE stdout/stderr", exc_info=True)
        try:
            native_trajectory = await self._download_artifact("trajectory.json")
        except Exception:
            native_trajectory = None
        result = native_trajectory or {}
        info = result.get("info", {})
        if termination.reason == "completed" or (termination.detail or "").startswith("mini-SWE command exited"):
            status = info.get("exit_status")
            last_extra = (result.get("messages") or [{}])[-1].get("extra", {})
            error = last_extra.get("exception_str", "")
            context_overflow = status == "BadRequestError" and (
                "context_length_exceeded" in error
                or "context length" in error.lower()
                or "maximum model length" in error.lower()
                or ("max_tokens" in error and "too large" in error.lower())
            )
            if context_overflow:
                termination = HarnessOutcome(reason="nonzero_exit", detail="ContextWindowExceededError")
            elif status == "Submitted":
                termination = HarnessOutcome(reason="completed")
            elif status in {"LimitsExceeded", "RepeatedFormatError", "ContextWindowExceededError", "TimeExceeded"}:
                termination = HarnessOutcome(reason="nonzero_exit", detail=status)
            else:
                termination = HarnessOutcome(
                    reason="infrastructure_error", detail=status or "mini-SWE produced no trajectory"
                )

        history, tool_history = project_native_trajectory(result)
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
        extra = {"harness_version": info.get("mini_version")}
        if native_trajectory is not None:
            extra["mini_swe_trajectory"] = native_trajectory
            termination.artifacts = [str(self.directory / "trajectory.json")]
        if (self.directory / "agent.log").exists():
            termination.artifacts.append(str(self.directory / "agent.log"))
        if self.observability_enabled:
            invocation = AgentInvocation(invocation_id=self.context.session_id)
            observations = AgentObservationBundle(source="miniswe", records=[invocation])
            trajectory = TrajectoryRecord(
                task_id=self.context.task_id or self.context.session_id,
                rollout_id=self.context.rollout_id or self.context.session_id,
            )
            if native_trajectory is None or termination.reason in {"cancelled", "timeout"}:
                trajectory.gaps.append(
                    ObservationGap(
                        code="native_trajectory_unavailable"
                        if native_trajectory is None
                        else "native_trajectory_partial",
                        invocation_id=invocation.invocation_id,
                        detail="Native mini-SWE saves after each step; an interrupted active step may be absent.",
                    )
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
                    status = termination.reason if termination.reason in {"cancelled", "timeout"} else "incomplete"
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
                        started_at=tool.get("started_at"),
                        completed_at=tool.get("completed_at"),
                        duration_ms=tool.get("duration_ms"),
                        timing_source="artifact",
                        status=status,
                        error_type=tool.get("error_type")
                        or (
                            termination.reason
                            if tool.get("status") == "incomplete" and termination.reason != "completed"
                            else None
                        ),
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
