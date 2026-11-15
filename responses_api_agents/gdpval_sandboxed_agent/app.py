#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenHands harness connected to a GDPVal resources server's remote sandbox."""

import asyncio
import json
import logging
import shutil
import stat
import tempfile
import time
import uuid
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from shlex import quote
from types import SimpleNamespace
from typing import Any, Optional

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field, PrivateAttr, field_validator

from nemo_gym.agents import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import get_server_url, raise_for_status
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.sandbox import AsyncSandbox, create_provider, resolve_provider_config
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.gdpval_sandboxed_agent.compaction import export_segments_from_trace
from responses_api_agents.gdpval_sandboxed_agent.sandbox_runtime import DEFAULT_PYTHON, install_runtime
from responses_api_agents.gdpval_sandboxed_agent.trajectory import export_trajectory


logger = logging.getLogger(__name__)
MAX_TRACE_FILES = 10_000
MAX_TRACE_BYTES = 1_073_741_824
MAX_TRACE_FILE_BYTES = 67_108_864


class TraceExportError(RuntimeError):
    """Saved controller output could not be safely reconstructed for training."""


try:  # the OpenSandbox SDK is only present in the agent's venv
    from opensandbox.exceptions.sandbox import SandboxApiException
except Exception:  # pragma: no cover

    class SandboxApiException(Exception):  # type: ignore[no-redef]
        """Placeholder so isinstance checks stay valid without the SDK."""


def _load_web_keys(path: Optional[str]) -> list[str]:
    """One key per line, blanks and ``#`` comments ignored; deduplicated, never logged."""
    if not path:
        raise ValueError("web_tools_enabled requires web_search_api_keys_file")
    keys: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key = line.split("#", 1)[0].strip()
        if key and key not in keys:
            keys.append(key)
    if not keys:
        raise ValueError(f"no web search keys in {path}")
    return keys


class GDPValSandboxedAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    resources_server: ResourcesServerRef
    # Each session stays on one named, single-worker resources server.
    resources_servers: list[ResourcesServerRef] = Field(default_factory=list)
    sandbox_provider: str = "sandbox"
    sandbox_python: str = DEFAULT_PYTHON
    remote_openhands_install_script_path: str = Field(
        description="Absolute sandbox path to the staged runtime installer."
    )
    remote_openhands_archive_path: str = Field(
        description="Absolute sandbox path to the runtime archive and .sha256 sidecar."
    )
    harness_install_timeout: int = Field(default=900, ge=31)
    # Total capacity across all Uvicorn workers; must divide evenly by num_workers.
    concurrency: int = Field(default=32, ge=1)
    # None removes the practical turn cap; agent_timeout still bounds execution.
    agent_max_turns: int | None = Field(default=200, ge=1)
    agent_timeout: int = Field(default=3600, ge=1)
    model_read_timeout_s: int = Field(default=1800, ge=1)
    # Re-issue a policy call that times out / is cut this many times (stateless request).
    model_read_retries: int = Field(default=0, ge=0)
    # Optional ``web_search`` / ``fetch_web_page`` (Exa backend) as an MCP stdio server inside
    # the sandbox (web_tools_mcp.py). The keys are shipped into the sandbox's agent config, where
    # the model has a shell: use a dedicated key set. Keys file: one key per line, # comments ok.
    web_tools_enabled: bool = False
    web_search_api_keys_file: Optional[str] = None
    web_search_max_sweeps: int = Field(default=1, ge=1)
    max_context: int = Field(default=196608, ge=1)
    max_output_tokens: int = Field(default=32768, ge=1)
    # Context compaction (default off). Like the Stirrup agent's field of the same name: when
    # the last reply's prompt + completion tokens reach this fraction of max_context, OpenHands
    # summarizes everything after the task prompt in one policy call (compaction_condenser.py)
    # and continues from system prompt + task + summary. /run then returns one response per
    # context segment (``responses``) so NeMo RL can train every segment as its own trace
    # (async_rl.multi_trace_enabled + env.nemo_gym.training_mode=compaction, token echo).
    context_summarization_cutoff: Optional[float] = Field(default=None, gt=0, le=1)
    # Per-event character budget in the summarization prompt (OpenHands max_event_length).
    context_summarization_max_event_chars: int = Field(default=10_000, ge=100)
    checkpoint: str

    @field_validator("remote_openhands_install_script_path", "remote_openhands_archive_path", "sandbox_python")
    @classmethod
    def require_absolute_runtime_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("Runtime paths must be absolute sandbox paths")
        return value


class GDPValRunResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


def _incomplete_response(model: str, tag: str) -> NeMoGymResponse:
    """Token-free placeholder: nothing is trained, the capture finalizer masks the row."""
    return NeMoGymResponse(
        id=f"gdpval-{tag}-{uuid.uuid4().hex[:12]}",
        created_at=time.time(),
        model=model,
        object="response",
        output=[],
        tools=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        status="incomplete",
    )


class GDPValSandboxedAgent(SimpleResponsesAPIAgent):
    config: GDPValSandboxedAgentConfig
    # No Ray inside this server (Gym would otherwise initialize it in every worker).
    ray_enabled = False
    _sem: asyncio.Semaphore = PrivateAttr()
    _resource_load: dict[str, int] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        workers = self.config.num_workers or 1
        if workers < 1 or self.config.concurrency % workers:
            raise ValueError("GDPVal concurrency must be divisible by positive num_workers")
        per_worker = self.config.concurrency // workers
        self._sem = asyncio.Semaphore(per_worker)
        logger.warning(
            "GDPVal agent capacity: workers=%d per_worker=%d total=%d",
            workers,
            per_worker,
            self.config.concurrency,
        )
        refs = self.config.resources_servers or [self.config.resources_server]
        self._resource_load = {ref.name: 0 for ref in refs}
        if len(self._resource_load) != len(refs):
            raise ValueError("GDPVal resources_servers must contain unique server names")

    @asynccontextmanager
    async def _resource_route(self) -> AsyncIterator[str]:
        # Counts are worker-local. The divided semaphore bounds total activity
        # across workers; least-loaded routing balances each worker independently.
        # No await between selection and increment: concurrent runs in this
        # event loop cannot claim the same free slot before it is accounted for.
        name = min(self._resource_load, key=self._resource_load.get)
        self._resource_load[name] += 1
        try:
            yield name
        finally:
            self._resource_load[name] -= 1

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        raise HTTPException(405, "Use /run for an artifact-producing GDPVal rollout")

    async def _attach(self, sandbox_id: str) -> AsyncSandbox:
        provider = create_provider(
            resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
        )
        try:
            return await AsyncSandbox.connect({"sandbox_id": sandbox_id, "workdir": "/workspace"}, provider=provider)
        except BaseException:
            await provider.aclose()
            raise

    @staticmethod
    async def _download_trace(sandbox: AsyncSandbox, local: Path) -> None:
        script = """from pathlib import Path
import zipfile
root = Path('/trace')
if root.is_symlink():
    raise ValueError('symlink trace root')
count = total = 0
with zipfile.ZipFile('/tmp/gdpval-trace.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in root.rglob('*'):
        if p.is_symlink():
            raise ValueError('symlink trace')
        if p.is_file():
            size = p.stat().st_size
            count += 1
            total += size
            if count > 10000 or total > 1073741824 or size > 67108864:
                raise ValueError('OpenHands trace exceeds extraction limits')
            z.write(p, p.relative_to('/'))
"""
        result = await sandbox.exec("python3 -c " + quote(script), timeout_s=300)
        if result.error_type is not None or result.return_code != 0:
            raise RuntimeError(f"Failed to package OpenHands trace: {result.error_type}: {result.stderr}")
        archive = local / "trace.zip"
        await sandbox.download("/tmp/gdpval-trace.zip", archive)
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if (
                len(members) > MAX_TRACE_FILES
                or sum(member.file_size for member in members) > MAX_TRACE_BYTES
                or any(member.file_size > MAX_TRACE_FILE_BYTES for member in members)
            ):
                raise ValueError("OpenHands trace exceeds extraction limits")
            for member in members:
                path = PurePosixPath(member.filename)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in member.filename
                    or stat.S_ISLNK(member.external_attr >> 16)
                ):
                    raise ValueError(f"Unsafe OpenHands trace entry: {member.filename}")
                if not path.parts or path.parts[0] != "trace" or member.is_dir():
                    continue
                destination = local.joinpath(*path.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1_048_576)

    async def _execute(self, sandbox: AsyncSandbox, body: BaseRunRequest) -> dict[str, Any]:
        cfg = self.config
        model_root = self.base_url_for_run(get_server_url(cfg.model_server.name), body)
        agent_config = {
            "policy_model": "gdpval-policy",
            "policy_checkpoint": cfg.checkpoint,
            "policy_proxy_url": model_root + "/v1",
            "max_context": cfg.max_context,
            "max_output_tokens": cfg.max_output_tokens,
            "max_turns": cfg.agent_max_turns,
            "model_read_timeout_s": cfg.model_read_timeout_s,
            "model_read_retries": cfg.model_read_retries,
            "token_capture": True,
        }
        if cfg.web_tools_enabled:
            agent_config["web_tools"] = {
                "enabled": True,
                "keys": _load_web_keys(cfg.web_search_api_keys_file),
                "max_sweeps": cfg.web_search_max_sweeps,
            }
        if cfg.context_summarization_cutoff is not None:
            agent_config["context_summarization"] = {
                "cutoff": cfg.context_summarization_cutoff,
                "max_event_chars": cfg.context_summarization_max_event_chars,
            }
        await install_runtime(
            sandbox,
            script_path=cfg.remote_openhands_install_script_path,
            archive_path=cfg.remote_openhands_archive_path,
            timeout_s=cfg.harness_install_timeout,
        )
        with tempfile.TemporaryDirectory(prefix="gdpval-agent-") as temporary:
            local = Path(temporary)
            config_path = local / "agent-config.json"
            config_path.write_text(json.dumps(agent_config) + "\n")
            await sandbox.upload(config_path, "/control/agent-config.json")
            await sandbox.upload(Path(__file__).with_name("run_openhands.py"), "/runner/run_openhands.py")
            await sandbox.upload(
                Path(__file__).with_name("direct_model_client.py"),
                "/runner/direct_model_client.py",
            )
            if cfg.web_tools_enabled:
                await sandbox.upload(Path(__file__).with_name("web_tools_mcp.py"), "/runner/web_tools_mcp.py")
            if cfg.context_summarization_cutoff is not None:
                await sandbox.upload(
                    Path(__file__).with_name("compaction_condenser.py"),
                    "/runner/compaction_condenser.py",
                )
            exec_timed_out = False
            try:
                result = await sandbox.exec(
                    f"timeout -k 10 {cfg.agent_timeout} {quote(cfg.sandbox_python)} /runner/run_openhands.py --config /control/agent-config.json "
                    "--workspace /workspace --trace /trace",
                    timeout_s=cfg.agent_timeout + 30,
                )
            except TimeoutError:
                exec_timed_out = True
                # The remote execution budget expired. Preserve whatever the agent
                # completed and report an incomplete attempt rather than rerun it.
                result = SimpleNamespace(return_code=-1, error_type="timeout", stdout="", stderr="agent_exec_timeout")
            log_path = local / "agent.log"
            try:
                log_path.write_text((result.stdout or "") + "\n" + (result.stderr or ""))
                await sandbox.upload(log_path, "/trace/agent.log")
                await self._download_trace(sandbox, local)
            except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
                raise TraceExportError(f"Cannot collect OpenHands trace: {error}") from error
            failure_reason = None
            incomplete_reason = None
            if result.return_code != 0 or result.error_type is not None:
                status_path = local / "trace/agent-status.json"
                try:
                    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
                    if not isinstance(status, dict):
                        raise ValueError("OpenHands status must be a JSON object")
                except (OSError, ValueError) as error:
                    raise TraceExportError(f"Invalid OpenHands status: {error}") from error
                error = str(status.get("last_error") or log_path.read_text()[-8000:] or "No controller output")
                if result.error_type is None and status.get("agent_state") == "error":
                    if "context length limit has been exceeded" in error:
                        failure_reason = "agent_context_limit"
                    elif "Agent reached maximum iteration" in error:
                        failure_reason = "agent_iteration_limit"
                    elif error in {
                        "Agent got stuck in a loop",
                        "AgentStuckInLoopError: Agent got stuck in a loop",
                    }:
                        failure_reason = "agent_stuck_in_loop"
                if failure_reason is None:
                    # Our provider reports a deadline kill as exit 124 with error_type "timeout".
                    deadline_killed = result.return_code == 124 or result.error_type == "timeout"
                    if result.error_type is not None or result.return_code == -1 or deadline_killed:
                        incomplete_reason = (
                            "agent_exec_timeout" if exec_timed_out or deadline_killed else "agent_exec_incomplete"
                        )
                    elif "SocketTimeoutError" in error or "APITimeoutError" in error:
                        incomplete_reason = "model_response_timeout"
                    else:
                        # Unknown controller failure (e.g. an OpenHands TypeError on a
                        # malformed tool argument). Masked, never trained on, and the
                        # error is recorded on the row (agent_error) and logged, so it
                        # stays visible without failing /run -- which aborted the
                        # group's rollout stream and re-dispatched every sibling.
                        incomplete_reason = "agent_exec_incomplete"
                        agent_config["agent_error"] = (
                            f"OpenHands failed with exit code {result.return_code}: {error[-600:]}"
                        )
                        logger.warning("GDPVal %s", agent_config["agent_error"])
            agent_config["policy_failure_reason"] = failure_reason
            if incomplete_reason and not list((local / "trace/completions").glob("*.json")):
                # No invented policy tokens. The capture finalizer can produce an
                # invalid placeholder row for this explicitly masked empty attempt.
                params = body.responses_create_params
                response = NeMoGymResponse(
                    id="gdpval-incomplete-" + local.name,
                    created_at=time.time(),
                    model=agent_config["policy_model"],
                    object="response",
                    output=[],
                    tools=[],
                    tool_choice="auto",
                    parallel_tool_calls=True,
                    status="incomplete",
                )
            else:
                try:
                    params, response = export_trajectory(local, agent_config)
                except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                    raise TraceExportError(f"Invalid OpenHands trajectory: {error}") from error
            task_id = (body.responses_create_params.metadata or {}).get("task_id")
            body_out = {
                "task_id": task_id,
                "responses_create_params": params.model_dump(mode="json"),
                "response": response.model_dump(mode="json"),
                "policy_failure_reason": failure_reason,
                "agent_finished": incomplete_reason is None,
                "incomplete_reason": incomplete_reason,
                "agent_error": agent_config.get("agent_error"),
            }
            if cfg.context_summarization_cutoff is not None and response.output:
                self._attach_segment_responses(body_out, local, model=agent_config["policy_model"], session_id=task_id)
            return body_out

    @staticmethod
    def _attach_segment_responses(body_out: dict, task: Path, *, model: str, session_id: Optional[str]) -> None:
        """Add ``responses`` (one per context segment) or mask the row when they cannot be built.

        ``task`` is the rollout's temp dir; the sandbox trace was downloaded to ``task/trace``
        and the completion logs are read from ``task/trace/completions`` like ``export_trajectory``.

        A segment that cannot be reconstructed must not train on a guess. The top-level
        response is logging-only in compaction mode (NeMo RL strips its token fields and the
        segments carry the targets), so on failure it becomes an empty ``incomplete``
        response that NeMo RL recovers as a masked row, exactly like the Stirrup agent.
        """
        try:
            body_out["responses"] = [
                r.model_dump(mode="json")
                for r in export_segments_from_trace(task, model=model, session_id=session_id or "main")
            ]
        except Exception as exc:
            logger.warning("GDPVal OpenHands segment export failed; masking rollout: %s", exc)
            body_out.pop("responses", None)
            body_out["agent_finished"] = False
            body_out["incomplete_reason"] = "agent_exec_incomplete"
            body_out["agent_error"] = f"segment export: {exc}"[:600]
            body_out["response"] = _incomplete_response(model, "segments").model_dump(mode="json")

    @staticmethod
    def _sandbox_lost_body(body: BaseRunRequest, error: BaseException) -> dict[str, Any]:
        """Verify payload for a rollout whose sandbox ended before the agent finished.

        No policy tokens are invented: the response is an empty ``incomplete``
        placeholder, exactly like the exec-timeout path, so the capture
        finalizer emits a masked placeholder row and the judge skips grading.
        """
        response = NeMoGymResponse(
            id="gdpval-sandbox-lost-" + uuid.uuid4().hex[:12],
            created_at=time.time(),
            model="gdpval-policy",
            object="response",
            output=[],
            tools=[],
            tool_choice="auto",
            parallel_tool_calls=True,
            status="incomplete",
        )
        return {
            "task_id": (body.responses_create_params.metadata or {}).get("task_id"),
            "responses_create_params": body.responses_create_params.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
            "policy_failure_reason": None,
            "agent_finished": False,
            "incomplete_reason": "sandbox_ended",
            "sandbox_error": str(error)[:500],
        }

    async def run(self, body: BaseRunRequest, request: Request) -> GDPValRunResponse:
        async with self._sem, self._resource_route() as resource_server:
            cookies = dict(request.cookies)
            sandbox = None
            seeded = False
            try:
                seed_response = await self.server_client.post(
                    server_name=resource_server,
                    url_path="/seed_session",
                    json=body.model_dump(mode="json"),
                    cookies=cookies,
                )
                cookies.update(seed_response.cookies)
                await raise_for_status(seed_response)
                seeded = True
                seed = await seed_response.json()
                try:
                    sandbox = await self._attach(seed["sandbox_handle"])
                    verify_body = await self._execute(sandbox, body)
                except TraceExportError as error:
                    logger.warning("GDPVal trace export failed; reporting masked attempt: %s", error)
                    verify_body = self._sandbox_lost_body(body, error)
                    verify_body["incomplete_reason"] = "agent_exec_incomplete"
                    verify_body["agent_error"] = verify_body.pop("sandbox_error")
                except SandboxApiException as error:
                    # Preserve a lost sandbox as an explicitly masked attempt
                    # so callers can distinguish it from a measured policy failure.
                    logger.warning("GDPVal sandbox lost mid-rollout; reporting masked attempt: %s", error)
                    verify_body = self._sandbox_lost_body(body, error)
                verify_response = await self.server_client.post(
                    server_name=resource_server,
                    url_path="/verify",
                    json=verify_body,
                    cookies=cookies,
                )
                await raise_for_status(verify_response)
                return GDPValRunResponse.model_validate(await verify_response.json())
            finally:
                if seeded:
                    try:
                        cleanup = await self.server_client.post(
                            server_name=resource_server,
                            url_path="/cleanup_session",
                            json={},
                            cookies=cookies,
                        )
                        await raise_for_status(cleanup)
                    except Exception:
                        logger.exception("GDPVal resources-server cleanup failed")
                if sandbox is not None:
                    try:
                        # Resources owns the sandbox; release only this process's borrowed connection.
                        await sandbox.disconnect()
                    except Exception:
                        logger.exception("GDPVal attached sandbox cleanup failed")


if __name__ == "__main__":
    GDPValSandboxedAgent.run_webserver()

elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = GDPValSandboxedAgent.run_webserver()
