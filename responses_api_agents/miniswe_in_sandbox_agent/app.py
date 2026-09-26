# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mini-SWE rollouts whose harness runs inside the resource-owned sandbox.

The existing ``miniswe_sandboxed_agent`` runs the mini-SWE loop on the Gym host and ships every bash command
through the sandbox exec API, which costs one control-plane round trip per step. This agent follows the OpenCode
agent's paradigm instead: it seeds a TB4 session, stages a stdlib-only runner (``runner/miniswe_runner.py``) plus a
vendored pure-Python Jinja2 into the sandbox, runs ONE exec as the task's agent identity for the whole episode,
and lets the runner call the Gym model server directly through a sandbox-reachable gateway. Afterwards it turns
the runner's records into a Gym response and submits the termination it observed to ``/verify``.
"""

import asyncio
import importlib.util
import json
import logging
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from shlex import quote
from time import monotonic, time
from typing import Literal
from uuid import uuid4

import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, TypeAdapter, field_validator

from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_correlation import rollout_context
from nemo_gym.sandbox import AsyncSandbox, create_provider, resolve_provider_config
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
    rollout_path_prefix,
)
from responses_api_agents.miniswe_in_sandbox_agent.models import (
    AgentExecutionResult,
    RunRequest,
    SandboxedVerifyRequest,
    SeedSessionResponse,
    Termination,
    VerifyResponse,
)


LOGGER = logging.getLogger(__name__)
RUNNER_DIR = Path(__file__).with_name("runner")
RUNNER_SCRIPT = RUNNER_DIR / "miniswe_runner.py"
MINI_TEMPLATES = RUNNER_DIR / "mini_2_4_6.yaml"
MINI_VERSION = "2.4.6"
HARNESS_VERSION = "miniswe-in-sandbox-runner/1 (mini-swe-agent 2.4.6 templates)"
VENDORED_PACKAGES = ("jinja2", "markupsafe")
RUNNER_FILES = ("trajectory.json", "output_items.json", "usages.json", "result.json", "runner.log")
OUTPUT_ITEMS = TypeAdapter(list[NeMoGymResponseOutputItem])


class RunnerConfig(BaseModel):
    step_limit: int = Field(default=0, ge=0)
    step_timeout_sec: int = Field(default=30, gt=0)
    max_consecutive_format_errors: int = Field(default=3, ge=0)
    http_retries: int = Field(default=3, ge=0)
    http_timeout_sec: float = Field(default=3600, gt=0)


class MiniSWEInSandboxConfig(BaseResponsesAPIAgentConfig):
    num_workers: Literal[1] = 1
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    # Sandbox-reachable origin that forwards to THIS Gym's model server (e.g. a relay in front of an ssh tunnel).
    # The runner appends Gym's rollout-capture prefix and ``/v1`` itself.
    model_gateway_url: str
    harness: RunnerConfig = Field(default_factory=RunnerConfig)
    artifacts_dir: Path = Path("results/miniswe_in_sandbox_agent")
    agent_max_timeout_sec: float | None = Field(default=None, gt=0)
    setup_timeout_sec: float = Field(default=360, gt=0)
    shutdown_timeout_sec: float = Field(default=30, ge=0)
    # Margin between the runner's own wall limit (clean exit with a saved trajectory) and the exec budget.
    runner_exit_margin_sec: float = Field(default=60, ge=0)
    # Appended to the task instruction (e.g. a teacher instruction), before the skills line.
    instruction_suffix: str = ""
    remote_dir_prefix: str = "/tmp/ng-miniswe-"
    python_executable: str = "python3"

    @field_validator("model_gateway_url")
    @classmethod
    def _origin(cls, value: str) -> str:
        from urllib.parse import urlsplit

        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.netloc or parts.path not in ("", "/") or parts.query:
            raise ValueError("model_gateway_url must be an origin such as http://10.0.0.1:24402 (no path)")
        return value.rstrip("/")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_response(params: NeMoGymResponseCreateParamsNonStreaming, model: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_" + uuid4().hex,
        created_at=int(time()),
        model=model,
        object="response",
        output=[],
        tool_choice=params.tool_choice,
        tools=params.tools,
        parallel_tool_calls=params.parallel_tool_calls,
    )


def build_vendor_zip(target: Path) -> Path:
    """Bundle the pure-Python Jinja2 + MarkupSafe from this venv; the C speedup is left out on purpose."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for package in VENDORED_PACKAGES:
            spec = importlib.util.find_spec(package)
            if spec is None or not spec.submodule_search_locations:
                raise RuntimeError(f"{package} is not installed in the agent environment")
            root = Path(next(iter(spec.submodule_search_locations)))
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix in (".so", ".pyd", ".pyc", ".c") or "__pycache__" in path.parts:
                    continue
                archive.write(path, f"{package}/{path.relative_to(root).as_posix()}")
    temporary.replace(target)
    return target


def runner_config(
    *,
    config: MiniSWEInSandboxConfig,
    seed: SeedSessionResponse,
    params: NeMoGymResponseCreateParamsNonStreaming,
    model_url: str,
    task: str,
    workdir: str,
    remote_dir: str,
    budget: float,
) -> dict:
    templates = yaml.safe_load(MINI_TEMPLATES.read_text())
    request_params = params.model_dump(exclude_none=True, exclude={"input", "tools"})
    return {
        "session_id": seed.session_id,
        "model_url": model_url,
        "headers": {"x-session-id": seed.session_id},
        "request_params": request_params,
        "task": task,
        "workdir": workdir,
        "env": templates["environment"]["env"],
        "templates": {
            "system_template": templates["agent"]["system_template"],
            "instance_template": templates["agent"]["instance_template"],
            "observation_template": templates["model"]["observation_template"],
            "format_error_template": templates["model"]["format_error_template"],
        },
        "step_limit": config.harness.step_limit,
        "step_timeout_sec": config.harness.step_timeout_sec,
        "max_consecutive_format_errors": config.harness.max_consecutive_format_errors,
        "http_retries": config.harness.http_retries,
        "http_timeout_sec": config.harness.http_timeout_sec,
        "budget_sec": max(30.0, budget - config.runner_exit_margin_sec),
        "pids_file": f"/tmp/{seed.session_id}.pids",
        "output_dir": remote_dir,
        "vendor_zip": f"{remote_dir}/vendor.zip",
        "mini_version": MINI_VERSION,
    }


def classify(result, runner_result: dict | None, log_tail: str) -> Termination:
    """Map the single exec's outcome and the runner's records to a TB4 termination (server-harness parity)."""
    if result is None or getattr(result, "error_type", None) == "timeout":
        return Termination(reason="timeout", detail="Runner exec reached the agent budget")
    if result.error_type:
        return Termination(reason="infrastructure_error", detail=f"Sandbox execution failed: {result.error_type}")
    if result.return_code != 0:
        return Termination(
            reason="infrastructure_error",
            exit_code=result.return_code,
            detail=f"Runner exited {result.return_code}: {log_tail[-1500:]}",
        )
    status = (runner_result or {}).get("exit_status")
    if runner_result is None:
        return Termination(reason="infrastructure_error", exit_code=0, detail="Runner left no result.json")
    if status == "Submitted":
        return Termination(reason="completed", exit_code=0)
    if status == "TimeExceeded":
        # The runner stopped itself at its wall limit (or on SIGTERM): the budget was hit, as the server harness reports.
        return Termination(reason="timeout", exit_code=0, detail="TimeExceeded")
    return Termination(reason="nonzero_exit", exit_code=0, detail=status or "unknown exit status")


class MiniSWEInSandboxAgent(SimpleResponsesAPIAgent):
    config: MiniSWEInSandboxConfig

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._runs = {}
        self._finalizers = set()
        self._closing = False
        self._shutdown_deadline: float | None = None
        if not RUNNER_SCRIPT.is_file() or not MINI_TEMPLATES.is_file():
            raise RuntimeError("The in-sandbox runner files are missing")
        self._vendor_zip = build_vendor_zip(self.config.artifacts_dir / "_runner" / "vendor.zip")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    async def shutdown(self) -> None:
        self._closing = True
        if self._shutdown_deadline is None:
            self._shutdown_deadline = monotonic() + self.config.shutdown_timeout_sec
        workers = [worker for _, worker in self._runs.values() if not worker.done()]
        for worker in workers:
            if not worker.cancelling():
                worker.cancel()
        if workers:
            await asyncio.wait(workers, timeout=max(0, self._shutdown_deadline - monotonic()))
        finalizers = set(self._finalizers)
        if finalizers:
            _, pending = await asyncio.wait(finalizers, timeout=max(0, self._shutdown_deadline - monotonic()))
            for task in pending:
                task.cancel()

    @staticmethod
    def _observe_background_task(task: asyncio.Task) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            LOGGER.error(
                "in-sandbox mini-SWE background operation failed", exc_info=(type(error), error, error.__traceback__)
            )

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        raise NotImplementedError("This agent requires /run")

    async def run(self, request: Request, body: RunRequest) -> VerifyResponse:
        if self._closing:
            raise HTTPException(503, "Agent server is shutting down")
        payload = body.model_dump(mode="json")
        rollout_id = self.rollout_id_from_run(body)
        payload["rollout_id"] = rollout_id or body.capture_rollout_id or payload.get("rollout_id") or uuid4().hex
        payload["client_session_id"] = request.session[SESSION_ID_KEY]
        if rollout_id:
            payload["_ng_rollout_id"] = rollout_id
        key = (payload["client_session_id"], payload["rollout_id"])
        if key not in self._runs:
            worker = asyncio.create_task(self._run(payload, dict(request.cookies), bool(rollout_id)))
            worker.add_done_callback(self._observe_background_task)
            self._runs[key] = (payload, worker)
        saved, worker = self._runs[key]
        if saved != payload:
            raise HTTPException(409, "Rollout identity is already bound to another request")
        return await asyncio.shield(worker)

    async def _run(self, payload: dict, cookies: dict, capture_model_calls: bool) -> VerifyResponse:
        seed_task = asyncio.create_task(
            self.server_client.post(
                server_name=self.config.resources_server.name, url_path="/seed_session", json=payload, cookies=cookies
            )
        )
        seed_task.add_done_callback(self._observe_background_task)
        cancelled = False
        try:
            seed_response = await asyncio.shield(seed_task)
        except asyncio.CancelledError:
            cancelled = True
            deadline = self._shutdown_deadline or (monotonic() + self.config.shutdown_timeout_sec)
            done, _ = await asyncio.wait({seed_task}, timeout=max(0, deadline - monotonic()))
            if not done:
                seed_task.cancel()
                raise
            seed_response = seed_task.result()
        await raise_for_status(seed_response)
        cookies = cookies | seed_response.cookies
        seed = SeedSessionResponse.model_validate(await get_response_json(seed_response))
        if seed.verified_response is not None:
            return seed.verified_response
        params = RunRequest.model_validate(payload).responses_create_params
        if cancelled:
            result = AgentExecutionResult(
                responses_create_params=params,
                response=empty_response(params, self.config.model_server.name),
                termination=Termination(reason="cancelled"),
            )
        else:
            result = await self.execute(
                seed,
                params,
                rollout_id=payload.get("_ng_rollout_id") or payload["rollout_id"],
                capture_model_calls=capture_model_calls,
                artifact_directory=Path(payload["artifact_directory"]) if payload.get("artifact_directory") else None,
            )
        verify_body = SandboxedVerifyRequest(session_id=seed.session_id, **result.model_dump())
        finalizer = asyncio.create_task(self._verify(verify_body, cookies))
        self._finalizers.add(finalizer)
        finalizer.add_done_callback(self._finalizers.discard)
        finalizer.add_done_callback(self._observe_background_task)
        return await asyncio.shield(finalizer)

    async def execute(
        self,
        seed: SeedSessionResponse,
        params: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: str,
        capture_model_calls: bool,
        artifact_directory: Path | None = None,
    ) -> AgentExecutionResult:
        """Stage the runner, run the whole episode in one exec as the task user, collect its records."""
        response = empty_response(params, self.config.model_server.name)
        termination = seed.termination or Termination(reason="infrastructure_error", detail="Setup did not complete")
        extra: dict = {"harness_version": HARNESS_VERSION}
        timings: dict = {}
        agent_started = False
        provider = None
        directory = artifact_directory or self.config.artifacts_dir / seed.session_id
        with rollout_context(rollout_id if capture_model_calls else None):
            try:
                if seed.termination is None:
                    timings["agent_setup"] = {"started_at": now()}
                    budget = min(seed.agent_timeout_sec, self.config.agent_max_timeout_sec or float("inf"))
                    async with asyncio.timeout(self.config.setup_timeout_sec):
                        provider = create_provider(resolve_provider_config(seed.sandbox_provider))
                        sandbox = await AsyncSandbox.connect(seed.sandbox_descriptor, provider=provider)
                        remote_dir, workdir = await self._stage(
                            sandbox, seed, params, rollout_id, capture_model_calls, budget
                        )
                    timings["agent_setup"]["finished_at"] = now()
                    extra["staging"] = getattr(self, "_staging_summary", None)
                    timings["agent_execution"] = {"started_at": now()}
                    agent_started = True
                    result = None
                    try:
                        async with asyncio.timeout(budget + 120):
                            result = await sandbox.exec(
                                self._exec_command(seed, remote_dir),
                                user=seed.user,
                                cwd=workdir,
                                timeout_s=budget,
                            )
                    except TimeoutError:
                        result = None
                    timings["agent_execution"]["finished_at"] = now()
                    if result is None or getattr(result, "error_type", None) == "timeout":
                        # The runner lives in its own session and survives the exec kill: stop it (it saves a
                        # TimeExceeded exit on SIGTERM) before the records are downloaded.
                        await self._stop_runner(sandbox, seed, workdir)
                    response, termination, extra = await self._collect(
                        sandbox, remote_dir, directory, params, result, extra
                    )
            except asyncio.CancelledError:
                termination = Termination(reason="cancelled")
            except Exception as exc:
                termination = Termination(
                    reason="timeout"
                    if isinstance(exc, TimeoutError) and not agent_started
                    else "infrastructure_error",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            finally:
                for timing in timings.values():
                    timing.setdefault("finished_at", now())
                if provider is not None:
                    try:
                        await provider.aclose()
                    except Exception:
                        LOGGER.exception("Failed to close the sandbox transport")
        return AgentExecutionResult(
            responses_create_params=params,
            response=response,
            termination=termination,
            agent_started=agent_started,
            agent_timings=timings,
            harness_metadata=extra,
        )

    async def _stop_runner(self, sandbox: AsyncSandbox, seed: SeedSessionResponse, workdir: str) -> None:
        pids = quote(f"/tmp/{seed.session_id}.pids")
        script = (
            f'if [ -f {pids} ]; then for p in $(cat {pids}); do kill -TERM -- -"$p" 2>/dev/null || true; done; '
            f'sleep 3; for p in $(cat {pids}); do kill -KILL -- -"$p" 2>/dev/null || true; done; fi'
        )
        try:
            await sandbox.exec(script, user=seed.user, cwd=workdir, timeout_s=30)
        except Exception as exc:
            LOGGER.warning("Could not stop the runner after the exec timeout: %s", exc)

    def _exec_command(self, seed: SeedSessionResponse, remote_dir: str) -> str:
        python = quote(self.config.python_executable)
        inner = (
            f"echo $$ >> /tmp/{seed.session_id}.pids; "
            f"exec {python} {quote(remote_dir + '/miniswe_runner.py')} --config {quote(remote_dir + '/config.json')} "
            f"> {quote(remote_dir + '/runner.log')} 2>&1"
        )
        return "setsid --wait bash -c " + quote(inner)

    async def _stage(
        self,
        sandbox: AsyncSandbox,
        seed: SeedSessionResponse,
        params: NeMoGymResponseCreateParamsNonStreaming,
        rollout_id: str,
        capture_model_calls: bool,
        budget: float,
    ) -> tuple[str, str]:
        if seed.mcp_servers:
            raise RuntimeError("Task MCP servers are not supported by the in-sandbox mini-SWE runner")
        cwd = await sandbox.exec("pwd", timeout_s=30, user=seed.user)
        if cwd.return_code:
            raise RuntimeError("Unable to determine the task working directory")
        workdir = cwd.stdout.strip()
        python = quote(self.config.python_executable)
        preflight = await sandbox.exec(
            f"command -v setsid && command -v {python} && {python} -c "
            + quote("import sys; assert sys.version_info >= (3, 9), sys.version; print(sys.version.split()[0])"),
            user=seed.user,
            cwd=workdir,
            timeout_s=60,
        )
        if preflight.return_code:
            raise RuntimeError(
                f"Sandbox lacks setsid or a Python >= 3.9 for the runner: {(preflight.stderr or preflight.stdout or '')[-500:]}"
            )
        remote_dir = f"{self.config.remote_dir_prefix}{seed.session_id}"
        default_uid = await sandbox.exec("id -u", timeout_s=30)
        created = await sandbox.exec(f"mkdir -p {quote(remote_dir)} && chmod 0755 {quote(remote_dir)}", timeout_s=30)
        if created.return_code:
            raise RuntimeError(f"Could not create the runner directory {remote_dir}: {created.stderr}")
        task = seed.instruction + self.config.instruction_suffix
        if seed.skills_dir:
            task += f"\nTask skills are in {seed.skills_dir}. Read the relevant SKILL.md files.\n"
        model_url = (
            self.config.model_gateway_url
            + rollout_path_prefix(
                rollout_id if capture_model_calls else None, token_capture=self._token_id_capture_enabled()
            )
            + "/v1"
        )
        config = runner_config(
            config=self.config,
            seed=seed,
            params=params,
            model_url=model_url,
            task=task,
            workdir=workdir,
            remote_dir=remote_dir,
            budget=budget,
        )
        staging = Path(self.config.artifacts_dir) / "_staging" / seed.session_id
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "config.json").write_text(json.dumps(config, indent=2))
        await sandbox.upload(RUNNER_SCRIPT, remote_dir + "/miniswe_runner.py")
        await sandbox.upload(self._vendor_zip, remote_dir + "/vendor.zip")
        await sandbox.upload(staging / "config.json", remote_dir + "/config.json")
        staged_owner = None
        if seed.user not in (None, "root", 0) and (default_uid.stdout or "").strip() == "0":
            owned = await sandbox.exec(f"chown -R {quote(str(seed.user))} {quote(remote_dir)}", timeout_s=30)
            if owned.return_code:
                raise RuntimeError(f"Could not hand the runner directory to {seed.user!r}: {owned.stderr}")
            staged_owner = seed.user
        self._staging_summary = {
            "bootstrap_uid": (default_uid.stdout or "").strip(),
            "staged_owner": staged_owner,
            "sandbox_python": ((preflight.stdout or "").strip().splitlines() or [None])[-1],
            "remote_dir": remote_dir,
            "workdir": workdir,
            "task_chars": len(task),
        }
        # The verify body carries the prompt the runner actually used, as the server-side harness does.
        params.input = [NeMoGymEasyInputMessage(role="user", content=task)]
        return remote_dir, workdir

    async def _collect(self, sandbox, remote_dir: str, directory: Path, params, result, extra: dict):
        directory.mkdir(parents=True, exist_ok=True)
        records: dict = {}
        for name in RUNNER_FILES:
            try:
                await sandbox.download(f"{remote_dir}/{name}", directory / name)
                records[name] = True
            except Exception as exc:
                records[name] = f"{type(exc).__name__}: {exc}"
        log_tail = ""
        if (directory / "runner.log").is_file():
            log_tail = (directory / "runner.log").read_text(errors="replace")[-4000:]
        runner_result = None
        if (directory / "result.json").is_file():
            try:
                runner_result = json.loads((directory / "result.json").read_text())
            except ValueError:
                runner_result = None
        response = empty_response(params, self.config.model_server.name)
        if (directory / "output_items.json").is_file():
            try:
                items = OUTPUT_ITEMS.validate_json((directory / "output_items.json").read_text())
                usage = None
                if (directory / "usages.json").is_file():
                    usages = json.loads((directory / "usages.json").read_text())
                    if usages and all(u is not None for u in usages):
                        usage = NeMoGymResponseUsage.sum_from_list(
                            [NeMoGymResponseUsage.model_validate(u) for u in usages]
                        )
                response = NeMoGymResponse(
                    id="resp_" + uuid4().hex,
                    created_at=int(time()),
                    model=self.config.model_server.name,
                    object="response",
                    output=items,
                    tool_choice=params.tool_choice,
                    tools=params.tools,
                    parallel_tool_calls=params.parallel_tool_calls,
                    usage=usage,
                )
            except Exception as exc:
                records["output_items.json"] = f"invalid: {type(exc).__name__}: {exc}"
        termination = classify(result, runner_result, log_tail)
        trajectory = None
        if (directory / "trajectory.json").is_file():
            try:
                trajectory = json.loads((directory / "trajectory.json").read_text())
            except ValueError:
                trajectory = None
        consistency = {}
        if (directory / "output_items.json").is_file() and runner_result is not None:
            try:
                raw_items = json.loads((directory / "output_items.json").read_text())
                calls = [i["call_id"] for i in raw_items if i.get("type") == "function_call"]
                outputs = [i["call_id"] for i in raw_items if i.get("type") == "function_call_output"]
                consistency = {
                    "outputs_match_calls": set(outputs) <= set(calls),
                    "function_calls": len(calls),
                    "function_call_outputs": len(outputs),
                    "n_calls": runner_result.get("n_calls"),
                }
            except Exception as exc:
                consistency = {"error": f"{type(exc).__name__}: {exc}"}
        extra = dict(extra)
        extra.update(
            {
                "consistency": consistency,
                "mini_swe_trajectory": trajectory,
                "runner_result": runner_result,
                "runner_exit_code": getattr(result, "return_code", None),
                "runner_error_type": getattr(result, "error_type", None),
                "runner_log_tail": log_tail[-1500:],
                "records": records,
                "exit_status": (runner_result or {}).get("exit_status"),
                "n_calls": (runner_result or {}).get("n_calls"),
                "steps": (runner_result or {}).get("steps"),
            }
        )
        termination.artifacts = [str(directory / "trajectory.json")] if trajectory is not None else []
        return response, termination, extra

    async def _verify(self, body: SandboxedVerifyRequest, cookies: dict) -> VerifyResponse:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump(mode="json"),
            cookies=cookies,
        )
        await raise_for_status(response)
        return VerifyResponse.model_validate(await get_response_json(response))


if __name__ == "__main__":
    MiniSWEInSandboxAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = MiniSWEInSandboxAgent.run_webserver()
