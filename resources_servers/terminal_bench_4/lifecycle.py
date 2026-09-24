# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource-owned preparation, deadlines, one finalizer, and cleanup."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from resources_servers.terminal_bench_4.collection import collect
from resources_servers.terminal_bench_4.environment import Environment
from resources_servers.terminal_bench_4.models import AgentTermination
from resources_servers.terminal_bench_4.oracle import stage_solution
from resources_servers.terminal_bench_4.shared_logs import SharedLogs
from resources_servers.terminal_bench_4.transfers import download_dir
from resources_servers.terminal_bench_4.verifier import restore, run_verifier


NATIVE_VERSION = "1"


def now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Session:
    identity: str
    owner: str
    request: Any
    session_id: str
    directory: Path
    phase: str = "preparing"
    subphase: str | None = None
    started: asyncio.Event = field(default_factory=asyncio.Event)
    execution: asyncio.Task | None = None
    finalization: asyncio.Task | None = None
    expiry_task: asyncio.Task | None = None
    agent_deadline: float | None = None
    task: Any = None
    environment: Any = None
    verifier_environment: Any = None
    shared_logs: Any = None
    termination: AgentTermination | None = None
    verify_body: Any = None
    verified_response: Any = None
    seed_response: Any = None
    result: dict = field(default_factory=dict)
    deadlines: dict = field(default_factory=dict)
    diagnostics: list = field(default_factory=list)
    recorded_resources: list = field(default_factory=list)
    owns_slot: bool = False
    slots: Any = None
    config: Any = None
    persist: Callable = field(default=lambda: None, repr=False)


def exception(session, error, error_type=None):
    record = {"exception_type": error_type or type(error).__name__, "exception_message": str(error)}
    session.result.setdefault("exception_info", record)
    session.diagnostics.append({"phase": session.phase, "subphase": session.subphase, **record})


async def cleanup(session):
    if session.expiry_task is not None:
        session.expiry_task.cancel()
    session.subphase = "cleanup"
    session.persist()
    for env in (session.environment, session.verifier_environment):
        if env is None or env.closed:
            continue
        try:
            await env.stop()
        except Exception as exc:
            exception(session, exc)
    if session.shared_logs is not None:
        try:
            # A failed sandbox deletion must not race removal of its live mount.
            await session.shared_logs.stop(
                remove_data=all(
                    env is None or env.closed for env in (session.environment, session.verifier_environment)
                )
            )
        except Exception as exc:
            exception(session, exc)
    if session.owns_slot:
        session.slots.release()
        session.owns_slot = False
    for key in ("environment_setup", "agent_setup", "agent_execution", "verifier"):
        if key in session.result:
            session.result[key].setdefault("finished_at", now())
    session.result["diagnostics"] = session.diagnostics
    session.result["finished_at"] = now()
    session.phase = "closed"
    session.subphase = None
    session.persist()


async def prepare_session(session, loader):
    # The reference creates these even for providers without host mounts. In
    # particular, the convention directory must remain a directory if a remote
    # type probe fails and the optional file download is attempted instead.
    for relative in ("agent", "verifier", "artifacts/logs/artifacts"):
        (session.directory / relative).mkdir(parents=True, exist_ok=True)
    session.result = {
        "runtime": "gym-tb4-native",
        "runtime_version": NATIVE_VERSION,
        "started_at": now(),
        "execution_mode": session.config.execution_mode,
    }
    await session.slots.acquire()
    session.owns_slot = True
    session.task = await loader.load(session.request.task_name, session.request.task_ref)
    session.environment = Environment(
        session.task,
        session.config.environment,
        session.session_id,
        session.directory,
        oracle=session.config.execution_mode == "oracle",
    )
    # Construct and validate the verifier configuration before allocating
    # either environment, but allocate its resources only after collection.
    session.verifier_environment = Environment(
        session.task,
        session.config.environment,
        session.session_id + "__verifier__trial",
        session.directory,
        verifier=True,
    )
    if session.config.environment.efs_logs_host_path:
        session.shared_logs = SharedLogs(session.environment)
        session.environment.shared_logs = session.shared_logs
        session.verifier_environment.shared_logs = session.shared_logs
        # Reject mount conflicts before allocating the helper or workloads.
        session.environment.build_spec()
        session.verifier_environment.build_spec()

    async def provision():
        if session.shared_logs:
            try:
                await session.shared_logs.start()
            except Exception as exc:
                if "VOLUME::HOST_PATH_NOT_ALLOWED" not in str(
                    exc
                ) or session.config.environment.efs_logs_host_path not in str(exc):
                    raise
                await session.shared_logs.stop()
                session.environment.shared_logs = None
                session.verifier_environment.shared_logs = None
                session.diagnostics.append({"operation": "efs_logs_fallback", "role": "helper", "error": str(exc)})
            session.persist()
        await session.environment.start()
        if getattr(session.environment, "efs_logs_fallback", None):
            session.verifier_environment.shared_logs = None
            session.diagnostics.append(
                {"operation": "efs_logs_fallback", "role": "agent", "error": session.environment.efs_logs_fallback}
            )

    session.result["environment_setup"] = {"started_at": now()}
    try:
        await asyncio.wait_for(provision(), session.task.config.environment.build_timeout_sec)
    except TimeoutError as exc:
        exception(session, exc, "EnvironmentStartTimeoutError")
        raise
    finally:
        session.result["environment_setup"]["finished_at"] = now()
    await session.environment.healthcheck()
    if session.config.execution_mode == "oracle":
        # Only resources sees the trusted host package. The agent receives the
        # already-staged sandbox, never a caller-supplied solution path.
        await stage_solution(session.environment.main, session.task.path / "solution")


async def finalize_session(session, *, grade):
    try:
        if not grade:
            if session.environment and session.environment.main and not session.environment.closed:
                await session.environment.quiesce_agent(session.session_id)
            return
        session.subphase = "quiesce"
        session.persist()
        await session.environment.quiesce_agent(session.session_id)
        if session.termination.reason != "completed":
            error_type = (
                "AgentTimeoutError" if session.termination.reason == "timeout" else "NonZeroAgentExitCodeError"
            )
            exception(session, session.termination.detail or session.termination.reason, error_type)
        session.subphase = "collect"
        session.persist()
        try:
            await download_dir(session.environment.main, "/logs/agent", session.directory / "agent")
        except Exception as exc:
            session.diagnostics.append({"operation": "agent_logs", "error": str(exc)})
        await collect(session.environment, session.directory / "artifacts", session.diagnostics)
        try:
            await session.environment.stop()
        except Exception as exc:
            exception(session, exc)
        if session.shared_logs and session.environment.closed:
            try:
                await session.shared_logs.prepare_verifier()
                session.diagnostics.append(
                    {
                        "operation": "efs_artifact_restore",
                        "snapshot_ready": bool(session.shared_logs.restored_archive),
                    }
                )
            except Exception as exc:
                session.diagnostics.append({"operation": "efs_artifact_restore", "error": str(exc)})
        session.subphase = "verifier_setup"
        session.result["verifier"] = {"started_at": now()}
        session.persist()
        try:
            # Compatibility: both startups use the task's environment build
            # budget; verifier healthchecks are not run by the reference.
            await asyncio.wait_for(
                session.verifier_environment.start(),
                session.task.config.environment.build_timeout_sec,
            )
            if getattr(session.verifier_environment, "efs_logs_fallback", None):
                session.diagnostics.append(
                    {
                        "operation": "efs_logs_fallback",
                        "role": "verifier",
                        "error": session.verifier_environment.efs_logs_fallback,
                    }
                )
            await restore(session.verifier_environment, session.directory / "artifacts")
            session.subphase = "verifier_execution"
            session.persist()
            session.result["verifier_result"] = await run_verifier(
                session.verifier_environment,
                session.directory,
                session.diagnostics,
            )
        finally:
            session.result["verifier"]["finished_at"] = now()
    except asyncio.CancelledError:
        exception(session, "Resources shutdown interrupted evaluation", "CancelledError")
    except Exception as exc:
        exception(session, exc)
    finally:
        await cleanup(session)


async def shutdown(sessions: list[Session], timeout: float) -> None:
    preparations = []
    finalizers = []
    expiries = []
    for session in sessions:
        if session.expiry_task is not None:
            session.expiry_task.cancel()
            expiries.append(session.expiry_task)
        if session.execution and not session.execution.done():
            await session.started.wait()
            session.execution.cancel()
            preparations.append(session.execution)
        if session.finalization and not session.finalization.done():
            finalizers.append(session.finalization)
    if expiries:
        await asyncio.gather(*expiries, return_exceptions=True)
    if preparations:
        await asyncio.gather(*preparations, return_exceptions=True)
    if finalizers:
        _, pending = await asyncio.wait(finalizers, timeout=timeout)
        for session in sessions:
            if session.finalization in pending and session.subphase != "cleanup":
                session.finalization.cancel()
        await asyncio.gather(*finalizers, return_exceptions=True)
    # Includes provisioned sessions whose agent disconnected or died before verification.
    for session in sessions:
        if session.phase != "closed":
            await cleanup(session)
