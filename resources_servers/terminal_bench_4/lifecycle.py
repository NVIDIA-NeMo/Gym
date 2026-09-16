# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource-owned preparation, deadlines, one finalizer, and cleanup."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from fastapi import HTTPException

from resources_servers.terminal_bench_4.collection import collect
from resources_servers.terminal_bench_4.environment import Environment
from resources_servers.terminal_bench_4.handoff import AgentTermination, SandboxedSeedResponse
from resources_servers.terminal_bench_4.shared_logs import SharedLogs
from resources_servers.terminal_bench_4.transfers import download_dir
from resources_servers.terminal_bench_4.verifier import restore, run_verifier


NATIVE_VERSION = "1"
SETUP_TIMEOUT_SEC = 360


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
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    preparation: asyncio.Task | None = None
    finalization: asyncio.Task | None = None
    watchdog: asyncio.Task | None = None
    task: Any = None
    environment: Any = None
    verifier_environment: Any = None
    shared_logs: Any = None
    seed: SandboxedSeedResponse | None = None
    termination: AgentTermination | None = None
    verify_body: Any = None
    verified_response: Any = None
    result: dict = field(default_factory=dict)
    deadline: float | None = None
    setup_deadline: float | None = None
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


def stop_watchdog(session):
    watchdog, session.watchdog = session.watchdog, None
    if watchdog and watchdog is not asyncio.current_task():
        watchdog.cancel()


async def cleanup(session):
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
    stop_watchdog(session)
    session.persist()


async def prepare_session(session, loader):
    # The reference creates these even for providers without host mounts. In
    # particular, the convention directory must remain a directory if a remote
    # type probe fails and the optional file download is attempted instead.
    for relative in ("agent", "verifier", "artifacts/logs/artifacts"):
        (session.directory / relative).mkdir(parents=True, exist_ok=True)
    session.result = {"runtime": "gym-tb4-native", "runtime_version": NATIVE_VERSION, "started_at": now()}
    ready = False
    try:
        await session.slots.acquire()
        session.owns_slot = True
        session.task = await loader.load(session.request.task_name, session.request.task_ref)
        session.environment = Environment(
            session.task, session.config.environment, session.session_id, session.directory
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
        async with session.lock:
            session.phase = "ready"
            session.setup_deadline = monotonic() + SETUP_TIMEOUT_SEC
            session.deadlines["setup_started_at"] = now()
            session.result["agent_setup"] = {"started_at": now()}
            session.watchdog = asyncio.create_task(watch_deadline(session, setup=True))
            session.persist()
        # Like the reference, descriptor I/O is inside the setup allowance.
        connection = await session.environment.main_connection()
        async with session.lock:
            if session.phase != "ready":
                raise RuntimeError("Episode ended while preparing its descriptor")
            session.seed = SandboxedSeedResponse(
                session_id=session.session_id,
                sandbox=connection,
                instruction=session.task.instruction,
                user=session.task.config.agent.user,
                agent_timeout_sec=min(
                    session.task.config.agent.timeout_sec, session.config.agent_max_timeout_sec or float("inf")
                ),
                setup_timeout_sec=SETUP_TIMEOUT_SEC,
                mcp_servers=[s.model_dump() for s in session.task.config.environment.mcp_servers],
                skills_dir=session.task.config.environment.skills_dir,
            )
            ready = True
            session.persist()
    except asyncio.CancelledError:
        session.termination = session.termination or AgentTermination(reason="cancelled")
        exception(session, "Preparation or harness setup cancelled", "CancelledError")
    except Exception as exc:
        exception(session, exc)
    finally:
        if not ready:
            await cleanup(session)


async def start_session(session):
    async with session.lock:
        if session.finalization is not None:
            raise HTTPException(409, "Episode is already finishing")
        if session.phase == "ready" and session.seed:
            if monotonic() >= session.setup_deadline:
                finish = _request_finish(
                    session, AgentTermination(reason="timeout", detail="Harness setup deadline reached")
                )
            else:
                stop_watchdog(session)
                session.result["agent_setup"]["finished_at"] = now()
                session.result["agent_execution"] = {"started_at": now()}
                session.deadlines["agent_started_at"] = now()
                session.deadline = monotonic() + session.seed.agent_timeout_sec
                session.phase = "agent_running"
                session.watchdog = asyncio.create_task(watch_deadline(session))
                session.persist()
                return {"agent_timeout_sec": max(0, session.deadline - monotonic())}
        elif session.phase == "agent_running":
            if monotonic() < session.deadline:
                return {"agent_timeout_sec": max(0, session.deadline - monotonic())}
            finish = _request_finish(
                session, AgentTermination(reason="timeout", detail="Resources agent deadline reached")
            )
        else:
            raise HTTPException(409, "Episode is not ready or running")
    await asyncio.shield(finish)
    raise HTTPException(409, "Episode deadline reached")


async def watch_deadline(session, setup=False):
    deadline = session.setup_deadline if setup else session.deadline
    await asyncio.sleep(max(0, deadline - monotonic()))
    detail = "Harness setup deadline reached" if setup else "Resources agent deadline reached"
    # A watchdog only requests finalization; it never owns or awaits it.
    await request_finish(session, AgentTermination(reason="timeout", detail=detail))


def _request_finish(session, termination):
    """Called only with the lock held. First accepted finish wins, except that
    an already elapsed execution deadline always precedes a newly handled request.
    """
    if session.finalization is not None or session.phase == "closed":
        return session.finalization
    grade = session.phase == "agent_running"
    if grade and monotonic() >= session.deadline:
        termination = AgentTermination(
            reason="timeout", detail="Resources agent deadline reached", artifacts=termination.artifacts
        )
    session.termination = session.termination or termination.model_copy(deep=True)
    stop_watchdog(session)
    if grade:
        session.result["agent_execution"]["finished_at"] = now()
        session.phase = "verifying"
    elif termination.reason == "timeout":
        exception(session, termination.detail or "Harness setup timed out", "AgentSetupTimeoutError")
    session.finalization = asyncio.create_task(finalize_session(session, grade=grade))
    session.persist()
    return session.finalization


async def request_finish(session, termination, body=None):
    async with session.lock:
        if body is not None:
            if session.phase in {"preparing", "ready"}:
                raise HTTPException(409, "Agent setup has not completed")
            if session.verify_body is None:
                session.verify_body = body.model_copy(deep=True)
                session.directory.mkdir(parents=True, exist_ok=True)
                (session.directory / "gym-agent.json").write_text(body.model_dump_json(indent=2))
        finish = _request_finish(session, termination)
        # Late worker artifacts remain references, even after deadline grading.
        if body is not None and session.termination is not None:
            session.termination.artifacts = list(
                dict.fromkeys(session.termination.artifacts + body.termination.artifacts)
            )
        session.persist()
        return finish


async def finalize_session(session, *, grade):
    try:
        if not grade:
            if session.preparation and not session.preparation.done():
                session.preparation.cancel()
                await session.preparation
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


async def shutdown(sessions, timeout):
    finalizers = []
    for session in sessions:
        finish = await request_finish(session, AgentTermination(reason="cancelled", detail="Resources shutdown"))
        if finish:
            finalizers.append(finish)
    if finalizers:
        _, pending = await asyncio.wait(finalizers, timeout=timeout)
        for task in pending:
            task.cancel()
        await asyncio.gather(*finalizers, return_exceptions=True)
