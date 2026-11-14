# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one Terminus-2 episode inside a Gym task sandbox, under the uploaded process supervisor.

The runner runs on the private Python runtime the Agent Server uploads, next to ``terminus.py`` and ``llm.py``.
Terminus drives a tmux session in the task's working directory with the task's environment, and calls the Gym
Model Server directly. ``--ensure-tmux`` installs tmux with Harbor's own installer during session setup.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4


# Tool shells get the environment the runner started with, before the settings below.
_TASK_ENVIRONMENT = dict(os.environ)
# LiteLLM fetches its model price list at import unless told to use the copy it ships.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)


def _runtime() -> dict[str, Any]:
    return {"hostname": os.uname().nodename, "pid": os.getpid(), "python": sys.executable}


def _classify_error(error: BaseException) -> tuple[str, int | None]:
    """The kind of error that ended the run, and the model's HTTP status when it answered with one."""
    import httpx
    from harbor.llms.base import ContextLengthExceededError, OutputLengthExceededError

    if isinstance(error, ContextLengthExceededError):
        return "context_limit", None
    if isinstance(error, OutputLengthExceededError):
        return "output_limit", None
    if isinstance(error, httpx.HTTPStatusError):
        return "model_request_error", error.response.status_code
    if isinstance(error, httpx.TransportError):
        return "model_transport_error", None
    return "harness_error", None


async def _ensure_tmux() -> int:
    """Install tmux the way Harbor's Terminus does when the image has none; the sandbox needs a package index."""
    try:
        from .terminus import LocalEnvironment, LocalTmuxSession
    except ImportError:
        from terminus import LocalEnvironment, LocalTmuxSession

    environment = LocalEnvironment(Path("/"), 600, base_env=_TASK_ENVIRONMENT, login_shell=False)
    session = LocalTmuxSession(
        session_name="unused",
        environment=environment,
        logging_path=Path("/dev/null"),
        local_asciinema_recording_path=None,
        remote_asciinema_recording_path=None,
    )
    await session._attempt_tmux_installation()
    check = await environment.exec("tmux -V")
    print(check.stdout or check.stderr, end="")
    return check.return_code


async def _run(payload: dict[str, Any], output_path: Path) -> dict[str, Any]:
    try:
        from .llm import NemoGymLLM
        from .terminus import LocalEnvironment, StandaloneTerminus2
    except ImportError:
        from llm import NemoGymLLM
        from terminus import LocalEnvironment, StandaloneTerminus2
    from harbor.models.agent.context import AgentContext

    logs_dir = Path(payload["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    settings = payload["terminus"]
    llm = NemoGymLLM(
        model_name=payload["model"],
        api_base=payload["model_base_url"],
        collect_rollout_details=settings["collect_rollout_details"],
        model_info=settings["model_info"],
        timeout_sec=payload["model_timeout_sec"],
        chat_params=payload["chat_params"],
        headers={"x-session-id": payload["invocation_id"]},
    )
    agent = StandaloneTerminus2(
        logs_dir=logs_dir,
        model_name=payload["model"],
        api_base=payload["model_base_url"],
        llm=llm,
        record_terminal_session=False,
        **settings,
    )
    environment = LocalEnvironment(
        Path(payload["workdir"]),
        payload["command_timeout_sec"],
        base_env=_TASK_ENVIRONMENT,
        login_shell=False,
    )
    context = AgentContext()
    stop_reason: str | None = None

    def result(reason: str) -> dict[str, Any]:
        trajectory_path = logs_dir / "trajectory.json"
        try:
            trajectory = json.loads(trajectory_path.read_text())
        except (OSError, ValueError):
            trajectory = {"steps": []}
        error = agent.run_error
        error_kind, error_status = _classify_error(error) if error is not None else (None, None)
        return {
            "stop_reason": reason,
            "trajectory": trajectory,
            "context": context.model_dump(mode="json"),
            "finished_naturally": agent.finished_naturally,
            "context_length_exceeded": bool(getattr(agent, "context_length_exceeded", False))
            or llm.context_length_exceeded,
            "error": f"{type(error).__name__}: {error}"[:4000] if error is not None else None,
            "error_type": type(error).__name__ if error is not None else None,
            "error_kind": error_kind,
            "error_status": error_status,
            "terminal_response_id": llm.last_response_id,
            "response_ids": list(llm.response_ids),
            "calls_without_usage": llm.calls_without_usage,
            "cached_usage_known": llm.cached_usage_known,
            "compactions": agent.compactions,
            "runtime": _runtime(),
        }

    main = asyncio.current_task()

    def interrupt() -> None:
        nonlocal stop_reason
        # Close writes the stop request before the supervisor signals; a deadline or a rollout stop does not.
        stop_reason = "cancelled" if Path(payload["stop_request_path"]).exists() else "wall_time"
        # Save first: unwinding a blocked command may not finish before the supervisor's hard stop.
        try:
            agent._dump_trajectory()
            _write_atomic(output_path, result(stop_reason))
        except Exception:
            traceback.print_exc()
        main.cancel()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, interrupt)
    try:
        await agent.setup(environment)
        await agent.run(payload["instruction"], environment, context)
    except asyncio.CancelledError:
        if stop_reason is None:
            raise
    finally:
        # A late signal changes nothing once the run has ended. Replacing the handler, rather than removing it, never
        # restores the default action, which would end the runner before it saves the result below.
        loop.add_signal_handler(signal.SIGTERM, lambda: None)
    # The tmux terminal and the commands running in it, such as a service, stay up after the runner ends, also after
    # a deadline, rollout stop or close, until the sandbox's owner stops the sandbox, as in Harbor. They remain
    # available to verification.
    if stop_reason is None:
        if agent.run_error is not None:
            stop_reason = "error"
        elif agent.finished_naturally:
            stop_reason = "completed"
        elif (
            settings["max_turns"] is not None
            and (context.metadata or {}).get("n_episodes", 0) >= settings["max_turns"]
        ):
            stop_reason = "max_turns"
        else:
            # Harbor ends the loop when the terminal session is gone.
            stop_reason = "terminal_exited"
    output = result(stop_reason)
    # Saved while the event loop still handles SIGTERM: closing the loop restores the signal's default action.
    _write_atomic(output_path, output)
    return output


def main() -> int:
    """Run the worker; the uploaded process supervisor owns its lifetime."""
    if sys.argv[1:] == ["--ensure-tmux"]:
        return asyncio.run(_ensure_tmux())
    if len(sys.argv) != 3:
        print("usage: sandbox_runner.py INPUT_JSON OUTPUT_JSON | --ensure-tmux", file=sys.stderr)
        return 2
    input_path, output_path = map(Path, sys.argv[1:])
    try:
        asyncio.run(_run(json.loads(input_path.read_text()), output_path))
    except BaseException as error:
        _write_atomic(
            output_path,
            {
                "stop_reason": "error",
                "error": f"{type(error).__name__}: {error}"[:4000],
                "error_type": type(error).__name__,
                "error_kind": "harness_error",
                "traceback": traceback.format_exc()[-8000:],
                "runtime": _runtime(),
            },
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
