# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one mini-SWE episode inside a Gym task sandbox.

The agent server uploads this file and runs it with the private runtime's Python (``python -E -s``) under Gym's
process supervisor. It builds mini-SWE's ``DefaultAgent`` from the pinned ``mini.yaml`` and the episode's overlay, the
way the ``mini`` command does, with mini-SWE's local environment, and runs the task in the working directory. mini-SWE
saves its native trajectory after every step. This runner adds ``output.json``: how the run ended, the task MCP servers
it started, and which process ran it.

Exit statuses: 0 when mini-SWE's loop ended (any exit status, including a step limit), 1 when it raised, 3 when a
required MCP server did not connect (no model call was made), and 143 when the supervisor stopped it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4


# Set for each bash command while task MCP servers are running, so mcp_cli.py can record the call that made it.
TOOL_CALL_ID_VARIABLE = "NEMO_GYM_MINI_SWE_TOOL_CALL_ID"
# mcp_cli.py connects to every server at once, each within this many seconds.
MCP_CONNECT_TIMEOUT_S = 30
MCP_UNAVAILABLE_EXIT = 3
TERMINATED_EXIT = 143


class Terminated(BaseException):
    """The supervisor's SIGTERM: at the deadline, or on close after it wrote the stop request.

    A ``BaseException``, so neither mini-SWE's step handling nor its model client turns it into a model error.
    mini-SWE still saves its trajectory on the way out.
    """


class MCPUnavailable(Exception):
    """A required task MCP server did not connect."""


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)


def start_mcp(cli: Path, *, required: list[str], log: Path) -> tuple[Any, dict, list]:
    """Start the MCP CLI's server, wait until every server connected or failed, and return its listing.

    The server process stays in this process group, so the supervisor stops it with the harness. Returns the process,
    the tools of each connected server, and the servers that did not connect. Raises ``MCPUnavailable`` before any
    model call when a required server did not connect.
    """
    with log.open("ab") as output:
        process = subprocess.Popen(
            [sys.executable, "-E", "-s", str(cli), "serve"], stdin=subprocess.DEVNULL, stdout=output, stderr=output
        )
    try:
        return _await_mcp(process, cli, required=required, log=log)
    except BaseException:
        # The caller gets no process to stop, so stop the CLI here: its stdio servers must not outlive the run.
        stop_mcp(process)
        raise


def stop_mcp(process: Any) -> None:
    """Stop the MCP CLI with SIGTERM, which closes its server sessions, and kill it if it does not exit in time."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _await_mcp(process: Any, cli: Path, *, required: list[str], log: Path) -> tuple[Any, dict, list]:
    socket = cli.with_name("mcp.sock")
    deadline = time.monotonic() + MCP_CONNECT_TIMEOUT_S + 30
    while not socket.exists():
        if process.poll() is not None:
            raise MCPUnavailable(
                f"The MCP CLI exited with {process.returncode}: {log.read_text(errors='replace')[-2000:]}"
            )
        if time.monotonic() > deadline:
            raise MCPUnavailable("The MCP CLI did not start in time")
        time.sleep(0.1)
    # The headers the grants carry are no longer needed on disk once every server has connected.
    cli.with_name("mcp_servers.json").unlink(missing_ok=True)
    listed = subprocess.run(
        [sys.executable, "-E", "-s", str(cli), "list"],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
        check=False,
    )
    if listed.returncode != 0:
        raise MCPUnavailable(f"Listing MCP tools failed: {listed.stderr.strip()}")
    listing = json.loads(listed.stdout)
    unavailable = [{"server": name, "error": entry["error"]} for name, entry in listing.items() if "error" in entry]
    missing = sorted(name for name in required if "error" in listing.get(name, {"error": "absent"}))
    if missing:
        reasons = (f"{name} ({str(listing.get(name, {}).get('error', 'absent'))[:500]})" for name in missing)
        raise MCPUnavailable("Required MCP servers did not connect: " + "; ".join(reasons))
    tools = {name: entry["tools"] for name, entry in listing.items() if "tools" in entry}
    return process, tools, unavailable


def run(payload: dict[str, Any], session_dir: Path, output_path: Path) -> int:
    runtime = {"hostname": os.uname().nodename, "pid": os.getpid(), "python": sys.executable}
    output: dict[str, Any] = {"runtime": runtime, "stop_reason": None, "error": None, "mcp": None}
    stop_request = Path(payload["stop_request_path"])

    def terminate(*_: object) -> None:
        # Only the first signal raises: the exception must not interrupt mini-SWE's own save on the way out.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        output["stop_reason"] = "cancelled" if stop_request.exists() else "wall_time"
        raise Terminated

    signal.signal(signal.SIGTERM, terminate)
    mcp_process = None
    exit_code = 0
    try:
        # mini-SWE reads these at import and on each model call; its global config stays out of the task's HOME.
        os.environ.update(
            MSWEA_GLOBAL_CONFIG_DIR=str(session_dir / "config"),
            MSWEA_CONFIGURED="true",
            MSWEA_SILENT_STARTUP="true",
            # Gym's Model Server retries; LiteLLM and mini-SWE do not retry on their own.
            MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT="1",
            LITELLM_LOCAL_MODEL_COST_MAP="true",
        )
        task = payload["task"]
        if payload.get("skills_dir"):
            task += f"\nTask skills are in {payload['skills_dir']}. Read the relevant SKILL.md files.\n"
        mcp = payload.get("mcp")
        if mcp is not None:
            cli = session_dir / "mcp_cli.py"
            mcp_process, tools, unavailable = start_mcp(cli, required=mcp["required"], log=session_dir / "mcp.log")
            output["mcp"] = {"servers": sorted(tools), "unavailable": unavailable}
            if tools:
                task += (
                    f"\nTask MCP tools (JSON schemas): {json.dumps(tools)}\n"
                    f"Call with: {sys.executable} -E -s {cli} call SERVER TOOL 'JSON_ARGUMENTS'.\n"
                )
        agent = build_agent(payload, attribute_calls=mcp is not None)
        result = agent.run(task)
        output["exit_status"] = result.get("exit_status")
    except Terminated:
        exit_code = TERMINATED_EXIT
    except MCPUnavailable as error:
        output["error"] = {"type": "mcp_unavailable", "message": str(error)}
        exit_code = MCP_UNAVAILABLE_EXIT
    except BaseException as error:
        output["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        # LiteLLM's exception class does not say the HTTP status (a 403 and a 413 are both APIError); its attribute does.
        status_code = getattr(error, "status_code", None)
        if type(status_code) is int:
            output["error"]["status_code"] = status_code
        exit_code = 1
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if mcp_process is not None:
            stop_mcp(mcp_process)
        write_atomic(output_path, output)
    return exit_code


def build_agent(payload: dict[str, Any], *, attribute_calls: bool):
    """mini-SWE's ``DefaultAgent`` with ``mini.yaml``, the episode's overlay, and a local environment that times commands."""
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.config import get_config_from_spec
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.exceptions import Submitted
    from minisweagent.models import get_model
    from minisweagent.models.utils.actions_toolcall_response import BASH_TOOL_RESPONSE_API
    from minisweagent.utils.serialize import recursive_merge

    class TimedLocalEnvironment(LocalEnvironment):
        """mini-SWE's local environment, adding when each command started and ended.

        mini-SWE copies ``output["extra"]`` into the saved tool observation. A submission raises ``Submitted`` after
        its command exited 0 and saves no observation, so its output, timing and tool call ID go on the exit message.
        """

        def _check_finished(self, output: dict) -> None:
            self.finished_output = output.get("output", "")
            super()._check_finished(output)

        def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
            started_at, started = time.time(), time.monotonic()
            environment = self.config.env
            if attribute_calls:
                self.config.env = {**environment, TOOL_CALL_ID_VARIABLE: str(action.get("tool_call_id") or "")}
            try:
                output = super().execute(action, cwd, timeout=timeout)
            except Submitted as submitted:
                extra = submitted.messages[0].setdefault("extra", {})
                extra.update(
                    timing(started_at, started), tool_call_id=action.get("tool_call_id"), output=self.finished_output
                )
                raise
            finally:
                self.config.env = environment
            output["extra"] = {**output.get("extra", {}), **timing(started_at, started)}
            return output

    class Agent(DefaultAgent):
        """``DefaultAgent`` whose step-by-step save cannot leave a half-written trajectory behind a forced stop."""

        def execute_actions(self, message: dict) -> list[dict]:
            # Saved before the commands run: a command that kills this process leaves the model's call on disk.
            self.save(self.config.output_path)
            return super().execute_actions(message)

        def serialize(self, *extra_dicts: dict) -> dict:
            data = super().serialize(*extra_dicts)
            # The model's commands can read the saved trajectory, which the runner's input is unlinked to protect:
            # leave out the rollout's model URL, its capture key and the API key.
            model_kwargs = data["info"]["config"]["model"].get("model_kwargs", {})
            for key in ("api_base", "api_key", "extra_headers"):
                model_kwargs.pop(key, None)
            return data

        def save(self, path: Path | None, *extra_dicts: dict) -> dict:
            data = self.serialize(*extra_dicts)
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                write_atomic(path, data)
            return data

    config = recursive_merge(get_config_from_spec("mini.yaml"), payload["config"])
    model_config = config["model"]
    # Strict false: Gym's Model Server checks the tool definitions it forwards.
    model_config["model_kwargs"] = {
        **model_config.get("model_kwargs", {}),
        "extra_body": {
            **model_config.get("model_kwargs", {}).get("extra_body", {}),
            "tools": [{**BASH_TOOL_RESPONSE_API, "strict": False}],
        },
    }
    environment_config = {key: value for key, value in config["environment"].items() if key != "environment_class"}
    agent_config = {key: value for key, value in config["agent"].items() if key != "agent_class"}
    agent_config["output_path"] = Path(payload["trajectory_path"])
    return Agent(get_model(config=model_config), TimedLocalEnvironment(**environment_config), **agent_config)


def timing(started_at: float, started: float) -> dict[str, float]:
    return {"started_at": started_at, "completed_at": time.time(), "duration_ms": (time.monotonic() - started) * 1000}


def main() -> int:
    """Run the episode in ``INPUT_JSON``; the uploaded process supervisor owns this process's lifetime."""
    if len(sys.argv) != 3:
        print("usage: sandbox_runner.py INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        return 2
    input_path, output_path = map(Path, sys.argv[1:])
    payload = json.loads(input_path.read_text())
    # The staged input is read once.
    input_path.unlink()
    return run(payload, input_path.parent, output_path)


if __name__ == "__main__":
    raise SystemExit(main())
