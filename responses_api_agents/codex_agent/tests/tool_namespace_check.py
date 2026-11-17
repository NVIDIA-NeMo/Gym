# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end check that agent tool commands cannot kill the Codex harness (Linux, real bwrap).

Builds a real bwrap provider sandbox, starts the real supervisor and runner through
``CodexSandboxSession``, and lets "Codex" start the tool host exactly as Codex does with
``tool_pid_namespace`` (``environments.toml``'s program and args). The tool command then runs

    pkill -f "python3 -"; pkill -f python; pkill -f node; pkill -9 -f nemo-gym; kill -9 -1

and the harness must survive: the supervisor writes a normal receipt and Codex completes its turn.

* ``fake`` mode needs no Codex: a Python stand-in named ``node`` plays Codex and a shell
  script plays the exec-server. ``--control`` runs the same tool command in the harness's
  namespace instead (expected to kill the harness), which shows the check is sensitive.
* ``--codex-runtime DIR`` (a directory installed by install_codex_runtime.sh) runs the real
  Codex CLI against a scripted Responses endpoint served from this process (the sandbox
  shares the host network), so real ``exec_command`` calls go through the real exec-server.

    python -m responses_api_agents.codex_agent.tests.tool_namespace_check --bwrap /path/to/bwrap \\
        [--codex-runtime DIR] [--control] [--base-dir DIR]

Prints a JSON report and exits 0 when every expectation holds.
"""

import argparse
import asyncio
import http.server
import json
import os
import re
import shlex
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from nemo_gym.agents import (
    HARNESS_RUNNER_FILE,
    HARNESS_RUNNER_SCRIPT,
    AgentSeedSessionRequest,
    SandboxCommand,
    SandboxSession,
    shell_assignments,
)
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.providers.bwrap import BwrapProvider
from responses_api_agents.codex_agent.app import toml_dumps
from responses_api_agents.codex_agent.sandbox import CodexSandboxSession
from responses_api_agents.codex_agent.tool_host import (
    codex_config_overrides,
    environments_toml,
    tool_host_argv,
    tool_host_paths,
    tool_host_setup_command,
)


SESSION_DIR = "/tmp/nemo-gym-codex-sessions/tool-namespace-check"
WORKDIR = "/workspace"
RUNTIME_MOUNT = "/nemo-gym-codex-runtime"
FAKE_RUNTIME = "/tmp/nemo-gym-tool-namespace-check-runtime"
# Run from a file: a `sh -c` string holding these patterns would match (and kill) its own shell.
KILL_SCRIPT = """\
r=/workspace/kill-report.txt; : > $r
pkill -f "python3 -"; echo "pkill -f 'python3 -' $?" >> $r
pkill -f python; echo "pkill -f python $?" >> $r
pkill -f node; echo "pkill -f node $?" >> $r
pkill -9 -f nemo-gym; echo "pkill -9 -f nemo-gym $?" >> $r
kill -9 -1; echo "kill -9 -1 $?" >> $r
"""
VIEW_SCRIPT = """\
{ echo "uid $(id -u) pid $$"
  echo "pid1 $(tr '\\0' ' ' < /proc/1/cmdline)"
  echo "pids $(ls /proc | grep -cE '^[0-9]+$')"
  echo "nvidia $(ls /dev | grep -c nvidia)"
  echo "workspace $(touch /workspace/.w && echo rw || echo ro)"
} > /workspace/view-report.txt; cat /workspace/view-report.txt
"""
# Plays the native Codex binary: --version for the setup probe, exec-server runs the tool command.
FAKE_CODEX = """#!/bin/sh
case "$1" in
  --version) echo "codex-cli fake" ;;
  exec-server) sh /workspace/kill.sh; sh /workspace/view.sh > /dev/null ;;
esac
"""
# Plays Codex (named node): starts the tool host as Codex would (or, for --control, runs the tool
# command itself), then reports whether its runner and supervisor are still alive.
FAKE_NODE = """
import json, os, subprocess, sys
argv = json.loads(sys.argv[1])
subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
runner = os.getppid()
supervisor = int(open(f"/proc/{runner}/stat").read().rsplit(")", 1)[1].split()[1])
alive = {}
for name, pid in (("runner", runner), ("supervisor", supervisor)):
    try:
        os.kill(pid, 0)
        alive[name] = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\\0", b" ").decode()
    except OSError:
        alive[name] = None
print(json.dumps({"type": "item.completed", "item": {"id": "check", "type": "agent_message", "text": json.dumps(alive)}}))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}))
"""


class ScriptedResponses:
    """A Responses endpoint: the kill script, the view script, an interactive session, then an answer."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

        def session_id() -> int:
            match = re.search(r"session ID (\d+)", self.tool_outputs()[-1])
            return int(match.group(1)) if match else -1

        calls = [
            ("exec_command", lambda: {"cmd": "sh /workspace/kill.sh; echo done", "yield_time_ms": 3000}),
            ("exec_command", lambda: {"cmd": "sh /workspace/view.sh", "yield_time_ms": 3000}),
            # exec_command + write_stdin: an interactive session through the tool host.
            ("exec_command", lambda: {"cmd": "read line; echo got:$line", "tty": True, "yield_time_ms": 500}),
            ("write_stdin", lambda: {"session_id": session_id(), "chars": "hello\n", "yield_time_ms": 3000}),
        ]
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(body)
                n = len(owner.requests) - 1
                if n < len(calls):
                    name, make_args = calls[n]
                    args = make_args()
                    item = {
                        "type": "function_call",
                        "id": f"fc_{n}",
                        "call_id": f"call_{n}",
                        "name": name,
                        "arguments": json.dumps(args),
                        "status": "completed",
                    }
                else:
                    item = {
                        "type": "message",
                        "id": f"msg_{n}",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "done", "annotations": []}],
                    }
                response = {
                    "id": f"resp_{n}",
                    "object": "response",
                    "status": "completed",
                    "output": [item],
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                }
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in (
                    {"type": "response.created", "response": {**response, "output": [], "status": "in_progress"}},
                    {"type": "response.output_item.done", "output_index": 0, "item": item},
                    {"type": "response.completed", "response": response},
                ):
                    self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()

            def log_message(self, *_: Any) -> None:
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def tool_outputs(self) -> list[str]:
        if not self.requests:
            return []
        return [
            item.get("output", "") for item in self.requests[-1]["input"] if item.get("type") == "function_call_output"
        ]


async def _exec(sandbox: AsyncSandbox, command: str, timeout: float = 120) -> str:
    result = await sandbox.exec(command, cwd=WORKDIR, timeout_s=timeout)
    if result.return_code != 0 or result.error_type:
        raise RuntimeError(f"{command[:200]!r} failed: exit={result.return_code} {result.stderr}{result.stdout}")
    return result.stdout or ""


async def run_check(
    bwrap: str, *, base_dir: str, codex_runtime: str | None = None, control: bool = False
) -> dict[str, Any]:
    """Run one scenario and return its report; ``report["ok"]`` holds the verdict."""
    mounts = [{"src": codex_runtime, "dst": RUNTIME_MOUNT, "ro": True}] if codex_runtime else []
    provider = BwrapProvider(
        bwrap_path=bwrap,
        base_dir=base_dir,
        network="host",
        mounts=mounts,
        allow_nested_userns=True,
    )
    sandbox = await AsyncSandbox(provider).start(SandboxSpec(workdir=WORKDIR))
    report: dict[str, Any] = {
        "mode": "codex" if codex_runtime else "fake",
        "isolated": not control,
        "gym_uid": os.getuid(),
        "bwrap": bwrap,
    }
    server = None
    scratch_dir = tempfile.TemporaryDirectory(prefix="nemo-gym-tool-namespace-check-")
    scratch = scratch_dir.name
    try:
        runtime = RUNTIME_MOUNT if codex_runtime else FAKE_RUNTIME
        report["sandbox_uid"] = (await _exec(sandbox, "id -u")).strip()
        await _exec(sandbox, f"mkdir -p {SESSION_DIR}/home/.codex")
        for name, text in (("kill.sh", KILL_SCRIPT), ("view.sh", VIEW_SCRIPT)):
            await sandbox.upload(_scratch(scratch, text), f"{WORKDIR}/{name}")
        if not codex_runtime:
            fake = (
                f"{FAKE_RUNTIME}/codex/node_modules/@openai/codex-linux-fake/vendor/fake-unknown-linux-musl/bin/codex"
            )
            await sandbox.upload(_scratch(scratch, FAKE_CODEX, mode=0o755), fake)
        await sandbox.upload(Path(bwrap), tool_host_paths(SESSION_DIR)[0])
        setup = await sandbox.exec(tool_host_setup_command(SESSION_DIR, runtime), timeout_s=60)
        report["setup"] = {"exit": setup.return_code, "output": f"{setup.stderr or ''}{setup.stdout or ''}".strip()}
        if setup.return_code != 0 or setup.error_type:
            report["ok"] = False
            return report

        session = CodexSandboxSession(
            request=AgentSeedSessionRequest(
                agent_session_id="tool-namespace-check",
                episode_id=EpisodeId(rollout_id="tool-namespace-check", attempt=0),
                task_id=TaskId(taskset="check", task_id="check"),
            ),
            session=SandboxSession(sandbox=sandbox, session_dir=SESSION_DIR, workdir=WORKDIR, harness="Codex"),
            runtime=runtime,
        )
        await session.session.sandbox.upload(HARNESS_RUNNER_SCRIPT, f"{SESSION_DIR}/{HARNESS_RUNNER_FILE}")
        env = {"HOME": f"{SESSION_DIR}/home", "CODEX_HOME": f"{SESSION_DIR}/home/.codex", "OPENAI_API_KEY": "check"}
        if codex_runtime:
            server = ScriptedResponses()
            config = _codex_config(server.base_url)
            if not control:
                config.update(codex_config_overrides())
                await session.upload_text("home/.codex/environments.toml", environments_toml(SESSION_DIR))
            await session.upload_text("home/.codex/config.toml", toml_dumps(config))
            command = [
                f"{RUNTIME_MOUNT}/node/bin/node",
                f"{RUNTIME_MOUNT}/codex/node_modules/@openai/codex/bin/codex.js",
                *("exec", "--json", "--ephemeral", "--skip-git-repo-check", "--cd", WORKDIR, "--", "-"),
            ]
        else:
            tool = tool_host_argv(SESSION_DIR) if not control else ["sh", f"{WORKDIR}/kill.sh"]
            await sandbox.upload(_scratch(scratch, FAKE_NODE), f"{SESSION_DIR}/fake_node.py")
            # A Codex stand-in whose command line says node, as the real one's does.
            await _exec(sandbox, f'ln -sfn "$(command -v python3)" {SESSION_DIR}/bin/node')
            command = [f"{SESSION_DIR}/bin/node", f"{SESSION_DIR}/fake_node.py", json.dumps(tool)]
        # The runner starts this command under the real supervisor, as CodexSandboxSession.stage_activation does
        # for the Codex launcher.
        params = {
            "cwd": WORKDIR,
            "command": command,
            "env": [f"{name}={value}" for name, value in env.items()],
            "stdin": f"{SESSION_DIR}/prompt.txt",
            "stdout": f"{SESSION_DIR}/stdout.log",
            "stderr": f"{SESSION_DIR}/stderr.log",
            "events": f"{SESSION_DIR}/events.jsonl",
        }

        async def stage_activation() -> SandboxCommand:
            await session.upload_text("prompt.txt", "check")
            await session.upload_text("params.sh", shell_assignments(params))
            return SandboxCommand(argv=["bash", f"{SESSION_DIR}/{HARNESS_RUNNER_FILE}", f"{SESSION_DIR}/params.sh"])

        session.capture_budget_s = 30
        try:
            raw = await session.session.execute(
                stage_activation=stage_activation, collect=session.collect_artifacts, timeout=180, close_timeout=30
            )
        except Exception as error:  # noqa: BLE001 - the report carries it
            raw = session.session.artifacts or ""
            report["execute_error"] = f"{type(error).__name__}: {error}"
        events = [json.loads(line)[1] for line in raw.splitlines() if line.strip()]
        report["cleanup"] = session.session.cleanup
        report["cleanup_receipt_missing"] = session.session.cleanup_receipt_missing
        report["launch_return_code"] = session.session.launch_return_code
        report["turn_completed"] = any(event.get("type") == "turn.completed" for event in events)
        report["kill_report"] = await _read(sandbox, f"{WORKDIR}/kill-report.txt")
        report["tool_view"] = await _read(sandbox, f"{WORKDIR}/view-report.txt")
        if server is not None:
            report["model_requests"] = len(server.requests)
            report["tool_outputs"] = [output[-400:] for output in server.tool_outputs()]
        else:
            messages = [e["item"]["text"] for e in events if e.get("item", {}).get("type") == "agent_message"]
            report["harness_alive_after_tool"] = json.loads(messages[-1]) if messages else None
        report["checks"] = _checks(report, codex=bool(codex_runtime))
        expected = all(report["checks"].values())
        if control:
            # The control run passes only when the check itself ran, the kill commands ran, and the unprotected
            # harness then died: no completed turn, plus a lost receipt or a supervisor that reported a failure.
            # A crash before the turn is not that.
            cleanup = report["cleanup"] or {}
            report["ok"] = (
                "execute_error" not in report
                and "kill -9 -1" in (report["kill_report"] or "")
                and not report["turn_completed"]
                and (bool(report["cleanup_receipt_missing"]) or cleanup.get("return_code") not in (0, None))
            )
        else:
            report["ok"] = expected and "execute_error" not in report
        return report
    finally:
        if server is not None:
            server.server.shutdown()
        await sandbox.stop()
        scratch_dir.cleanup()


def _checks(report: dict[str, Any], *, codex: bool) -> dict[str, bool]:
    cleanup = report.get("cleanup") or {}
    view = report.get("tool_view") or ""
    checks = {
        "turn_completed": report["turn_completed"],
        "supervisor_receipt_clean": bool(cleanup)
        and cleanup.get("return_code") == 0
        and not cleanup.get("timed_out")
        and not cleanup.get("error")
        and not report["cleanup_receipt_missing"],
        "tool_signals_never_reached_harness": not cleanup.get("foreign_signals"),
        "kill_commands_ran": "kill -9 -1" in (report.get("kill_report") or ""),
        # The fake tool host is a shell script, so its command line starts with the shell.
        "tool_namespace_pid1_is_tool_host": bool(re.search(r"^pid1 .*nemo-gym-tool-host exec-server", view, re.M)),
        "no_gpu_devices": "nvidia 0" in view,
        "workspace_writable": "workspace rw" in view,
    }
    if codex:
        outputs = report.get("tool_outputs") or []
        checks["interactive_session_works"] = bool(outputs) and "got:hello" in outputs[-1]
    else:
        alive = report.get("harness_alive_after_tool") or {}
        checks["runner_and_supervisor_alive"] = bool(alive.get("runner")) and bool(alive.get("supervisor"))
    return checks


def _codex_config(base_url: str) -> dict[str, Any]:
    return {
        "model_provider": "gym",
        "model": "tool-namespace-check",
        "approval_policy": "never",
        "sandbox_mode": "danger-full-access",
        "web_search": "disabled",
        "check_for_update_on_startup": False,
        "analytics": {"enabled": False},
        "history": {"persistence": "none"},
        "features": {"multi_agent": False, "code_mode": False},
        "model_providers": {
            "gym": {"name": "gym", "base_url": base_url, "env_key": "OPENAI_API_KEY", "wire_api": "responses"}
        },
    }


def _scratch(directory: str, text: str, mode: int = 0o644) -> Path:
    path = Path(directory) / f"{len(os.listdir(directory))}"
    path.write_text(text)
    path.chmod(mode)
    return path


async def _read(sandbox: AsyncSandbox, path: str) -> str | None:
    result = await sandbox.exec(f"cat {shlex.quote(path)}", cwd=WORKDIR, timeout_s=30)
    return result.stdout if result.return_code == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bwrap", required=True, help="bwrap binary for the sandbox and the tool namespace")
    parser.add_argument("--codex-runtime", help="runtime installed by install_codex_runtime.sh: run the real Codex")
    parser.add_argument(
        "--control", action="store_true", help="run tool commands unisolated (should kill the harness)"
    )
    parser.add_argument("--base-dir", default=None, help="node-local parent for the sandbox (default: a temp dir)")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="nemo-gym-tool-namespace-sbx-") as default_base:
        report = asyncio.run(
            run_check(
                os.path.abspath(args.bwrap),
                base_dir=args.base_dir or default_base,
                codex_runtime=os.path.abspath(args.codex_runtime) if args.codex_runtime else None,
                control=args.control,
            )
        )
    print(json.dumps(report, indent=1, default=str))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
