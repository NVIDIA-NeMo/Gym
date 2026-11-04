# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture Codex JSON events; the supervisor bounds execution; the sandbox owner stops task services."""

import json
import os
import selectors
import signal
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from time import time


OOM_SCORE_ADJ = "/proc/self/oom_score_adj"


def prefer_for_oom_kill(path: str = OOM_SCORE_ADJ) -> str:
    """Make this process and its children the memory-cgroup OOM killer's preferred victims.

    Raise oom_score_adj to 1000, or to 0 when the sandbox starts it below zero (a Guaranteed
    pod's -997), so the sandbox never becomes the node's first victim in a global OOM. Raising
    needs no privilege; a failure leaves the score unchanged. Returns what happened.
    """
    try:
        with open(path) as handle:
            before = int(handle.read())
        after = 0 if before < 0 else 1000
        with open(path, "w") as handle:
            handle.write(str(after))
    except (OSError, ValueError) as error:
        return f"unchanged: {error}"
    return f"{before} -> {after}"


def run(params: dict) -> int:
    """Run Codex with isolated input/environment and flush timestamped events as they arrive."""
    directory = Path(params["directory"])
    (directory / "runtime.json").write_text(json.dumps({"hostname": os.uname().nodename, "pid": os.getpid()}))
    (directory / "prompt.txt").write_text(params["prompt"])
    if params.get("raise_oom_score_adj"):
        # Codex and its tools inherit the score; execd and the supervisor keep the sandbox's own.
        print(f"oom_score_adj: {prefer_for_oom_kill()}", file=sys.stderr, flush=True)
    stopping = False

    def interrupt(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, interrupt)
    with ExitStack() as stack:
        command = list(params["command"])
        if params.get("model_base_url"):
            # -I intentionally excludes the session directory from sys.path.
            # Load only the adapter file staged beside this worker.
            import runpy

            proxy = runpy.run_path(str(Path(__file__).with_name("sandbox_transport.py")))["rate_limit_proxy"]
            base_url = stack.enter_context(
                proxy(
                    params["model_base_url"],
                    params["invocation_id"],
                    directory / "model_attempts.jsonl",
                    timeout=params["timeout"],
                )
            )
            command[2:2] = ["-c", "model_providers.gym.base_url=" + json.dumps(base_url)]
        stderr = stack.enter_context((directory / "stderr.log").open("wb"))
        stdin = stack.enter_context((directory / "prompt.txt").open("rb"))
        events = stack.enter_context((directory / "events.jsonl").open("w"))
        # By default Codex gets the sandbox's PATH, locale, HOME and XDG directories;
        # inherit_env passes the task's whole environment, as a harness started in the container sees it.
        inherited = (
            dict(os.environ)
            if params.get("inherit_env")
            else {
                key: os.environ[key]
                for key in (
                    "PATH",
                    "LANG",
                    "LC_ALL",
                    "HOME",
                    "XDG_CONFIG_HOME",
                    "XDG_DATA_HOME",
                    "XDG_CACHE_HOME",
                    "XDG_STATE_HOME",
                    # Lets the supervisor find Codex's tools after they leave its process tree.
                    "NEMO_GYM_SUPERVISOR",
                )
                if key in os.environ
            }
        )
        process = subprocess.Popen(
            command,
            cwd=params["cwd"],
            env={**inherited, **params["env"]},
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        try:
            pending = bytearray()

            def capture(line: bytes) -> None:
                try:
                    event = json.loads(line.decode(errors="replace"))
                except (ValueError, RecursionError):
                    return
                if isinstance(event, dict):
                    events.write(json.dumps([time(), event]) + "\n")
                    events.flush()

            # A detached tool can inherit stdout. Drain available data after Codex exits,
            # without waiting for that tool to close its pipe. Resources may retain
            # task services through verification.
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    if stopping and process.poll() is None:
                        process.terminate()
                        stopping = False
                    ready = selector.select(timeout=0.05)
                    if ready:
                        chunk = os.read(process.stdout.fileno(), 64 * 1024)
                        if not chunk:
                            break
                        pending.extend(chunk)
                        while b"\n" in pending:
                            line, _, pending = pending.partition(b"\n")
                            capture(line)
                    elif process.poll() is not None:
                        break
            if pending:
                capture(pending)
            return process.wait()
        finally:
            # An I/O/capture error must not leave the CLI changing the task.
            # Only stop our direct CLI; task services belong to the sandbox owner.
            if process.poll() is None:
                process.kill()
            process.wait()


def main() -> None:
    if sys.argv[1] == "--snapshot":
        directory = Path(sys.argv[2])
        files = sorted((directory / "home/.codex/sessions").glob("**/*.jsonl"))
        if files:
            (directory / "native_session.jsonl").write_text(
                "\n".join(line for path in files for line in path.read_text(errors="replace").splitlines())
            )
        return
    params = json.loads(Path(sys.argv[1]).read_text())
    raise SystemExit(run(params))


if __name__ == "__main__":
    main()
