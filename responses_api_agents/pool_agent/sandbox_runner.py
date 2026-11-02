# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Pool runner; Python's standard library is sufficient inside the sandbox."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def strip_cache_hints(body: dict[str, object]) -> None:
    """Remove Pool's provider hints, preserving tool schemas, arguments and message text."""
    for message in body.get("messages", []):
        message.pop("cache_control", None)
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                part.pop("cache_control", None)
    for tool in body.get("tools", []):
        tool.pop("cache_control", None)
        if isinstance(tool.get("function"), dict):
            tool["function"].pop("cache_control", None)


def model_proxy(base_url: str, *, timeout: float = 300, calls_path: Path | None = None) -> ThreadingHTTPServer:
    """Bridge Pool's nonstreaming requests to Gym using a loopback-only threaded server."""

    completed = 0
    lock = threading.Lock()
    if calls_path is not None:
        calls_path.write_text(json.dumps({"completed": 0}))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_POST(self) -> None:
            nonlocal completed
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                strip_cache_hints(body)
                request = Request(
                    base_url.rstrip("/") + "/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer dummy_key",
                    },  # pragma: allowlist secret
                )
                try:
                    upstream = urlopen(request, timeout=timeout)
                except HTTPError as error:
                    upstream = error
                with upstream:
                    status, payload = upstream.status, upstream.read()
                    content_type = upstream.headers.get("Content-Type", "application/json")
                if 200 <= status < 300 and calls_path is not None:
                    # Persist before delivery so a killed CLI cannot erase completed inference.
                    with lock:
                        completed += 1
                        temporary = calls_path.with_suffix(".tmp")
                        temporary.write_text(json.dumps({"completed": completed}))
                        temporary.replace(calls_path)
            except Exception as error:
                status, content_type = 502, "application/json"
                payload = json.dumps({"error": str(error)}).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler)


def run(directory: Path, workdir: str) -> int:
    """Keep Pool state outside the task repository and retain stdout, stderr and exit status."""
    env = dict(os.environ)
    task_env_path = directory / "task_env.json"
    if task_env_path.exists():
        env.update(json.loads(task_env_path.read_text()))
    # Do not inherit Pool remote sinks or provider overrides from the task image.
    for key in list(env):
        if key.startswith(("POOL_", "POOLSIDE_")) or key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
            env.pop(key)
    task_home = env.get("HOME") or str(Path.home())
    task_directories = {"HOME": task_home}
    for key, suffix in (
        ("XDG_CONFIG_HOME", ".config"),
        ("XDG_STATE_HOME", ".local/state"),
        ("XDG_DATA_HOME", ".local/share"),
        ("XDG_CACHE_HOME", ".cache"),
    ):
        task_directories[key] = env.get(key) or str(Path(task_home) / suffix)
    (directory / "task_directories.json").write_text(json.dumps(task_directories))
    config_path = directory / "agent_config.json"
    config = json.loads(config_path.read_text())
    # Shells inherit the task HOME, while the CLI keeps its own configuration private.
    config["tools"]["shell"].update(
        {
            "shell": sys.executable,
            "shell_args": ["-I", str(Path(__file__).resolve()), "--task-shell", str(directory)],
        }
    )
    provider = config["model"]["provider"]["openai"]
    proxy = model_proxy(
        provider["base_url"], timeout=float(config["http_timeout"][:-1]), calls_path=directory / "model_calls.json"
    )
    provider["base_url"] = f"http://127.0.0.1:{proxy.server_port}/v1"
    config_path.write_text(json.dumps(config))
    env.update(
        {
            "HOME": str(directory / "home"),
            "XDG_CONFIG_HOME": str(directory / "config"),
            "XDG_STATE_HOME": str(directory / "state"),
            "XDG_DATA_HOME": str(directory / "data"),
            "XDG_CACHE_HOME": str(directory / "cache"),
            "POOLSIDE_API_KEY": "gym-model-server",  # pragma: allowlist secret
        }
    )
    for key in ("HOME", "XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    (directory / "runtime.json").write_text(json.dumps({"hostname": socket.gethostname(), "pid": os.getpid()}))
    command = [
        str(directory / "bin/pool"),
        "exec",
        "-o",
        "json",
        "--sandbox",
        "disabled",
        "--unsafe-auto-allow",
        "--keep-running-shells",
        "--run-id",
        "gym-root",
        "--agent-config-file",
        str(directory / "agent_config.json"),
        "-f",
        str(directory / "prompt.txt"),
    ]
    stopping = False

    def interrupt(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    previous_handler = signal.signal(signal.SIGTERM, interrupt)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    try:
        with (directory / "stdout.jsonl").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
            if stopping:
                return_code = -signal.SIGTERM
            else:
                process = subprocess.Popen(
                    command, cwd=workdir, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr
                )
                while process.poll() is None:
                    if stopping:
                        try:
                            process.terminate()
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                        break
                    try:
                        process.wait(timeout=0.1)
                    except subprocess.TimeoutExpired:
                        pass
                return_code = process.returncode
        (directory / "exit.json").write_text(json.dumps(return_code))
        return return_code
    finally:
        proxy.shutdown()
        proxy.server_close()
        thread.join(timeout=5)
        signal.signal(signal.SIGTERM, previous_handler)


def snapshot(directory: Path) -> None:
    """Collect only the explicitly named root session after the supervisor confirms cleanup."""
    candidates = list(directory.rglob("trajectory-*_gym-root.ndjson"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one Pool root trajectory, found {len(candidates)}")
    (directory / "trajectory.ndjson").write_bytes(candidates[0].read_bytes())


if __name__ == "__main__":
    if sys.argv[1] == "--task-shell":
        task_env = dict(os.environ)
        task_env.update(json.loads((Path(sys.argv[2]) / "task_directories.json").read_text()))
        os.execvpe("bash", ["bash", "-c", sys.argv[3]], task_env)
    elif sys.argv[1] == "--snapshot":
        snapshot(Path(sys.argv[2]))
    else:
        sys.exit(run(Path(sys.argv[1]), sys.argv[2]))
