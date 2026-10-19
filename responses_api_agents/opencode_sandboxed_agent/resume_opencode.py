# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resume OpenCode's existing user turn without appending a checkpoint prompt.

Runs inside the sandbox. OpenCode 1.17.11's prompt API upserts ``messageID`` and
leaves existing parts in place when ``parts`` is empty, then enters the session
loop. Its CLI instead requires a new prompt, which changes the request and makes
a saved generation prefix ineligible for reuse.
"""

from __future__ import annotations

import base64
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import uuid4


def resume_payload(messages: list[dict]) -> dict:
    """Re-enter the last user message, retaining its model and agent settings."""
    users = [message["info"] for message in messages if message["info"]["role"] == "user"]
    if not users:
        raise ValueError("Cannot resume an OpenCode session without a saved user turn")
    # IDs establish conversation order; a resume updates the user message's timestamp.
    user = max(users, key=lambda info: info["id"])
    payload = {"messageID": user["id"], "parts": []}
    for name in ("agent", "system", "format", "tools"):
        if user.get(name) is not None:
            payload[name] = user[name]
    model = user.get("model")
    if model:
        payload["model"] = {name: model[name] for name in ("providerID", "modelID")}
        if model.get("variant") is not None:
            payload["variant"] = model["variant"]
    return payload


def resume_session(base_url: str, authorization: str, directory: str) -> None:
    """Resume the newest root session in this rollout's private data directory."""

    def request(path: str, payload: dict | None = None) -> object:
        url = base_url + path + "?" + urllib.parse.urlencode({"directory": directory})
        headers = {"Authorization": authorization, "Content-Type": "application/json"}
        data = None if payload is None else json.dumps(payload).encode()
        with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers)) as response:
            return json.load(response)

    sessions = request("/session")
    roots = [session for session in sessions if not session.get("parentID")]
    if not roots:
        raise ValueError("Cannot resume: the rollout has no saved OpenCode root session")
    session = max(roots, key=lambda item: item["time"]["updated"])
    session_path = "/session/" + urllib.parse.quote(session["id"], safe="")
    messages = request(session_path + "/message")
    before_users = {
        message["info"]["id"]: message["parts"] for message in messages if message["info"]["role"] == "user"
    }
    result = request(session_path + "/message", resume_payload(messages))
    if result["info"].get("error"):
        raise RuntimeError("The resumed OpenCode session ended with an execution error")
    after = request(session_path + "/message")
    after_users = {message["info"]["id"]: message["parts"] for message in after if message["info"]["role"] == "user"}
    if after_users != before_users:
        raise RuntimeError(
            "OpenCode changed the saved user turns while resuming; prefix recovery requires unchanged input"
        )
    print("OpenCode resumed the existing user turn without adding a prompt", flush=True)


def run() -> None:
    """Start an authenticated loopback server and stop it when the run finishes."""
    password = uuid4().hex
    username = "opencode"
    authorization = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
    env = {**os.environ, "OPENCODE_SERVER_USERNAME": username, "OPENCODE_SERVER_PASSWORD": password}
    with tempfile.TemporaryDirectory(prefix="nemo-gym-opencode-resume-") as temporary:
        log_path = Path(temporary) / "server.log"
        with log_path.open("w") as log:
            server = subprocess.Popen(
                ["opencode", "serve", "--hostname", "127.0.0.1", "--port", "0"],
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        try:
            deadline = time.monotonic() + 60
            while True:
                match = re.search(r"opencode server listening on (http://127\.0\.0\.1:\d+)", log_path.read_text())
                if match:
                    break
                if server.poll() is not None:
                    raise RuntimeError("OpenCode resume server exited before it became ready")
                if time.monotonic() >= deadline:
                    raise TimeoutError("OpenCode resume server did not become ready within 60 seconds")
                time.sleep(0.1)
            # OpenCode's CLI uses PWD, including a shell's logical symlink path.
            resume_session(match.group(1), authorization, os.environ.get("PWD", os.getcwd()))
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGINT)
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


def interrupted(signum: int, frame: object) -> None:
    """Let checkpoint signals unwind the helper and stop its server process group."""
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        run()
    except KeyboardInterrupt:
        sys.exit(130)
