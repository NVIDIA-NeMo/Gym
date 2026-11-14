# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host proxy for the original Python tool; model code never runs in this process."""

import atexit
import base64
import io
import json
import os
import re
import selectors
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4


MAX_BYTES = 64 * 1024 * 1024


class SandboxFailure(RuntimeError):
    """Sandbox or transport failure, never a model-quality zero."""


def write_request(process, data, timeout):
    deadline = time.monotonic() + timeout
    fd = process.stdin.fileno()
    os.set_blocking(fd, False)
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_WRITE)
        while data:
            if not selector.select(max(0, deadline - time.monotonic())):
                raise SandboxFailure("Code sandbox request deadline")
            try:
                sent = os.write(fd, data[:65536])
                data = data[sent:]
            except BlockingIOError:
                continue


def read_reply(process, timeout):
    deadline = time.monotonic() + timeout
    data = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while b"\n" not in data:
            if not selector.select(max(0, deadline - time.monotonic())):
                raise SandboxFailure("Code sandbox response deadline")
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                raise SandboxFailure("Code sandbox terminated")
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise SandboxFailure("Code sandbox response exceeds byte limit")
    try:
        result = json.loads(data)
    except (ValueError, TypeError) as exc:
        raise SandboxFailure("Invalid code sandbox response") from exc
    if result.get("error"):
        raise SandboxFailure("Code sandbox operation failed: " + str(result["error"])[:300])
    return result


class CodeSandbox:
    def __init__(self, config, output):
        self.output = Path(output) / "sandbox_images"
        self.output.mkdir(parents=True, exist_ok=True)
        self.timeout = min(float(config.get("code_sandbox_timeout", 120)), 600)
        self.process = None
        self.lock = threading.Lock()
        self.failed = False
        atexit.register(self.close)
        self.config = {
            "output": str(Path(output).resolve()),
            "source_root": config.get("source_root") or os.environ["AGENTVISTA_ROOT"],
            "kernel_python": sys.executable,
            "episode_timeout": config.get("episode_timeout", 6000),
            "blocked_paths": [str(p) for p in (config.get("dataset"), config.get("image_folder"), output) if p],
        }
        allowed = {
            "PATH",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "TMPDIR",
            "HOME",
            "LANG",
            "LC_ALL",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_STATE_HOME",
        }
        env = {k: v for k, v in os.environ.items() if k in allowed}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            self.process = subprocess.Popen(
                [
                    config.get("code_sandbox_python") or os.environ.get("AGENTVISTA_SANDBOX_PYTHON") or sys.executable,
                    "-m",
                    "responses_api_agents.agentvista_agent.runtime.sandbox_bridge",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
            write_request(self.process, json.dumps(self.config).encode() + b"\n", 30)
            ready = read_reply(self.process, 60)
            if ready.get("isolation") != "bubblewrap-namespaces-v1":
                raise SandboxFailure("Code sandbox did not acknowledge isolation")
        except BaseException:
            self.failed = True
            (self.output.parent / "code_sandbox_failed.json").write_text(json.dumps({"sandbox_failed": True}))
            self.close()
            raise

    def call(self, params, image_map=None, **kwargs):
        from PIL import Image

        with self.lock:
            try:
                images = {}
                for name, image in (image_map or {}).items():
                    stream = io.BytesIO()
                    image.save(stream, format="PNG")
                    images[name] = base64.b64encode(stream.getvalue()).decode()
                request = json.dumps({"params": params, "images": images}).encode() + b"\n"
                if len(request) > MAX_BYTES:
                    raise SandboxFailure("Code sandbox request exceeds byte limit")
                if self.failed or self.process is None or self.process.poll() is not None:
                    raise SandboxFailure("Code sandbox is unavailable")
                write_request(self.process, request, self.timeout)
                reply = read_reply(self.process, self.timeout)
                text = reply["result"]
                if not isinstance(text, str) or not isinstance(reply["images"], dict) or len(reply["images"]) > 32:
                    raise SandboxFailure("Invalid code sandbox output")
                paths = {}
                for reference, data in reply["images"].items():
                    # Never interpret a sandbox-supplied host path or transfer a symlink.
                    image = Image.open(io.BytesIO(base64.b64decode(data, validate=True)))
                    image.load()
                    path = self.output / (uuid4().hex + ".png")
                    with path.open("xb") as stream:
                        image.save(stream, format="PNG")
                    paths[reference] = str(path)

                # The upstream host image parser follows markdown paths. Reject every
                # reference not materialized from returned image bytes by this proxy.
                def image_reference(match):
                    if match[2] not in paths:
                        return "[unavailable sandbox image]"
                    # The original default harness also uses labels in output filenames.
                    label = match[1]
                    if not re.fullmatch(r"tool_image_\d+|observation_\d+", label):
                        label = "tool_image_" + str(int(uuid4().hex[:8], 16))
                    return f"![{label}]({paths[match[2]]})"

                return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image_reference, text)
            except BaseException:
                self.failed = True
                (self.output.parent / "code_sandbox_failed.json").write_text(json.dumps({"sandbox_failed": True}))
                self.close()
                raise

    def close(self):
        if self.process is None:
            return
        process, self.process = self.process, None
        if process.stdin:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout):
            if stream:
                stream.close()


def install_code_sandbox(config, output):
    """Replace the registry constructor before either harness can instantiate it."""
    from tools import TOOL_REGISTRY

    original = TOOL_REGISTRY["code_interpreter"]
    sandbox = CodeSandbox(config, output)

    class IsolatedInterpreter:
        name = original.name
        description = original.description
        parameters = original.parameters

        def __init__(self, config=None):
            pass

        def call(self, *args, **kwargs):
            return sandbox.call(*args, **kwargs)

    TOOL_REGISTRY["code_interpreter"] = IsolatedInterpreter

    def close():
        TOOL_REGISTRY["code_interpreter"] = original
        sandbox.close()

    return sandbox, close
