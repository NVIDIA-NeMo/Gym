# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym supervises Bubblewrap; the local provider alone supplies no isolation."""

import asyncio
import contextlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from responses_api_agents.agentvista_agent.runtime.source import verify_source


LIMIT = 64 * 1024 * 1024


def sandbox_command(config, private):
    binary = shutil.which("bwrap")
    if not binary:
        raise RuntimeError("Bubblewrap is required; host execution is forbidden")
    python = Path(config["kernel_python"])
    # Runtime directories are operator-controlled executables/libraries, never
    # source checkouts, datasets, episode folders, home directories, or mounts of /.
    probe = subprocess.check_output(
        [str(python), "-I", "-c", "import sys,json; print(json.dumps([sys.prefix,sys.base_prefix]))"],
        text=True,
        timeout=15,
    )
    prefixes = json.loads(probe)
    runtime = {Path(p) for p in prefixes} | {Path(p).resolve() for p in prefixes}
    runtime.add(python.parent.parent)
    runtime.add(python.resolve().parent.parent)
    link = python
    while link.is_symlink():
        target = Path(os.readlink(link))
        link = target if target.is_absolute() else link.parent / target
        runtime.add(link.parent.parent)
    forbidden = [
        Path(config["source_root"]).resolve(),
        private.parent.resolve(),
        *[Path(p).resolve() for p in config.get("blocked_paths", [])],
    ]
    runtime |= {Path(p) for p in ("/usr/lib", "/lib", "/lib64") if Path(p).exists()}
    for path in runtime:
        if path in {Path("/"), Path("/home"), Path("/root")} or any(
            p.is_relative_to(path.resolve()) for p in forbidden
        ):
            raise ValueError("Unsafe Python runtime prefix")
    args = [binary, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL", "--clearenv"]
    for path in sorted(runtime):
        args += ["--ro-bind", str(path.resolve()), str(path)]
    # A venv's executable can be a symlink into the base runtime; keep the original
    # venv path so Python resolves its site-packages without host PYTHONPATH.
    args += [
        "--ro-bind",
        str(private / "app"),
        "/app",
        "--bind",
        str(private / "channel"),
        "/channel",
        "--tmpfs",
        "/workspace",
        "--tmpfs",
        "/tmp",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--chdir",
        "/workspace",
    ]
    for key, value in {
        "HOME": "/workspace",
        "PYTHONPATH": "/app",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MPLCONFIGDIR": "/workspace/matplotlib",
        "IPYTHONDIR": "/workspace/ipython",
        "JUPYTER_RUNTIME_DIR": "/workspace/jupyter",
    }.items():
        args += ["--setenv", key, value]
    args += [str(python), "/app/sandbox_runtime.py"]
    return shlex.join(args) + " </dev/null"


async def run(config):
    source = Path(config["source_root"]).resolve(strict=True)
    verify_source(source)
    with tempfile.TemporaryDirectory(prefix="code-sandbox-", dir=config["output"]) as folder:
        private = Path(folder)
        app = private / "app"
        (app / "tools").mkdir(parents=True)
        (private / "channel").mkdir()
        (app / "tools/__init__.py").write_text("")
        for name in ("code_interpreter.py", "base.py", "tool_registry.py"):
            shutil.copyfile(source / "tools" / name, app / "tools" / name)
        shutil.copyfile(Path(__file__).with_name("sandbox_runtime.py"), app / "sandbox_runtime.py")
        command = sandbox_command(config, private) + " >/dev/null 2>&1"
        # Only this fixed namespace launcher runs through local exec. Neither a
        # model string nor a model-selected shell command reaches the host shell.
        async with AsyncSandbox({"local": {}}, SandboxSpec(workdir=str(private))) as sandbox:
            await sandbox.start()
            channel_fd = os.open(private / "channel", os.O_RDONLY | os.O_DIRECTORY)
            task = asyncio.create_task(sandbox.exec(command, timeout_s=config.get("episode_timeout", 6000)))
            writer = None
            try:
                deadline = asyncio.get_running_loop().time() + 45
                while True:
                    if task.done():
                        result = await task
                        raise RuntimeError(
                            f"Code sandbox startup failed (exit {result.return_code}, {result.error_type})"
                        )
                    try:
                        reader, writer = await asyncio.open_unix_connection(
                            f"/proc/self/fd/{channel_fd}/tool.sock", limit=LIMIT + 1
                        )
                        break
                    except (FileNotFoundError, ConnectionRefusedError):
                        if asyncio.get_running_loop().time() > deadline:
                            raise TimeoutError("Code sandbox readiness deadline")
                        await asyncio.sleep(0.1)
                sys.stdout.buffer.write(await asyncio.wait_for(reader.readline(), 30))
                sys.stdout.buffer.flush()
                while raw := await asyncio.to_thread(sys.stdin.buffer.readline, LIMIT + 1):
                    if len(raw) > LIMIT:
                        raise ValueError("Oversized tool request")
                    writer.write(raw)
                    await writer.drain()
                    response = await asyncio.wait_for(reader.readline(), 600)
                    if not response:
                        raise RuntimeError("Code sandbox disconnected")
                    sys.stdout.buffer.write(response)
                    sys.stdout.buffer.flush()
                writer.close()
                await writer.wait_closed()
            finally:
                if writer is not None:
                    writer.close()
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(writer.wait_closed(), 1)
                os.close(channel_fd)
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


def main():
    config = json.loads(sys.stdin.buffer.readline(LIMIT + 1))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    task = loop.create_task(run(config))
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    try:
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": str(exc)[:600]}) + "\n")
        sys.stdout.flush()
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        # A cancelled stdin to_thread can still be waiting on a pipe. The proxy
        # closes stdin on termination; do not wait indefinitely for that thread.
        loop.close()


if __name__ == "__main__":
    main()
