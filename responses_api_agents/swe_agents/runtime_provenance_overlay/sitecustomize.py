# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe the pinned OpenHands lifecycle before executable setup or policy actions."""

import atexit
import importlib.abc
import importlib.machinery
import json
import os
import runpy
import socket
import sys
import threading
from pathlib import Path


class Observer:
    def __init__(self, role):
        self.lock = threading.Lock()
        self.nonce = os.environ["NEMO_GYM_RUNTIME_NONCE"]
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect(os.environ["NEMO_GYM_RUNTIME_SOCKET"])
        self.stream = self.sock.makefile("rb")
        self.send("hello", role=role)
        atexit.register(self.complete)

    def send(self, event, **fields):
        with self.lock:
            try:
                payload = {"nonce": self.nonce, "event": event, **fields}
                self.sock.sendall(json.dumps(payload).encode() + b"\n")
                if self.stream.readline(32) != b"ok\n":
                    raise RuntimeError("Runtime provenance observer did not acknowledge the event")
            except Exception:
                self.sock.close()
                raise

    def complete(self):
        try:
            self.send("complete")
        except Exception:
            pass
        self.stream.close()
        self.sock.close()


def install():
    if not os.getenv("NEMO_GYM_RUNTIME_NONCE"):
        return
    args = getattr(sys, "orig_argv", [])
    is_server = any(args[i : i + 2] == ["-m", "openhands.runtime.action_execution_server"] for i in range(len(args)))
    is_controller = any(
        arg.endswith("/evaluation/benchmarks/swe_bench/run_infer.py")
        or arg == "evaluation/benchmarks/swe_bench/run_infer.py"
        for arg in args
    )
    if not (is_server or is_controller):
        return
    observer = Observer("server" if is_server else "controller")
    # Both controller helpers and runtime setup can execute shell/profile code.
    # Only the wrapped, identity-tracked runtime bootstrap may cross this boundary.
    installed = False
    creating_runtime = False

    def audit(event, args):
        nonlocal installed
        if event == "nemo_gym.runtime_observer_probe":
            installed = True
        elif event in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.fork", "os.forkpty"}:
            if event == "subprocess.Popen" and creating_runtime and not is_server:
                command = args[1]
                if isinstance(command, (list, tuple)) and any(
                    list(command[i : i + 2]) == ["-m", "openhands.runtime.action_execution_server"]
                    for i in range(len(command))
                ):
                    return
            # The synchronous ACK happens before execution. Failures abort the operation.
            observer.send("execution_started")

    sys.addaudithook(audit)
    sys.audit("nemo_gym.runtime_observer_probe")
    if not installed:
        raise RuntimeError("Runtime provenance audit hook was not installed")
    if is_server:
        observer.send("ready")
        return

    target = "openhands.runtime.impl.local.local_runtime"

    class Loader:
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            runtime = module.LocalRuntime
            execute = runtime.execute_action
            wait_alive = runtime._wait_until_alive
            create_server = module._create_server
            processes = {}

            def observed_create(*args, **kwargs):
                nonlocal creating_runtime
                creating_runtime = True
                try:
                    result = create_server(*args, **kwargs)
                except BaseException:
                    observer.send("invalid")
                    raise
                finally:
                    creating_runtime = False
                process = result[0].process
                try:
                    start = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
                except (OSError, ValueError, IndexError):
                    start = None
                processes[process] = start
                observer.send("server_started", pid=process.pid, start=start)
                return result

            async def observed_execute(self, *args, **kwargs):
                observer.send("execution_started")
                return await execute(self, *args, **kwargs)

            def observed_wait(self, *args, **kwargs):
                try:
                    return wait_alive(self, *args, **kwargs)
                finally:
                    process = self.server_process
                    if process is not None and process.poll() is not None:
                        observer.send(
                            "runtime_exit",
                            pid=process.pid,
                            start=processes.get(process),
                            returncode=process.returncode,
                        )

            module._create_server = observed_create
            runtime.execute_action = observed_execute
            runtime._wait_until_alive = observed_wait
            observer.send("ready")

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "openhands.runtime.impl.local.local_runtime":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = Loader(spec.loader)
            return spec

    if target in sys.modules:
        raise RuntimeError("Runtime imported before provenance instrumentation")
    sys.meta_path.insert(0, Finder())


try:
    install()
finally:
    # A missing observer must not disable the existing training-token routing.
    if os.getenv("NEMO_GYM_MODEL_SERVER_BASE_URL") and os.path.isfile("/nemo_gym_capture_overlay/sitecustomize.py"):
        runpy.run_path("/nemo_gym_capture_overlay/sitecustomize.py")
