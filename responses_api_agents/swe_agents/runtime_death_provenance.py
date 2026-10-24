# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-owned evidence for OpenHands deaths before execution-capable initialization."""

import asyncio
import json
import os
import socket
import struct
import tempfile
import uuid
from pathlib import Path


SUPPORTED_OPENHANDS_COMMIT = "54d2a95edc666551b854a7e5d4205ec69e60ff23"
PROVENANCE_MOUNT = "/nemo_gym_runtime_provenance"
SOCKET_MOUNT = "/nemo_gym_runtime_provenance.sock"


def _identity(pid: int) -> tuple[int, str]:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[1]), fields[19]


class RuntimeDeathMonitor:
    """Accept only this container's peers; execution is a monotonic host decision.

    Evidence stays in host memory. Only the socket is mounted in the container.
    Missing, incomplete, or contradictory observations never justify masking.
    """

    def __init__(self) -> None:
        self.nonce = uuid.uuid4().hex
        self.directory = tempfile.TemporaryDirectory(prefix="ng-runtime-", dir="/tmp")
        self.path = str(Path(self.directory.name) / "observer.sock")
        self.root: tuple[int, str] | None = None
        self.root_ready = asyncio.Event()
        self.server = None
        self.tasks: set[asyncio.Task] = set()
        self.peers: dict[tuple[int, str], dict] = {}
        self.started = False
        self.invalid = False
        self.controller = None
        self.expected: set[tuple[int, str]] = set()
        self.exits: set[tuple[int, str]] = set()
        self.last_started = None
        self.last_exit = None

    async def start(self) -> None:
        self.server = await asyncio.start_unix_server(self._accept, path=self.path, limit=2048)

    def set_root(self, pid: int) -> None:
        try:
            self.root = (pid, _identity(pid)[1])
        except (OSError, ValueError, IndexError):
            self.invalid = True
        finally:
            self.root_ready.set()

    def _belongs_to_run(self, pid: int) -> bool:
        seen = set()
        try:
            while pid > 1 and pid not in seen:
                seen.add(pid)
                parent, start = _identity(pid)
                if (pid, start) == self.root:
                    return True
                pid = parent
        except (OSError, ValueError, IndexError):
            pass
        return False

    async def _accept(self, reader, writer) -> None:
        task = asyncio.current_task()
        self.tasks.add(task)
        peer = None
        try:
            await asyncio.wait_for(self.root_ready.wait(), 5)
            pid, uid, _ = struct.unpack(
                "3i", writer.get_extra_info("socket").getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            )
            if uid != os.getuid() or not self._belongs_to_run(pid):
                return
            identity = (pid, _identity(pid)[1])
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            is_server = any(
                argv[i : i + 2] == [b"-m", b"openhands.runtime.action_execution_server"] for i in range(len(argv))
            )
            is_controller = any(
                arg.endswith(b"/evaluation/benchmarks/swe_bench/run_infer.py")
                or arg == b"evaluation/benchmarks/swe_bench/run_infer.py"
                for arg in argv
            )
            if not (is_server or is_controller) or identity in self.peers:
                return
            nspid = next(
                line for line in Path(f"/proc/{pid}/status").read_text().splitlines() if line.startswith("NSpid:")
            ).split()[-1]
            hello = json.loads(await asyncio.wait_for(reader.readline(), 5))
            role = "server" if is_server else "controller"
            if hello != {"nonce": self.nonce, "event": "hello", "role": role}:
                return
            peer = {"role": role, "nspid": int(nspid), "start": identity[1], "ready": False, "complete": False}
            self.peers[identity] = peer
            if role == "controller":
                if self.controller is not None:
                    self.invalid = True
                self.controller = identity
            writer.write(b"ok\n")
            await writer.drain()
            while line := await reader.readline():
                event = json.loads(line)
                if event.get("nonce") != self.nonce or peer["complete"]:
                    self.invalid = True
                    break
                kind = event.get("event")
                if kind == "ready":
                    peer["ready"] = True
                elif kind == "execution_started":
                    self.started = True
                elif kind in {"server_started", "runtime_exit"} and role == "controller" and peer["ready"]:
                    if type(event.get("pid")) is not int or not isinstance(event.get("start"), str):
                        self.invalid = True
                        break
                    key = (event["pid"], event["start"])
                    if kind == "server_started":
                        self.expected.add(key)
                        self.last_started = key
                    elif key in self.expected and type(event.get("returncode")) is int:
                        self.exits.add(key)
                        self.last_exit = key
                    else:
                        self.invalid = True
                        break
                elif kind == "complete" and peer["ready"]:
                    peer["complete"] = True
                else:
                    self.invalid = True
                    break
                writer.write(b"ok\n")
                await writer.drain()
        except (OSError, ValueError, IndexError, StopIteration, asyncio.TimeoutError):
            if peer is not None:
                self.invalid = True
        finally:
            if peer is not None and not peer["complete"]:
                self.invalid = True
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
            self.tasks.discard(task)

    async def finish(self) -> str:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        if self.tasks:
            done, pending = await asyncio.wait(self.tasks, timeout=1)
            if pending:
                self.invalid = True
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        servers = [p for p in self.peers.values() if p["role"] == "server"]
        verified = (
            not self.invalid
            and not self.started
            and self.controller is not None
            and servers
            and all(p["ready"] and p["complete"] for p in self.peers.values())
            and bool(self.exits)
            and self.expected
            and self.last_started == self.last_exit
            and self.expected == {(p["nspid"], p["start"]) for p in servers}
        )
        self.directory.cleanup()
        return "pre_execution" if verified else "unknown"
