# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Proof fixture, not part of the benchmark: cut off the /run reply of chosen rollouts mid-body.

Use as ``+rollout_collection_driver=benchmarks.mmlu_prox.body_cutoff_proof:collect`` with
``+body_cutoff_plan=<plan.json>``. The driver puts a loopback proxy in front of each environment server for this
collection process only (servers and other clients are untouched), then runs the normal collection. The proxy forwards
every /run unchanged to the real environment server and reads its whole reply. For a planned rollout it then sends the
reply's status and headers, with the full Content-Length, and only ``cut_after_bytes`` bytes of the body before it
aborts the connection, as a connection reset after a 200 does. The rollout is named by the request body
(``episode_id.rollout_id``, or ``_ng_task_index``-``_ng_rollout_index``).

Plan::

    {"cut": {"30443-0": 1}, "cut_after_bytes": 4096, "max_forwarded": 40, "ledger": "<ledger.jsonl>"}

``cut`` gives how many replies of each rollout are cut, counted across resumed runs from the ledger, so a resume can
receive the reply. The ledger has one line per request (rollout id, action, status, bytes), never headers or bodies.
Requests past ``max_forwarded`` are not forwarded and get a 503 marked ``X-NG-Fault-Injected: over_budget``.
"""

import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import aiohttp

import nemo_gym.rollout_collection as rollout_collection


INJECTED_HEADER = "X-NG-Fault-Injected"
_KEPT_REPLY_HEADERS = ("content-type", "x-ng-failure-report")
_HOP_BY_HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive"}


def rollout_id(body: bytes) -> str | None:
    """The rollout a /run request body names, or ``None``."""
    try:
        request = json.loads(body)
    except ValueError:
        return None
    if not isinstance(request, dict):
        return None
    episode = request.get("episode_id")
    if isinstance(episode, dict) and isinstance(episode.get("rollout_id"), str):
        return episode["rollout_id"]
    task, rollout = request.get("_ng_task_index"), request.get("_ng_rollout_index")
    return f"{task}-{rollout}" if task is not None and rollout is not None else None


class BodyCutoffProxy:
    def __init__(self, *, upstream: str, plan: Mapping[str, Any], ledger: Path):
        self.upstream = upstream.rstrip("/")
        self.cut = {str(k): int(v) for k, v in plan["cut"].items()}
        self.cut_after_bytes = int(plan["cut_after_bytes"])
        self.max_forwarded = int(plan["max_forwarded"])
        self.ledger_path = ledger
        prior = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        self.cuts_done = Counter(entry["rollout_id"] for entry in prior if entry["action"] == "cut")
        self.forwarded = sum(entry["action"] in ("cut", "forwarded") for entry in prior)
        self.ledger = ledger.open("a", buffering=1)
        self.session: aiohttp.ClientSession | None = None
        self.server: asyncio.base_events.Server | None = None

    def _record(self, **fields: Any) -> None:
        self.ledger.write(json.dumps({"utc": time.time(), "upstream": self.upstream, **fields}, sort_keys=True) + "\n")

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            request_line, *header_lines = head.decode("latin-1").split("\r\n")
            method, path, _ = request_line.split(" ", 2)
            headers = dict(line.split(":", 1) for line in header_lines if ":" in line)
            headers = {k.strip(): v.strip() for k, v in headers.items()}
            length = int(next((v for k, v in headers.items() if k.lower() == "content-length"), "0"))
            body = await reader.readexactly(length)
            rollout = rollout_id(body)
            if self.forwarded >= self.max_forwarded:
                self._record(rollout_id=rollout, path=path, action="over_budget", status=503)
                data = b'{"error": "over_budget"}'
                writer.write(
                    f"HTTP/1.1 503 Service Unavailable\r\n{INJECTED_HEADER}: over_budget\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode()
                    + data
                )
                await writer.drain()
                return
            self.forwarded += 1
            assert self.session is not None
            forward_headers = {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}
            async with self.session.request(
                method, f"{self.upstream}{path}", headers=forward_headers, data=body
            ) as reply:
                reply_body = await reply.read()
                status, reason = reply.status, reply.reason or ""
                kept = {k: v for k, v in reply.headers.items() if k.lower() in _KEPT_REPLY_HEADERS}
            cut = rollout is not None and self.cuts_done[rollout] < self.cut.get(rollout, 0)
            header_block = "".join(f"{k}: {v}\r\n" for k, v in kept.items())
            writer.write(
                f"HTTP/1.1 {status} {reason}\r\n{header_block}Content-Length: {len(reply_body)}\r\n"
                f"{'' if cut else 'Connection: close' + chr(13) + chr(10)}\r\n".encode()
            )
            if cut:
                self.cuts_done[rollout] += 1
                writer.write(reply_body[: self.cut_after_bytes])
                await writer.drain()
                self._record(
                    rollout_id=rollout,
                    path=path,
                    action="cut",
                    status=status,
                    bytes_sent=min(self.cut_after_bytes, len(reply_body)),
                    bytes_total=len(reply_body),
                )
                writer.transport.abort()
                return
            writer.write(reply_body)
            await writer.drain()
            self._record(rollout_id=rollout, path=path, action="forwarded", status=status, bytes_total=len(reply_body))
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            if not writer.transport.is_closing():
                writer.close()

    async def start(self) -> str:
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()
        if self.session is not None:
            await self.session.close()
        self.ledger.close()


async def collect(rollout_collection_config: Any, global_config_dict: Mapping[str, Any]) -> None:
    """Run the normal collection with every environment server reached through a cut-off proxy."""
    plan = json.loads(Path(global_config_dict["body_cutoff_plan"]).read_text())
    client = rollout_collection.setup_server_client_utils()
    proxies = []
    for name, entry in global_config_dict.items():
        if isinstance(entry, Mapping) and "environment_servers" in entry:
            proxy = BodyCutoffProxy(upstream=client._resolve_base_url(name), plan=plan, ledger=Path(plan["ledger"]))
            client._server_base_urls[name] = await proxy.start()
            proxies.append(proxy)
    original = rollout_collection.setup_server_client_utils
    rollout_collection.setup_server_client_utils = lambda *args, **kwargs: client
    try:
        await rollout_collection.RolloutCollectionHelper().run_from_config(rollout_collection_config)
    finally:
        rollout_collection.setup_server_client_utils = original
        for proxy in proxies:
            await proxy.stop()
