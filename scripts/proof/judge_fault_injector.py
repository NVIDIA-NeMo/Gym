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
"""Loopback proxy that injects judge provider failures for chosen rollouts, for a judge-only recovery proof.

A judge model server points its ``openai_base_url`` at this proxy. Each request names its rollout in
``X-Request-Id`` (``<rollout id>-<8 hex>``, set by the model client inside a rollout context). The plan decides,
per rollout id, how many requests get an injected HTTP status before the proxy forwards them unchanged to the
real provider; ``exhaust`` rollouts are never forwarded. Every injected answer carries ``X-NG-Fault-Injected``
and an ``injected_fault`` error body, so it cannot be mistaken for a provider answer, and the ledger records one
line per request (rollout id, action, status), never headers or bodies. Requests past ``--max-forwarded`` are not
forwarded and get an injected answer marked ``over_budget``.

``actions`` gives a rollout one action per request, in order: ``forward``, ``status:<code>[:<error code>]
[:retry_after=<s>]`` (an injected error whose body names the error code, for example
``status:422:budget_exceeded``), or ``delay:<s>`` (hold the request, then forward it if the caller still waits). A
request past its rollout's list, or past ``--max-requests`` in all, gets a non-retryable 400 marked ``over_plan`` or
``over_budget``. With actions, the ledger has a ``start`` and an ``end`` line per request (sequence, time, rollout,
request id, action, outcome), so a request the caller gave up on, or that shutdown ended, is still counted.
SIGTERM or SIGINT ends held requests and stops the proxy.

Plan file::

    {"status": 500, "transient": {"0-1": 1}, "exhaust": ["0-3"],
     "actions": {"0-5": ["status:429::retry_after=1", "forward"], "0-1": ["delay:120"]}}

Usage::

    python scripts/proof/judge_fault_injector.py --port 18555 --upstream https://provider.example \\
        --plan plan.json --ledger ledger.jsonl --max-forwarded 40
"""

import argparse
import asyncio
import hashlib
import json
import re
import signal
import time
from collections import Counter
from pathlib import Path

import aiohttp
from aiohttp import web


INJECTED_HEADER = "X-NG-Fault-Injected"
_REQUEST_SUFFIX = re.compile(r"-[0-9a-f]{8}$")
_HOP_BY_HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "accept-encoding"}


def rollout_id(request_id: str | None) -> str | None:
    """The rollout id a model client's ``X-Request-Id`` names; ``None`` outside a rollout."""
    if not request_id or not _REQUEST_SUFFIX.search(request_id):
        return None
    return _REQUEST_SUFFIX.sub("", request_id)


class Injector:
    def __init__(
        self,
        *,
        upstream: str,
        plan: dict,
        ledger: Path,
        max_forwarded: int,
        upstream_timeout: float,
        max_requests: int | None = None,
    ):
        self.upstream = upstream.rstrip("/")
        self.status = int(plan.get("status", 500))
        self.transient = {str(k): int(v) for k, v in (plan.get("transient") or {}).items()}
        self.exhaust = {str(k) for k in plan.get("exhaust") or ()}
        self.actions = {str(k): [str(a) for a in v] for k, v in (plan.get("actions") or {}).items()}
        self.max_requests = max_requests
        self.requests = 0
        self.plan_sha256 = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        self.ledger = ledger.open("a", buffering=1)
        self.max_forwarded = max_forwarded
        self.upstream_timeout = upstream_timeout
        self.seen: Counter = Counter()
        self.forwarded = 0
        self.seq = 0
        self.session: aiohttp.ClientSession | None = None

    def _record(self, **fields) -> None:
        self.seq += 1
        self.ledger.write(json.dumps({"seq": self.seq, "utc": time.time(), **fields}, sort_keys=True) + "\n")

    def _injected(self, reason: str) -> web.Response:
        body = {
            "error": {
                "message": f"injected by judge_fault_injector ({reason})",
                "type": "injected_fault",
                "code": "injected_fault",
            }
        }
        return web.json_response(body, status=self.status, headers={INJECTED_HEADER: reason})

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"plan_sha256": self.plan_sha256, "forwarded": self.forwarded, "requests": self.seq})

    def _planned_error(self, action: str) -> web.Response:
        """An injected error named by ``status:<code>[:<error code>][:retry_after=<s>]``."""
        _, code, *rest = action.split(":")
        error_code = rest[0] if rest and rest[0] else "injected_fault"
        headers = {INJECTED_HEADER: action}
        for option in rest[1:]:
            if option.startswith("retry_after="):
                headers["Retry-After"] = option.split("=", 1)[1]
        body = {
            "error": {
                "message": f"injected by judge_fault_injector ({action})",
                "type": error_code,
                "code": error_code,
            }
        }
        return web.json_response(body, status=int(code), headers=headers)

    async def _act(self, request: web.Request, action: str, base: dict) -> web.StreamResponse:
        """Answer one request of a rollout that has an action list; the caller writes the ledger lines."""
        if action.startswith("status:"):
            base["outcome"], base["status"] = "injected", int(action.split(":")[1])
            return self._planned_error(action)
        if action.startswith("delay:"):
            await asyncio.sleep(float(action.split(":", 1)[1]))
        return await self._forward(request, base)

    async def _forward(self, request: web.Request, base: dict) -> web.StreamResponse:
        if self.forwarded >= self.max_forwarded:
            base["outcome"], base["status"] = "over_budget", 400
            return web.json_response({"error": "over_budget"}, status=400, headers={INJECTED_HEADER: "over_budget"})
        self.forwarded += 1
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
        assert self.session is not None
        async with self.session.request(
            request.method, f"{self.upstream}{request.path_qs}", headers=headers, data=await request.read()
        ) as upstream:
            body = await upstream.read()
            base["outcome"], base["status"] = "forwarded", upstream.status
            content_type = upstream.headers.get("Content-Type", "application/json")
            return web.Response(status=upstream.status, body=body, headers={"Content-Type": content_type})

    async def handle(self, request: web.Request) -> web.StreamResponse:
        request_id = request.headers.get("X-Request-Id")
        rollout = rollout_id(request_id)
        self.seen[rollout] += 1
        nth = self.seen[rollout]
        base = {
            "rollout_id": rollout,
            "request_id": request_id,
            "method": request.method,
            "path": request.path,
            "nth": nth,
        }
        if rollout in self.actions or self.max_requests is not None:
            return await self._handle_planned(request, rollout, nth, base)
        if rollout in self.exhaust or (rollout in self.transient and nth <= self.transient[rollout]):
            self._record(action="injected", status=self.status, **base)
            return self._injected("plan")
        if self.forwarded >= self.max_forwarded:
            self._record(action="over_budget", status=self.status, **base)
            return self._injected("over_budget")
        self.forwarded += 1
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
        started = time.monotonic()
        assert self.session is not None
        async with self.session.request(
            request.method, f"{self.upstream}{request.path_qs}", headers=headers, data=await request.read()
        ) as upstream:
            body = await upstream.read()
            self._record(
                action="forwarded", status=upstream.status, upstream_s=round(time.monotonic() - started, 3), **base
            )
            content_type = upstream.headers.get("Content-Type", "application/json")
            return web.Response(status=upstream.status, body=body, headers={"Content-Type": content_type})

    async def _handle_planned(
        self, request: web.Request, rollout: str | None, nth: int, base: dict
    ) -> web.StreamResponse:
        """Answer with a start and an end ledger line, also when the caller leaves or the proxy stops."""
        self.requests += 1
        plan = self.actions.get(rollout)
        if self.max_requests is not None and self.requests > self.max_requests:
            action = "over_budget"
        elif plan is None:
            action = "forward"
        else:
            action = plan[nth - 1] if nth <= len(plan) else "over_plan"
        started = time.monotonic()
        self._record(event="start", action=action, **base)
        try:
            if action in ("over_budget", "over_plan"):
                base["outcome"], base["status"] = action, 400
                return web.json_response({"error": action}, status=400, headers={INJECTED_HEADER: action})
            return await self._act(request, action, base)
        except asyncio.CancelledError:
            # The caller closed the connection (its timeout) or the proxy is stopping.
            base["outcome"] = "caller_closed" if not self.stopping else "stopped"
            raise
        finally:
            self._record(event="end", action=action, elapsed_s=round(time.monotonic() - started, 3), **base)

    async def start(self, port: int) -> web.AppRunner:
        self.stopping = False
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.upstream_timeout))
        app = web.Application(client_max_size=1024**3)
        app.router.add_get("/_injector/health", self.health)
        app.router.add_route("*", "/{tail:.*}", self.handle)
        # A short shutdown timeout closes held connections at once, so a waiting caller sees the proxy go away.
        runner = web.AppRunner(app, access_log=None, handler_cancellation=True, shutdown_timeout=1.0)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        return runner


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--upstream", required=True, help="provider origin; the request path (for example /v1/responses) is appended"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--max-forwarded", type=int, required=True)
    parser.add_argument("--upstream-timeout", type=float, default=1800.0)
    parser.add_argument("--max-requests", type=int, default=None, help="all requests the proxy answers, any action")
    args = parser.parse_args()
    injector = Injector(
        upstream=args.upstream,
        plan=json.loads(args.plan.read_text()),
        ledger=args.ledger,
        max_forwarded=args.max_forwarded,
        upstream_timeout=args.upstream_timeout,
        max_requests=args.max_requests,
    )
    runner = await injector.start(args.port)
    print(json.dumps({"listening": f"http://127.0.0.1:{args.port}", "plan_sha256": injector.plan_sha256}), flush=True)
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    await stop.wait()
    injector.stopping = True
    await runner.cleanup()
    assert injector.session is not None
    await injector.session.close()


if __name__ == "__main__":
    asyncio.run(main())
