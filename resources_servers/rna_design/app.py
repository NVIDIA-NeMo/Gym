# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RNA design tools for an outside harness such as Claude Code, served over MCP.

The same campaign as rna_design_agent, with the harness choosing the calls. Each rollout gets its own tool
worker process holding its campaign. verify re-scores the pool with the suite's own metric and gates, the
same reward as the agent.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import sys
import tempfile
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSessionFence,
    SimpleResourcesServer,
)
from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import failure_kinds
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.rna_design.verifier_fixture import create_server, invoke


LOG = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
AGENT_DIR = HERE.parent.parent / "responses_api_agents" / "rna_design_agent"
SUITES_DIR = AGENT_DIR / "suites"
# Claude Code's last message when the model endpoint failed and its own retries ran out.
API_ERROR = re.compile(r"^API Error: \d{3}\b", re.MULTILINE)


class RnaDesignResourcesServerConfig(BaseResourcesServerConfig):
    worker_python: Optional[str] = Field(
        default=None, description="Python with torch for gRNAde, from prepare.py --setup. Unset uses this server's."
    )
    work_dir: str = Field(
        default="", description="Where run directories go and are kept. Empty uses a temp dir, removed after scoring."
    )
    call_timeout_s: float = Field(default=1800.0, description="Wall-clock limit for one tool call.")
    session_ttl_s: float = Field(
        default=7200.0,
        description="A session not verified within this many seconds is closed when the next one starts, so a "
        "rollout that died before /verify does not leave its worker running.",
    )


class RnaDesignSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")
    suite: str
    target: dict[str, Any]


class RnaDesignVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    suite: str
    target: dict[str, Any]


class RnaDesignVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    suite: str
    target_name: str
    best: Optional[float] = None
    solved: bool = False
    pool_size: int = 0
    n_passing_gates: int = 0
    rounds: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    environment_faults: int = 0
    transcript: list[str] = Field(default_factory=list)


class ToolWorker:
    """One rollout's campaign process. Calls are serialized because the campaign state is not thread-safe.

    Every request carries an id and its reply echoes it. A harness may give up on a slow call while the
    worker still runs it, so the next request skips that late reply instead of taking it as its own.
    """

    def __init__(self, proc: asyncio.subprocess.Process, timeout_s: float, scratch: Optional[Path] = None) -> None:
        self.proc, self.timeout_s, self.lock, self.next_id = proc, timeout_s, asyncio.Lock(), 0
        self.scratch, self.started = scratch, time.monotonic()  # scratch is removed on close

    async def ask(self, request: dict) -> dict:
        async with self.lock:
            self.next_id += 1
            self.proc.stdin.write((json.dumps({**request, "id": self.next_id}) + "\n").encode())
            await self.proc.stdin.drain()
            while True:
                line = await asyncio.wait_for(self.proc.stdout.readline(), timeout=self.timeout_s)
                if not line:
                    raise RuntimeError("the tool worker exited")
                reply = json.loads(line)
                if reply.get("id") == self.next_id:
                    return reply

    async def close(self) -> None:
        # The worker and the gRNAde runs it started, even when the worker itself has already exited.
        with suppress(ProcessLookupError):
            os.killpg(self.proc.pid, signal.SIGKILL)
        await self.proc.wait()
        if self.scratch:
            shutil.rmtree(self.scratch, ignore_errors=True)


class RnaDesignSeedSessionResponse(BaseSeedSessionResponse):
    # Names the rollout's session, so a harness that gives up on the rollout can release it at /close_session.
    resources_session_id: str


class RnaDesignResourcesServer(SimpleResourcesServer):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: RnaDesignResourcesServerConfig
    workers: dict[str, ToolWorker] = Field(default_factory=dict)
    tool_names: dict[str, str] = Field(default_factory=dict)  # model-facing name -> registry name
    _fence: ResourcesSessionFence = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._fence = ResourcesSessionFence(self.config.session_close_retry_window_seconds)
        sys.path.insert(0, str(AGENT_DIR))
        from prepare import ensure_suites
        from rds import registry
        from setup_eternafold import ensure_eternafold

        ensure_eternafold()
        ensure_suites()
        self.tool_names = {spec.llm_name: spec.name for spec in registry.all_tools()}

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/campaign_state")(self.campaign_state)
        app.post("/{path}")(self.call_tool)
        serve = app.router.lifespan_context

        @asynccontextmanager
        async def close_workers_on_exit(app):
            # Workers run in their own process groups, so they would outlive this server.
            async with serve(app) as state:
                yield state
            for worker in self.workers.values():
                await worker.close()
            self.workers.clear()

        app.router.lifespan_context = close_workers_on_exit
        return app

    def mcp_tools(self, harvested, catchall):
        # One tool per rds tool, with the schema the agent's model sees, all dispatched through POST /{path}.
        from rds import registry

        tools = [(spec.llm_name, spec.llm_tool()["input_schema"], spec.summary) for spec in registry.all_tools()]
        return harvested + [catchall.tool(name, schema, summary) for name, schema, summary in tools]

    def mcp_allowed_tools_for_session(self, seed_body: dict[str, Any]) -> Optional[list[str]]:
        """The tools this rollout's task offers, the same the agent's model sees, plus the free campaign_state."""
        from rds.campaign import applicable_tools
        from rds.suite import Suite
        from rds.targets import from_dict, register

        name = str((seed_body.get("target") or {}).get("name", ""))
        suite = Suite.load(self._suite_path(str(seed_body.get("suite", ""))))
        campaign = next((c for c in suite.campaigns() if c.inline_target["name"] == name), None)
        if campaign is None:
            raise ValueError(f"target {name!r} is not in suite {suite.name!r}")
        register(from_dict(campaign.inline_target))
        return ["campaign_state"] + [spec.llm_name for spec in applicable_tools(campaign)]

    def _suite_path(self, suite: str) -> Path:
        known = {p.stem: p for p in SUITES_DIR.glob("*.yaml")}
        if suite not in known:
            raise ValueError(f"unknown suite {suite!r}; known suites: {sorted(known)}")
        return known[suite]

    async def seed_session(self, request: Request, body: RnaDesignSeedSessionRequest) -> RnaDesignSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        # One seed or close per session at a time, and no seed may reopen a session that was just closed.
        async with self._fence.locked(session_id):
            self._fence.raise_if_closed(session_id)
            suite_path = self._suite_path(body.suite)  # before any process starts, so a bad row leaves nothing running
            # A rollout that fails before /verify never closes its session. Close it on a re-seed, or once it is stale.
            now = time.monotonic()
            stale = [
                sid
                for sid, w in self.workers.items()
                if sid == session_id or now - w.started > self.config.session_ttl_s
            ]
            for sid in stale:
                if worker := self.workers.pop(sid, None):
                    await worker.close()
            if stale:
                LOG.warning("closed %d rna_design session(s) that never reached /verify", len(stale))
            base = Path(self.config.work_dir) if self.config.work_dir else Path(tempfile.gettempdir()) / "rna_design"
            base.mkdir(parents=True, exist_ok=True)
            run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=base))
            proc = await asyncio.create_subprocess_exec(
                self.config.worker_python or sys.executable,
                "-I",
                str(HERE / "tool_worker.py"),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
                limit=2**26,  # one reply is one JSON line, and a transcript can pass asyncio's 64 KiB default
                # A tool that outlives the call limit would answer the next call instead, so tools stop a minute short.
                env={**os.environ, "RDS_TOOL_TIMEOUT_S": str(max(self.config.call_timeout_s - 60, 60))},
            )
            worker = ToolWorker(proc, self.config.call_timeout_s, scratch=None if self.config.work_dir else run_dir)
            try:
                opened = await worker.ask(
                    {
                        "op": "open",
                        "suite_path": str(suite_path),
                        "target_name": str(body.target.get("name", "")),
                        "run_dir": str(run_dir),
                    }
                )
                if not opened["ok"]:
                    raise RuntimeError(opened["error"])
            except BaseException:
                await worker.close()  # whatever failed, the process and its run directory go with it
                raise
            self.workers[session_id] = worker
        return RnaDesignSeedSessionResponse(resources_session_id=str(session_id))

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Release a rollout's tool worker, named by its seed reply's resources_session_id, as a harness does when it
        gives up on the rollout. Closing a session that is gone or was never seeded succeeds."""
        async with self._fence.locked(body.resources_session_id):
            if worker := self.workers.get(body.resources_session_id):
                await worker.close()
                self.workers.pop(body.resources_session_id, None)  # only once closed, so a failed close can be retried
            self._fence.mark_closed(body.resources_session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    async def campaign_state(self, request: Request, body: dict) -> dict:
        """Show the campaign: the target and its constraints, how designs are ranked, the budget left, the best
        designs so far and recent failures. It costs nothing, so read it before you start and whenever you need."""
        worker = self.workers.get(request.session[SESSION_ID_KEY])
        if worker is None:
            return {"error": "no campaign for this session; seed_session was not called"}
        reply = await worker.ask({"op": "view"})
        return reply["result"] if reply["ok"] else {"error": reply["error"]}

    async def call_tool(self, request: Request, path: str) -> dict:
        """Run one rds tool for this rollout and return the campaign state after it."""
        worker = self.workers.get(request.session[SESSION_ID_KEY])
        if worker is None:
            return {"error": "no campaign for this session; seed_session was not called"}
        tool = self.tool_names.get(path, path)
        reply = await worker.ask({"op": "call", "tool": tool, "arguments": await request.json()})
        return reply["result"] if reply["ok"] else {"error": reply["error"]}

    async def verify(self, request: Request, body: RnaDesignVerifyRequest) -> RnaDesignVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        worker = self.workers.pop(session_id, None)
        base = {**body.model_dump(), "suite": body.suite, "target_name": str(body.target.get("name", ""))}
        if worker is None:
            return RnaDesignVerifyResponse(
                **base,
                reward=0.0,
                mask_sample=True,
                failure_kind=failure_kinds.AGENT_RUN_ERROR,
                failure_reason="no campaign for this session",
            )
        try:
            reply = await worker.ask({"op": "score"})
        except Exception as exc:  # a scoring call that times out or finds the worker dead says nothing about the model
            reply = {"ok": False, "error": f"scoring failed: {type(exc).__name__}: {exc}"}
        finally:
            await worker.close()
        if not reply["ok"]:
            return RnaDesignVerifyResponse(
                **base,
                reward=0.0,
                mask_sample=True,
                failure_kind=failure_kinds.AGENT_RUN_ERROR,
                failure_reason=reply["error"],
            )
        result = reply["result"]
        usage = body.response.usage
        last = next((item for item in reversed(body.response.output) if getattr(item, "type", "") == "message"), None)
        last_text = "".join(getattr(part, "text", "") for part in getattr(last, "content", None) or [])
        if result["tool_calls"] == 0 and not (usage and usage.input_tokens):
            # The model was never reached (a dead or wrong endpoint), so this is no measurement of it.
            failure = "the harness made no model call and no tool call"
        elif not result["solved"] and API_ERROR.search(last_text):
            # The endpoint failed partway, so an unsolved pool says nothing about the model. A solved one still counts.
            failure = "the model endpoint failed during the rollout"
        elif not result["solved"] and result["environment_faults"]:
            # A tool the machine could not run. Calls that failed on the model's own arguments are measured.
            failure = f"{result['environment_faults']} tool call(s) could not run on this machine"
        else:
            failure = None
        return RnaDesignVerifyResponse(
            **base,
            **{
                k: result[k]
                for k in (
                    "best",
                    "solved",
                    "pool_size",
                    "n_passing_gates",
                    "rounds",
                    "tool_calls",
                    "tool_failures",
                    "environment_faults",
                    "transcript",
                )
            },
            reward=1.0 if result["solved"] else 0.0,
            mask_sample=failure is not None,
            failure_kind=failure_kinds.AGENT_RUN_ERROR if failure else None,
            failure_reason=failure,
        )


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_server,
    request_model=RnaDesignVerifyRequest,
    cases_path=HERE / "tests" / "verifier_cases.jsonl",
    invoke=invoke,
)


if __name__ == "__main__":
    RnaDesignResourcesServer.run_webserver()
