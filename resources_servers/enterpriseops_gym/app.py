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
"""EnterpriseOps-Gym resources server.

Adapts the ServiceNow EnterpriseOps-Gym benchmark (external MCP gym servers + SQL/judge
verifiers) to NeMo Gym:

- ``/seed_session`` seeds a fresh database per rollout on each task gym server (SQL content
  cached in memory, seeding bounded by a per-gym semaphore) and pins {gym -> database_id}
  to the NeMo Gym session cookie.
- A catch-all ``POST /{tool_name}`` route proxies tool calls to the right MCP gym server
  with the session's ``x-database-id`` and task context headers. The response body is the
  MCP JSON-RPC ``result`` object, byte-compatible with what the EOG harness feeds its model.
- ``/verify`` runs the task's verifiers via the ported EOG verifier engine. The headline
  ``reward`` uses EOG's name-collapsed all-pass semantics for leaderboard parity; strict
  every-verifier metrics are also emitted for RL reward shaping. Verify is idempotent
  (results are cached per session) and always deletes the rollout's databases.
- A TTL janitor deletes databases of sessions that never reached verify (killed rollouts).
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, raise_for_status
from resources_servers.enterpriseops_gym.mcp_client import MCPGymClient, load_seed_sql
from resources_servers.enterpriseops_gym.verifier_engine import VerifierEngine


logger = logging.getLogger(__name__)


RESERVED_ROUTES = {"seed_session", "verify", "aggregate_metrics"}


class EnterpriseOpsGymResourcesServerConfig(BaseResourcesServerConfig):
    # Maps EOG gym server names (e.g. "sn-csm-server") to base URLs, overriding the
    # mcp_server_url baked into the dataset rows. Useful for pointing at replicas or
    # non-default hosts without re-converting the dataset.
    gym_url_overrides: Dict[str, str] = Field(default_factory=dict)

    # Replica pools: gym server name -> list of base URLs. Each seed_session pins its
    # rollout to one replica round-robin (all tool calls / SQL / delete stay on it).
    # This is the horizontal-scale lever for RL rollout collection — run N copies of a
    # domain's MCP container and list them here. Takes precedence over gym_url_overrides.
    gym_url_pools: Dict[str, List[str]] = Field(default_factory=dict)

    # Root directory for resolving relative seed_database_file paths from the dataset
    # (EOG task paths look like "Domain Wise DBs and Task-DB Mappings/csm/dbs/db_x.sql").
    seed_sql_root: str = "../enterpriseops-gym"

    # Database seeding executes a full SQL dump on the gym server; bound concurrent seeds
    # per gym server so large rollout batches don't stampede it.
    max_concurrent_seeds: int = 4

    # Sessions that never reach /verify (killed rollouts) get their databases deleted by
    # the janitor after this TTL. Verified sessions are dropped after the TTL as well.
    session_ttl_seconds: float = 3600.0
    janitor_interval_seconds: float = 300.0

    # False (default) = EOG leaderboard parity: reward is all-pass over NAME-COLLAPSED
    # verifier results (duplicate-named verifiers overwrite; see verifier_engine.py).
    # True = strict RL mode: reward is all-pass over every defined verifier.
    strict_verifiers: bool = False

    tool_call_timeout_seconds: float = 30.0
    sql_timeout_seconds: float = 30.0

    # Judge model for response_check verifiers. EOG uses the policy model as its own judge;
    # point this at policy_model for parity, or pin a fixed judge for RL.
    judge_model_server: Optional[ModelServerRef] = None
    # Extra responses-create params merged into judge calls (e.g. temperature, max_output_tokens).
    judge_responses_create_params: Dict[str, Any] = Field(default_factory=dict)


class SessionGym(BaseModel):
    gym_name: str
    base_url: str
    mcp_endpoint: str = "/mcp"
    database_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    auth_config: Optional[Dict[str, Any]] = None


class SessionState(BaseModel):
    gyms: Dict[str, SessionGym] = Field(default_factory=dict)
    tool_to_gym: Dict[str, str] = Field(default_factory=dict)
    created_at: float
    dbs_deleted: bool = False
    verify_result: Optional[Dict[str, Any]] = None
    # Per-tool-call latency records, in execution order: {tool, gym, latency_ms}.
    tool_latencies: List[Dict[str, Any]] = Field(default_factory=list)


class EnterpriseOpsSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Dict[str, Any]


class EnterpriseOpsSeedSessionResponse(BaseSeedSessionResponse):
    databases: Dict[str, Optional[str]] = Field(default_factory=dict)


class EnterpriseOpsVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Dict[str, Any]


class EnterpriseOpsVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    # EOG-parity scoring (name-collapsed verifier results; see verifier_engine.py).
    overall_success: bool
    verifier_pass_rate: float
    verification_results: Dict[str, Any]

    # Strict scoring over every defined verifier (RL reward shaping).
    strict_success: bool
    strict_pass_rate: float

    num_verifiers_defined: int
    num_verifiers_scored: int
    num_tool_calls: int

    # Per-tool-call latencies in execution order (this server proxies every MCP call,
    # so it can time them — something the upstream harness cannot).
    tool_latencies_ms: List[Dict[str, Any]] = Field(default_factory=list)


class EnterpriseOpsGymResourcesServer(SimpleResourcesServer):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: EnterpriseOpsGymResourcesServerConfig
    sessions: Dict[str, SessionState] = Field(default_factory=dict)

    gym_clients: Dict[str, MCPGymClient] = None
    seed_semaphores: Dict[str, asyncio.Semaphore] = None
    replica_counters: Dict[str, int] = None
    janitor_task: Optional[asyncio.Task] = None

    def model_post_init(self, __context: Any) -> None:
        self.gym_clients = {}
        self.seed_semaphores = {}
        self.replica_counters = {}

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Catch-all tool proxy. Registered after the base routes, so /seed_session,
        # /verify, and /aggregate_metrics keep precedence (Starlette matches in order).
        app.post("/{tool_name}")(self.call_tool)

        return app

    # ------------------------------------------------------------------
    # Gym client / session helpers
    # ------------------------------------------------------------------

    def _get_gym_client(self, gym: SessionGym) -> MCPGymClient:
        key = json.dumps([gym.base_url, gym.mcp_endpoint, gym.auth_config], sort_keys=True)
        client = self.gym_clients.get(key)
        if client is None:
            client = MCPGymClient(
                base_url=gym.base_url,
                auth_config=gym.auth_config,
                mcp_endpoint=gym.mcp_endpoint,
                tool_call_timeout_seconds=self.config.tool_call_timeout_seconds,
                sql_timeout_seconds=self.config.sql_timeout_seconds,
            )
            self.gym_clients[key] = client
        return client

    def _get_seed_semaphore(self, base_url: str) -> asyncio.Semaphore:
        semaphore = self.seed_semaphores.get(base_url)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.config.max_concurrent_seeds)
            self.seed_semaphores[base_url] = semaphore
        return semaphore

    def _resolve_seed_sql_path(self, seed_database_file: str) -> Optional[Path]:
        if not seed_database_file:
            return None
        path = Path(seed_database_file)
        if path.is_absolute():
            return path
        # ng_run forks servers with CWD = the server dir, but users naturally write
        # seed_sql_root relative to the gym repo root — try both.
        candidates = [
            Path(self.config.seed_sql_root) / path,
            PARENT_DIR / self.config.seed_sql_root / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _parse_session_gyms(self, verifier_metadata: Dict[str, Any]) -> Dict[str, SessionGym]:
        gym_servers_config = verifier_metadata.get("gym_servers_config") or []
        if not gym_servers_config:
            raise ValueError("verifier_metadata.gym_servers_config is missing or empty")

        gyms: Dict[str, SessionGym] = {}
        for idx, server_config in enumerate(gym_servers_config):
            for field in ("mcp_server_name", "mcp_server_url"):
                if field not in server_config:
                    raise ValueError(f"gym_servers_config[{idx}] missing required field: '{field}'")

            gym_name = server_config["mcp_server_name"]
            pool = self.config.gym_url_pools.get(gym_name)
            if pool:
                # Pin this rollout to one replica, round-robin across sessions.
                replica_index = self.replica_counters.get(gym_name, 0)
                self.replica_counters[gym_name] = replica_index + 1
                base_url = pool[replica_index % len(pool)]
            else:
                base_url = self.config.gym_url_overrides.get(gym_name, server_config["mcp_server_url"])
            gyms[gym_name] = SessionGym(
                gym_name=gym_name,
                base_url=base_url,
                mcp_endpoint=server_config.get("mcp_endpoint", "/mcp"),
                context=server_config.get("context") or {},
                auth_config=server_config.get("auth_config"),
            )
        return gyms

    async def _seed_gym_database(self, gym: SessionGym, seed_database_file: str) -> None:
        """Seed one gym's database. Mirrors EOG: a missing seed file logs an error and
        proceeds with database_id=None instead of failing the rollout."""
        seed_path = self._resolve_seed_sql_path(seed_database_file)
        if seed_path is None or not seed_path.exists():
            logger.error(f"SQL seed file not found for gym '{gym.gym_name}': {seed_path}")
            return

        sql_content = load_seed_sql(str(seed_path))
        client = self._get_gym_client(gym)
        gym.database_id = await client.seed_database(
            sql_content, seed_semaphore=self._get_seed_semaphore(gym.base_url)
        )

    async def _delete_session_dbs(self, state: SessionState) -> None:
        if state.dbs_deleted:
            return
        state.dbs_deleted = True
        for gym in state.gyms.values():
            if gym.database_id:
                try:
                    await self._get_gym_client(gym).delete_database(gym.database_id)
                except Exception as e:
                    logger.error(f"Failed to delete database {gym.database_id} for gym '{gym.gym_name}': {e}")

    # ------------------------------------------------------------------
    # Janitor
    # ------------------------------------------------------------------

    def _ensure_janitor(self) -> None:
        # janitor_interval_seconds <= 0 disables the background janitor (tests call
        # cleanup_expired_sessions directly).
        if self.config.janitor_interval_seconds <= 0:
            return
        if self.janitor_task is None or self.janitor_task.done():
            self.janitor_task = asyncio.create_task(self._janitor_loop())

    async def _janitor_loop(self) -> None:  # pragma: no cover
        while True:
            await asyncio.sleep(self.config.janitor_interval_seconds)
            try:
                await self.cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"Janitor sweep failed: {e}")

    async def cleanup_expired_sessions(self, now: Optional[float] = None) -> int:
        """Delete databases and drop state for sessions older than the TTL. Returns count."""
        now = now if now is not None else time.time()
        expired = [
            session_id
            for session_id, state in self.sessions.items()
            if now - state.created_at > self.config.session_ttl_seconds
        ]
        for session_id in expired:
            state = self.sessions.pop(session_id)
            if not state.dbs_deleted:
                logger.warning(f"Janitor deleting databases for expired session {session_id} (rollout never verified)")
            await self._delete_session_dbs(state)
        return len(expired)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    async def seed_session(
        self, request: Request, body: EnterpriseOpsSeedSessionRequest
    ) -> EnterpriseOpsSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]

        gyms = self._parse_session_gyms(body.verifier_metadata)
        gym_servers_config = body.verifier_metadata.get("gym_servers_config") or []

        # Seed all gyms' databases concurrently (bounded by per-gym-server semaphores).
        await asyncio.gather(
            *(
                self._seed_gym_database(
                    gyms[server_config["mcp_server_name"]], server_config.get("seed_database_file", "")
                )
                for server_config in gym_servers_config
            )
        )

        state = SessionState(
            gyms=gyms,
            tool_to_gym=body.verifier_metadata.get("tool_to_gym") or {},
            created_at=time.time(),
        )
        self.sessions[session_id] = state
        self._ensure_janitor()

        return EnterpriseOpsSeedSessionResponse(databases={name: gym.database_id for name, gym in gyms.items()})

    async def call_tool(self, tool_name: str, request: Request) -> Response:
        """Catch-all MCP tool proxy.

        The response body is exactly ``json.dumps(<jsonrpc result>)`` (or ``{}`` on transport
        error) — the same observation string the EOG harness feeds its model.
        """
        if tool_name in RESERVED_ROUTES:  # pragma: no cover - starlette routes these first
            return Response(content=json.dumps({}), media_type="application/json", status_code=404)

        session_id = request.session[SESSION_ID_KEY]
        state = self.sessions.get(session_id)
        if state is None:
            return Response(
                content=json.dumps({"error": "No seeded session found. Call /seed_session first."}),
                media_type="application/json",
                status_code=400,
            )

        try:
            arguments = json.loads((await request.body()) or b"{}")
        except json.JSONDecodeError as e:
            return Response(
                content=json.dumps({"error": f"Invalid JSON tool arguments: {e}"}),
                media_type="application/json",
                status_code=400,
            )

        # Route the tool to its gym: dataset-provided mapping, else the only gym.
        gym_name = state.tool_to_gym.get(tool_name)
        if gym_name is None and len(state.gyms) == 1:
            gym_name = next(iter(state.gyms))
        gym = state.gyms.get(gym_name)
        if gym is None:
            return Response(
                content=json.dumps({"error": f"Tool '{tool_name}' is not mapped to any gym server"}),
                media_type="application/json",
                status_code=400,
            )

        client = self._get_gym_client(gym)
        tool_t0 = time.monotonic()
        tool_result = await client.call_tool(tool_name, arguments, database_id=gym.database_id, context=gym.context)
        state.tool_latencies.append(
            {
                "tool": tool_name,
                "gym": gym.gym_name,
                "latency_ms": round((time.monotonic() - tool_t0) * 1000, 1),
            }
        )

        # EOG parity: the model observes json.dumps(tool_result.get("result", {})).
        return Response(content=json.dumps(tool_result.get("result", {})), media_type="application/json")

    async def _judge(self, system_prompt: str, user_prompt: str) -> str:
        """response_check judge call. EOG judges with a plain system+user chat completion."""
        judge_body = {
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        } | dict(self.config.judge_responses_create_params)

        response = await self.server_client.post(
            server_name=self.config.judge_model_server.name,
            url_path="/v1/responses",
            json=judge_body,
        )
        await raise_for_status(response)
        judge_response = NeMoGymResponse.model_validate(await get_response_json(response))

        texts = [
            content.text
            for item in judge_response.output
            if item.type == "message"
            for content in item.content
            if content.type == "output_text"
        ]
        return "".join(texts)

    @staticmethod
    def build_model_response(response: NeMoGymResponse) -> Tuple[Dict[str, Any], int]:
        """Build the EOG-shape model_response dict from a Responses API trajectory.

        EOG passes {"content": <final response>, "tool_calls": [{"name", "args"}, ...]} to its
        verifiers, where the final response is the last AI text (or, when the loop ended on a
        tool step, the last tool observation).
        """
        final_response = ""
        tool_calls: List[Dict[str, Any]] = []
        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "message" and getattr(item, "role", None) == "assistant":
                final_response = "".join(
                    content.text for content in item.content if getattr(content, "type", None) == "output_text"
                )
            elif item_type == "function_call":
                try:
                    args = json.loads(item.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = item.arguments
                tool_calls.append({"name": item.name, "args": args})
            elif item_type == "function_call_output":
                final_response = item.output

        return {"content": final_response, "tool_calls": tool_calls}, len(tool_calls)

    async def verify(self, request: Request, body: EnterpriseOpsVerifyRequest) -> EnterpriseOpsVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        state = self.sessions.get(session_id)
        if state is None:
            raise ValueError("No seeded session found for verify. Call /seed_session first.")

        # Idempotency: the harness may retry verify if the response was lost; the databases
        # are gone after the first verify, so replay the cached result instead of re-running.
        if state.verify_result is not None:
            return EnterpriseOpsVerifyResponse.model_validate(state.verify_result)

        try:
            verifiers = body.verifier_metadata.get("verifiers") or []
            model_response, num_tool_calls = self.build_model_response(body.response)

            engine = VerifierEngine(
                gym_clients={name: self._get_gym_client(gym) for name, gym in state.gyms.items()},
                judge_fn=self._judge if self.config.judge_model_server else None,
            )
            verification_results, all_verifier_results = await engine.run_verifiers(
                verifiers=verifiers,
                model_response=model_response,
                gym_databases={name: gym.database_id for name, gym in state.gyms.items()},
                gym_contexts={name: gym.context for name, gym in state.gyms.items()},
            )
        finally:
            await self._delete_session_dbs(state)

        # EOG-parity scoring over the name-collapsed results (executor.py semantics).
        total_verifiers = len(verification_results)
        passed_verifiers = sum(1 for v in verification_results.values() if v.get("passed", False))
        overall_success = all(v["passed"] for v in verification_results.values())
        verifier_pass_rate = passed_verifiers / total_verifiers if total_verifiers > 0 else 0.0

        # Strict scoring over every defined verifier (skipped verifiers count as failed).
        strict_passed = [entry["result"].get("passed", False) for entry in all_verifier_results]
        strict_success = all(strict_passed)
        strict_pass_rate = sum(strict_passed) / len(strict_passed) if strict_passed else 0.0

        reward = float(strict_success) if self.config.strict_verifiers else float(overall_success)

        verify_response = EnterpriseOpsVerifyResponse(
            **body.model_dump(),
            reward=reward,
            overall_success=overall_success,
            verifier_pass_rate=verifier_pass_rate,
            verification_results=verification_results,
            strict_success=strict_success,
            strict_pass_rate=strict_pass_rate,
            num_verifiers_defined=len(verifiers),
            num_verifiers_scored=total_verifiers,
            num_tool_calls=num_tool_calls,
            tool_latencies_ms=state.tool_latencies,
        )
        state.verify_result = verify_response.model_dump()
        return verify_response

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Per-domain breakdown mirroring the EOG leaderboard (per-domain success + macro avg)."""
        domain_rewards: Dict[str, List[float]] = {}
        domain_pass_rates: Dict[str, List[float]] = {}
        domain_strict: Dict[str, List[float]] = {}
        for task_group in tasks:
            for rollout in task_group:
                domain = (rollout.get("verifier_metadata") or {}).get("domain") or "unknown"
                domain_rewards.setdefault(domain, []).append(rollout.get("reward", 0.0))
                domain_pass_rates.setdefault(domain, []).append(rollout.get("verifier_pass_rate", 0.0))
                domain_strict.setdefault(domain, []).append(rollout.get("strict_pass_rate", 0.0))

        metrics: Dict[str, Any] = {}
        for domain, rewards in sorted(domain_rewards.items()):
            metrics[f"{domain}/success_rate"] = sum(rewards) / len(rewards)
            metrics[f"{domain}/verifier_pass_rate"] = sum(domain_pass_rates[domain]) / len(domain_pass_rates[domain])
            metrics[f"{domain}/strict_pass_rate"] = sum(domain_strict[domain]) / len(domain_strict[domain])
            metrics[f"{domain}/num_rollouts"] = len(rewards)

        per_domain_success = [metrics[f"{d}/success_rate"] for d in domain_rewards]
        if per_domain_success:
            metrics["macro_success_rate"] = sum(per_domain_success) / len(per_domain_success)
        return metrics


if __name__ == "__main__":
    EnterpriseOpsGymResourcesServer.run_webserver()
