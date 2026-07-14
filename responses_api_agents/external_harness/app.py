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

"""The external-harness agent: runs any registered CLI harness in a sandbox.

The orchestration is harness-agnostic. Per rollout it seeds the task, creates a
sandbox through nemo_gym.sandbox, asks the harness adapter to install and
prepare itself, launches it pointed at a rollout-scoped model URL, harvests the
outcome, verifies, waits for capture, and builds trajectories. Which CLI runs
is entirely behind the adapter (adapters/); nothing here names one.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import time
from collections.abc import Mapping
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.observability.capture_reader import HttpCaptureReader
from nemo_gym.observability.records import ModelCallRecord
from nemo_gym.rollout_id import is_valid_rollout_id, make_rollout_id
from nemo_gym.sandbox import SandboxResources, resolve_provider_config
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers import SandboxSpec
from nemo_gym.sandbox_client import make_remote_provider
from nemo_gym.server_utils import request
from nemo_gym.trajectory.builder import (
    assert_nemo_rl_contiguity,
    build_trajectories,
    project_main_chain_response,
)
from nemo_gym.trajectory.registry import get_builder
from nemo_gym.trajectory.sources import CaptureTokenSource
from responses_api_agents.external_harness.adapters import (
    RolloutEndpoints,
    SeedResult,
    get_adapter,
)
from responses_api_agents.external_harness.sandbox_utils import sandbox_read_text


class HarnessSettings(BaseModel):
    """Everything about *how* to run a harness, shared verbatim between the
    config-facing agent config and the resolved runtime config so neither
    restates the other. Endpoints (resolved URLs) live on the runtime config."""

    # Registered adapter name — no default, the config must name the harness.
    harness: str
    harness_spec: dict = Field(default_factory=dict)  # validated by the adapter's spec_model
    # External harnesses are multi-turn and NeMo-RL consumes one trajectory per
    # rollout, so prefix_merging (all turns merged into one contiguous main
    # chain) is the complete choice — per_request would deliver only the first
    # call. assert_nemo_rl_contiguity guards against mis-merges. Switch to
    # per_request for harnesses that rewrite/compact context between calls.
    builder: str = "prefix_merging"  # per_request | prefix_merging
    training_mode: bool = False
    # Sandbox selection, resolved through nemo_gym.sandbox.resolve_provider_config:
    # either the name of a sandbox block from a separately-composed provider
    # config, or an inline single-key {provider: {...}} mapping. Optional when a
    # sandbox_server is named instead (below).
    sandbox_provider: str | dict | None = None
    # Alternatively, the NAME of a sandbox server to create the sandbox THROUGH.
    # When set, the box is owned by that server, so the verifier can operate the
    # SAME live box (live_ref) or spin up its own eval box using the server URL
    # passed to verify. Exactly one of sandbox_provider / sandbox_server is used.
    sandbox_server: str | None = None
    # SandboxSpec fields for the run (image, resources, provider_options, ...).
    # No image default: the config supplies the image. workdir is optional
    # (the orchestrator creates a per-rollout dir under it).
    sandbox_spec: dict = Field(default_factory=dict)


class ExternalHarnessConfig(BaseModel):
    """Resolved runtime config: harness settings plus the concrete endpoint URLs
    this rollout talks to."""

    settings: HarnessSettings
    model_server_url: str
    model_api_key: str = "dummy_key"
    policy_model_name: str = ""
    capture_dir: Optional[str] = None  # local fast path when co-located
    # Resolved base URL of the sandbox server (when settings.sandbox_server is
    # set). Passed to verify so an env can spin up its own eval box.
    sandbox_server_url: Optional[str] = None
    # The verifier: seeds the task and owns /verify (exactly one). Base URL.
    verify_url: Optional[str] = None
    # Tool providers: base URLs of resources servers that ONLY lend tools over
    # MCP for this rollout (seeded with tool_only=true, no task materialized).
    # Their tools are namespaced by server name so they can't clash.
    tool_provider_urls: list[str] = Field(default_factory=list)


class RolloutResult(BaseModel):
    rollout_id: str
    responses_create_params: dict
    response: dict
    reward: float = 0.0
    is_resolved: bool = False
    ng_model_call_capture: dict = Field(default_factory=dict)
    ng_trajectories: list[dict] = Field(default_factory=list)
    harness_outcome: dict = Field(default_factory=dict)


def _task_prompt(task_row: dict) -> str:
    rcp = task_row.get("responses_create_params") or {}
    inp = rcp.get("input")
    if isinstance(inp, str):
        return inp
    if isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content")
                if isinstance(content, str):
                    return content
    return str(inp)


def _sandbox_reachable(base_url: str) -> str:
    """Under ``--network host`` the container shares the host network namespace,
    so host services are reachable at 127.0.0.1. Normalize a wildcard bind host."""
    return base_url.replace("://0.0.0.0", "://127.0.0.1")


def _mcp_servers_from_seed(seed_json: dict, resources_base_url: str) -> dict[str, dict]:
    """Turn a resources server's /seed_session MCP metadata into the sandbox-side
    server map: {server_name: {type, url, headers}} with a reachable url."""
    md = seed_json.get("mcp") or {}
    server_name = md.get("server_name")
    if not server_name:
        return {}
    url = _sandbox_reachable(resources_base_url.rstrip("/")) + md.get("url_path", "/mcp")
    return {server_name: {"type": md.get("transport", "http"), "url": url, "headers": md.get("headers", {})}}


async def _seed_rollout(cfg: "ExternalHarnessConfig", rollout_id: str, task_row: dict) -> SeedResult:
    """Ask the resources server to define this rollout's world and hand back any
    tools it lends over MCP. Falls back to the dataset row's files when no
    resources server is wired (standalone launchers)."""
    files = (task_row.get("verifier_metadata") or {}).get("files", {}) or {}
    mcp_servers: dict[str, dict] = {}
    harvest: dict = {}
    sharing = "none"
    if cfg.verify_url:
        resp = await request(
            "POST",
            cfg.verify_url.rstrip("/") + "/seed_session",
            json={
                "ng_rollout_id": rollout_id,
                "responses_create_params": task_row.get("responses_create_params", {}),
                "verifier_metadata": task_row.get("verifier_metadata", {}),
            },
        )
        resp.raise_for_status()
        seed_json = await resp.json()
        seeded = ((seed_json.get("sandbox_spec") or {}).get("files")) or {}
        if seeded:
            files = seeded
        mcp_servers = _mcp_servers_from_seed(seed_json, cfg.verify_url)
        # The verifier declares what to pull out of the sandbox to grade
        # (files to read and/or commands to run); empty for response-scored envs.
        harvest = seed_json.get("harvest") or {}
        sharing = str(seed_json.get("sandbox_sharing") or "none")

    # Tool providers lend tools only: seed each with tool_only=true (no task
    # materialized) and merge its MCP server, keyed by its own server name so
    # tools never clash across providers (mcp__<server>__<tool>).
    for provider_url in cfg.tool_provider_urls:
        resp = await request(
            "POST",
            provider_url.rstrip("/") + "/seed_session",
            json={"ng_rollout_id": rollout_id, "tool_only": True},
        )
        resp.raise_for_status()
        provider_servers = _mcp_servers_from_seed(await resp.json(), provider_url)
        for name, entry in provider_servers.items():
            if name in mcp_servers:
                raise ValueError(f"tool provider MCP name {name!r} collides with another server")
            mcp_servers[name] = entry
    return SeedResult(
        files=files,
        mcp_servers=mcp_servers,
        harvest_files=[str(f) for f in (harvest.get("files") or [])],
        harvest_commands=[str(c) for c in (harvest.get("commands") or [])],
        sandbox_sharing=sharing,
    )


def _text_from_step_response(resp: dict) -> str:
    """Extract the assistant text from a captured step response, across all
    three dialects (Responses output items, chat choices, Anthropic content)."""
    if not isinstance(resp, dict):
        return ""
    parts: list[str] = []
    for item in resp.get("output") or []:  # Responses dialect
        if isinstance(item, dict) and item.get("type") == "message":
            content = item.get("content")
            if isinstance(content, list):
                parts += [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "output_text"]
            elif isinstance(content, str):
                parts.append(content)
    for choice in resp.get("choices") or []:  # chat dialect
        msg = (choice or {}).get("message") or {}
        if isinstance(msg, dict) and msg.get("content"):
            parts.append(msg["content"])
    for block in resp.get("content") or []:  # Anthropic dialect
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(p for p in parts if p)


def _assemble_semantic_response(steps: list, rollout_id: str, model: str, summary: dict) -> "NeMoGymResponse":
    """Assemble the captured assistant turns into a Responses-format
    NeMoGymResponse — the semantic view sent to /verify (response-scored envs
    like reasoning_gym read the answer text off ``response.output``) and
    returned for eval."""
    output_items = []
    for i, step in enumerate(steps):
        text = _text_from_step_response(getattr(step, "response", None) or {})
        if text:
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg-{rollout_id}-{i}",
                    content=[NeMoGymResponseOutputText(annotations=[], text=text)],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )
    tin = int(summary.get("tokens_in", 0) or 0)
    tout = int(summary.get("tokens_out", 0) or 0)
    return NeMoGymResponse(
        id=f"resp-{rollout_id}",
        created_at=int(time.time()),
        model=model or "",
        object="response",
        output=output_items,
        tool_choice="auto",
        tools=[],
        parallel_tool_calls=False,
        usage=NeMoGymResponseUsage(
            input_tokens=tin,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
            output_tokens=tout,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=tin + tout,
        ),
    )


def resolve_rollout_id(task_row: dict, namespace: str, task_index: int, rollout_index: int, attempt: int) -> str:
    """Resolve this rollout's id. The reserved, namespaced ``ng_rollout_id`` key
    wins when the rollout-collection layer stamped one (regex-validated; a bad
    value is a 400 at the caller). Otherwise mint a UNIQUE id: some drivers
    (e.g. NeMo-RL) reuse the same row for every generation without varying the
    indices, so a purely deterministic id would collide and merge captures. The
    deterministic part stays as a readable prefix (``namespace``, e.g. the
    harness name, so captures are self-describing), with a unique suffix."""
    provided = str(task_row.get("ng_rollout_id") or task_row.get("rollout_id") or "").strip()
    if provided:
        if not is_valid_rollout_id(provided):
            raise ValueError(f"invalid ng_rollout_id {provided!r}")
        return provided
    return f"{make_rollout_id(namespace, task_index, rollout_index, attempt)}.{uuid4().hex[:8]}"


async def run_external_harness_rollout(
    cfg: ExternalHarnessConfig,
    task_row: dict,
    task_index: int,
    rollout_index: int,
    attempt: int,
    named_configs: Optional[Mapping[str, Any]] = None,
) -> RolloutResult:
    settings = cfg.settings
    rollout_id = resolve_rollout_id(task_row, settings.harness, task_index, rollout_index, attempt)
    adapter = get_adapter(settings.harness)
    spec = adapter.spec_model(name=settings.harness, **settings.harness_spec)

    # 1. seed: the resources server defines the world (files) and hands back any
    #    tools it lends over MCP (url + signed per-rollout session token).
    task_prompt = _task_prompt(task_row)
    # SandboxSpec fields from config; workdir/files are set per-rollout here so
    # they never come from the shared config. Per-rollout workdir keeps parallel
    # rollouts apart (container providers isolate anyway, but a host-sharing
    # provider does not).
    spec_config = dict(settings.sandbox_spec)
    base_workdir = spec_config.pop("workdir", None) or "/tmp/nemo-gym-harness"
    spec_config.pop("files", None)
    workdir = f"{base_workdir.rstrip('/')}/{rollout_id}"
    seed = await _seed_rollout(cfg, rollout_id, task_row)
    files = seed.files

    # 2. sandbox: pick where the box lives. With a sandbox_server the box is
    #    owned by that server (so the verifier can operate the same live box or
    #    spin up its own eval box); otherwise it's an in-process provider (name
    #    ref or inline mapping). Seed task files as absolute paths under the
    #    workdir so the harness's cwd and its files line up.
    seeded = {f"{workdir.rstrip('/')}/{rel}": content for rel, content in files.items()}
    if settings.sandbox_server:
        if not cfg.sandbox_server_url:
            raise ValueError("settings.sandbox_server is set but no sandbox server URL was resolved")
        provider: Any = make_remote_provider(cfg.sandbox_server_url)
    else:
        if not settings.sandbox_provider:
            raise ValueError("one of settings.sandbox_provider or settings.sandbox_server is required")
        provider = resolve_provider_config(settings.sandbox_provider, named_configs)
    # Bind the box to this rollout: the sandbox server refuses to mint a lease
    # whose rollout id differs, so a leaked ref can't touch another rollout's
    # box. In-process providers ignore metadata.
    metadata = dict(spec_config.pop("metadata", {}))
    metadata["ng_rollout_id"] = rollout_id
    sandbox_spec = SandboxSpec(
        image=spec_config.pop("image", None),
        workdir=workdir,
        files=seeded,
        ttl_s=spec_config.pop("ttl_s", None),
        ready_timeout_s=spec_config.pop("ready_timeout_s", None),
        env=spec_config.pop("env", {}),
        metadata=metadata,
        resources=SandboxResources.from_mapping(spec_config.pop("resources", {})),
        provider_options=spec_config.pop("provider_options", {}),
    )
    sandbox = await AsyncSandbox(provider, sandbox_spec).start()

    # The box stays up through /verify: a live_ref env inspects it there. Capture
    # read and verify live inside the try so the box is alive for both; the
    # trajectory build (step 9) needs only the capture and runs after stop().
    reader = HttpCaptureReader(
        cfg.model_server_url, api_key=cfg.model_api_key, local_dir=cfg.capture_dir, local_read="auto"
    )
    steps: list = []
    summary: dict = {}
    response: dict = {}
    resp_by_reqid: dict = {}
    reward, is_resolved, info = 0.0, False, {}
    sandbox_ref: Optional[dict] = None
    try:
        # Ensure the workdir exists. The docker provider creates it via `run -w`,
        # but others (e.g. opensandbox) do not, and an env with no seeded files
        # never triggers the upload path that would mkdir it. Run from "/" since
        # the default cwd is the (not-yet-existing) workdir.
        await sandbox.exec(f"mkdir -p {shlex.quote(workdir)}", cwd="/", timeout_s=60)
        endpoints = RolloutEndpoints(
            model_base_url=f"{cfg.model_server_url.rstrip('/')}/ng-rollout/{rollout_id}",
            api_key=cfg.model_api_key,
            model_name=cfg.policy_model_name,
        )

        # 3. install the harness into the fresh sandbox.
        runtime = adapter.runtime(spec)
        for setup_cmd in runtime.setup_cmds:
            result = await sandbox.exec(shlex.join(setup_cmd), timeout_s=spec.setup_timeout_s)
            if result.return_code != 0:
                raise RuntimeError(
                    f"harness setup command {setup_cmd!r} failed (rc={result.return_code}): {(result.stderr or '')[-500:]}"
                )

        # 4. prepare (write the harness's config files) and launch it.
        await adapter.prepare(sandbox, spec, seed, endpoints, workdir)
        cmd, env = adapter.launch(spec, seed, endpoints, task_prompt, workdir)
        run = await sandbox.exec(shlex.join(cmd), env=env, timeout_s=spec.timeout_s)
        exit_code, out, err = run.return_code or 0, run.stdout or "", run.stderr or ""

        # 5. outcome harvest. Prefer a standard ng_outcome.json the harness/env
        #    writes; else base the outcome on harness stdout/stderr. On top of
        #    that, harvest any files/commands the env asked for — file-artifact
        #    envs (e.g. "write the answer to answer.txt") and patch envs (git
        #    diff) grade from these. response-scored envs declare none. The
        #    extraction spec lives with the env, not the agent.
        raw = await sandbox_read_text(sandbox, f"{workdir.rstrip('/')}/ng_outcome.json")
        outcome = json.loads(raw) if raw else {"harness_stdout": out[-1000:], "error": err[-1000:]}
        harvested_files: dict[str, str] = {}
        for rel in seed.harvest_files:
            content = await sandbox_read_text(sandbox, f"{workdir.rstrip('/')}/{rel}")
            if content is not None:
                harvested_files[rel] = content
        command_outputs: dict[str, dict] = {}
        for cmd in seed.harvest_commands:
            res = await sandbox.exec(cmd, cwd=workdir, timeout_s=300)
            command_outputs[cmd] = {
                "stdout": res.stdout or "",
                "stderr": res.stderr or "",
                "returncode": res.return_code,
            }
        if harvested_files:
            outcome.setdefault("harvested_files", harvested_files)
        if command_outputs:
            outcome.setdefault("command_outputs", command_outputs)
        if harvested_files:
            outcome.setdefault("outcome_text", next(iter(harvested_files.values())))
        elif command_outputs:
            outcome.setdefault("outcome_text", next(iter(command_outputs.values()))["stdout"])
        outcome["harness_exit_code"] = exit_code

        # If the env grades from the LIVE box, mint an operate co-lease for the
        # verifier and keep the box up (the finally below stops it only after
        # verify returns). The verifier attaches to the SAME physical box.
        if seed.sandbox_sharing == "live_ref" and settings.sandbox_server:
            sandbox_ref = (await sandbox.grant(scope="operate")).to_dict()

        # 6. wait for capture, then read it back. Capture is the source of truth
        #    for what the harness's model calls produced.
        await asyncio.sleep(0.25)
        await reader.prefetch(rollout_id)
        steps = [r for r in reader.records(rollout_id, kinds={"model_call"}) if isinstance(r, ModelCallRecord)]
        summary = reader.summary(rollout_id)

        # 7. assemble the semantic response from capture (final-answer text
        #    across turns, all dialects). This is what response-scored envs grade.
        semantic_response = _assemble_semantic_response(steps, rollout_id, cfg.policy_model_name, summary)
        response = semantic_response.model_dump(mode="json")
        resp_by_reqid = {s.request_id: s.response for s in steps}

        # 8. verify (resources server owns reward authority). Send the full task
        #    row plus the assembled response, the harvested artifact, and — when
        #    server-backed — the sandbox server URL (so the env can create its
        #    own eval box) and a live sandbox_ref (so it can operate the agent's
        #    box). The env uses whichever fits its grading shape.
        if cfg.verify_url:
            verify_payload = dict(task_row)
            verify_payload.update(
                {
                    "rollout_id": rollout_id,
                    "ng_rollout_id": rollout_id,
                    "responses_create_params": task_row.get("responses_create_params", {}),
                    "response": response,
                    "blackbox_outcome": outcome,
                    "sandbox_server_url": cfg.sandbox_server_url,
                    "sandbox_ref": sandbox_ref,
                }
            )
            verify_resp = await request("POST", cfg.verify_url.rstrip("/") + "/verify", json=verify_payload)
            verify_resp.raise_for_status()
            verify_json = await verify_resp.json()
            reward = float(verify_json.get("reward", 0.0))
            is_resolved = bool(verify_json.get("is_resolved", reward >= 1.0))
            info = verify_json.get("info", {}) or {}
    finally:
        await sandbox.stop()

    # 9. in training mode, project the main chain (token ids) as the response.
    trajectories = []
    if settings.training_mode:
        source = CaptureTokenSource(reader)
        trajectories = build_trajectories(
            rollout_id,
            source,
            model_call_records=steps,
            builder=settings.builder,
            reward=reward,
            is_resolved=is_resolved,
            policy_model=cfg.policy_model_name,
            aux_fingerprints=tuple(spec.aux_fingerprints) or None,
        )
        if trajectories:
            out_builder = get_builder(settings.builder)(source.entries(rollout_id))
            projection = project_main_chain_response(
                rollout_id, out_builder.chains, resp_by_reqid, model=cfg.policy_model_name
            )
            assert_nemo_rl_contiguity(projection)
            response = projection

    return RolloutResult(
        rollout_id=rollout_id,
        responses_create_params=task_row.get("responses_create_params", {}),
        response=response,
        reward=reward,
        is_resolved=is_resolved,
        ng_model_call_capture=summary,
        ng_trajectories=[t.model_dump() for t in trajectories],
        harness_outcome={**outcome, **info},
    )


# ---------------------------------------------------------------------------
# Gym-native agent server — runs under `gym env start` / `gym eval run`.
# ---------------------------------------------------------------------------
from fastapi import HTTPException  # noqa: E402

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse  # noqa: E402
from nemo_gym.base_responses_api_agent import (  # noqa: E402
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef  # noqa: E402
from nemo_gym.openai_utils import (  # noqa: E402
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import ServerClient, get_server_url, is_nemo_gym_fastapi_entrypoint  # noqa: E402


class ExternalHarnessAgentConfig(BaseResponsesAPIAgentConfig):
    # The verifier: seeds the task and owns /verify (verification authority).
    resources_server: ResourcesServerRef
    # Zero or more resources servers that ONLY lend tools over MCP for the
    # rollout (seeded tool_only; no task, never a verifier). Their tools are
    # namespaced by server name so, e.g., web_search.search and
    # retail_env.lookup can't clash.
    tool_providers: list[ResourcesServerRef] = Field(default_factory=list)
    model_server: ModelServerRef
    policy_model_name: str = ""
    # How to run the harness (harness name, sandbox, builder, ...). Shared with
    # the runtime config so the two never restate each other.
    settings: HarnessSettings


class ExternalHarnessRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ExternalHarnessVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    response: dict = {}


class ExternalHarnessAgent(SimpleResponsesAPIAgent):
    config: ExternalHarnessAgentConfig

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError("external_harness serves /run, not /v1/responses")

    async def run(self, body: ExternalHarnessRunRequest = Body()) -> ExternalHarnessVerifyResponse:
        row = body.model_dump()
        # The merged global config resolves a named sandbox_provider to its block.
        global_config_dict = ServerClient.load_from_global_config().global_config_dict
        sandbox_server_name = self.config.settings.sandbox_server
        cfg = ExternalHarnessConfig(
            settings=self.config.settings,
            model_server_url=get_server_url(self.config.model_server.name),
            verify_url=get_server_url(self.config.resources_server.name),
            tool_provider_urls=[get_server_url(p.name) for p in self.config.tool_providers],
            policy_model_name=self.config.policy_model_name,
            capture_dir=None,
            sandbox_server_url=get_server_url(sandbox_server_name) if sandbox_server_name else None,
        )
        try:
            result = await run_external_harness_rollout(
                cfg,
                row,
                task_index=row.get("task_index", 0),
                rollout_index=row.get("rollout_index", 0),
                attempt=row.get("attempt", 0),
                named_configs=global_config_dict,
            )
        except ValueError as e:
            # A malformed reserved ng_rollout_id is a client error, not a 500.
            raise HTTPException(status_code=400, detail=str(e)) from e
        return ExternalHarnessVerifyResponse(**result.model_dump())


if __name__ == "__main__":
    ExternalHarnessAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ExternalHarnessAgent.run_webserver()  # noqa: F401
