#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDPVal RL rollouts with the Stirrup (AA v2) agent on the resources server's sandbox.

Sibling of ``gdpval_sandboxed_agent`` (OpenHands). Same ``/run`` protocol with
``resources_servers/gdpval_disaggregated``: seed the session, attach to its
OpenSandbox, run the agent, call ``/verify``, always ``/cleanup_session``.
What differs is the harness: the Stirrup agent loop runs *in this process*
(it only needs HTTP to the policy and the sandbox), with the single
``code_exec`` tool, the ``finish`` / ``abandon_task_finish`` tools and the AA v2
GDPVal prompt that the GDPVal evaluation harness uses. Deliverables are the
files the model names in ``finish``; they are copied into
``/workspace/deliverables`` so the resources server collects and judges exactly
what the model submitted.

Policy calls go to the rollout-specific Gym model URL, so training token
capture stays external (the model server's ledger); the exported response
reuses the served id of the last policy call for terminal attribution.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from shlex import quote
from typing import Any, Optional

from fastapi import HTTPException, Request
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from openai.types.responses import FunctionTool
from nemo_gym.sandbox import AsyncSandbox, create_provider
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.server_utils import get_server_url, is_nemo_gym_fastapi_entrypoint, raise_for_status
from pydantic import ConfigDict, Field, PrivateAttr

logger = logging.getLogger(__name__)

try:  # the OpenSandbox SDK is only present in the agent's venv
    from opensandbox.exceptions.sandbox import SandboxApiException
except Exception:  # pragma: no cover

    class SandboxApiException(Exception):  # type: ignore[no-redef]
        """Placeholder so isinstance checks stay valid without the SDK."""


AGENT_NAME = "gdpval_stirrup_agent"
PROMPT_TEMPLATE_PATH = Path(__file__).with_name("prompts") / "gdpval_user_prompt.txt"
_PROMPT_TEMPLATE: Optional[str] = None


# ---------------------------------------------------------------------------
# Config / request types
# ---------------------------------------------------------------------------


class GDPValStirrupAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    resources_server: ResourcesServerRef
    # Each session stays on one named, single-worker resources server.
    resources_servers: list[ResourcesServerRef] = Field(default_factory=list)
    sandbox_provider: str = "sandbox"
    # Total capacity across all Uvicorn workers; must divide evenly by num_workers.
    concurrency: int = Field(default=32, ge=1)
    # Stirrup turn budget (one policy call per turn). The eval harness runs 250.
    agent_max_turns: int = Field(default=200, ge=1)
    # Wall-clock budget for the whole rollout (seconds); masked as agent_exec_timeout.
    agent_timeout: int = Field(default=4800, ge=1)
    # Per policy call: read timeout and stateless re-issues on timeout/connection loss.
    model_read_timeout_s: float = Field(default=600.0, gt=0)
    model_read_retries: int = Field(default=1, ge=0)
    # Policy context window and the per-call completion cap / floor / safety buffer.
    max_context: int = Field(default=196608, ge=1024)
    max_output_tokens: int = Field(default=32768, ge=256)
    min_completion_tokens: int = Field(default=1024, ge=1)
    completion_token_buffer: int = Field(default=1000, ge=0)
    # Sampling. The training model server applies its own sampling_overrides; these are
    # what the agent requests (and what a plain vLLM endpoint honours in a smoke test).
    temperature: float = 1.0
    top_p: float = 1.0
    enable_thinking: bool = True
    # "metadata": Gym's policy model server (training/profiling) merges metadata.chat_template_kwargs;
    # "extra_body": a raw vLLM endpoint (smoke tests against vllm serve).
    policy_chat_template_kwargs_transport: str = "metadata"
    # Sandbox command budget (AA v2: 5 minutes) and layout fixed by the resources server.
    shell_timeout: int = Field(default=300, ge=1)
    working_dir: str = "/workspace"
    reference_dir: str = "/workspace/reference_files"
    deliverables_dir: str = "/workspace/deliverables"
    trace_dir: str = "/trace"
    # Stirrup starts warning the model this many turns before agent_max_turns.
    turns_remaining_warning_threshold: int = Field(default=20, ge=0)
    # Context compaction (eval parity: Stirrup summarizes at 0.7 of the window). None keeps
    # one contiguous conversation. When set, the run result also carries one response per
    # context segment (``responses``) so NeMo RL can train every segment as its own trace
    # (async_rl.multi_trace_enabled + env.nemo_gym.training_mode=compaction, token echo).
    context_summarization_cutoff: Optional[float] = Field(default=None, gt=0, le=1)
    # Token echo with Gym-side exact-prefix supply: Gym's file lineage store keeps every call's
    # tokens per rollout and nothing retires them in training. When set, the agent deletes its
    # own rollout's records (<token_id_capture.dir>/<rollout_id>.*) once /run has its result.
    capture_records_cleanup: bool = False
    # Web tools (eval parity: web_search + fetch_web_page). Off by default. Keys are Exa API
    # keys, read from ``web_search_api_keys_file`` (one per line) and/or the env var named by
    # ``web_search_api_keys_env`` (comma-separated); they are never logged or echoed.
    web_tools_enabled: bool = False
    web_search_api_keys_file: Optional[str] = None
    web_search_api_keys_env: str = "EXA_API_KEYS"
    web_search_max_sweeps: int = Field(default=1, ge=1)
    policy_model_name: str = "gdpval-policy"
    checkpoint: str


class GDPValStirrupRunRequest(BaseRunRequest):
    """Run request that keeps the row's top-level task fields (prompt, sector, ...)."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class GDPValRunResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Helpers (pure; unit-tested)
# ---------------------------------------------------------------------------


def _template() -> str:
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        _PROMPT_TEMPLATE = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return _PROMPT_TEMPLATE


def build_task_text(prompt: str, sector: str = "", occupation: str = "") -> str:
    """Same task wrapper the Stirrup eval agent uses for GDPVal rows."""
    return f"Sector: {sector or ''}\nOccupation: {occupation or ''}\n\n{prompt}"


def build_user_prompt(task_text: str, reference_files: list[str], *, working_dir: str) -> str:
    """Render the AA v2 GDPVal user prompt.

    ``reference_files`` are paths relative to ``working_dir`` (the resources
    server stages them under ``<working_dir>/reference_files/<name>``).
    """
    files_section = "".join(f"- {rel}\n" for rel in sorted(reference_files)) or "None"
    return _template().format(task=task_text, reference_files=files_section, working_dir=working_dir.rstrip("/"))


def task_fields_from_request(body: GDPValStirrupRunRequest, seed: dict[str, Any] | None) -> dict[str, Any]:
    """Prompt / sector / occupation / reference names: seed response first, then the row."""
    metadata = dict(getattr(body.responses_create_params, "metadata", None) or {})
    extra = dict(getattr(body, "model_extra", None) or {})
    seed_task = dict((seed or {}).get("task") or {})

    def pick(key: str, default: Any = None) -> Any:
        for source in (seed_task, extra, metadata):
            value = source.get(key)
            if value not in (None, ""):
                return value
        return default

    names = pick("reference_files", [])
    if isinstance(names, str):
        try:
            names = json.loads(names)
        except json.JSONDecodeError:
            names = [names] if names.strip() else []
    return {
        "task_id": pick("task_id"),
        "prompt": pick("prompt"),
        "sector": pick("sector", ""),
        "occupation": pick("occupation", ""),
        "reference_files": [str(n) for n in (names or [])],
    }


def classify_outcome(
    *,
    finish_params: Any,
    error: BaseException | None,
    timed_out: bool,
) -> tuple[Optional[str], Optional[str], Optional[str], str]:
    """Map how the Stirrup loop ended to the verify request's failure fields.

    Returns ``(policy_failure_reason, incomplete_reason, agent_error, finish_kind)``.
    ``finish_kind`` is one of ``finish`` / ``abandon`` / ``max_turns`` / ``context_limit`` /
    ``timeout`` / ``model_timeout`` / ``error``.
    """
    from responses_api_agents.gdpval_stirrup_agent.policy_client import PolicyContextLimitError

    if timed_out:
        return None, "agent_exec_timeout", None, "timeout"
    if error is not None:
        from stirrup.core.exceptions import ContextOverflowError

        if isinstance(error, (PolicyContextLimitError, ContextOverflowError)):
            return "agent_context_limit", None, None, "context_limit"
        name = type(error).__name__
        if name in ("APITimeoutError", "APIConnectionError", "ReadTimeout", "ConnectTimeout"):
            return None, "model_response_timeout", f"{name}: {error}"[:600], "model_timeout"
        return None, "agent_exec_incomplete", f"{name}: {error}"[:600], "error"
    if finish_params is None:
        return "agent_iteration_limit", None, None, "max_turns"
    if getattr(finish_params, "paths", None) is None:
        return None, None, None, "abandon"
    return None, None, None, "finish"


def export_response(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    response_id: Optional[str],
    model: str,
    status: str,
    metadata: dict[str, str] | None = None,
) -> tuple[NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponse]:
    """Chat-completions history -> Gym (create params, response), OpenHands-exporter style."""
    from nemo_gym.responses_converter import VLLMConverter, split_responses_input_output_items
    from openai.types.responses import FunctionTool

    converter = VLLMConverter(return_token_id_information=False)
    items = converter.chat_completions_messages_to_responses_items([dict(m) for m in messages])
    initial, output = split_responses_input_output_items(items)
    function_tools = [FunctionTool.model_validate(t["function"] | {"type": "function"}) for t in (tools or [])]
    params = NeMoGymResponseCreateParamsNonStreaming(
        input=initial,
        tools=[t.model_dump(exclude_unset=True) | {"strict": t.strict} for t in function_tools],
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    response = NeMoGymResponse(
        id=response_id or f"gdpval-stirrup-{uuid.uuid4().hex[:12]}",
        created_at=time.time(),
        model=model,
        object="response",
        output=output,
        tools=function_tools,
        tool_choice="auto",
        parallel_tool_calls=True,
        status=status,
        metadata={k: str(v)[:512] for k, v in (metadata or {}).items()} or None,
    )
    return params, response


def export_segment_responses(client: Any, *, model: str, session_id: str) -> list[NeMoGymResponse]:
    """One Gym response per context segment, with echoed token ids on every assistant item.

    The metadata follows NeMo RL's compaction-trace contract (all values strings):
    ``session_id``, ``parent_session_id`` (empty: main session only), ``segment_index``
    and ``segment_boundary_reason`` ("" / "compaction" / "post_compaction"). Output items
    cover only the messages after the segment's first request (its context prefix), so a
    summary that re-sends earlier assistant text is never a training target twice.
    """
    from nemo_gym.responses_converter import VLLMConverter

    responses = []
    for segment in client.segments():
        messages, prefix = client.segment_messages(segment)
        with_tokens = any("prompt_token_ids" in m for m in messages[prefix:])
        converter = VLLMConverter(return_token_id_information=with_tokens)
        output = converter.chat_completions_messages_to_responses_items(messages[prefix:])
        tools = segment["calls"][-1][1].get("tools") or []
        responses.append(
            NeMoGymResponse(
                id=f"gdpval-stirrup-{session_id}-seg{segment['index']}",
                created_at=time.time(),
                model=model,
                object="response",
                output=output,
                tools=[FunctionTool.model_validate(t["function"] | {"type": "function"}) for t in tools],
                tool_choice="auto",
                parallel_tool_calls=True,
                status="completed",
                # NeMo RL's session-trace validator (training_mode=compaction): one main
                # session that is its own root/parent, and every model call owned by exactly
                # one segment through request_ids (the served chat-completion ids).
                metadata={
                    "training_mode": "compaction",
                    "session_id": str(session_id),
                    "root_session_id": str(session_id),
                    "parent_session_id": "",
                    "session_role": "parent",
                    "spawn_call_id": "",
                    "segment_index": str(segment["index"]),
                    "segment_boundary_reason": segment["reason"],
                    "request_ids": json.dumps(
                        [
                            str(call.get("response_id") or f"{session_id}:call{position}")
                            for position, call in segment["calls"]
                        ]
                    ),
                    "policy_calls": str(len(segment["calls"])),
                },
            )
        )
    return responses


def _incomplete_response(model: str, tag: str) -> NeMoGymResponse:
    """Token-free placeholder: nothing was generated, the row is masked."""
    return NeMoGymResponse(
        id=f"gdpval-stirrup-{tag}-{uuid.uuid4().hex[:12]}",
        created_at=time.time(),
        model=model,
        object="response",
        output=[],
        tools=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        status="incomplete",
    )


# Copies the files the model submitted into the deliverables dir (flattened by
# basename, like Stirrup's save_output_files). Runs inside the sandbox.
_COLLECT_SCRIPT = """import json, os, shutil, sys
paths = json.loads(sys.argv[1]); dest = sys.argv[2]; wd = sys.argv[3]
os.makedirs(dest, exist_ok=True)
copied, missing, skipped = [], [], []
for p in paths:
    src = p if os.path.isabs(p) else os.path.join(wd, p)
    if not os.path.isfile(src):
        (skipped if os.path.isdir(src) else missing).append(p); continue
    target = os.path.join(dest, os.path.basename(src))
    if os.path.realpath(src) != os.path.realpath(target):
        shutil.copy2(src, target)
    copied.append(p)
print(json.dumps({"copied": copied, "missing": missing, "skipped_dirs": skipped}))
"""


def build_web_tools(cfg: "GDPValStirrupAgentConfig"):
    """Return the Exa-backed web tool provider, or ``None`` when web tools are off."""
    if not cfg.web_tools_enabled:
        return None
    import os

    from responses_api_agents.gdpval_stirrup_agent.exa_search import ExaToolProvider, parse_keys

    keys = parse_keys(os.environ.get(cfg.web_search_api_keys_env, "")) if cfg.web_search_api_keys_env else []
    try:
        return ExaToolProvider(
            api_keys=keys,
            api_keys_file=cfg.web_search_api_keys_file or None,
            max_sweeps=cfg.web_search_max_sweeps,
        )
    except ValueError as exc:
        raise ValueError(
            "web_tools_enabled but no web search keys: set web_search_api_keys_file or the "
            f"{cfg.web_search_api_keys_env} env var ({exc})"
        ) from None


class _QuietLogger:
    """Stirrup AgentLoggerBase implementation that records nothing to the console."""

    name: str = AGENT_NAME
    model: Optional[str] = None
    max_turns: Optional[int] = None
    depth: int = 0
    finish_params: Any = None
    run_metadata: Any = None
    output_dir: Optional[str] = None

    def __init__(self) -> None:
        self.turns = 0
        self.tool_calls = 0
        self.compactions = 0
        self.warnings: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    def on_step(self, step: int, tool_calls: int = 0, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.turns = step
        self.tool_calls = tool_calls

    def assistant_message(self, turn, max_turns, assistant_message) -> None:
        return None

    def user_message(self, user_message) -> None:
        return None

    def task_message(self, task) -> None:
        return None

    def tool_result(self, tool_message) -> None:
        return None

    def context_summarization_start(self, pct_used, cutoff) -> None:
        self.compactions += 1

    def context_summarization_complete(self, summary, bridge) -> None:
        return None

    def debug(self, message, *args) -> None:
        return None

    def info(self, message, *args) -> None:
        return None

    def warning(self, message, *args) -> None:
        self.warnings.append(str(message) % args if args else str(message))

    def error(self, message, *args) -> None:
        self.warnings.append(str(message) % args if args else str(message))

    def pause_live(self) -> None:
        return None

    def resume_live(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Agent server
# ---------------------------------------------------------------------------


def _disable_stirrup_disk_cache() -> None:
    """Make Stirrup's ~/.cache/stirrup state cache a no-op for training rollouts.

    The 16 sibling rollouts of a prompt group share one task hash, so their
    CacheManager instances race on the same state.json.tmp rename and
    clear_cache; the first Stirrup training run lost rollouts to
    FileNotFoundError there. Nothing in training ever resumes from that cache.
    """
    from stirrup.core import agent as stirrup_agent_module
    from stirrup.core.cache import CacheManager

    class _NoDiskCache(CacheManager):
        def save_state(self, *args, **kwargs):  # noqa: D401 - intentional no-op
            return None

        def load_state(self, *args, **kwargs):
            return None

        def clear_cache(self, *args, **kwargs):
            return None

        def restore_files(self, *args, **kwargs):
            return None

    stirrup_agent_module.CacheManager = _NoDiskCache


_disable_stirrup_disk_cache()


class GDPValStirrupAgent(SimpleResponsesAPIAgent):
    config: GDPValStirrupAgentConfig
    _sem: asyncio.Semaphore = PrivateAttr()
    _resource_load: dict[str, int] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        workers = self.config.num_workers or 1
        if workers < 1 or self.config.concurrency % workers:
            raise ValueError("GDPVal concurrency must be divisible by positive num_workers")
        per_worker = self.config.concurrency // workers
        self._sem = asyncio.Semaphore(per_worker)
        logger.warning(
            "GDPVal Stirrup agent capacity: workers=%d per_worker=%d total=%d max_turns=%d timeout=%ds",
            workers, per_worker, self.config.concurrency, self.config.agent_max_turns, self.config.agent_timeout,
        )
        refs = self.config.resources_servers or [self.config.resources_server]
        self._resource_load = {ref.name: 0 for ref in refs}
        if len(self._resource_load) != len(refs):
            raise ValueError("GDPVal resources_servers must contain unique server names")
        if not PROMPT_TEMPLATE_PATH.is_file():
            raise FileNotFoundError(PROMPT_TEMPLATE_PATH)
        web = build_web_tools(self.config)  # raises at startup when enabled without keys
        logger.warning("GDPVal Stirrup web tools: %s", f"on ({web.num_keys} keys)" if web else "off")

    @asynccontextmanager
    async def _resource_route(self):
        name = min(self._resource_load, key=self._resource_load.get)
        self._resource_load[name] += 1
        try:
            yield name
        finally:
            self._resource_load[name] -= 1

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        raise HTTPException(405, "Use /run for an artifact-producing GDPVal rollout")

    async def _attach(self, sandbox_id: str) -> AsyncSandbox:
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, get_global_config_dict()))
        try:
            return await AsyncSandbox.connect(
                {"sandbox_id": sandbox_id, "workdir": self.config.working_dir}, provider=provider
            )
        except BaseException:
            await provider.aclose()
            raise

    # ------------------------------------------------------------------

    def _capture_store_root(self) -> Optional[Path]:
        """Gym's token-capture directory from the global config, if the file store is in use."""
        block = (get_global_config_dict() or {}).get("token_id_capture") or {}
        if not block.get("enabled") or block.get("sink") or not block.get("dir"):
            return None
        return Path(str(block["dir"]))

    async def _cleanup_capture_records(self, body: BaseRunRequest, last_response_id: Optional[str] = None) -> None:
        """Delete this rollout's Gym token records, unless another handler is still appending to them.

        Gym re-sends ``/run`` with the same body (same ``_ng_rollout_id``) after a dropped
        connection, so two handlers can share one rollout id; deleting the records then
        orphans the other handler's next continuation (2026-10-19, lineage "no indexed
        parent"). The served id of our last policy call tells the store whose records
        these are; ``None`` (no policy call made here) deletes unconditionally.
        """
        root = self._capture_store_root()
        rollout_id = self.rollout_id_from_run(body)
        if root is None or not rollout_id:
            return
        try:
            from nemo_gym.token_id_capture.store import TokenCaptureStore

            await asyncio.to_thread(TokenCaptureStore(root).delete_unless_foreign, rollout_id, last_response_id)
        except Exception:
            logger.warning("GDPVal capture-record cleanup failed for rollout %s", rollout_id, exc_info=True)

    async def _reference_files(self, sandbox: AsyncSandbox, names: list[str]) -> list[str]:
        """Reference paths relative to the working dir; listed from the sandbox when unknown."""
        cfg = self.config
        wd = cfg.working_dir.rstrip("/")
        ref = cfg.reference_dir.rstrip("/")
        rel_ref = ref[len(wd) + 1 :] if ref.startswith(wd + "/") else ref
        if names:
            return [f"{rel_ref}/{n}" for n in names]
        listed = await sandbox.exec(f"find {quote(ref)} -type f 2>/dev/null | sort", timeout_s=60)
        out = []
        for line in (listed.stdout or "").splitlines():
            line = line.strip()
            if line.startswith(wd + "/"):
                out.append(line[len(wd) + 1 :])
            elif line:
                out.append(line)
        return out

    async def _collect_deliverables(self, sandbox: AsyncSandbox, paths: list[str]) -> dict[str, Any]:
        cfg = self.config
        result = await sandbox.exec(
            "python3 -c "
            + quote(_COLLECT_SCRIPT)
            + f" {quote(json.dumps(paths))} {quote(cfg.deliverables_dir)} {quote(cfg.working_dir)}",
            timeout_s=300,
        )
        if result.return_code:
            return {"error": (result.stderr or result.stdout or "")[-800:], "copied": [], "missing": paths}
        try:
            return json.loads((result.stdout or "").strip().splitlines()[-1])
        except Exception:
            return {"error": "unparseable collect output", "copied": [], "missing": paths}

    async def _upload_trace(self, sandbox: AsyncSandbox, files: dict[str, Any]) -> None:
        cfg = self.config
        remote_dir = f"{cfg.trace_dir.rstrip('/')}/stirrup"
        await sandbox.exec(f"mkdir -p {quote(remote_dir)}", timeout_s=60)
        with tempfile.TemporaryDirectory(prefix="gdpval-stirrup-trace-") as tmp:
            for name, payload in files.items():
                local = Path(tmp) / name
                local.write_text(
                    payload if isinstance(payload, str) else json.dumps(payload, default=str) + "\n",
                    encoding="utf-8",
                )
                await sandbox.upload(local, f"{remote_dir}/{name}")

    async def _execute(self, sandbox: AsyncSandbox, body: GDPValStirrupRunRequest, seed: dict[str, Any]) -> dict:
        from responses_api_agents.gdpval_stirrup_agent.opensandbox_provider import SandboxCodeExecToolProvider
        from responses_api_agents.gdpval_stirrup_agent.policy_client import GDPValPolicyClient
        from responses_api_agents.stirrup_agent.finish_tool_coercing import ABANDON_FINISH_TOOL, COERCING_FINISH_TOOL
        from responses_api_agents.stirrup_agent.nemo_agent import NeMoAgent

        cfg = self.config
        task = task_fields_from_request(body, seed)
        if not task["prompt"]:
            raise HTTPException(422, "GDPVal run request has no task prompt (seed response or row fields)")
        reference_files = await self._reference_files(sandbox, task["reference_files"])
        prompt = build_user_prompt(
            build_task_text(task["prompt"], task["sector"], task["occupation"]),
            reference_files,
            working_dir=cfg.working_dir,
        )

        params = body.responses_create_params
        model_root = self.base_url_for_run(get_server_url(cfg.model_server.name), body)
        client = GDPValPolicyClient(
            model=cfg.policy_model_name,
            base_url=model_root + "/v1",
            context_window=cfg.max_context,
            max_completion_tokens_cap=min(cfg.max_output_tokens, getattr(params, "max_output_tokens", None) or cfg.max_output_tokens),
            min_completion_tokens=cfg.min_completion_tokens,
            completion_token_buffer=cfg.completion_token_buffer,
            temperature=getattr(params, "temperature", None) if getattr(params, "temperature", None) is not None else cfg.temperature,
            top_p=getattr(params, "top_p", None) if getattr(params, "top_p", None) is not None else cfg.top_p,
            enable_thinking=cfg.enable_thinking,
            chat_template_kwargs_transport=cfg.policy_chat_template_kwargs_transport,
            read_timeout_s=cfg.model_read_timeout_s,
            read_retries=cfg.model_read_retries,
        )
        provider = SandboxCodeExecToolProvider(sandbox, working_dir=cfg.working_dir, shell_timeout=cfg.shell_timeout)
        web = build_web_tools(cfg)
        quiet = _QuietLogger()
        agent = NeMoAgent(
            client=client,
            name=AGENT_NAME,
            max_turns=cfg.agent_max_turns,
            tools=[provider] + ([web] if web is not None else []),
            finish_tool=[COERCING_FINISH_TOOL, ABANDON_FINISH_TOOL],
            tool_response_as_user=True,
            skip_input_file_listing=True,
            # Summarization only when configured (each context becomes its own trace);
            # never unwind completed turns, that would silently rewrite a context.
            context_summarization_cutoff=cfg.context_summarization_cutoff or 2.0,
            recover_from_context_overflow=False,
            turns_remaining_warning_threshold=cfg.turns_remaining_warning_threshold,
            logger=quiet,
        )
        agent._cache_on_interrupt = False  # no SIGINT handlers / cache files per rollout
        agent._clear_cache_on_success = False

        started = time.time()
        finish_params = None
        run_metadata: dict[str, Any] = {}
        error: BaseException | None = None
        timed_out = False
        try:
            async with asyncio.timeout(cfg.agent_timeout):
                try:
                    async with agent.session() as session:
                        finish_params, _history, run_metadata = await session.run(prompt)
                except TypeError as exit_error:
                    # Stirrup's session __aexit__ can trip in _log_finish; keep the results.
                    if agent._last_finish_params is None and not client.calls:
                        raise
                    finish_params = agent._last_finish_params
                    run_metadata = agent._last_run_metadata or {}
                    quiet.warnings.append(f"session exit raised {exit_error!r}")
        except TimeoutError:
            timed_out = True
        except SandboxApiException:
            raise
        except Exception as exc:  # classified below; never fails /run
            error = exc
        elapsed = time.time() - started

        failure_reason, incomplete_reason, agent_error, finish_kind = classify_outcome(
            finish_params=finish_params, error=error, timed_out=timed_out
        )
        if agent_error:
            logger.warning("GDPVal Stirrup rollout %s: %s", finish_kind, agent_error)

        submitted = list(getattr(finish_params, "paths", None) or [])
        collected = await self._collect_deliverables(sandbox, submitted) if submitted else {"copied": [], "missing": []}

        token_usage = {
            "policy_calls": len(client.calls),
            "prompt_tokens": client.total_prompt_tokens,
            "completion_tokens": client.total_completion_tokens,
            "length_finishes": client.length_finishes,
        }
        summary = {
            "task_id": task["task_id"],
            "finish_kind": finish_kind,
            "finish_reason": getattr(finish_params, "reason", None),
            "submitted_paths": submitted,
            "collected": collected,
            "turns": quiet.turns,
            "tool_calls": quiet.tool_calls,
            "compactions": quiet.compactions,
            "commands_run": provider.commands_run,
            "commands_timed_out": provider.commands_timed_out,
            "web_searches": web.searches if web is not None else 0,
            "web_fetches": web.fetches if web is not None else 0,
            "web_key_rotations": web.key_rotations if web is not None else 0,
            "elapsed_seconds": round(elapsed, 1),
            "policy_failure_reason": failure_reason,
            "incomplete_reason": incomplete_reason,
            "agent_error": agent_error,
            "warnings": quiet.warnings[-20:],
            **token_usage,
        }
        try:
            await self._upload_trace(
                sandbox,
                {
                    "summary.json": summary,
                    "finish_params.json": finish_params.model_dump() if hasattr(finish_params, "model_dump") else finish_params,
                    "run_metadata.json": {k: v for k, v in run_metadata.items() if not k.startswith("_")},
                    "policy_calls.json": [
                        {k: v for k, v in call.items() if k not in ("messages", "tools")} for call in client.calls
                    ],
                    "messages.json": client.final_messages(),
                },
            )
        except Exception:
            logger.exception("GDPVal Stirrup trace upload failed (continuing)")

        messages = client.final_messages()
        has_assistant = any(m.get("role") == "assistant" for m in messages)
        if not has_assistant:
            # No policy tokens exist for this attempt; report a masked, unfinished row.
            if incomplete_reason is None and failure_reason is None:
                incomplete_reason = "agent_exec_incomplete"
                agent_error = agent_error or "no policy completion was produced"
            params_out = body.responses_create_params
            response = _incomplete_response(cfg.policy_model_name, "empty")
        else:
            params_out, response = export_response(
                messages=messages,
                tools=client.calls[-1].get("tools"),
                response_id=client.last_response_id,
                model=cfg.policy_model_name,
                status="completed" if incomplete_reason is None else "incomplete",
                metadata={
                    "harness": "stirrup",
                    "finish_kind": finish_kind,
                    "turns": quiet.turns,
                    "policy_calls": len(client.calls),
                    "submitted": len(submitted),
                },
            )
        body_out = {
            "task_id": task["task_id"] or (getattr(params, "metadata", None) or {}).get("task_id"),
            "responses_create_params": params_out.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
            "policy_failure_reason": failure_reason,
            "agent_finished": incomplete_reason is None,
            "incomplete_reason": incomplete_reason,
            "agent_error": agent_error,
            # Consumed by run() before /verify: the served id of this handler's last policy call,
            # so capture-record cleanup leaves records another handler is still appending to.
            "_capture_last_response_id": client.last_response_id,
        }
        if cfg.context_summarization_cutoff is not None and has_assistant:
            try:
                body_out["responses"] = [
                    r.model_dump(mode="json")
                    for r in export_segment_responses(client, model=cfg.policy_model_name, session_id=task["task_id"] or "main")
                ]
            except Exception as exc:
                # A segment that cannot be reconstructed must not train on a guess: mask the row.
                logger.warning("GDPVal Stirrup segment export failed; masking rollout: %s", exc)
                body_out["agent_finished"] = False
                body_out["incomplete_reason"] = "agent_exec_incomplete"
                body_out["agent_error"] = f"segment export: {exc}"[:600]
                # The top-level response is text-only by design (segments carry the
                # tokens); without segments NeMo RL would reject it as a result with
                # no generation data, so report an empty output it recovers as masked.
                body_out["response"] = _incomplete_response(cfg.policy_model_name, "segments").model_dump(mode="json")
        return body_out

    @staticmethod
    def _sandbox_lost_body(body: BaseRunRequest, error: BaseException, model: str) -> dict:
        return {
            "task_id": (getattr(body.responses_create_params, "metadata", None) or {}).get("task_id"),
            "responses_create_params": body.responses_create_params.model_dump(mode="json"),
            "response": _incomplete_response(model, "sandbox-lost").model_dump(mode="json"),
            "policy_failure_reason": None,
            "agent_finished": False,
            "incomplete_reason": "sandbox_ended",
            "sandbox_error": str(error)[:500],
        }

    async def run(self, body: GDPValStirrupRunRequest, request: Request) -> GDPValRunResponse:
        async with self._sem, self._resource_route() as resource_server:
            cookies = dict(request.cookies)
            sandbox = None
            seeded = False
            try:
                seed_response = await self.server_client.post(
                    server_name=resource_server,
                    url_path="/seed_session",
                    json={"responses_create_params": body.responses_create_params.model_dump(mode="json")},
                    cookies=cookies,
                )
                cookies.update(seed_response.cookies)
                await raise_for_status(seed_response)
                seeded = True
                seed = await seed_response.json()
                try:
                    sandbox = await self._attach(seed["sandbox_handle"])
                    verify_body = await self._execute(sandbox, body, seed)
                except SandboxApiException as error:
                    logger.warning("GDPVal sandbox lost mid-rollout; reporting masked attempt: %s", error)
                    verify_body = self._sandbox_lost_body(body, error, self.config.policy_model_name)
                last_response_id = verify_body.pop("_capture_last_response_id", None)
                verify_response = await self.server_client.post(
                    server_name=resource_server, url_path="/verify", json=verify_body, cookies=cookies
                )
                await raise_for_status(verify_response)
                result = GDPValRunResponse.model_validate(await verify_response.json())
                if self.config.capture_records_cleanup:
                    await self._cleanup_capture_records(body, last_response_id)
                return result
            finally:
                if seeded:
                    try:
                        cleanup = await self.server_client.post(
                            server_name=resource_server, url_path="/cleanup_session", json={}, cookies=cookies
                        )
                        await raise_for_status(cleanup)
                    except Exception:
                        logger.exception("GDPVal resources-server cleanup failed")
                if sandbox is not None:
                    try:
                        await sandbox.stop()
                    except Exception:
                        logger.exception("GDPVal attached sandbox cleanup failed")


if __name__ == "__main__":
    GDPValStirrupAgent.run_webserver()

elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = GDPValStirrupAgent.run_webserver()
