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

"""CVDP agentic agent: runs Claude Code inside an Apptainer sandbox.

This agent reuses the host-only ``claude_code_agent`` for everything except the
sandboxed execution path. For tasks that declare ``target_files`` (RTL the agent
must edit on disk), it:

1. asks the :class:`ApptainerProvider` for a fresh sandbox whose host staging dir
   is bind mounted at ``container_workdir`` (default ``/code``),
2. seeds that workspace with the task's ``context_files`` only (never the hidden
   grading harness),
3. runs ``claude`` inside the container (the host Node+Claude prefix is bind
   mounted at ``/opt/claude_node``; the prompt is fed via stdin so large CVDP
   specs do not hit the kernel argv limit),
4. reads back the HDL files the agent produced and passes them to ``/verify`` as
   ``rtl_files`` so the resources server grades the actual on-disk artifact.

Tasks without ``target_files`` (e.g. code-comprehension Q&A) fall back to the
host-only flow inherited from ``ClaudeCodeAgent``.
"""

import asyncio
import json
import logging
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from time import time
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.sandbox.providers.apptainer import (
    ApptainerCreateConfig,
    ApptainerExecConfig,
    ApptainerProvider,
)
from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.claude_code_agent.app import (
    ClaudeCodeAgent,
    ClaudeCodeAgentConfig,
    ClaudeCodeAgentRunRequest,
    ClaudeCodeAgentVerifyResponse,
    _extract_instruction,
    parse_stream_json,
)


LOG = logging.getLogger(__name__)

# Source files the agent may legitimately create or modify, captured for grading.
# Scoped so we don't sweep in build artifacts (.out, .vcd, ...) or the on-disk
# claude config.
_HDL_EXTENSIONS = (".sv", ".svh", ".v", ".vh")
_AGENT_SOURCE_DIRS = ("rtl", "verif")

# Self-contained Node.js + Claude Code prefix, bind-mounted into the sandbox at
# /opt/claude_node (the sim image has no Node/claude of its own). Built once on the
# host, reused by every rollout. Override via config.claude_node_dir.
_CLAUDE_PKG = "@anthropic-ai/claude-code"
_NODE_VERSION = "22.15.0"
_NODE_DIST_URL = f"https://nodejs.org/dist/v{_NODE_VERSION}/node-v{_NODE_VERSION}-linux-x64.tar.xz"
_NODE_PREFIX = Path(__file__).parent / ".claude_node"

# Claude's mutable state (config, sessions, tool-results, caches) lives here —
# deliberately OUTSIDE the task workspace (container_workdir). If it lived under
# the workdir, the model's first `find <workdir>` would surface Claude's own
# internal files and it would rabbit-hole into them instead of doing the task.
# This path is in the container's writable tmpfs (instances start --writable-tmpfs).
_CONTAINER_STATE_DIR = "/tmp/claude_state"

# Claude's stream-json stdout is tee'd to this dir, which is bind-mounted from a
# host temp dir. It lives outside container_workdir (same rationale as state dir)
# so the model never sees it. On a timeout the provider discards the captured
# pipe output, so this on-disk copy is the only way to recover the partial
# trajectory of a run that ran over its wall-clock budget.
_CONTAINER_LOG_DIR = "/tmp/claude_logs"
_STREAM_LOG_NAME = "stream.jsonl"


def _install_node_locally() -> Path:
    """Download a relocatable Node.js (incl. npm) into ``_NODE_PREFIX``; return its bin dir."""
    if (_NODE_PREFIX / "bin" / "node").is_file():
        return _NODE_PREFIX / "bin"

    _NODE_PREFIX.mkdir(parents=True, exist_ok=True)
    tarball = _NODE_PREFIX / "node.tar.xz"
    LOG.info("downloading Node.js %s", _NODE_VERSION)
    urllib.request.urlretrieve(_NODE_DIST_URL, tarball)  # noqa: S310
    with tarfile.open(tarball, "r:xz") as tf:
        tf.extractall(_NODE_PREFIX, filter="data")

    nested = next(p for p in _NODE_PREFIX.iterdir() if p.is_dir() and p.name.startswith("node-"))
    for item in nested.iterdir():
        item.rename(_NODE_PREFIX / item.name)
    nested.rmdir()
    tarball.unlink(missing_ok=True)
    return _NODE_PREFIX / "bin"


def ensure_claude_node_prefix(version: str | None = None) -> Path:
    """Ensure a self-contained Node.js + Claude Code prefix exists; return its path."""
    claude_bin = _NODE_PREFIX / "bin" / "claude"
    if claude_bin.is_file():
        return _NODE_PREFIX

    bin_dir = _install_node_locally()
    npm_bin = bin_dir / "npm"
    pkg = f"{_CLAUDE_PKG}@{version}" if version else f"{_CLAUDE_PKG}@latest"
    # Install into the same prefix (npm global prefix == node prefix) so claude is
    # self-contained: bin/claude -> lib/node_modules/@anthropic-ai/claude-code.
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
    subprocess.run([str(npm_bin), "install", "-g", "--prefix", str(_NODE_PREFIX), pkg], check=True, env=env)

    if not claude_bin.is_file():
        raise RuntimeError(f"claude-code install did not produce a self-contained binary at {claude_bin}")
    LOG.info("self-contained claude-code prefix ready at %s", _NODE_PREFIX)
    return _NODE_PREFIX


def _safe_workspace_path(base: Path, rel: str) -> Optional[Path]:
    """Resolve a workspace-relative path, rejecting absolute paths, ``..``
    traversal, and symlink escapes. Returns the absolute path inside ``base``, or
    None if the path would escape the workspace."""
    if not rel:
        return None
    try:
        candidate = (base / rel).resolve()
        base_resolved = base.resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    if candidate == base_resolved or base_resolved in candidate.parents:
        return candidate
    return None


def _is_harness_path(rel: str) -> bool:
    """True for paths that belong to the hidden grading harness and must never be
    seeded into the agent workspace (the test scripts in ``src/`` and the compose
    file)."""
    norm = rel.replace("\\", "/").strip("/")
    return (
        norm == "src"
        or norm.startswith("src/")
        or norm == "docker-compose.yml"
        or norm.endswith("/docker-compose.yml")
    )


def _summarize_claude_failure(stdout: str) -> str:
    """Best-effort one-line reason for a non-zero claude exit, pulled from its
    stream-json stdout. Claude reports API problems as ``api_retry`` events and a
    terminal ``result`` line (errors go to stdout, not stderr), so we collect the
    HTTP statuses it retried on plus the final result's error fields."""
    retry_statuses: list[str] = []
    result_summary = ""
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if obj.get("type") == "system" and obj.get("subtype") == "api_retry":
            status = obj.get("error_status")
            err = obj.get("error")
            retry_statuses.append(f"{status}/{err}" if err else str(status))
        elif obj.get("type") == "result":
            parts = []
            if obj.get("subtype"):
                parts.append(f"subtype={obj['subtype']}")
            if obj.get("is_error") is not None:
                parts.append(f"is_error={obj['is_error']}")
            if obj.get("api_error_status"):
                parts.append(f"api_error_status={obj['api_error_status']}")
            if obj.get("result"):
                parts.append(f"result={str(obj['result'])[:200]}")
            result_summary = " ".join(parts)
    bits = []
    if retry_statuses:
        from collections import Counter

        counts = Counter(retry_statuses)
        bits.append("api_retry=" + ", ".join(f"{k} x{v}" for k, v in counts.items()))
    if result_summary:
        bits.append(f"result_line[{result_summary}]")
    return "; ".join(bits)


class CvdpAgentConfig(ClaudeCodeAgentConfig):
    """Config for the sandboxed CVDP agent (extends the host-only agent's config)."""

    # Context window Claude Code should assume for the active model. Claude only
    # auto-detects the window for first-party (api.anthropic.com) base URLs; when
    # routing through anthropic_base_url it falls back to a 200K default, which
    # triggers premature auto-compaction on large-window models. Setting this
    # exports CLAUDE_CODE_MAX_CONTEXT_TOKENS (+ DISABLE_AUTO_COMPACT, required for
    # the override to take effect). None leaves Claude's default behavior.
    max_context_tokens: Optional[int] = 1_000_000

    # --- Apptainer execution ---
    # claude runs inside an Apptainer container built from ``sim_image`` (or an
    # explicit ``sif_path``) so it can self-test with the in-container EDA tools.
    # The host Node+Claude prefix is bind mounted in (claude itself is not in the
    # sim image).
    sim_image: str = "nvidia/cvdp-sim:v1.0.0"
    sif_path: Optional[str] = None  # explicit .sif; if unset, pull/convert sim_image
    sif_cache_dir: str = ""  # defaults to ~/.cache/nemo-gym/sif
    claude_node_dir: str = ""  # host Node+Claude prefix to bind (defaults to a built-in one)
    container_workdir: str = "/code"  # workspace mount point + cwd + HOME inside the container


class CvdpAgent(ClaudeCodeAgent):
    """Claude Code agent that executes inside an Apptainer sandbox for RTL tasks."""

    config: CvdpAgentConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        # The apptainer provider, host Node prefix, and SIF download are resolved
        # lazily on the first sandboxed run so the host-only path (and startup on
        # machines without apptainer) keeps working.
        self._provider: Optional[ApptainerProvider] = None
        self._node_bind_dir: Optional[str] = None
        self._provider_guard = asyncio.Lock()
        self._sif_locks: dict[str, asyncio.Lock] = {}
        self._sif_lock_guard = asyncio.Lock()
        self._sif_cache_dir = self.config.sif_cache_dir or os.path.join(Path.home(), ".cache", "nemo-gym", "sif")
        os.makedirs(self._sif_cache_dir, exist_ok=True)

    def _resolve_node_dir(self) -> str:
        """Locate (or build) a self-contained Node + Claude Code prefix to bind in."""
        if self.config.claude_node_dir:
            node_dir = self.config.claude_node_dir
            if not os.path.isdir(os.path.join(node_dir, "bin")):
                raise RuntimeError(
                    f"claude_node_dir not found at {node_dir!r}; expected a self-contained "
                    "Node+Claude prefix (bin/, lib/)."
                )
            return node_dir
        return str(ensure_claude_node_prefix(self.config.claude_code_version))

    async def _ensure_provider(self) -> ApptainerProvider:
        """Build the apptainer provider once, on first use."""
        if self._provider is not None:
            return self._provider
        async with self._provider_guard:
            if self._provider is None:
                node_dir = await asyncio.to_thread(self._resolve_node_dir)
                self._node_bind_dir = node_dir
                self._provider = ApptainerProvider(
                    create=ApptainerCreateConfig(
                        mount_point=self.config.container_workdir,
                        extra_start_args=["--writable-tmpfs"],
                    ),
                    exec=ApptainerExecConfig(
                        default_timeout_s=self.config.timeout,
                        default_binds=[f"{node_dir}:/opt/claude_node:ro"],
                        concurrency=self.config.concurrency,
                    ),
                )
                LOG.warning("apptainer provider ready: node bind=%s, sif cache=%s", node_dir, self._sif_cache_dir)
        return self._provider

    async def _ensure_sif(self, image: str) -> str:
        """Return a cached SIF for the Docker image, pulling it on first use."""
        safe_name = image.replace("/", "_").replace(":", "_") + ".sif"
        sif_path = os.path.join(self._sif_cache_dir, safe_name)
        if os.path.exists(sif_path):
            return sif_path
        async with self._sif_lock_guard:
            lock = self._sif_locks.setdefault(image, asyncio.Lock())
        async with lock:
            if os.path.exists(sif_path):
                return sif_path
            tmp_path = sif_path + ".pulling"
            proc = await asyncio.create_subprocess_exec(
                "apptainer",
                "pull",
                "--force",
                tmp_path,
                f"docker://{image}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise RuntimeError(
                    f"apptainer pull failed for {image} (exit {proc.returncode}): {stderr.decode(errors='replace')}"
                )
            os.rename(tmp_path, sif_path)
            return sif_path

    async def _resolve_sif(self) -> str:
        if self.config.sif_path:
            return self.config.sif_path
        return await self._ensure_sif(self.config.sim_image)

    def _seed_workspace(
        self,
        workdir: Path,
        context_files: Dict[str, str],
        harness_files: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create the CVDP workspace layout and write context files only (no harness/tests).

        Defends the hidden-test guarantee: any context path that is also a declared
        harness file, or that looks like harness (``src/**`` or a compose file), is
        skipped. Paths that escape the workspace (absolute or ``..``) are rejected."""
        for d in ("rtl", "verif", "docs", "src", "rundir"):
            (workdir / d).mkdir(parents=True, exist_ok=True)
        forbidden = set(harness_files or {})
        for filepath, content in (context_files or {}).items():
            if content is None:
                continue
            if filepath in forbidden or _is_harness_path(filepath):
                LOG.warning("skipping harness-like context file %s", filepath)
                continue
            dest = _safe_workspace_path(workdir, filepath)
            if dest is None:
                LOG.warning("skipping unsafe context file path %s", filepath)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                dest.write_text(content, encoding="utf-8")
            except Exception:
                LOG.warning("failed to seed context file %s", filepath)

    def _collect_produced_files(
        self, workdir: Path, context_files: Dict[str, str], target_files: list
    ) -> Dict[str, str]:
        """Capture every HDL source the agent created or modified.

        Returns {relpath: content} for files under the source dirs (``rtl``,
        ``verif``) with an HDL extension that are new or differ from what we seeded,
        plus any declared target files present on disk. Build artifacts and unchanged
        context files are skipped."""
        produced: Dict[str, str] = {}

        for src_dir in _AGENT_SOURCE_DIRS:
            base = workdir / src_dir
            if not base.is_dir():
                continue
            for fpath in base.rglob("*"):
                if not fpath.is_file() or fpath.suffix.lower() not in _HDL_EXTENSIONS:
                    continue
                rel = fpath.relative_to(workdir).as_posix()
                try:
                    content = fpath.read_text(encoding="utf-8")
                except Exception:
                    LOG.warning("could not read produced file %s", rel)
                    continue
                # New file, or modified relative to what we seeded.
                if context_files.get(rel) != content:
                    produced[rel] = content

        # Always include declared target files that exist (even if unchanged or
        # written outside the scanned source dirs).
        for tf in target_files:
            if tf in produced:
                continue
            fpath = _safe_workspace_path(workdir, tf)
            if fpath is None:
                LOG.warning("skipping unsafe target file path %s", tf)
                continue
            if fpath.is_file():
                try:
                    produced[tf] = fpath.read_text(encoding="utf-8")
                except Exception:
                    LOG.warning("could not read produced target file %s", tf)

        return produced

    def _build_claude_args(self, model: str, system_prompt: Optional[str]) -> list[str]:
        """Build the ``claude`` CLI args (no positional prompt; it is fed via stdin)."""
        args = [
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]
        if self.config.bare:
            args.append("--bare")
        args += ["--max-turns", str(self.config.max_turns), "--model", model]
        if self.config.mcp_config:
            args += ["--mcp-config", self.config.mcp_config]
        if system_prompt:
            args += ["--append-system-prompt", system_prompt]
        if self.config.allowed_tools:
            args += ["--allowedTools", self.config.allowed_tools]
        if self.config.disallowed_tools:
            args += ["--disallowedTools", self.config.disallowed_tools]
        if self.config.thinking:
            args += ["--thinking", self.config.thinking]
        if self.config.max_thinking_tokens is not None:
            args += ["--max-thinking-tokens", str(self.config.max_thinking_tokens)]
        return args

    def _container_env(self, model: str, base_url: str, api_key: str) -> dict[str, str]:
        """Environment exported into the container for the claude exec.

        HOME is intentionally not set here; apptainer rejects overriding HOME via
        ``--env``, so it is exported inside the shell command instead."""
        env = {
            "ANTHROPIC_API_KEY": api_key,  # pragma: allowlist secret
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "CLAUDE_CODE_SUBAGENT_MODEL": model,
            "IS_SANDBOX": "1",
            # Keep Claude's config/session tree out of the task workspace.
            "CLAUDE_CONFIG_DIR": f"{_CONTAINER_STATE_DIR}/.claude_config",
            "PATH": "/opt/claude_node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
        if self.config.max_context_tokens is not None:
            env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(self.config.max_context_tokens)
            env["DISABLE_AUTO_COMPACT"] = "1"
        if base_url:
            env["ANTHROPIC_BASE_URL"] = base_url
            env["ANTHROPIC_AUTH_TOKEN"] = api_key or "local"
        return env

    @staticmethod
    def _read_partial_stream(log_dir: Optional[Path]) -> str:
        """Read whatever claude tee'd before it was killed (empty string if none)."""
        if log_dir is None:
            return ""
        log_file = log_dir / _STREAM_LOG_NAME
        try:
            return log_file.read_text(errors="replace")
        except OSError:
            return ""

    async def _run_claude_in_sandbox(
        self, handle: Any, instruction: str, system_prompt: Optional[str], log_dir: Optional[Path] = None
    ) -> tuple[str, str]:
        """Run claude inside the sandbox; return (stdout, model_name).

        When ``log_dir`` (a host dir bind-mounted at ``_CONTAINER_LOG_DIR``) is
        given, claude's stdout is tee'd to a file there so a run that times out
        can still surface its partial trajectory.
        """
        provider = await self._ensure_provider()
        base_url = self._resolve_base_url()
        # Keep full model name for local/custom endpoints; strip provider prefix for real Anthropic API.
        model = self.config.model if base_url else self.config.model.split("/")[-1]

        wd = self.config.container_workdir
        # Claude's config dir lives in the container's tmpfs, NOT under the task
        # workspace, so `find <wd>` only shows task files. Seed settings.json there
        # at launch (the dir isn't host-visible, so we write it in-shell). The JSON
        # is small and quote-free, so embedding it in the command is safe.
        config_dir = f"{_CONTAINER_STATE_DIR}/.claude_config"
        settings_json = json.dumps(self._build_settings())

        claude_args = self._build_claude_args(model, system_prompt)
        inner = shlex.join(["/opt/claude_node/bin/claude", *claude_args])
        # Tee stdout to a host-visible log so a timed-out run (whose pipe output the
        # provider discards) still leaves its partial trajectory on disk. `exec` is
        # dropped because the shell must outlive claude to run the tee pipe.
        container_log = f"{_CONTAINER_LOG_DIR}/{_STREAM_LOG_NAME}"
        sink = f"| tee {shlex.quote(container_log)}" if log_dir is not None else ""
        # HOME is set in-shell (apptainer forbids overriding HOME via --env) and points
        # outside the workspace too, so stray dotfiles don't pollute it. The prompt is
        # fed via stdin, not argv, since CVDP prompts can exceed the kernel's per-arg
        # limit (MAX_ARG_STRLEN, ~128KB -> E2BIG).
        command = (
            f"export HOME={shlex.quote(_CONTAINER_STATE_DIR)} && "
            f"mkdir -p {shlex.quote(config_dir)} && "
            f"printf %s {shlex.quote(settings_json)} > {shlex.quote(config_dir + '/settings.json')} && "
            f"cd {shlex.quote(wd)} && {inner} {sink}"
        )

        result = await provider.exec(
            handle,
            command,
            env=self._container_env(model, base_url, self.config.anthropic_api_key),
            stdin=instruction.encode(),
            timeout_s=self.config.timeout,
        )

        if result.error_type == "timeout":
            partial = self._read_partial_stream(log_dir)
            if partial:
                LOG.warning(
                    "claude-code timed out after %ds; recovered %d bytes of partial trajectory from %s",
                    self.config.timeout,
                    len(partial),
                    log_dir,
                )
            else:
                LOG.warning("claude-code timed out after %ds (no partial trajectory captured)", self.config.timeout)
            return partial, model

        stdout = result.stdout or ""
        if result.return_code != 0:
            reason = _summarize_claude_failure(stdout)
            LOG.warning(
                "claude-code exited %s: %s (stderr=%r)",
                result.return_code,
                reason or stdout[-800:],
                (result.stderr or "")[:400],
            )
        return stdout, model

    async def run(self, request: Request, body: ClaudeCodeAgentRunRequest) -> ClaudeCodeAgentVerifyResponse:
        meta = (body.model_extra or {}).get("verifier_metadata") or {}
        target_files = meta.get("target_files") or []
        if not target_files:
            # No on-disk target: use the inherited host-only seed/respond/verify flow.
            return await super().run(request, body)
        async with self.sem:
            return await self._run_sandboxed(request, body, meta, target_files)

    async def _run_sandboxed(
        self,
        request: Request,
        body: ClaudeCodeAgentRunRequest,
        meta: dict,
        target_files: list,
    ) -> ClaudeCodeAgentVerifyResponse:
        """Seed a workspace with context files only, run claude in the container so
        it can read the spec / companion RTL and self-test with the in-container EDA
        tools, then read the produced files back and grade them via ``/verify``."""
        cookies = request.cookies
        context_files = meta.get("context_files") or {}

        inp = body.responses_create_params.input
        if isinstance(inp, str):
            user_message, input_system = inp, None
        else:
            user_message, input_system = _extract_instruction(inp)
        system_parts = [p for p in [self.config.system_prompt, input_system] if p]
        system_prompt = "\n\n".join(system_parts) if system_parts else None

        provider = await self._ensure_provider()
        sif = await self._resolve_sif()
        # Host dir for claude's tee'd stream, bind-mounted into the container outside
        # the workspace. Kept on timeout (so the partial trajectory is inspectable),
        # removed otherwise since the full stdout already lands in the rollout.
        log_dir = Path(tempfile.mkdtemp(prefix="cvdp_traj_"))
        handle = await provider.create(
            SandboxSpec(image=sif, provider_options={"binds": [f"{log_dir}:{_CONTAINER_LOG_DIR}"]})
        )
        workdir = handle.raw.staging_dir
        try:
            self._seed_workspace(workdir, context_files, meta.get("harness_files"))

            stdout, model_name = await self._run_claude_in_sandbox(
                handle, user_message, system_prompt, log_dir=log_dir
            )
            output_items, usage = parse_stream_json(stdout)
            # Prefer Claude Code's authoritative num_turns (what --max-turns bounds);
            # fall back to counting assistant text messages if the result event is absent
            # (e.g. older CLI or a truncated stream). The message count undercounts because
            # tool-only turns produce no text message.
            turns = usage.get("num_turns")
            if turns is None:
                turns = sum(
                    1
                    for it in output_items
                    if getattr(it, "type", None) == "message" and getattr(it, "role", None) == "assistant"
                )

            rtl_files = self._collect_produced_files(workdir, context_files, target_files)
            produced_targets = [tf for tf in target_files if (workdir / tf).is_file()]
            naturally = bool(target_files) and len(produced_targets) == len(target_files)
            if not rtl_files:
                LOG.warning(
                    "claude-code produced no HDL files on disk (targets %s); "
                    "falling back to text extraction from the response",
                    target_files,
                )

            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            response = NeMoGymResponse(
                id=f"resp_{uuid4().hex}",
                created_at=int(time()),
                model=model_name,
                object="response",
                output=output_items,
                tool_choice=body.responses_create_params.tool_choice,
                tools=body.responses_create_params.tools,
                parallel_tool_calls=body.responses_create_params.parallel_tool_calls,
                usage=NeMoGymResponseUsage(
                    input_tokens=input_tokens,
                    input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                    output_tokens=output_tokens,
                    output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                    total_tokens=input_tokens + output_tokens,
                ),
            )
            agent_resp_json = response.model_dump()

            # Only send rtl_files when claude actually wrote files; otherwise omit it
            # so the resources server falls back to parsing RTL from the response text.
            verify_payload = body.model_dump() | {"response": agent_resp_json}
            if rtl_files:
                verify_payload["rtl_files"] = rtl_files

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_payload,
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            return ClaudeCodeAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )
        finally:
            await provider.close(handle)
            shutil.rmtree(log_dir, ignore_errors=True)


if __name__ == "__main__":
    CvdpAgent.run_webserver()
