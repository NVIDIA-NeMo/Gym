# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""NeMo Gym agent wrapping the PinchBench OpenClaw benchmark.

External benchmark, integrated at the agent-server level (mirrors
`swe_agents` / `harbor_agent`): one JSONL record == one PinchBench task, and
each `/run` launches **one self-contained sandbox per task** (via Gym's provider-neutral
Sandbox API, PR #1377) that runs the stock PinchBench `benchmark.py` for that single task
through OpenClaw, tars its result + transcript under the per-sandbox working mount, and exits. The
sandbox is the per-task isolation boundary (own filesystem → own `~/.openclaw` → own
gateway), which is how SWE-bench/Terminus avoid cross-rollout races (see README).

The skill is NOT vendored: the image (Dockerfile.benchmark) clones PinchBench at a
pinned tag (`v2.0.0`) and applies `setup_scripts/nvidia-pinchbench.patch` (the NVIDIA
OpenAI-compatible-endpoint + judge integration), and bakes in `run_task.sh` at
`/opt/run_task.sh` — mirroring how `harbor_agent`/`mini_swe_agent` pin a framework
commit rather than vendoring.

The sandbox provider is config-selected (`sandbox_provider`): `apptainer` (Slurm/HPC) or
`opensandbox` (cluster), etc. Each per-task sandbox starts its OWN gateway daemon, so it
never hits the shared-gateway WorkspaceVanishedError cliff. Results are pulled back via
`AsyncSandbox.download` (no host bind-mount).

See README.md for design + findings (skill patch, gateway, parity).
"""

import asyncio
import glob
import json
import logging
import shutil
import tarfile
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from fastapi import Request, Response
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.rollout_observability import (
    AgentObservationBundle,
    ObservationGap,
    SandboxObservation,
    link_tool_calls_to_sandbox,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxCreateError, SandboxExecResult, SandboxResources, SandboxSpec
from responses_api_agents.openclaw_agent.app import openclaw_session_conversation
from responses_api_agents.openclaw_agent.observability import (
    build_openclaw_observation_tree,
    build_openclaw_observations,
    discover_openclaw_session_tree,
)


LOG = logging.getLogger(__name__)


class PinchBenchAgentConfig(BaseResponsesAPIAgentConfig):
    # Policy model OpenClaw runs against (streaming-capable endpoint, NOT a Gym
    # non-streaming model server — see README).
    model_base_url: str
    model_api_key: str
    model_name: str
    model_server: Optional[ModelServerRef] = None

    # Judge for hybrid / llm_judge tasks (OpenAI-compatible endpoint).
    judge_model: str
    judge_base_url: str
    judge_api_key: str
    judge_model_server: Optional[ModelServerRef] = None

    # Each task runs in its OWN sandbox with its OWN in-sandbox OpenClaw gateway, so the
    # gateway never shares a workspace across tasks (avoids the WorkspaceVanishedError cliff
    # a shared 147-task gateway hits). gym-nano scored 0.583 (n=3), at parity with vanilla
    # standalone PinchBench (0.564). At openclaw 2026.6.5 `openclaw agent` needs a gateway to
    # persist transcripts, so this is the only supported mode.
    openclaw_mode: Literal["gateway"] = "gateway"
    gateway_token: str = "pinchbench-local"  # in-sandbox OpenClaw gateway token

    # Per-task sandbox via Gym's provider-neutral Sandbox API (PR #1377), replacing direct
    # docker/apptainer calls. `sandbox_provider` selects + configures the provider (e.g.
    # {"apptainer": {...}} or {"opensandbox": {...}}); `sandbox_spec` carries the image
    # (.sif path or docker:// ref), resources, ttl, etc. env + task_id metadata are injected
    # per task.
    sandbox_provider: dict[str, Any] = {}
    sandbox_spec: dict[str, Any] = {}
    task_timeout_s: int = 1800  # per-task exec timeout (PinchBench tasks can be long)
    # Writable, per-sandbox-isolated working mount inside the sandbox. run_task.sh puts
    # the skill copy, OpenClaw's $HOME, $TMPDIR and benchmark.py's run-root here, and we
    # pull results from <base>/out/out.tgz. Default matches the apptainer provider's
    # mount_point (/sandbox); if you override that, set this to match.
    sandbox_work_base: str = "/sandbox"

    web_search_provider: str = "brave"
    brave_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None

    timeout_multiplier: float = 3.0
    max_concurrent: int = 4
    max_tokens: int = 16384
    context_window: int = 131072
    work_root: str = "/tmp/pinchbench_gym"
    # Where per-task transcripts are archived (kept on disk for inspection, like
    # swe_agents' persistent_dir). `raw_rollout` keeps a pointer to this archive.
    transcripts_dir: str = "/tmp/pinchbench_gym/transcripts"


# Failure-routing sentinels read by the rollout dispatcher (nemo_gym.rollout_collection).
NG_FAILURE_CLASS_KEY = "_ng_failure_class"
NG_NO_PERSIST_KEY = "_ng_no_persist"
NG_TERMINAL_KEY = "_ng_failure_terminal"


class SandboxKilledError(RuntimeError):
    """Sandbox process died by signal (walltime SIGTERM / preemption / OOM kill)."""


def _classify_task_failure(exc: BaseException) -> str:
    """Map a task failure onto the dispatcher's routing classes."""
    if isinstance(exc, SandboxKilledError):
        return "kill_shaped"
    if isinstance(exc, TimeoutError):
        return "timeout_exceeded"
    return "legitimate"


def _sandbox_observation_from_exec(provider: str, result: SandboxExecResult) -> SandboxObservation:
    if result.error_type == "timeout":
        outcome = "timeout"
    elif result.error_type is not None:
        outcome = "sandbox_error"
    elif result.return_code == 0:
        outcome = "completed"
    else:
        outcome = "failed"

    return SandboxObservation(
        role="agent",
        provider=provider,
        outcome=outcome,
        # A provider error may use a non-process sentinel return code.
        exit_code=None if result.error_type is not None else result.return_code,
        error_type=result.error_type,
    )


class PinchBenchRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class PinchBenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    task_id: str
    grading_type: str
    grading_breakdown: dict
    grading_notes: str
    status: str
    raw_rollout: dict  # transcript archive location + compact metadata
    ng_agent_observations: AgentObservationBundle | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class PinchBenchAgent(SimpleResponsesAPIAgent):
    config: PinchBenchAgentConfig

    def model_post_init(self, context):
        self._sem = asyncio.Semaphore(self.config.max_concurrent)
        return super().model_post_init(context)

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        raise NotImplementedError("PinchBench is an external benchmark; use /run.")

    # --- task env ----------------------------------------------------------
    def _task_env(self, task_id: str, rollout_id: Optional[str] = None) -> dict:
        model_base_url = (
            self.resolve_model_base_url(self.config.model_server.name, rollout_id)
            if self.config.model_server is not None
            else self.config.model_base_url
        )
        judge_base_url = (
            self.resolve_model_base_url(self.config.judge_model_server.name, rollout_id)
            if self.config.judge_model_server is not None
            else self.config.judge_base_url
        )
        env = {
            "TASK_ID": task_id,
            "MODEL_NAME": self.config.model_name,
            "MODEL_BASE_URL": model_base_url,
            "MODEL_API_KEY": self.config.model_api_key,
            "JUDGE_MODEL": self.config.judge_model,
            "JUDGE_BASE_URL": judge_base_url,
            "JUDGE_API_KEY": self.config.judge_api_key,
            "OPENAI_API_KEY": self.config.model_api_key,
            "PINCHBENCH_WEB_SEARCH_PROVIDER": self.config.web_search_provider,
            "PINCHBENCH_MAX_TOKENS": str(self.config.max_tokens),
            "PINCHBENCH_CONTEXT_WINDOW": str(self.config.context_window),
            "TIMEOUT_MULT": str(self.config.timeout_multiplier),
            "PINCHBENCH_WORK_BASE": self.config.sandbox_work_base,
        }
        # Each per-task container starts its OWN OpenClaw gateway daemon (per-task, so
        # it never hits the shared-workspace WorkspaceVanishedError cliff). The client
        # in-container picks up the token from this env var.
        env["OPENCLAW_GATEWAY_TOKEN"] = self.config.gateway_token
        if self.config.brave_api_key:
            env["BRAVE_API_KEY"] = self.config.brave_api_key
        if self.config.tavily_api_key:
            env["TAVILY_API_KEY"] = self.config.tavily_api_key
        if rollout_id is not None:
            env["NEMO_GYM_OBSERVABILITY_ENABLED"] = "1"
        return env

    # --- per-task sandbox (Gym Sandbox API; provider-neutral) ---------------
    def _build_spec(self, task_id: str, rollout_id: Optional[str] = None) -> SandboxSpec:
        cfg = dict(self.config.sandbox_spec)
        return SandboxSpec(
            image=cfg.get("image"),
            ttl_s=cfg.get("ttl_s"),
            ready_timeout_s=cfg.get("ready_timeout_s"),
            workdir=cfg.get("workdir"),
            resources=SandboxResources.from_mapping(cfg.get("resources", {})),
            provider_options=cfg.get("provider_options", {}),
            env=self._task_env(task_id, rollout_id),
            metadata={"task_id": task_id},
        )

    async def _run_in_sandbox(
        self,
        task_id: str,
        out_dir: Path,
        rollout_id: Optional[str] = None,
        observation_collector: Optional[Callable[[SandboxObservation], None]] = None,
    ) -> Optional[SandboxObservation]:
        """Run one PinchBench task and pull its /out archive back."""
        provider = self.config.sandbox_provider or {}
        apptainer_cfg = provider.get("apptainer") if isinstance(provider, dict) else None
        if isinstance(apptainer_cfg, dict) and apptainer_cfg.get("direct_exec"):
            return await self._run_in_apptainer_direct(
                task_id,
                out_dir,
                apptainer_cfg,
                rollout_id=rollout_id,
                observation_collector=observation_collector,
            )

        if not self.config.sandbox_provider:
            raise ValueError("pinchbench requires sandbox_provider (see configs/pinchbench.yaml)")
        provider_name = next(iter(self.config.sandbox_provider))
        archive = f"{self.config.sandbox_work_base.rstrip('/')}/out/out.tgz"
        sb = AsyncSandbox(self.config.sandbox_provider)
        observation: Optional[SandboxObservation] = None
        try:
            await sb.start(self._build_spec(task_id, rollout_id))
            sandbox_id = sb.sandbox_id
            observation = SandboxObservation(role="agent", provider=provider_name, sandbox_id=sandbox_id)
            if observation_collector is not None:
                observation_collector(observation)
            exec_result = await sb.exec("bash /opt/run_task.sh", timeout_s=self.config.task_timeout_s)
            observation = _sandbox_observation_from_exec(provider_name, exec_result)
            observation.sandbox_id = sandbox_id
            if observation_collector is not None:
                observation_collector(observation)
            await sb.download(archive, out_dir / "out.tgz")
        finally:
            try:
                if observation is not None and rollout_id is not None:
                    try:
                        usage = await sb.resource_usage()
                    except Exception:
                        LOG.warning("sandbox resource usage is unavailable", exc_info=True)
                    else:
                        observation.wall_time_s = usage.wall_time_s
                        observation.cpu_time_s = usage.cpu_time_s
                        observation.peak_memory_mib = usage.peak_memory_mib
                        observation.resource_usage_source = usage.source
            finally:
                await sb.stop()
        with tarfile.open(out_dir / "out.tgz") as tf:
            tf.extractall(out_dir)  # noqa: S202 -- trusted, in-sandbox-produced archive
        return observation

    def _write_direct_exec_wrapper(self, staging_dir: Path) -> Path:
        wrapper_path = staging_dir / "run_task_efb.sh"
        wrapper = textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            BASE="${PINCHBENCH_WORK_BASE:-/sandbox}"
            HOME_DIR="$BASE/home"
            mkdir -p "$HOME_DIR/.openclaw" "$BASE/out"

            python3 - <<'PYCFG'
            import json
            import os
            from pathlib import Path

            base = os.environ.get("PINCHBENCH_WORK_BASE", "/sandbox")
            home = Path(base) / "home"
            cfg_path = home / ".openclaw" / "openclaw.json"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                cfg = json.loads(cfg_path.read_text("utf-8-sig")) if cfg_path.exists() else {}
            except Exception:
                cfg = {}

            model_id = os.environ["MODEL_NAME"]
            base_url = os.environ["MODEL_BASE_URL"].rstrip("/")
            api_key = os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
            max_tokens = int(os.environ.get("PINCHBENCH_MAX_TOKENS", "65536"))
            context_window = int(os.environ.get("PINCHBENCH_CONTEXT_WINDOW", "131072"))
            runtime_params = {
                "temperature": 1,
                "top_p": 0.95,
                "seed": 0,
                "skip_special_tokens": False,
                "chat_template_kwargs": {"enable_thinking": True},
                "maxTokens": max_tokens,
                "max_tokens": max_tokens,
                "max_completion_tokens": max_tokens,
            }
            custom_provider = {
                "baseUrl": base_url,
                "apiKey": api_key,
                "api": "openai-completions",
                "models": [
                    {
                        "id": model_id,
                        "name": model_id,
                        "input": ["text"],
                        "reasoning": False,
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                        "contextWindow": context_window,
                        "contextTokens": context_window,
                        "maxTokens": max_tokens,
                        "params": runtime_params,
                        "compat": {
                            "requiresStringContent": True,
                            "supportsUsageInStreaming": True,
                            "maxTokensField": "max_tokens",
                        },
                    }
                ],
            }
            models = cfg.setdefault("models", {})
            models["mode"] = "merge"
            models.setdefault("providers", {})["custom"] = custom_provider
            agents = cfg.setdefault("agents", {})
            defaults = agents.setdefault("defaults", {})
            agent_model = f"custom/{model_id}"
            defaults.setdefault("models", {})[agent_model] = {"params": runtime_params}
            defaults.setdefault("model", {})["primary"] = agent_model
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), "utf-8")
            PYCFG

            PATCHED_RUN_TASK="$BASE/run_task_patched.sh"
            python3 - <<'PYSCRIPT'
            import os
            from pathlib import Path

            base = Path(os.environ.get("PINCHBENCH_WORK_BASE", "/sandbox"))
            src = Path("/opt/run_task.sh")
            dst = base / "run_task_patched.sh"
            text = src.read_text()
            text = text.replace(
                "pkill -9 -f openclaw 2>/dev/null || true",
                'kill -9 "$GW_PID" 2>/dev/null || true',
            )
            dst.write_text(text)
            dst.chmod(0o755)
            PYSCRIPT

            set +e
            bash "$PATCHED_RUN_TASK"
            RC=$?
            set -e
            exit "$RC"
            """
        )
        wrapper_path.write_text(wrapper)
        wrapper_path.chmod(0o755)
        return wrapper_path

    async def _run_in_apptainer_direct(
        self,
        task_id: str,
        out_dir: Path,
        apptainer_cfg: dict[str, Any],
        rollout_id: Optional[str] = None,
        observation_collector: Optional[Callable[[SandboxObservation], None]] = None,
    ) -> SandboxObservation:
        image = self.config.sandbox_spec.get("image")
        if not image:
            raise ValueError("pinchbench sandbox_spec.image is required for direct Apptainer exec")

        work_base = self.config.sandbox_work_base.rstrip("/") or "/sandbox"
        if not work_base.startswith("/"):
            raise ValueError("pinchbench sandbox_work_base must be an absolute path")

        staging_dir = out_dir / "sandbox"
        staging_dir.mkdir(parents=True, exist_ok=True)
        wrapper_path = self._write_direct_exec_wrapper(staging_dir)
        archive = staging_dir / "out" / "out.tgz"

        direct_args = apptainer_cfg.get("direct_exec_args")
        if direct_args is None:
            direct_args = ["--cleanenv", "--no-home"]
        elif isinstance(direct_args, str):
            direct_args = direct_args.split()

        task_env = self._task_env(task_id, rollout_id)
        argv = ["apptainer", "exec", *[str(arg) for arg in direct_args]]
        argv += ["--bind", f"{staging_dir}:{work_base}"]
        for key, value in task_env.items():
            argv += ["--env", f"{key}={value}"]
        argv += [str(image), "bash", f"{work_base}/{wrapper_path.name}"]

        stdout_path = staging_dir / "apptainer.stdout.log"
        stderr_path = staging_dir / "apptainer.stderr.log"
        with stdout_path.open("wb") as stdout_f, stderr_path.open("wb") as stderr_f:
            process_started_at = time.perf_counter()
            proc = await asyncio.create_subprocess_exec(*argv, stdout=stdout_f, stderr=stderr_f)
            sandbox_id = f"pinchbench:{out_dir.name}"
            try:
                await asyncio.wait_for(proc.wait(), timeout=self.config.task_timeout_s)
            except asyncio.TimeoutError as exc:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
                observation = SandboxObservation(
                    role="agent",
                    provider="apptainer",
                    sandbox_id=sandbox_id,
                    outcome="timeout",
                    exit_code=proc.returncode,
                    wall_time_s=time.perf_counter() - process_started_at,
                    error_type="timeout",
                )
                if observation_collector is not None:
                    observation_collector(observation)
                raise TimeoutError(f"direct apptainer exec timed out for task {task_id}") from exc

        return_code = proc.returncode or 0
        observation = SandboxObservation(
            role="agent",
            provider="apptainer",
            sandbox_id=sandbox_id,
            outcome="completed" if return_code == 0 else "failed",
            exit_code=return_code,
            wall_time_s=time.perf_counter() - process_started_at,
        )
        if observation_collector is not None:
            observation_collector(observation)
        stdout = stdout_path.read_text(errors="replace")[-4000:] if stdout_path.exists() else ""
        stderr = stderr_path.read_text(errors="replace")[-4000:] if stderr_path.exists() else ""
        if proc.returncode != 0 and not archive.exists():
            run_log = staging_dir / "out" / "run.log"
            run_tail = run_log.read_text(errors="replace")[-4000:] if run_log.exists() else ""
            detail = (stderr or stdout or run_tail or "no output").strip()
            # rc<0 or 137/143 = killed by signal (walltime/preemption), not a task failure.
            if proc.returncode is not None and (proc.returncode < 0 or proc.returncode in (137, 143)):
                raise SandboxKilledError(
                    f"direct apptainer exec killed (rc={proc.returncode}) for task {task_id}: {detail[:1000]}"
                )
            raise RuntimeError(f"direct apptainer exec failed for task {task_id}: {detail[:4000]}")
        if not archive.exists():
            raise RuntimeError(f"direct apptainer exec did not produce {archive} for task {task_id}")

        shutil.copy2(archive, out_dir / "out.tgz")
        with tarfile.open(out_dir / "out.tgz") as tf:
            tf.extractall(out_dir)  # noqa: S202 -- trusted, in-sandbox-produced archive
        return observation

    # --- result parsing -----------------------------------------------------
    def _parse_result(self, task_id: str, out_dir: Path) -> dict:
        results = [p for p in glob.glob(str(out_dir / "*.json")) if "transcript" not in p]
        if not results:
            return {"reward": 0.0, "grading_type": "unknown", "breakdown": {}, "notes": "", "status": "error"}
        data = json.loads(Path(results[0]).read_text())
        for t in data.get("tasks", []):
            if t.get("task_id") == task_id:
                g = t.get("grading") or {}
                run0 = (g.get("runs") or [{}])[0]
                return {
                    "reward": float(g.get("mean", 0.0)),
                    "grading_type": run0.get("grading_type", "unknown"),
                    "breakdown": run0.get("breakdown", {}),
                    "notes": run0.get("notes", ""),
                    "status": "success",
                }
        return {"reward": 0.0, "grading_type": "unknown", "breakdown": {}, "notes": "", "status": "missing_task"}

    @staticmethod
    def _content_text(content) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or item.get("output") or "")
            else:
                parts.append(getattr(item, "text", None) or getattr(item, "output", None) or "")
        return "\n".join(p for p in parts if p)

    @staticmethod
    def _reasoning_text(message: dict) -> str:
        parts = []
        for key in ("reasoning_content", "reasoning_text", "thinking"):
            value = message.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            parts.append(reasoning)
        elif isinstance(reasoning, dict):
            for key in ("text", "content", "summary"):
                value = reasoning.get(key)
                if isinstance(value, str) and value:
                    parts.append(value)

        content = message.get("content") or []
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict) or item.get("type") not in ("thinking", "reasoning"):
                    continue
                text = item.get("thinking") or item.get("text") or item.get("reasoning") or ""
                if text:
                    parts.append(text)
        return "\n".join(parts)

    @staticmethod
    def _tool_call_arguments(block: dict) -> str:
        args = block.get("arguments")
        if isinstance(args, str):
            return args
        if args is None:
            partial_args = block.get("partialArgs")
            if isinstance(partial_args, str):
                return partial_args
        if args is None:
            args = {}
        return json.dumps(args, ensure_ascii=False)

    @staticmethod
    def _usage_from_transcript(events: list[dict]) -> NeMoGymResponseUsage:
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        reasoning_tokens = 0

        def as_int(value) -> int:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        for event in events:
            message = event.get("message") or {}
            if message.get("role") != "assistant":
                continue
            usage = message.get("usage") or {}
            input_tokens += as_int(usage.get("input") or usage.get("input_tokens") or usage.get("prompt_tokens"))
            output_tokens += as_int(
                usage.get("output") or usage.get("output_tokens") or usage.get("completion_tokens")
            )
            cached_tokens += as_int(usage.get("cacheRead"))
            input_details = usage.get("input_tokens_details") or {}
            output_details = usage.get("output_tokens_details") or {}
            cached_tokens += as_int(input_details.get("cached_tokens"))
            reasoning_tokens += as_int(usage.get("reasoning") or output_details.get("reasoning_tokens"))

        return NeMoGymResponseUsage(
            input_tokens=input_tokens,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached_tokens),
            output_tokens=output_tokens,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=reasoning_tokens),
            total_tokens=input_tokens + output_tokens,
        )

    @staticmethod
    def _read_transcript_events(task_id: str, out_dir: Path) -> list[dict]:
        events: list[dict] = []
        tpath = out_dir / "0001_transcripts" / f"{task_id}.jsonl"
        if tpath.exists():
            for line in tpath.read_text().splitlines():
                try:
                    event = json.loads(line)
                    events.append(event if isinstance(event, dict) else {"raw": line})
                except (json.JSONDecodeError, RecursionError):
                    events.append({"raw": line})
        return events

    def _response_from_transcript_events(self, task_id: str, events: list[dict]) -> NeMoGymResponse:
        output_items = []
        seen_event_ids = set()

        for event in events:
            event_id = event.get("id")
            if event_id:
                if event_id in seen_event_ids:
                    continue
                seen_event_ids.add(event_id)

            if event.get("type") != "message":
                continue

            message = event.get("message") or {}
            role = message.get("role")
            if role == "assistant":
                reasoning = self._reasoning_text(message)
                if reasoning:
                    output_items.append(
                        NeMoGymResponseReasoningItem(
                            id=f"rs_{event_id or len(output_items)}",
                            summary=[NeMoGymSummary(text=reasoning, type="summary_text")],
                            type="reasoning",
                            encrypted_content=None,
                        )
                    )

                content = message.get("content")
                text = self._content_text(
                    [item for item in content if isinstance(item, dict) and item.get("type") == "text"]
                    if isinstance(content, list)
                    else content
                )
                if text:
                    output_items.append(
                        NeMoGymResponseOutputMessage(
                            id=f"msg_{event_id or len(output_items)}",
                            content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                            role="assistant",
                            status="completed",
                            type="message",
                        )
                    )

                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict) or block.get("type") != "toolCall":
                            continue
                        call_id = block.get("id") or f"call_{len(output_items)}"
                        output_items.append(
                            NeMoGymResponseFunctionToolCall(
                                arguments=self._tool_call_arguments(block),
                                call_id=call_id,
                                name=block.get("name") or "",
                                type="function_call",
                                id=call_id,
                                status="completed",
                            )
                        )
            elif role == "toolResult":
                call_id = message.get("toolCallId") or message.get("tool_call_id") or ""
                output_text = self._content_text(message.get("content"))
                if not output_text and message.get("details") is not None:
                    output_text = json.dumps(message["details"], ensure_ascii=False)
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=call_id,
                        output=output_text,
                        status="completed",
                    )
                )

        if not output_items:
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id="msg_0",
                    content=[NeMoGymResponseOutputText(type="output_text", text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        return NeMoGymResponse(
            id=task_id,
            created_at=1.0,
            model=self.config.model_name,
            object="response",
            output=output_items,
            parallel_tool_calls=False,
            tools=[],
            tool_choice="auto",
            usage=self._usage_from_transcript(events),
        )

    def _response_from_transcript(self, task_id: str, out_dir: Path) -> NeMoGymResponse:
        return self._response_from_transcript_events(task_id, self._read_transcript_events(task_id, out_dir))

    def _collect_transcript(self, task_id: str, out_dir: Path, run_id: str) -> tuple[list, str]:
        """Read the full archived transcript and persist it to transcripts_dir
        (kept on disk for inspection, like swe_agents' persistent_dir)."""
        tdir = out_dir / "0001_transcripts"
        events = self._read_transcript_events(task_id, out_dir)
        archive = ""
        if tdir.exists():
            dest = Path(self.config.transcripts_dir) / f"{task_id}_{run_id}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copytree(tdir, dest, dirs_exist_ok=True)
                session_store = out_dir / "openclaw_sessions"
                if session_store.exists():
                    shutil.copytree(session_store, dest / "openclaw_sessions", dirs_exist_ok=True)
                archive = str(dest)
            except OSError:
                pass
        return events, archive

    def _empty_response(self, task_id: str) -> NeMoGymResponse:
        """Minimal valid response for the failure path, so /run can return 200
        with reward 0 (never 500) even when no transcript was ever produced."""
        return NeMoGymResponse(
            id=task_id,
            created_at=1.0,
            model=self.config.model_name,
            object="response",
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "id": "msg_0",
                    "content": [{"type": "output_text", "text": "", "annotations": []}],
                }
            ],
            parallel_tool_calls=False,
            tools=[],
            tool_choice="auto",
        )

    async def run(self, body: PinchBenchRunRequest = Body(), request: Request = None) -> PinchBenchVerifyResponse:
        record = body.model_dump()
        meta = record.get("verifier_metadata") or {}
        task_id = meta.get("task_id") or record.get("task_id")
        if not task_id:
            raise ValueError("record is missing verifier_metadata.task_id")

        run_id = uuid.uuid4().hex
        out_dir = Path(self.config.work_root) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        result = {"reward": 0.0, "grading_type": "unknown", "breakdown": {}, "notes": "", "status": "error"}
        routing: dict = {}
        response = self._empty_response(task_id)
        transcript_events: list = []
        archive_path = ""
        observations: Optional[AgentObservationBundle] = None
        sandbox_observation: Optional[SandboxObservation] = None
        rollout_id = self.rollout_id_from_run(body)
        observe = rollout_id is not None
        sandbox_started: Optional[float] = None

        def collect_sandbox_observation(value: SandboxObservation) -> None:
            nonlocal sandbox_observation
            sandbox_observation = value

        try:
            async with self._sem:
                sandbox_started = time.perf_counter()
                sandbox_observation = await self._run_in_sandbox(
                    task_id,
                    out_dir,
                    rollout_id=rollout_id,
                    observation_collector=collect_sandbox_observation,
                )  # one sandbox per task
                if sandbox_observation is not None and sandbox_observation.wall_time_s is None:
                    sandbox_observation.wall_time_s = time.perf_counter() - sandbox_started
            result = self._parse_result(task_id, out_dir)
            response = self._response_from_transcript(task_id, out_dir)
            transcript_events, archive_path = self._collect_transcript(task_id, out_dir, run_id)
        except Exception as exc:  # noqa: BLE001 -- never 500; one task must not abort the whole collection (ng_collect is fail-fast)
            failure_class = _classify_task_failure(exc)
            provider = self.config.sandbox_provider or {}
            provider_name = next(iter(provider)) if len(provider) == 1 else None
            if sandbox_observation is not None and sandbox_started is not None:
                if sandbox_observation.wall_time_s is None:
                    sandbox_observation.wall_time_s = time.perf_counter() - sandbox_started
                if sandbox_observation.outcome == "unknown":
                    if isinstance(exc, TimeoutError):
                        sandbox_observation.outcome = "timeout"
                    elif isinstance(exc, SandboxCreateError):
                        sandbox_observation.outcome = "sandbox_error"
                if sandbox_observation.error_type is None:
                    sandbox_observation.error_type = type(exc).__name__
            elif provider_name is not None and sandbox_started is not None:
                if isinstance(exc, TimeoutError):
                    sandbox_observation = SandboxObservation(
                        role="agent",
                        provider=provider_name,
                        outcome="timeout",
                        wall_time_s=time.perf_counter() - sandbox_started,
                        error_type=type(exc).__name__,
                    )
                elif isinstance(exc, SandboxCreateError):
                    sandbox_observation = SandboxObservation(
                        role="agent",
                        provider=provider_name,
                        outcome="sandbox_error",
                        wall_time_s=time.perf_counter() - sandbox_started,
                        error_type=type(exc).__name__,
                    )
                else:
                    sandbox_observation = SandboxObservation(
                        role="agent",
                        provider=provider_name,
                        outcome="unknown",
                        wall_time_s=time.perf_counter() - sandbox_started,
                        error_type=type(exc).__name__,
                    )
            print(f"[pinchbench-{failure_class}] {task_id}: {type(exc).__name__}: {exc}", flush=True)
            result = {
                "reward": 0.0,
                "grading_type": "unknown",
                "breakdown": {},
                "notes": f"run failed: {type(exc).__name__}: {exc}",
                "status": "error",
            }
            routing[NG_FAILURE_CLASS_KEY] = failure_class
            if failure_class == "kill_shaped":
                routing[NG_NO_PERSIST_KEY] = True
            elif failure_class == "timeout_exceeded":
                routing[NG_TERMINAL_KEY] = True
        finally:
            if observe:
                try:
                    transcript_available = any(
                        isinstance(event, dict) and event.get("type") == "message" for event in transcript_events
                    )
                    request_input = body.responses_create_params.input
                    request_items = (
                        [NeMoGymEasyInputMessage(role="user", content=request_input)]
                        if isinstance(request_input, str)
                        else list(request_input)
                    )
                    observed_output: list[Any] = []
                    if transcript_available:
                        observed_output = list(response.output)
                        if (
                            len(observed_output) == 1
                            and getattr(observed_output[0], "type", None) == "message"
                            and not self._content_text(getattr(observed_output[0], "content", None))
                        ):
                            observed_output = []
                    root_session_id = next(
                        (
                            event.get("id")
                            for event in transcript_events
                            if event.get("type") == "session" and isinstance(event.get("id"), str)
                        ),
                        run_id,
                    )
                    session_tree, tree_gaps = discover_openclaw_session_tree(
                        out_dir / "openclaw_sessions" / "agents",
                        root_session_id,
                    )
                    if session_tree:
                        tree_inputs = []
                        for invocation_id, parent_id, events in session_tree:
                            invocation_conversation = openclaw_session_conversation(
                                events,
                                input_items=request_items if parent_id is None else None,
                                fallback_output=observed_output if parent_id is None else None,
                            )
                            tree_inputs.append((invocation_id, parent_id, invocation_conversation, events))
                        observations = build_openclaw_observation_tree(
                            tree_inputs,
                            source="pinchbench",
                            model_ref=self.config.model_server,
                        )
                    else:
                        transcript_conversation = openclaw_session_conversation(
                            transcript_events,
                            input_items=request_items,
                            fallback_output=observed_output,
                        )
                        observations = build_openclaw_observations(
                            root_session_id,
                            transcript_conversation,
                            transcript_events,
                            transcript_available=transcript_available,
                            source="pinchbench",
                            model_ref=self.config.model_server,
                        )
                    observations.gaps.extend(tree_gaps)
                except Exception:
                    LOG.exception("failed to build PinchBench observations")
                    observations = AgentObservationBundle(
                        source="pinchbench",
                        gaps=[ObservationGap(code="observation_capture_failed")],
                    )
                if sandbox_observation is not None:
                    if sandbox_observation.sandbox_id is not None:
                        link_tool_calls_to_sandbox(observations, sandbox_observation.sandbox_id)
                    else:
                        observations.gaps.append(ObservationGap(code="sandbox_identity_unavailable"))
                    observations.records.append(sandbox_observation)
                    if sandbox_observation.cpu_time_s is None:
                        observations.gaps.append(ObservationGap(code="sandbox_cpu_time_unavailable"))
                    if sandbox_observation.peak_memory_mib is None:
                        observations.gaps.append(ObservationGap(code="sandbox_memory_usage_unavailable"))
                else:
                    observations.gaps.append(ObservationGap(code="sandbox_observation_unavailable"))
                if self.config.model_server is None:
                    observations.gaps.append(ObservationGap(code="policy_model_capture_unavailable"))
                if self.config.judge_model_server is None:
                    observations.gaps.append(ObservationGap(code="judge_model_capture_unavailable"))
            shutil.rmtree(out_dir, ignore_errors=True)

        return PinchBenchVerifyResponse(
            **record,
            reward=result["reward"],
            response=response,
            task_id=task_id,
            grading_type=result["grading_type"],
            grading_breakdown=result["breakdown"],
            grading_notes=result["notes"],
            status=result["status"],
            raw_rollout={
                "transcript_event_count": len(transcript_events),
                "archived_to": archive_path,
                "run_id": run_id,
            },
            **({"ng_agent_observations": observations.model_dump(mode="json")} if observations is not None else {}),
            **routing,
        )


if __name__ == "__main__":
    PinchBenchAgent.run_webserver()
