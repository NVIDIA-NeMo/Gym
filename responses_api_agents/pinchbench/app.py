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
import itertools
import json
import shutil
import tarfile
import textwrap
import uuid
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

from fastapi import Request, Response
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.openai_utils import (
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
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec


class PinchBenchAgentConfig(BaseResponsesAPIAgentConfig):
    # Policy model OpenClaw runs against (streaming-capable endpoint, NOT a Gym
    # non-streaming model server — see README).
    model_base_url: str
    model_api_key: str
    model_name: str

    # Judge for hybrid / llm_judge tasks (OpenAI-compatible endpoint).
    judge_model: str
    judge_base_url: str
    judge_api_key: str

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
    # Single key, comma-separated keys, or a list; multiple keys rotate across task
    # sandboxes to spread provider rate limits.
    tavily_api_key: Optional[Union[str, List[str]]] = None

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


class PinchBenchAgent(SimpleResponsesAPIAgent):
    config: PinchBenchAgentConfig

    def model_post_init(self, context):
        self._sem = asyncio.Semaphore(self.config.max_concurrent)
        self._tavily_rr = itertools.count()
        return super().model_post_init(context)

    def _next_tavily_key(self) -> str:
        raw = self.config.tavily_api_key
        keys = raw if isinstance(raw, list) else [k.strip() for k in raw.split(",")]
        keys = [k for k in keys if k]
        return keys[next(self._tavily_rr) % len(keys)]

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        raise NotImplementedError("PinchBench is an external benchmark; use /run.")

    # --- task env ----------------------------------------------------------
    def _task_env(self, task_id: str) -> dict:
        env = {
            "TASK_ID": task_id,
            "MODEL_NAME": self.config.model_name,
            "MODEL_BASE_URL": self.config.model_base_url,
            "MODEL_API_KEY": self.config.model_api_key,
            "JUDGE_MODEL": self.config.judge_model,
            "JUDGE_BASE_URL": self.config.judge_base_url,
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
            env["TAVILY_API_KEY"] = self._next_tavily_key()
        return env

    # --- per-task sandbox (Gym Sandbox API; provider-neutral) ---------------
    def _build_spec(self, task_id: str) -> SandboxSpec:
        cfg = dict(self.config.sandbox_spec)
        return SandboxSpec(
            image=cfg.get("image"),
            ttl_s=cfg.get("ttl_s"),
            ready_timeout_s=cfg.get("ready_timeout_s"),
            workdir=cfg.get("workdir"),
            resources=SandboxResources.from_mapping(cfg.get("resources", {})),
            provider_options=cfg.get("provider_options", {}),
            env=self._task_env(task_id),
            metadata={"task_id": task_id},
        )

    async def _run_in_sandbox(self, task_id: str, out_dir: Path) -> None:
        """Run one PinchBench task and pull its /out archive back."""
        provider = self.config.sandbox_provider or {}
        apptainer_cfg = provider.get("apptainer") if isinstance(provider, dict) else None
        if isinstance(apptainer_cfg, dict) and apptainer_cfg.get("direct_exec"):
            await self._run_in_apptainer_direct(task_id, out_dir, apptainer_cfg)
            return

        if not self.config.sandbox_provider:
            raise ValueError("pinchbench requires sandbox_provider (see configs/pinchbench.yaml)")
        archive = f"{self.config.sandbox_work_base.rstrip('/')}/out/out.tgz"
        sb = AsyncSandbox(self.config.sandbox_provider)
        try:
            await sb.start(self._build_spec(task_id))
            await sb.exec("bash /opt/run_task.sh", timeout_s=self.config.task_timeout_s)
            await sb.download(archive, out_dir / "out.tgz")
        finally:
            await sb.stop()
        with tarfile.open(out_dir / "out.tgz") as tf:
            tf.extractall(out_dir)  # noqa: S202 -- trusted, in-sandbox-produced archive

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

    async def _run_in_apptainer_direct(self, task_id: str, out_dir: Path, apptainer_cfg: dict[str, Any]) -> None:
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

        task_env = self._task_env(task_id)
        argv = ["apptainer", "exec", *[str(arg) for arg in direct_args]]
        argv += ["--bind", f"{staging_dir}:{work_base}"]
        for key, value in task_env.items():
            argv += ["--env", f"{key}={value}"]
        argv += [str(image), "bash", f"{work_base}/{wrapper_path.name}"]

        stdout_path = staging_dir / "apptainer.stdout.log"
        stderr_path = staging_dir / "apptainer.stderr.log"
        with stdout_path.open("wb") as stdout_f, stderr_path.open("wb") as stderr_f:
            proc = await asyncio.create_subprocess_exec(*argv, stdout=stdout_f, stderr=stderr_f)
            try:
                await asyncio.wait_for(proc.wait(), timeout=self.config.task_timeout_s)
            except asyncio.TimeoutError as exc:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
                raise TimeoutError(f"direct apptainer exec timed out for task {task_id}") from exc

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
        partial_args = block.get("partialArgs")
        if isinstance(partial_args, str):
            return partial_args
        args = block.get("arguments")
        if isinstance(args, str):
            return args
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
                    events.append(json.loads(line))
                except json.JSONDecodeError:
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
        try:
            async with self._sem:
                await self._run_in_sandbox(task_id, out_dir)  # one sandbox per task
            result = self._parse_result(task_id, out_dir)
            response = self._response_from_transcript(task_id, out_dir)
            transcript_events, archive_path = self._collect_transcript(task_id, out_dir, run_id)
        except Exception as exc:  # noqa: BLE001 -- never 500; one task must not abort the whole collection (ng_collect is fail-fast)
            failure_class = _classify_task_failure(exc)
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
            **routing,
        )


if __name__ == "__main__":
    PinchBenchAgent.run_webserver()
