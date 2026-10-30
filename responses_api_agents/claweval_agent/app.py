# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym /run adapter for ClawEval tasks executed by Harbor and OpenCode."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.agentic_vbench_agent.app import AgenticVBenchAgent, AgenticVBenchConfig
from responses_api_agents.agentic_vbench_agent.capture_proxy import capture_output, load_records
from responses_api_agents.agentic_vbench_agent.core import run_process
from responses_api_agents.claweval_agent.core import (
    inventory,
    mean_metrics,
    read_result,
    stage_harbor_task,
)
from responses_api_agents.harbor_agent.utils import HarborAgentUtils


class ClawEvalConfig(AgenticVBenchConfig):
    task_roots: dict[str, str]
    claw_eval_source: str
    judge_base_url: str
    judge_model: str
    judge_api_key_env: str
    concurrency: int = Field(default=12, ge=1)


class ClawEvalRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any]


class ClawEvalVerifyResponse(BaseVerifyResponse):
    verifier_metadata: dict[str, Any]
    source_id: str
    task_id: str
    task_variant: str
    family: str
    status: str
    artifacts: str
    trajectory: dict[str, Any]
    protocol: str = "claweval-native-grader-opencode"
    completion: float | None = None
    robustness: float | None = None
    safety: float | None = None
    passed: bool | None = None
    terminal_response_id: str | None = None
    model_calls: int = 0
    session_traces: list[dict[str, Any]] = []
    compaction_summaries: int = 0
    generated_tokens: int = 0
    capture_records: str | None = None


class ClawEvalAgent(AgenticVBenchAgent):
    config: ClawEvalConfig

    def model_post_init(self, context: Any) -> None:
        # This intentionally mirrors the proven Agentic-VBench transport setup;
        # only task inventory/staging and verification differ.
        SimpleResponsesAPIAgent.model_post_init(self, context)
        config = self.config
        self._proxy_lock = asyncio.Lock()
        if (config.model_server is None) == (config.model_base_url is None):
            raise ValueError("Configure exactly one of model_server or model_base_url")
        if config.model_base_url is not None and not config.model_id:
            raise ValueError("model_id is required with model_base_url")
        self._semaphore = asyncio.Semaphore(config.concurrency)
        self._tasks = inventory(config.task_roots)
        self._benchmark_root = Path(config.claw_eval_source).resolve(strict=True)
        self._harbor_python = config.harbor_python or sys.executable
        if not Path(self._harbor_python).is_file():
            raise FileNotFoundError(self._harbor_python)
        self._output_root = self._resolve_output_root()
        for root in (*map(Path, config.task_roots.values()), Path(__file__).parent):
            if self._output_root.resolve().is_relative_to(root.resolve()):
                raise ValueError("Evaluation output must be outside ClawEval inputs")
        self._runtime_root = (
            Path(config.runtime_root) if config.runtime_root else Path(tempfile.mkdtemp(prefix="claweval-"))
        )
        self._env = self._resolve_environment()

    def _resolve_output_root(self) -> Path:
        if self.config.output_root:
            return Path(self.config.output_root)
        global_config = getattr(self.server_client, "global_config_dict", None) or {}
        rollouts = global_config.get("output_jsonl_fpath") if hasattr(global_config, "get") else None
        if not rollouts:
            raise ValueError("output_root is required when the run has no output_jsonl_fpath")
        return Path(str(rollouts)).with_suffix("") / "claweval_episodes"

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming):
        raise HTTPException(400, "Use /run with a ClawEval dataset row")

    async def run(self, body: ClawEvalRunRequest) -> ClawEvalVerifyResponse:
        endpoint, model = self._model_endpoint(body)
        identity = {"request": body.model_dump(), "model": model, "endpoint": endpoint}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if key not in self._inflight:
            self._inflight[key] = asyncio.create_task(self._run_once(body, key, endpoint, model))
        return await asyncio.shield(self._inflight[key])

    @staticmethod
    def _prompt(body: ClawEvalRunRequest) -> str:
        params = body.responses_create_params.model_dump(exclude_none=True)
        inputs = params.get("input")
        if not isinstance(inputs, list) or len(inputs) != 1 or inputs[0].get("role") != "user":
            raise HTTPException(400, "Expected one ClawEval user prompt")
        content = inputs[0].get("content")
        if isinstance(content, list):
            if len(content) != 1 or content[0].get("type") != "input_text":
                raise HTTPException(400, "ClawEval media must be staged in the sandbox")
            content = content[0].get("text")
        if not isinstance(content, str) or not content:
            raise HTTPException(400, "Empty ClawEval prompt")
        return content

    async def _run_once(
        self,
        body: ClawEvalRunRequest,
        key: str,
        endpoint: str,
        model: str,
    ) -> ClawEvalVerifyResponse:
        metadata = body.verifier_metadata
        task_id = metadata.get("task_id")
        task_variant = metadata.get("task_variant")
        source_id = metadata.get("source_id")
        if not all(isinstance(value, str) for value in (task_id, task_variant, source_id)):
            raise HTTPException(400, "ClawEval task variant, task_id, and source_id must be strings")
        task = self._tasks.get(f"{task_variant}::{task_id}")
        if task is None:
            raise HTTPException(400, "Unknown ClawEval task")
        if not source_id.startswith(f"{task_variant}__"):
            raise HTTPException(400, "ClawEval source_id and task variant disagree")
        for metadata_key in ("task_variant", "family", "task_package_sha256"):
            if metadata.get(metadata_key) != task[metadata_key]:
                raise HTTPException(400, f"Mismatched ClawEval metadata: {metadata_key}")
        prompt = self._prompt(body)
        if metadata.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest():
            raise HTTPException(400, "ClawEval prompt hash mismatch")

        async with self._semaphore:
            episode = f"{task_id}-{key}"
            output = self._output_root.resolve() / episode
            cached = output / "gym_result.json"
            if cached.is_file():
                return ClawEvalVerifyResponse.model_validate_json(cached.read_text())
            if output.exists():
                await asyncio.to_thread(self._archive_incomplete, output)
            output.mkdir(parents=True, exist_ok=False)
            staged_task = await asyncio.to_thread(
                stage_harbor_task,
                task,
                prompt,
                output / "harbor_task",
                claw_eval_source=self.config.claw_eval_source,
            )
            runtime = self._runtime_root / key[:16]
            capture = self._token_id_capture_enabled()
            record_dir = output / "model_calls"
            agent_endpoint = endpoint
            proxy = None
            if capture:
                proxy = await self._ensure_proxy(endpoint)
                agent_endpoint = proxy.register(
                    key[:16],
                    endpoint.removesuffix("/v1"),
                    record_dir,
                    local_step_cap_reply=self.config.local_step_cap_reply,
                    require_native_session_headers=self.config.opencode_subagents_enabled,
                    require_delegation=self.config.opencode_training_mode == "subagent",
                )
            command = [
                self._harbor_python,
                str(Path(__file__).with_name("harbor_runner.py")),
                "--task-path",
                str(staged_task),
                "--endpoint",
                agent_endpoint,
                "--model",
                model,
                "--output",
                str(output),
                "--runtime-root",
                str(runtime),
                "--context-tokens",
                str(self.config.model_context_tokens),
                "--output-tokens",
                str(self.config.model_output_capability_tokens),
                "--backend",
                self.config.backend,
                "--verifier-timeout-multiplier",
                str(self.config.verifier_timeout_multiplier),
                "--model-timeout-ms",
                str(self.config.model_call_timeout_ms),
                "--judge-base-url",
                self.config.judge_base_url,
                "--judge-model",
                self.config.judge_model,
                "--judge-api-key-env",
                self.config.judge_api_key_env,
            ]
            if self.config.agent_timeout_multiplier is not None:
                command.extend(["--agent-timeout-multiplier", str(self.config.agent_timeout_multiplier)])
            if self.config.opencode_tool_output_max_lines is not None:
                command.extend(["--tool-output-max-lines", str(self.config.opencode_tool_output_max_lines)])
            if self.config.opencode_tool_output_max_bytes is not None:
                command.extend(["--tool-output-max-bytes", str(self.config.opencode_tool_output_max_bytes)])
            if self.config.max_turns is not None:
                command.extend(["--max-turns", str(self.config.max_turns)])
            if self.config.credentials_file:
                command.extend(["--credentials-file", self.config.credentials_file])
            if self.config.opencode_compaction_auto is not None:
                command.extend(["--compaction-auto", str(self.config.opencode_compaction_auto).lower()])
            if self.config.opencode_compaction_prune is not None:
                command.extend(["--compaction-prune", str(self.config.opencode_compaction_prune).lower()])
            if self.config.opencode_disable_title:
                command.append("--disable-title")
            if self.config.opencode_training_mode is not None:
                command.extend(["--training-mode", self.config.opencode_training_mode])
            if self.config.opencode_subagents_enabled:
                command.extend(["--subagents", "--subagent-max-depth", str(self.config.opencode_subagent_max_depth)])
            if self.config.opencode_subagent_max_active is not None:
                command.extend(["--subagent-max-active", str(self.config.opencode_subagent_max_active)])
            if self.config.opencode_delegation_prompt is not None:
                command.extend(["--delegation-prompt", self.config.opencode_delegation_prompt])
            if self.config.opencode_subagent_prompt is not None:
                command.extend(["--subagent-prompt", self.config.opencode_subagent_prompt])
            try:
                exit_code = await run_process(command, output, runtime, self._env)
            except asyncio.CancelledError:
                await asyncio.shield(asyncio.to_thread(self._cleanup_containers, output))
                raise
            finally:
                if capture and proxy is not None:
                    proxy.unregister(key[:16])
            if exit_code:
                await asyncio.to_thread(self._cleanup_containers, output)
                raise RuntimeError(f"ClawEval Harbor runner exited {exit_code}; inspect {output / 'runner.log'}")

            result = await asyncio.to_thread(read_result, output, task)
            response = HarborAgentUtils.get_default_response_object()
            extra: dict[str, Any] = {}
            if capture:
                items, stats = capture_output(
                    load_records(record_dir),
                    include_compaction_summaries=self.config.train_compaction_summaries,
                )
                if self.config.opencode_training_mode == "subagent" and not any(
                    trace.get("session_role") == "child" for trace in stats.session_traces
                ):
                    (output / "delegation_required_error.json").write_text(
                        json.dumps(
                            {
                                "error": "strict subagent rollout completed without a verified child session",
                                "session_traces": stats.session_traces,
                            },
                            indent=2,
                        )
                    )
                    await asyncio.to_thread(self._cleanup_containers, output)
                    raise RuntimeError("Strict subagent rollout completed without a verified child session")
                response.update(model=model, output=items)
                extra = {
                    "terminal_response_id": stats.terminal_response_id,
                    "model_calls": stats.model_calls,
                    "session_traces": stats.session_traces,
                    "compaction_summaries": stats.compaction_summaries,
                    "generated_tokens": stats.generated_tokens,
                    "capture_records": str(record_dir),
                }
                if stats.failed_calls:
                    extra.update(mask_sample=True, failure_kind="claweval:model_call_failed")
                print(
                    f"[claweval] source={source_id} reward={result['reward']:.3f} "
                    f"calls={stats.model_calls} compactions={stats.compaction_summaries} "
                    f"children={max(0, len(stats.session_traces) - 1)} failed={stats.failed_calls} "
                    f"generated_tokens={stats.generated_tokens}",
                    flush=True,
                )
            else:
                response.update(
                    model=model,
                    output=HarborAgentUtils.trial_result_to_responses({}, result["trajectory"]),
                )
            response["usage"] = None
            verified = ClawEvalVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=response,
                verifier_metadata=metadata,
                source_id=source_id,
                task_id=task_id,
                task_variant=task_variant,
                family=task["family"],
                **result,
                **extra,
            )
            temporary = output / "gym_result.json.tmp"
            temporary.write_text(verified.model_dump_json())
            temporary.replace(cached)
            return verified

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return mean_metrics([row for task in tasks for row in task if "reward" in row])

    def compute_repeat_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return self.compute_metrics(tasks)


if __name__ == "__main__":
    ClawEvalAgent.run_webserver()
