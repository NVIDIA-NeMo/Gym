# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym /run adapter for the pinned official-compatible Harbor/OpenCode workflow."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from responses_api_agents.agentic_vbench_agent.core import (
    inventory,
    read_result,
    run_process,
)
from responses_api_agents.harbor_agent.utils import HarborAgentUtils


class AgenticVBenchConfig(BaseResponsesAPIAgentConfig):
    benchmark_root: str
    evalkit_root: str
    output_root: str
    runtime_root: str
    model_base_url: str
    model_id: str
    concurrency: int = Field(default=2, ge=1)
    credentials_file: str | None = None
    model_context_tokens: Literal[131072] = 131072
    model_output_capability_tokens: Literal[32000] = 32000


class AgenticVBenchRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict[str, Any]


class AgenticVBenchVerifyResponse(BaseVerifyResponse):
    verifier_metadata: dict[str, Any]
    task_id: str
    family: str
    status: str
    artifacts: str
    trajectory: dict[str, Any]
    protocol: str = "official-compatible-opencode"


class AgenticVBenchAgent(SimpleResponsesAPIAgent):
    config: AgenticVBenchConfig
    _semaphore: asyncio.Semaphore = PrivateAttr()
    _tasks: dict = PrivateAttr()
    _inflight: dict[str, asyncio.Task] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._semaphore = asyncio.Semaphore(self.config.concurrency)
        self._tasks = inventory(Path(self.config.benchmark_root))
        runner = Path(self.config.evalkit_root) / "shell/run_agentic_vbench_harbor_trial.sh"
        if not runner.is_file():
            raise FileNotFoundError(runner)
        for root in (self.config.benchmark_root, self.config.evalkit_root):
            if Path(self.config.output_root).resolve().is_relative_to(Path(root).resolve()):
                raise ValueError("Evaluation output must be outside source checkouts")

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        raise HTTPException(400, "Use /run with an Agentic-VBench dataset row")

    async def run(self, body: AgenticVBenchRunRequest) -> AgenticVBenchVerifyResponse:
        # Gym's HTTP retries must never create a second trajectory for a completed zero score.
        identity = {
            "request": body.model_dump(),
            "model": self.config.model_id,
            "endpoint": self.config.model_base_url,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if key not in self._inflight:
            self._inflight[key] = asyncio.create_task(self._run_once(body, key))
        return await asyncio.shield(self._inflight[key])

    async def _run_once(self, body: AgenticVBenchRunRequest, key: str) -> AgenticVBenchVerifyResponse:
        task_id = body.verifier_metadata.get("task_id")
        if not isinstance(task_id, str):
            raise HTTPException(400, "task_id must be a string")
        task = self._tasks.get(task_id)
        if task is None:
            raise HTTPException(400, "Unknown benchmark task")
        for metadata_key in ("family", "benchmark_revision", "prompt_sha256"):
            if body.verifier_metadata.get(metadata_key) != task[metadata_key]:
                raise HTTPException(400, f"Mismatched task metadata: {metadata_key}")
        params = body.responses_create_params.model_dump(exclude_none=True)
        inputs = params["input"]
        if not isinstance(inputs, list) or len(inputs) != 1 or inputs[0].get("role") != "user":
            raise HTTPException(400, "Expected the verbatim benchmark prompt")
        content = inputs[0]["content"]
        if isinstance(content, list):
            if len(content) != 1 or content[0].get("type") != "input_text":
                raise HTTPException(400, "Initial media injection is not supported")
            content = content[0]["text"]
        if content != task["prompt"]:
            raise HTTPException(400, "Prompt differs from pinned benchmark")
        async with self._semaphore:
            episode = f"{task_id}-{key}"
            output = Path(self.config.output_root).resolve() / episode
            cached = output / "gym_result.json"
            if cached.is_file():
                return AgenticVBenchVerifyResponse.model_validate_json(cached.read_text())
            if output.exists():
                raise RuntimeError(f"Incomplete existing episode; inspect before any infrastructure retry: {output}")
            output.mkdir(parents=True, exist_ok=False)
            command = [
                "bash",
                str(Path(self.config.evalkit_root) / "shell/run_agentic_vbench_harbor_trial.sh"),
                "--agentic-vbench-root",
                self.config.benchmark_root,
                "--vlmevalkit-src",
                self.config.evalkit_root,
                "--model-base-url",
                self.config.model_base_url.rstrip("/"),
                "--model-id",
                self.config.model_id,
                "--output-dir",
                str(output),
                "--tasks",
                task_id,
                "--max-parallel",
                "1",
                "--setup-max-attempts",
                "1",
                "--official-compatible",
                "--opencode-version",
                "1.14.39",
                "--harbor-version",
                "0.6.6",
                "--model-context-tokens",
                str(self.config.model_context_tokens),
                "--model-output-capability-tokens",
                str(self.config.model_output_capability_tokens),
            ]
            if self.config.credentials_file:
                command.extend(["--credentials-file", self.config.credentials_file])
            exit_code = await run_process(command, output, Path(self.config.runtime_root) / key[:16])
            # A nonzero exit never becomes an invented zero reward. Preserve the original artifacts.
            if exit_code:
                raise RuntimeError(f"Harbor runner exited {exit_code}; inspect {output / 'runner.log'}")
            result = await asyncio.to_thread(read_result, output, task)
            response = HarborAgentUtils.get_default_response_object()
            response.update(
                model=self.config.model_id,
                output=HarborAgentUtils.trial_result_to_responses({}, result["trajectory"]),
            )
            response["usage"] = None
            verified = AgenticVBenchVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=response,
                verifier_metadata=body.verifier_metadata,
                task_id=task_id,
                family=task["family"],
                **result,
            )
            temporary = output / "gym_result.json.tmp"
            temporary.write_text(verified.model_dump_json())
            temporary.replace(cached)
            return verified


if __name__ == "__main__":
    AgenticVBenchAgent.run_webserver()
