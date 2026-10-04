# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym /run adapter for the pinned official-compatible Harbor/OpenCode workflow."""

import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_server_url
from responses_api_agents.agentic_vbench_agent.core import (
    FAMILIES,
    backend_environment,
    ensure_checkout,
    ensure_docker_cli,
    equal_family_mean,
    equal_family_weight,
    inventory,
    probe_docker,
    read_result,
    remove_episode_containers,
    run_process,
)
from responses_api_agents.harbor_agent.utils import HarborAgentUtils


class AgenticVBenchConfig(BaseResponsesAPIAgentConfig):
    # The policy is Gym's model server by default: OpenCode calls its OpenAI-compatible
    # chat route, so the served model name, seed and sampling overrides apply. A direct
    # endpoint remains available for launchers that serve the model themselves.
    model_server: ModelServerRef | None = None
    model_base_url: str | None = None
    model_id: str | None = None
    # Inputs: defaults are derived inside the Gym job; explicit paths serve other launchers.
    benchmark_root: str | None = None
    harbor_python: str | None = None
    output_root: str | None = None
    runtime_root: str | None = None
    docker_cli_dir: str | None = None
    credentials_file: str | None = None
    # Task containers run on a remote rootless Podman backend whose `client.env` appears
    # under backend_dir; otherwise DOCKER_* must already be in the environment.
    backend: Literal["podman", "remote"] = "remote"
    backend_dir: str | None = None
    backend_wait_seconds: float = Field(default=1800.0, ge=0)
    judge_protocol: str | None = "nvinference-hybrid"
    concurrency: int = Field(default=12, ge=1)
    max_turns: int | None = Field(default=None, ge=1)
    verifier_timeout_multiplier: float = Field(default=3.0, ge=1.0)
    model_context_tokens: int = Field(default=262144, gt=0)
    model_output_capability_tokens: int = Field(default=100000, gt=0)
    model_call_timeout_ms: int = Field(default=3_600_000, gt=0)


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
    # reward * 25 / family size: its task-weighted mean is the equal-family leaderboard mean.
    equal_family_reward: float
    reward_repair: float | None = None
    reward_assembly: float | None = None
    reward_sequencing: float | None = None
    reward_repurpose: float | None = None


class AgenticVBenchAgent(SimpleResponsesAPIAgent):
    config: AgenticVBenchConfig
    _semaphore: asyncio.Semaphore = PrivateAttr()
    _tasks: dict = PrivateAttr()
    _inflight: dict[str, asyncio.Task] = PrivateAttr(default_factory=dict)
    _benchmark_root: Path = PrivateAttr()
    _harbor_python: str = PrivateAttr()
    _output_root: Path = PrivateAttr()
    _runtime_root: Path = PrivateAttr()
    _env: dict[str, str] = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        config = self.config
        if (config.model_server is None) == (config.model_base_url is None):
            raise ValueError("Configure exactly one of model_server or model_base_url")
        if config.model_base_url is not None and not config.model_id:
            raise ValueError("model_id is required with model_base_url")
        self._semaphore = asyncio.Semaphore(config.concurrency)
        self._benchmark_root = ensure_checkout(Path(config.benchmark_root) if config.benchmark_root else None)
        self._tasks = inventory(self._benchmark_root)
        self._harbor_python = config.harbor_python or sys.executable
        if not Path(self._harbor_python).is_file():
            raise FileNotFoundError(self._harbor_python)
        self._output_root = self._resolve_output_root()
        for root in (self._benchmark_root, Path(__file__).parent):
            if self._output_root.resolve().is_relative_to(Path(root).resolve()):
                raise ValueError("Evaluation output must be outside source checkouts")
        self._runtime_root = (
            Path(config.runtime_root) if config.runtime_root else Path(tempfile.mkdtemp(prefix="agentic-vbench-"))
        )
        self._env = self._resolve_environment()

    def _resolve_output_root(self) -> Path:
        if self.config.output_root:
            return Path(self.config.output_root)
        global_config = getattr(self.server_client, "global_config_dict", None) or {}
        rollouts = global_config.get("output_jsonl_fpath") if hasattr(global_config, "get") else None
        if not rollouts:
            raise ValueError("output_root is required when the run has no output_jsonl_fpath")
        return Path(str(rollouts)).with_suffix("") / "agentic_vbench_episodes"

    def _resolve_environment(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.config.backend == "remote":
            cli_dir = ensure_docker_cli(Path(self.config.docker_cli_dir) if self.config.docker_cli_dir else None)
            if cli_dir is not None:
                env["PATH"] = os.pathsep.join([str(cli_dir), env.get("PATH", "")])
                env["DOCKER_CONFIG"] = str(cli_dir)
            if self.config.backend_dir:
                env.update(backend_environment(Path(self.config.backend_dir), self.config.backend_wait_seconds))
            if not env.get("DOCKER_HOST"):
                raise ValueError("Remote backend needs backend_dir or DOCKER_HOST in the environment")
            if shutil.which("docker", path=env["PATH"]) is None:
                raise FileNotFoundError("docker CLI not found for the remote backend")
            probe_docker(env)
        return env

    def _model_endpoint(self, body: AgenticVBenchRunRequest) -> tuple[str, str]:
        if self.config.model_base_url is not None:
            return self.config.model_base_url.rstrip("/"), str(self.config.model_id)
        server = self.config.model_server
        assert server is not None
        base_url = self.base_url_for_run(get_server_url(server.name), body.model_dump()) + "/v1"
        global_config = getattr(self.server_client, "global_config_dict", None) or {}
        model = self.config.model_id or (
            global_config.get("policy_model_name") if hasattr(global_config, "get") else None
        )
        return base_url, str(model or server.name)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        raise HTTPException(400, "Use /run with an Agentic-VBench dataset row")

    async def run(self, body: AgenticVBenchRunRequest) -> AgenticVBenchVerifyResponse:
        endpoint, model = self._model_endpoint(body)
        # Gym's HTTP retries must never create a second trajectory for a completed zero score.
        identity = {"request": body.model_dump(), "model": model, "endpoint": endpoint}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if key not in self._inflight:
            self._inflight[key] = asyncio.create_task(self._run_once(body, key, endpoint, model))
        return await asyncio.shield(self._inflight[key])

    async def _run_once(
        self, body: AgenticVBenchRunRequest, key: str, endpoint: str, model: str
    ) -> AgenticVBenchVerifyResponse:
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
            output = self._output_root.resolve() / episode
            cached = output / "gym_result.json"
            if cached.is_file():
                return AgenticVBenchVerifyResponse.model_validate_json(cached.read_text())
            if output.exists():
                # A resumed run finds the episode a killed job left behind. Keep its artifacts
                # aside, drop its containers, and run the task again: the model trajectory
                # was never scored, so this is an infrastructure retry, not a rerun.
                await asyncio.to_thread(self._archive_incomplete, output)
            output.mkdir(parents=True, exist_ok=False)
            runtime = self._runtime_root / key[:16]
            command = [
                self._harbor_python,
                str(Path(__file__).with_name("harbor_runner.py")),
                "--task-path",
                str(self._benchmark_root / "tasks" / f"agentic_vbench_{task['family']}" / task_id),
                "--endpoint",
                endpoint,
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
            ]
            if self.config.judge_protocol:
                command.extend(["--judge-protocol", self.config.judge_protocol])
            if self.config.max_turns is not None:
                command.extend(["--max-turns", str(self.config.max_turns)])
            if self.config.credentials_file:
                command.extend(["--credentials-file", self.config.credentials_file])
            try:
                exit_code = await run_process(command, output, runtime, self._env)
            except asyncio.CancelledError:
                await asyncio.shield(asyncio.to_thread(self._cleanup_containers, output))
                raise
            # A nonzero exit never becomes an invented zero reward. Preserve the original artifacts.
            if exit_code:
                await asyncio.to_thread(self._cleanup_containers, output)
                raise RuntimeError(f"Harbor runner exited {exit_code}; inspect {output / 'runner.log'}")
            result = await asyncio.to_thread(read_result, output, task)
            response = HarborAgentUtils.get_default_response_object()
            response.update(
                model=model,
                output=HarborAgentUtils.trial_result_to_responses({}, result["trajectory"]),
            )
            response["usage"] = None
            family = task["family"]
            verified = AgenticVBenchVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=response,
                verifier_metadata=body.verifier_metadata,
                task_id=task_id,
                family=family,
                equal_family_reward=result["reward"] * equal_family_weight(family),
                **{f"reward_{family}": result["reward"]},
                **result,
            )
            temporary = output / "gym_result.json.tmp"
            temporary.write_text(verified.model_dump_json())
            temporary.replace(cached)
            return verified

    def _archive_incomplete(self, output: Path) -> None:
        self._cleanup_containers(output)
        archive = output.with_name(f"{output.name}.incomplete-{time.strftime('%Y%m%dT%H%M%S')}")
        output.rename(archive)

    def _cleanup_containers(self, output: Path) -> None:
        if self.config.backend != "remote":
            return
        try:
            remove_episode_containers(output, self._env)
        except Exception as error:  # noqa: BLE001 - cleanup must not mask the episode's own failure
            print(f"Backend container cleanup failed for {output}: {error}", file=sys.stderr)

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Leaderboard metric: the equal mean of the four family means, exact on any subset."""
        rows = [row for task in tasks for row in task if row.get("family") in FAMILIES]
        return equal_family_mean(rows)

    def compute_repeat_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return self.compute_metrics(tasks)


if __name__ == "__main__":
    AgenticVBenchAgent.run_webserver()
