# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import tarfile
import tempfile
from asyncio import Semaphore
from pathlib import Path
from typing import Any

from fastapi import Request
from omegaconf import OmegaConf
from pydantic import ConfigDict, Field

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata


NEMORL_REVISION = "1cee83587d0f0d2ba82e7cdeced9772641fddbe3"
GYM_REVISION = "fd5e84d6b1c485c80e7ae61553bbd485611c03b4"
INSTRUCTIONS = """Improve post-training by editing recipe.yaml and NeMo-RL/.
Gym is inside NeMo-RL/3rdparty/Gym-workspace/Gym/. New files and core changes,
including new loss functions, are accepted. Wire changes into training and test them.
After you finish, we automatically train Qwen/Qwen2.5-1.5B-Instruct on one GPU
within a {train_minutes}-minute execution budget, including authored dependency builds, startup, training and export.
Training uses 8 prompts × 8 responses per step; then clean code evaluates the weights.
Reward is AIME25 avg@8 accuracy (eight sampled answers per problem, 32K context, boxed answers).
Use train_math.jsonl; do not obtain or train on evaluation examples.
The model architecture, 8×8 batch, GPU/time budget, and evaluator are fixed.
You have {research_minutes} minutes and 200 turns. Changes outside recipe.yaml and NeMo-RL/ are ignored."""


class NeMoRLEnvConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    sandbox_provider: str | dict[str, Any] = "sandbox"
    image: str
    author_sandbox_provider: str | dict[str, Any] = "author_sandbox"
    author_image: str
    author_image_auth: dict[str, str] | None = None
    source_repository: str = "https://github.com/NVIDIA-NeMo/RL.git"
    aime25_path: str = "benchmarks/aime25/data/aime25_benchmark.jsonl"
    train_seconds: int = Field(default=3600, ge=60, le=3600)
    research_seconds: int = Field(default=3600, ge=60, le=3600)
    max_turns: int = Field(default=200, ge=1, le=200)
    concurrency: int = 1
    eval_concurrency: int = Field(default=30, ge=1)


class NeMoRLEnvRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class NeMoRLEnvResponse(BaseVerifyResponse):
    aime25_exact: float = 0.0
    aime25_avg_at_8: float = 0.0
    train_reward: float = 0.0
    completed: int = 0
    inner_steps: int = 0
    wandb_url: str = ""
    patch: str = ""
    failure_reason: str = ""


class NeMoRLEnvAgent(SimpleResponsesAPIAgent):
    config: NeMoRLEnvConfig
    _author_sem: Semaphore | None = None
    _verifier_sem: Semaphore | None = None
    _transfer_sem: Semaphore | None = None
    _provider: dict[str, Any] | None = None
    _author_provider: dict[str, Any] | None = None
    _metadata: dict[str, Any] | None = None

    def model_post_init(self, context: Any) -> None:
        if self._token_id_capture_enabled():
            raise ValueError("nemorl_env currently supports rollout evaluation only; disable training-token capture")
        self._author_provider = resolve_provider_config(
            self.config.author_sandbox_provider, self.server_client.global_config_dict
        )
        self._author_sem = Semaphore(self.config.concurrency)
        self._verifier_sem = Semaphore(self.config.concurrency)
        self._transfer_sem = Semaphore(1)
        self._provider = resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
        self._metadata = resolve_provider_metadata(self.config.sandbox_provider, self.server_client.global_config_dict)
        super().model_post_init(context)

    @staticmethod
    def _task_files() -> dict[str, str]:
        root = Path(__file__).with_name("task_environment")
        return {f"/testbed/{p.name}": p.read_text() for p in root.iterdir() if p.is_file()}

    async def _author(
        self, body: NeMoGymResponseCreateParamsNonStreaming, rollout_id: str | None
    ) -> tuple[NeMoGymResponse, str]:
        model = OmegaConf.to_container(
            get_first_server_config_dict(self.server_client.global_config_dict, self.config.model_server.name),
            resolve=True,
        )
        job = {
            "body": body.model_dump(mode="json"),
            "source_repository": self.config.source_repository,
            "research_seconds": self.config.research_seconds,
            "train_seconds": self.config.train_seconds,
            "max_turns": self.config.max_turns,
            "model": {key: model[key] for key in ("base_url", "api_key", "model", "uses_reasoning_parser")},
        }

        with tempfile.TemporaryDirectory(prefix="nemorl_env_author_bundle_") as tmp:
            archive_path = Path(tmp) / "author.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                paths = [PARENT_DIR / name for name in ("pyproject.toml", "README.md", "LICENSE")]
                for directory in (
                    "nemo_gym",
                    "responses_api_models/vllm_model",
                    "responses_api_agents/claude_code_agent",
                ):
                    paths.extend(
                        path
                        for path in (PARENT_DIR / directory).rglob("*.py")
                        if not any(
                            part.startswith(".") or part == "tests" for part in path.relative_to(PARENT_DIR).parts
                        )
                    )
                root = Path(__file__).parent
                paths.extend(root / name for name in ("__init__.py", "app.py", "author_worker.py"))
                paths.extend(root / "task_environment" / name for name in ("recipe.yaml", "train_math.jsonl"))
                for path in paths:
                    archive.add(path, arcname="Gym/" + str(path.relative_to(PARENT_DIR)), recursive=False)
            spec = SandboxSpec(
                image=self.config.author_image,
                ttl_s=self.config.research_seconds + 1800,
                ready_timeout_s=1200,
                workdir="/rollout",
                resources=SandboxResources(cpu=4, memory_mib=16384, disk_gib=30),
                provider_options={"image_auth": self.config.author_image_auth}
                if self.config.author_image_auth
                else {},
                env={
                    "PYTHONPATH": "/rollout/Gym",
                    "NEMO_GYM_EXTRA_ROOTS": "/rollout/Gym",
                    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "262144",
                    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32768",
                    "DISABLE_COMPACT": "1",
                },
                files={"/rollout/job.json": json.dumps(job)},
                metadata={**(self._metadata or {}), "benchmark": "nemorl-env-author"},
            )
            async with AsyncSandbox(self._author_provider, spec) as sandbox:
                await sandbox.start()
                await sandbox.upload(archive_path, "/rollout/author.tar.gz")
                async with asyncio.timeout(self.config.research_seconds + 600):
                    result = await sandbox.exec(
                        "tar xzf /rollout/author.tar.gz -C /rollout && "
                        "/agent_deps_mount/bin/python -m responses_api_agents.nemorl_env.author_worker",
                        timeout_s=self.config.research_seconds + 600,
                    )
                if result.return_code != 0 or result.error_type:
                    raise RuntimeError(f"Author worker failed: {result.error_type or result.stderr}")
                response_path, patch_path = Path(tmp) / "response.json", Path(tmp) / "change.diff"
                await sandbox.download("/rollout/response.json", response_path)
                await sandbox.download("/rollout/change.diff", patch_path)
                return NeMoGymResponse.model_validate_json(response_path.read_text()), patch_path.read_text()

    def _sandbox_spec(self, extra: dict[str, str], *, evaluation: bool = False) -> SandboxSpec:
        files = self._task_files()
        if evaluation:
            files.pop("/testbed/launch_inner.py")
            files.pop("/testbed/recipe.yaml")
        else:
            files.pop("/testbed/evaluate.py")
        return SandboxSpec(
            image=self.config.image,
            ttl_s=18000,
            ready_timeout_s=1200,
            workdir="/testbed",
            env={
                "NEMORL_ROOT": "/testbed/NeMo-RL",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                **({"NEMORL_ENV_EVAL_CONCURRENCY": str(self.config.eval_concurrency)} if evaluation else {}),
                **{
                    name: os.environ[name]
                    for name in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT", "WANDB_NAME")
                    if evaluation and name in os.environ
                },
            },
            files={**files, **extra},
            metadata={**(self._metadata or {}), "benchmark": "nemorl-env-aime25"},
            resources=SandboxResources(cpu=8, memory_mib=65536, disk_gib=100, gpu=1),
        )

    async def _evaluate(self, patch: str) -> dict[str, Any]:
        aime = PARENT_DIR / self.config.aime25_path
        if not aime.is_file():
            raise FileNotFoundError("Run `gym eval prepare --benchmark aime25` first")
        with tempfile.TemporaryDirectory(prefix="nemorl_env_weights_") as tmp:
            weights = Path(tmp) / "model.safetensors"
            diagnostics = {}
            for stage in ("train", "eval"):
                evaluation = stage == "eval"
                extra = {"/root/aime25.jsonl": aime.read_text()} if evaluation else {"/root/change.diff": patch}
                spec = self._sandbox_spec(extra, evaluation=evaluation)
                async with AsyncSandbox(self._provider, spec) as sandbox:
                    await sandbox.start()
                    if evaluation:
                        async with self._transfer_sem:
                            await sandbox.upload(weights, "/testbed/model.safetensors")
                        weights.unlink()
                    setup = await sandbox.exec(
                        f"apt-get update -qq && apt-get install -y -qq git curl xz-utils && bash run.sh {stage} setup",
                        cwd="/testbed",
                        timeout_s=7200,
                        user="root",
                    )
                    if setup.return_code != 0 or setup.error_type:
                        raise RuntimeError(f"{stage} setup failed: {setup.error_type or setup.stderr}\n{setup.stdout}")
                    budget = 7200 if evaluation else self.config.train_seconds

                    async with asyncio.timeout(budget):
                        result = await sandbox.exec(
                            f"INNER_TRAIN_SECONDS={max(1, self.config.train_seconds - 120)} "
                            f"bash run.sh {stage} execute",
                            cwd="/testbed",
                            timeout_s=budget,
                            user="root",
                        )
                    if result.return_code != 0 or result.error_type:
                        detail = "\n".join(part for part in (result.stdout, result.stderr, result.error_type) if part)[
                            -8000:
                        ]
                        raise RuntimeError(f"{stage} execution failed: {detail}")
                    prefix = "NEMORL_ENV_RESULT=" if evaluation else "NEMORL_ENV_TRAIN_RESULT="
                    records = [
                        line[len(prefix) :] for line in (result.stdout or "").splitlines() if line.startswith(prefix)
                    ]
                    if not records:
                        raise RuntimeError(f"missing {prefix} record")
                    payload = json.loads(records[-1])
                    if evaluation:
                        score = payload
                    else:
                        diagnostics = {
                            "inner_steps": int(payload["inner_steps"]),
                            "train_reward": float(payload["train_reward"]),
                        }
                        async with self._transfer_sem:
                            await sandbox.download("/testbed/model.safetensors", weights)

            return {**score, **diagnostics}

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        async with self._author_sem:
            response, _ = await self._author(body, request.path_params.get("rollout_id"))
        return response

    async def run(self, body: NeMoRLEnvRunRequest = Body()) -> NeMoRLEnvResponse:
        async with self._author_sem:
            rollout_id = self.rollout_id_from_run(body)
            response, patch = await self._author(body.responses_create_params, rollout_id)
        async with self._verifier_sem:
            score = await self._evaluate(patch) if patch.strip() else {"reward": 0.0, "failure_reason": "no_patch"}
        return NeMoRLEnvResponse(
            responses_create_params=body.responses_create_params, response=response, patch=patch, **score
        )


if __name__ == "__main__":
    os.chdir(PARENT_DIR)
    NeMoRLEnvAgent.run_webserver()
