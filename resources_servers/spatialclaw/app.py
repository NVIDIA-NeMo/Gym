# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource server backed by SpatialClaw's native benchmark evaluators."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from responses_api_agents.spatialclaw_agent.app import (
    SPATIALCLAW_COMMIT,
    SPATIALCLAW_URL,
    _config_path,
    _install_source_path,
    _validate_spatialclaw_checkout,
    ensure_spatialclaw_checkout,
)


class SpatialClawResourcesServerConfig(BaseResourcesServerConfig):
    spatialclaw_url: str = SPATIALCLAW_URL
    spatialclaw_commit: str = SPATIALCLAW_COMMIT
    spatialclaw_root: str | None = None
    source_cache_root: str | None = None
    dataset_config: str
    data_root: str | None = None


class SpatialClawRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    sample_id: str
    answer: Any = None


class SpatialClawVerifyRequest(SpatialClawRunRequest, BaseVerifyRequest):
    pass


class SpatialClawVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    sample_id: str
    prediction: str
    extracted_answer: str
    native_score: float | None
    scored: bool


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float | int]:
    """Flatten native result summaries while dropping row-level detail blobs."""
    flattened: dict[str, float | int] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"detailed_results", "details"}:
                continue
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            flattened.update(_flatten_numeric(child, child_prefix))
    elif isinstance(value, bool):
        flattened[prefix] = int(value)
    elif isinstance(value, (int, float)):
        flattened[prefix] = value
    return flattened


class SpatialClawResourcesServer(SimpleResourcesServer):
    """Use the pinned loader for row scoring and its full evaluator for metrics."""

    config: SpatialClawResourcesServerConfig
    _benchmark_instance: Any = PrivateAttr(default=None)
    _sample_by_id: dict[str, Any] = PrivateAttr(default_factory=dict)
    _load_lock: asyncio.Lock | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self._load_lock = asyncio.Lock()

    def _load_benchmark_sync(self) -> Any:
        if self._benchmark_instance is not None:
            return self._benchmark_instance

        if self.config.spatialclaw_root:
            root = _validate_spatialclaw_checkout(Path(self.config.spatialclaw_root), self.config.spatialclaw_commit)
        else:
            cache_root = (
                Path(self.config.source_cache_root).expanduser().resolve()
                if self.config.source_cache_root
                else Path(__file__).parent / ".sources"
            )
            root = ensure_spatialclaw_checkout(self.config.spatialclaw_url, self.config.spatialclaw_commit, cache_root)
        _install_source_path(root)

        from spatial_agent.config import SpatialAgentConfig, set_config
        from spatial_agent.evals.factory import BenchmarkFactory

        spatial_config = SpatialAgentConfig()
        spatial_config._load_from_envs()
        spatial_config.update_from_dataset_json(_config_path(root, self.config.dataset_config, "dataset"))
        set_config(spatial_config)
        data_root = (
            Path(self.config.data_root).expanduser().resolve() if self.config.data_root else (root / "data").resolve()
        )
        benchmark = BenchmarkFactory.create_benchmark(
            spatial_config.benchmark,
            data_root=str(data_root),
            question_type=spatial_config.question_type,
        )
        if benchmark is None:
            raise RuntimeError(f"SpatialClaw dataset config selected no benchmark: {self.config.dataset_config}")

        sample_by_id: dict[str, Any] = {}
        for sample in benchmark.data:
            sample_id = str(sample.sample_id)
            if sample_id in sample_by_id:
                raise RuntimeError(f"Duplicate SpatialClaw sample id: {sample_id}")
            sample_by_id[sample_id] = sample
        self._benchmark_instance = benchmark
        self._sample_by_id = sample_by_id
        return benchmark

    async def _load_benchmark(self) -> Any:
        if self._benchmark_instance is not None:
            return self._benchmark_instance
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        async with self._load_lock:
            return await asyncio.to_thread(self._load_benchmark_sync)

    async def verify(self, body: SpatialClawVerifyRequest) -> SpatialClawVerifyResponse:
        benchmark = await self._load_benchmark()
        sample = self._sample_by_id.get(str(body.sample_id))
        if sample is None:
            raise ValueError(f"Unknown SpatialClaw sample id: {body.sample_id}")

        prediction = (body.response.output_text or "").strip()
        score = benchmark.evaluate_single(sample, prediction)
        native_score = None if score is None else float(score)
        extracted = benchmark.extract_answer(prediction)
        return SpatialClawVerifyResponse(
            **body.model_dump(),
            reward=0.0 if native_score is None else native_score,
            prediction=prediction,
            extracted_answer=extracted,
            native_score=native_score,
            scored=native_score is not None,
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        benchmark = self._load_benchmark_sync()
        selected_samples = []
        predictions: dict[Any, str] = {}
        for rollouts in tasks:
            if not rollouts:
                continue
            rollout = rollouts[0]
            sample = self._sample_by_id.get(str(rollout.get("sample_id", "")))
            if sample is None:
                continue
            selected_samples.append(sample)
            predictions[sample.sample_id] = str(rollout.get("prediction", ""))

        original_data = benchmark.data
        benchmark.data = selected_samples
        try:
            native_results = benchmark.evaluate(predictions, output_dir=None)
        finally:
            benchmark.data = original_data

        metrics, _, _, _ = compute_pass_majority_metrics(
            tasks,
            score_fn=lambda rollout: (
                {"native_score": float(rollout["native_score"])} if rollout.get("native_score") is not None else {}
            ),
            answer_key="extracted_answer",
        )
        metrics.update({f"spatialclaw/{key}": value for key, value in _flatten_numeric(native_results).items()})
        metrics["spatialclaw/num_evaluated_tasks"] = len(selected_samples)
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        key = {
            name: value
            for name, value in agent_metrics.items()
            if name
            in {
                "spatialclaw/overall_accuracy",
                "spatialclaw/overall_accuracy_pct",
                "spatialclaw/overall_score",
                "spatialclaw/overall_score_pct",
            }
        }
        key.update(
            highest_k_metrics(
                agent_metrics,
                "pass@1[avg-of-{k}]",
                score_names=["native_score"],
            )
        )
        return key


if __name__ == "__main__":
    SpatialClawResourcesServer.run_webserver()
