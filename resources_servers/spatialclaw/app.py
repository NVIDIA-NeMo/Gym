# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verification and native metric aggregation for SpatialClaw evaluations."""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME


class SpatialClawResourcesServerConfig(BaseResourcesServerConfig):
    spatialclaw_root: str = ""
    data_root: str = ""
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class SpatialClawVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    expected_answer: str = ""
    benchmark: str | None = None
    sample_id: str | int | None = None
    dataset_config: str | None = None
    data_root: str | None = None
    scoring_mode: Literal["auto", "mcqa", "exact", "token_f1", "native"] = "native"


class SpatialClawVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    expected_answer: str
    extracted_answer: str
    prediction: str
    scoring_mode_used: str
    scorer_supported: bool = True


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _visible_answer(value: Any) -> str:
    """Remove balanced or malformed private thinking spans from an answer."""
    text = str(value or "")
    visible: list[str] = []
    hidden_depth = 0
    cursor = 0
    for match in re.finditer(r"(?is)</?think>", text):
        if hidden_depth == 0:
            visible.append(text[cursor : match.start()])
        if match.group().casefold() == "<think>":
            hidden_depth += 1
        elif hidden_depth:
            hidden_depth -= 1
        else:
            visible.clear()
        cursor = match.end()
    if hidden_depth == 0:
        visible.append(text[cursor:])
    return " ".join(part.strip() for part in visible if part.strip()).strip()


def _extract_choice(value: Any) -> str:
    text = _normalize(_visible_answer(value))
    patterns = (
        r"^([A-Za-z])$",
        r"\\boxed\{\s*([A-Za-z])\s*\}",
        r"(?i)(?:answer|choice|option)\s*(?:is|:)?\s*([A-Za-z])\b",
        r"ReturnAnswer\(\s*['\"]([A-Za-z])['\"]\s*\)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return str(matches[-1]).upper()
    return text.upper()


def _answer_tokens(value: Any) -> list[str]:
    text = _normalize(_visible_answer(value)).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return [token for token in text.split() if token not in {"a", "an", "the"}]


def _token_f1(prediction: Any, expected: Any) -> float:
    prediction_tokens = _answer_tokens(prediction)
    expected_tokens = _answer_tokens(expected)
    if not prediction_tokens or not expected_tokens:
        return float(prediction_tokens == expected_tokens and bool(expected_tokens))
    overlap = sum((Counter(prediction_tokens) & Counter(expected_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(expected_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    """Flatten scalar native metrics while excluding large per-sample payloads."""
    flattened: dict[str, float] = {}
    if isinstance(value, bool):
        return flattened
    if isinstance(value, (int, float)):
        if prefix:
            flattened[prefix] = float(value)
        return flattened
    if not isinstance(value, dict):
        return flattened
    for key, child in value.items():
        if key in {"detailed_results", "results", "predictions"}:
            continue
        child_prefix = f"{prefix}/{key}" if prefix else str(key)
        flattened.update(_flatten_numeric(child, child_prefix))
    return flattened


class SpatialClawResourcesServer(SimpleResourcesServer):
    config: SpatialClawResourcesServerConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _benchmark_cache: ClassVar[dict[str, Any]] = {}

    def _root(self) -> Path:
        root_value = self.config.spatialclaw_root or os.environ.get("SPATIALCLAW_ROOT", "")
        if not root_value:
            raise RuntimeError("SPATIALCLAW_ROOT or resources-server spatialclaw_root is required")
        root = Path(root_value).expanduser().resolve()
        if not (root / "spatial_agent" / "evals" / "factory.py").is_file():
            raise RuntimeError(f"Invalid SpatialClaw checkout for verifier: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        return root

    def _resolve_dataset_config(self, root: Path, value: str | None) -> Path | None:
        if not value:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / "spatial_agent" / "config" / "dataset" / path
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        if not path.is_file():
            raise FileNotFoundError(f"SpatialClaw dataset config not found: {path}")
        return path

    def _benchmark(self, benchmark_name: str, dataset_config: str | None, data_root: str | None) -> Any:
        root = self._root()
        resolved_config = self._resolve_dataset_config(root, dataset_config)
        resolved_data_root = data_root or self.config.data_root or str(root / "data")
        key = json.dumps(
            {
                "benchmark": benchmark_name,
                "dataset_config": str(resolved_config or ""),
                "data_root": resolved_data_root,
            },
            sort_keys=True,
        )
        cached = self._benchmark_cache.get(key)
        if cached is not None:
            return cached

        from spatial_agent.config import SpatialAgentConfig, set_config
        from spatial_agent.evals.factory import BenchmarkFactory

        spatial_config = SpatialAgentConfig()
        spatial_config._load_from_envs()
        if resolved_config:
            spatial_config.update_from_dataset_json(str(resolved_config))
        set_config(spatial_config)
        benchmark = BenchmarkFactory.create_benchmark(
            benchmark_name,
            data_root=resolved_data_root,
            question_type=getattr(spatial_config, "question_type", None),
        )
        if benchmark is None:
            raise RuntimeError(f"SpatialClaw benchmark {benchmark_name!r} is not runnable")
        self._benchmark_cache[key] = benchmark
        return benchmark

    @staticmethod
    def _prediction(body: SpatialClawVerifyRequest) -> str:
        metadata = body.response.metadata or {}
        if isinstance(metadata, dict) and "spatialclaw_final_answer" in metadata:
            return str(metadata["spatialclaw_final_answer"] or "")
        return body.response.output_text

    def _native_score(self, body: SpatialClawVerifyRequest, prediction: str) -> float | None:
        if not body.benchmark or body.sample_id is None:
            return None
        benchmark = self._benchmark(body.benchmark, body.dataset_config, body.data_root)
        sample = next(
            (sample for sample in benchmark if str(sample.sample_id) == str(body.sample_id)),
            None,
        )
        if sample is None:
            raise KeyError(f"SpatialClaw sample {body.sample_id!r} not found in {body.benchmark!r}")
        return benchmark.evaluate_single(sample, prediction)

    async def verify(self, body: SpatialClawVerifyRequest) -> SpatialClawVerifyResponse:
        prediction = self._prediction(body)
        expected = _normalize(body.expected_answer)
        mode = body.scoring_mode
        score: float | None = None
        mode_used = mode

        if mode in {"native", "auto"} and body.benchmark and body.sample_id is not None:
            score = self._native_score(body, prediction)
            mode_used = "native"
        if score is None and mode in {"mcqa", "auto"} and len(_extract_choice(expected)) == 1:
            score = float(bool(prediction) and _extract_choice(prediction) == _extract_choice(expected))
            mode_used = "mcqa"
        if score is None and mode in {"exact", "auto"}:
            score = float(bool(prediction) and _normalize(prediction).casefold() == expected.casefold())
            mode_used = "exact"
        if score is None and mode == "token_f1":
            score = _token_f1(prediction, expected) if prediction else 0.0
            mode_used = "token_f1"

        return SpatialClawVerifyResponse(
            **body.model_dump(exclude={"expected_answer"}),
            reward=float(score or 0.0),
            expected_answer=expected,
            extracted_answer=_normalize(prediction),
            prediction=prediction,
            scoring_mode_used=mode_used,
            scorer_supported=score is not None,
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Run SpatialClaw's dataset-level evaluator once per rollout repeat."""
        grouped: dict[tuple[str, str | None, str | None], dict[int, dict[str, str]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for task_rollouts in tasks:
            for rollout in task_rollouts:
                benchmark = rollout.get("benchmark")
                sample_id = rollout.get("sample_id")
                if not benchmark or sample_id is None:
                    continue
                key = (str(benchmark), rollout.get("dataset_config"), rollout.get("data_root"))
                repeat = int(rollout.get(ROLLOUT_INDEX_KEY_NAME, 0) or 0)
                grouped[key][repeat][str(sample_id)] = str(rollout.get("prediction", ""))

        metrics: dict[str, Any] = {}
        for (benchmark_name, dataset_config, data_root), repeats in grouped.items():
            benchmark = self._benchmark(benchmark_name, dataset_config, data_root)
            config_name = Path(dataset_config).stem if dataset_config else benchmark_name
            namespace = f"native/{config_name}"
            values_by_metric: dict[str, list[float]] = defaultdict(list)
            for repeat, predictions in sorted(repeats.items()):
                selected = copy.copy(benchmark)
                selected.data = self._selected_native_samples(benchmark_name, benchmark.data, predictions)
                if not selected.data:
                    continue
                native = _flatten_numeric(selected.evaluate(predictions, output_dir=None))
                metrics[f"{namespace}/repeat_{repeat}/coverage"] = (
                    len(selected.data) / len(benchmark.data) if benchmark.data else 0.0
                )
                for metric_name, value in native.items():
                    metrics[f"{namespace}/repeat_{repeat}/{metric_name}"] = value
                    values_by_metric[metric_name].append(value)
            for metric_name, values in values_by_metric.items():
                metrics[f"{namespace}/{metric_name}"] = sum(values) / len(values)
        return metrics

    @staticmethod
    def _selected_native_samples(
        benchmark_name: str,
        samples: list[Any],
        predictions: dict[str, str],
    ) -> list[Any]:
        """Keep valid native units when aggregating a smoke or interrupted run.

        Video-MME-v2 scores consecutive four-question groups. Dropping one row and
        concatenating the remainder would change every following group boundary, so
        only complete canonical groups may enter a partial aggregate. Other SpatialClaw
        benchmarks are independently scored and can safely retain individual rows.
        """
        if benchmark_name == "videommev2":
            complete_groups: list[Any] = []
            for offset in range(0, len(samples), 4):
                group = samples[offset : offset + 4]
                if len(group) == 4 and all(str(sample.sample_id) in predictions for sample in group):
                    complete_groups.extend(group)
            return complete_groups
        return [sample for sample in samples if str(sample.sample_id) in predictions]

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        key_metrics = {
            key: value
            for key, value in agent_metrics.items()
            if key in {"mean/reward", "mean/input_tokens", "mean/output_tokens"}
        }
        primary_names = {
            "overall_accuracy",
            "overall_score",
            "overall_score_pct",
            "mean_score",
            "mean_of_question_type_scores",
        }
        for metric_key, value in agent_metrics.items():
            if metric_key.startswith("native/") and "/repeat_" not in metric_key:
                if metric_key.rsplit("/", 1)[-1] in primary_names or metric_key.endswith("/final_rating/total"):
                    key_metrics[metric_key] = value
        return key_metrics


if __name__ == "__main__":
    SpatialClawResourcesServer.run_webserver()
