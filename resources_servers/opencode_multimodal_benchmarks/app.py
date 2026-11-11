# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym-native verification for VStarBench, HRBench8K, VTCBench and ZeroBench."""

import asyncio
import os
from collections.abc import Mapping
from typing import Any, ClassVar

import aiohttp
from pydantic import Field, PrivateAttr, model_validator

from nemo_gym.client import request
from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME, AggregateMetrics, AggregateMetricsRequest
from nemo_gym.core import NeMoGymChatCompletion
from nemo_gym.metrics import compute_aggregate_metrics
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.utils import error_to_http_status, explicit_causes, is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import JUDGE_FAILURES, JudgeError, call_judge, is_judge_failure
from resources_servers.opencode_multimodal_benchmarks.metrics import average_repeats, pass_at_k
from resources_servers.opencode_multimodal_benchmarks.reporting import benchmark_report
from resources_servers.opencode_multimodal_benchmarks.verifier import (
    VisualVerifier,
)
from resources_servers.opencode_multimodal_benchmarks.verifier import (
    VisualVerifyRequest as VisualVerifyRequest,
)
from resources_servers.opencode_multimodal_benchmarks.verifier import (
    VisualVerifyResponse as VisualVerifyResponse,
)
from resources_servers.opencode_multimodal_benchmarks.verifier_fixtures import visual_fixture


VERIFIER_FIXTURE = visual_fixture()


class JudgeHTTPError(JudgeError):
    """HTTP status without credential-bearing request headers or URLs."""

    def __init__(self, status: int) -> None:
        super().__init__(f"Visual judge HTTP {status}")
        self.status_code = status


class VisualBenchmarksConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    judge_required: bool = False
    judge_endpoint: str | None = None
    judge_model: str | None = None
    # Keep the secret out of saved Gym configs and the OpenCode sandbox.
    judge_api_key_env: str = "OPENAI_API_KEY"
    # A model server enables judge-call correlation and capture.
    # None selects direct HTTP judging via judge_endpoint.
    judge_server: str | None = None
    judge_concurrency: int = Field(default=1024, gt=0)
    judge_timeout: float = Field(default=120, gt=0)
    judge_attempts: int = Field(default=3, gt=0)
    vtc_llm_fallback: bool = True

    @model_validator(mode="before")
    @classmethod
    def reject_unknown_judge_settings(cls, value: Any) -> Any:
        # Gym also supplies catalog metadata such as datasets, verified and description.
        if isinstance(value, Mapping):
            unknown = sorted(key for key in value if key.startswith("judge_") and key not in cls.model_fields)
            if unknown:
                raise ValueError(f"Unknown judge settings: {', '.join(unknown)}")
        return value


class VisualBenchmarksServer(VisualVerifier, SimpleResourcesServer):
    config: VisualBenchmarksConfig
    # Direct HTTP is a compatibility transport without Gym model-call observations.
    grading_may_use_uninstrumented_model: ClassVar[bool] = True
    _judge_sem: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._judge_sem = asyncio.Semaphore(self.config.judge_concurrency)
        if self.config.judge_required:
            if bool(self.config.judge_server) == bool(self.config.judge_endpoint):
                raise JudgeError("Configure exactly one of judge_server or judge_endpoint")
            policy_servers = {"policy_model"}
            global_config = getattr(self.server_client, "global_config_dict", {})
            if isinstance(global_config, Mapping):
                for block in global_config.values():
                    if not isinstance(block, Mapping):
                        continue
                    agents = block.get("responses_api_agents", {})
                    for agent in agents.values():
                        if isinstance(agent, Mapping):
                            reference = agent.get("model_server") or {}
                            if isinstance(reference, Mapping) and reference.get("name"):
                                policy_servers.add(reference["name"])
            if self.config.judge_server in policy_servers:
                raise JudgeError("judge_server must not point at the policy model")
            if self.config.judge_server is not None:
                block = global_config.get(self.config.judge_server) if isinstance(global_config, Mapping) else None
                models = block.get("responses_api_models") if isinstance(block, Mapping) else None
                if (
                    not isinstance(models, Mapping)
                    or len(models) != 1
                    or not all(isinstance(model, Mapping) for model in models.values())
                ):
                    raise JudgeError(
                        f"judge_server {self.config.judge_server!r} must name one configured model server"
                    )
            self._judge_credentials()

    def _judge_credentials(self) -> str | None:
        """Require explicit routing; never select a deployment on the caller's behalf."""
        if not self.config.judge_model:
            raise JudgeError("Configure judge_model before using LLM verification")
        if self.config.judge_server is not None:
            return None
        if not self.config.judge_endpoint:
            raise JudgeError("Configure judge_server or judge_endpoint before using LLM verification")
        key = os.environ.get(self.config.judge_api_key_env)
        if not key:
            raise JudgeError(f"Missing judge credential: {self.config.judge_api_key_env}")
        return key

    async def _judge(self, prompt: str, *, temperature: float = 0) -> str:
        return await self._judge_completion(
            {
                "model": self.config.judge_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 128,
                "stream": False,
            }
        )

    async def _judge_completion(self, payload: dict[str, Any]) -> str:
        """Share transport and retry ownership; release concurrency slots during backoff."""
        key = self._judge_credentials()
        for attempt in range(self.config.judge_attempts):
            try:
                async with self._judge_sem:
                    if self.config.judge_server is not None:
                        async with asyncio.timeout(self.config.judge_timeout):
                            completion = await call_judge(
                                self.server_client,
                                server_name=self.config.judge_server,
                                url_path="/v1/chat/completions",
                                json=payload,
                                response_model=NeMoGymChatCompletion,
                                max_attempts=1,
                            )
                        content = completion.choices[0].message.content
                    else:
                        response = await request(
                            "POST",
                            self.config.judge_endpoint,
                            json=payload,
                            headers={"Authorization": f"Bearer {key}"},
                            timeout=aiohttp.ClientTimeout(total=self.config.judge_timeout),
                            _max_num_tries=1,
                        )
                        async with response:
                            if response.status != 200:
                                raise JudgeHTTPError(response.status)
                            value = await response.json()
                        content = value["choices"][0]["message"]["content"]
                    if content is not None and not isinstance(content, str):
                        raise JudgeError("Visual judge content must be a string or null")
                    return content or ""
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                *JUDGE_FAILURES,
                KeyError,
                IndexError,
                ValueError,
                TypeError,
            ) as error:
                if isinstance(error, JUDGE_FAILURES) and not is_judge_failure(error):
                    raise
                causes = list(explicit_causes(error))
                # The HTTP carrier may be a proxy 500 around a permanent provider 400.
                # Typed remote failures preserve that original explicit cause separately.
                statuses = [status for cause in causes if (status := error_to_http_status(cause)) is not None]
                status = (
                    statuses[-1] if statuses else error_to_http_status(getattr(error, "http_error", None) or error)
                )
                permanent = (
                    (isinstance(status, int) and 400 <= status < 500 and status not in (408, 429))
                    or any(isinstance(cause, (KeyError, IndexError, ValueError, TypeError)) for cause in causes)
                    or (isinstance(error, JudgeError) and not isinstance(error, JudgeHTTPError))
                )
                if permanent or attempt + 1 == self.config.judge_attempts:
                    if is_judge_failure(error):
                        raise
                    raise JudgeError(str(error) or type(error).__name__) from error
            await asyncio.sleep(min(30, 2 ** (attempt + 1)))
        raise AssertionError("Judge attempts exhausted")

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Compare valid measurements with the resolved attempt plan, including absent repeats."""
        expected_repeats: dict[tuple[str, str], set[int]] = {}
        for row in body.verify_responses:
            metadata = row["verifier_metadata"]
            identity = (metadata["benchmark"], metadata["index"])
            expected_repeats.setdefault(identity, set()).update(
                (body.expected_rollout_indices or {}).get(row[TASK_INDEX_KEY_NAME], [])
            )
        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=lambda tasks: self.compute_metrics(tasks, expected_repeats=expected_repeats),
            get_key_metrics_fn=self.get_key_metrics,
            compute_repeat_metrics_fn=self.compute_repeat_metrics,
        )

    def compute_metrics(
        self,
        tasks: list[list[dict[str, Any]]],
        *,
        expected_repeats: Mapping[tuple[str, str], set[int]] | None = None,
    ) -> dict[str, Any]:
        """Report headline scores only when every received attempt has a valid measurement."""
        grouped = {}
        for repeats in tasks:
            if repeats:
                benchmark = repeats[0]["verifier_metadata"]["benchmark"]
                grouped.setdefault(benchmark, []).append(repeats)
        metrics = {}
        for benchmark, benchmark_tasks in grouped.items():
            averaged, balanced = average_repeats(benchmark_tasks)
            rows = [row["verifier_metadata"] | {"hit": row["reward"]} for row in averaged]
            report = benchmark_report(benchmark, rows)
            value = sum(row["hit"] for row in rows) / len(rows) if rows else None
            if benchmark == "HRBench8K" and rows:
                value = next(row["accuracy"] for row in report if row["cycle"] == "Average" and row["type"] == "all")
            mean_accuracy = value
            if benchmark == "ZeroBench":
                value = pass_at_k(benchmark_tasks, k=5)
                report = [{"split": "zerobench", "count": len(rows), "pass@5": value, "pass@1": mean_accuracy}]
            metadata_rows = [row["verifier_metadata"] for repeats in benchmark_tasks for row in repeats]
            if any("expected_samples" not in metadata for metadata in metadata_rows):
                raise ValueError("Benchmark metrics require expected_samples on every task")
            expected_values = {metadata["expected_samples"] for metadata in metadata_rows}
            if len(expected_values) > 1:
                raise ValueError("Inconsistent prepared dataset sample counts")
            expected = next(iter(expected_values))
            received_repeats_complete = True
            if expected_repeats is not None:
                observed = {
                    (benchmark, repeats[0]["verifier_metadata"]["index"]): {
                        row.get(ROLLOUT_INDEX_KEY_NAME, 0) for row in repeats
                    }
                    for repeats in benchmark_tasks
                }
                received = {
                    identity: indices for identity, indices in expected_repeats.items() if identity[0] == benchmark
                }
                received_repeats_complete = observed == received
            complete = balanced and received_repeats_complete and len(rows) == expected and value is not None
            metrics[f"{benchmark}/completed"] = len(rows)
            metrics[f"{benchmark}/expected"] = expected
            metrics[f"{benchmark}/complete"] = complete
            if value is not None:
                metrics[f"{benchmark}/" + ("accuracy" if complete else "partial_accuracy")] = value
            if benchmark == "ZeroBench":
                if mean_accuracy is not None:
                    metrics["ZeroBench/pass@1"] = mean_accuracy
                if value is not None:
                    metrics["ZeroBench/" + ("pass@5" if complete else "partial_pass@5")] = value
            metrics[f"{benchmark}/report"] = report
        if len(grouped) == 1:
            benchmark = next(iter(grouped))
            if metrics[f"{benchmark}/complete"]:
                metrics["accuracy"] = metrics[f"{benchmark}/accuracy"]
        return metrics

    def compute_repeat_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return self.compute_metrics(tasks)

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in agent_metrics.items()
            if key == "accuracy" or key.endswith("/accuracy") or key == "ZeroBench/pass@5"
        }


if __name__ == "__main__":
    VisualBenchmarksServer.run_webserver()

elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VisualBenchmarksServer.run_webserver()  # noqa: F401
