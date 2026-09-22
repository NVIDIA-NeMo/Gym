# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native, judge-free MathArena AIME verification with bounded safe parsing."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.reward_profile import compute_aggregate_metrics
from resources_servers.matharena_aime.setup_parser import DIRECTORY, UPSTREAM_REVISION, ensure_parser_runtime


def visible_answer(response: NeMoGymResponse) -> str:
    """Select only the last assistant final message, never an earlier repaired answer."""
    for item in reversed(response.output):
        if getattr(item, "role", None) == "user":
            return ""
        if item.type != "message" or item.role != "assistant":
            continue
        text = "\n".join(part.text for part in item.content if part.type == "output_text")
        text = re.sub(r"<(think|thinking)>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.split(r"</(?:think|thinking)>", text, flags=re.IGNORECASE)[-1]
        return re.split(r"<(?:think|thinking)>", text, flags=re.IGNORECASE)[0].strip()
    return ""


def is_truncated(response: NeMoGymResponse) -> bool:
    """Keep generation-budget exhaustion visible independently of parsed correctness."""
    return (
        response.status == "incomplete"
        or bool(response.incomplete_details)
        or any(item.type == "message" and item.status == "incomplete" for item in response.output)
    )


class MathArenaAIMEConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    parser_timeout_seconds: float = Field(default=5.0, gt=0)
    parser_max_concurrency: int = Field(default=8, gt=0)
    expected_num_repeats: int = Field(default=4, gt=0)
    parser_memory_limit_mb: int = Field(default=2048, ge=256)
    parser_python: str | None = None


class FormatRetryRequest(BaseModel):
    response: NeMoGymResponse


class FormatRetryResponse(BaseModel):
    needs_format_retry: bool = False
    parser_warning: int = 3
    extracted_answer: str | None = None
    valid: bool
    verifier_error: str | None = None


class MathArenaAIMEVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    expected_answer: int = Field(ge=0, le=999, strict=True)
    task_id: str = Field(min_length=1)
    language: str = Field(min_length=1)
    problem_idx: int = Field(ge=1, le=30, strict=True)
    turn_responses: list[NeMoGymResponse] = Field(default_factory=list)
    format_retry_check: FormatRetryResponse | None = None


class MathArenaAIMEVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    extracted_answer: str | None
    parser_warning: int
    valid: bool
    verifier_error: str | None
    candidate_truncated: bool
    review_required: bool
    upstream_revision: str = UPSTREAM_REVISION


class MathArenaAIMEResourcesServer(SimpleResourcesServer):
    """Use the official extraction/equality path without an external judge model."""

    config: MathArenaAIMEConfig
    _parser_semaphore: asyncio.Semaphore = PrivateAttr()
    _parser_python: Path = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._parser_semaphore = asyncio.Semaphore(self.config.parser_max_concurrency)
        self._parser_python = (
            Path(self.config.parser_python)
            if self.config.parser_python
            else ensure_parser_runtime(DIRECTORY / ".parser-venv")
        )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/needs_format_retry")(self.needs_format_retry)
        return app

    async def parse(
        self, text: str, *, strict: bool, expected: int | None = None, output_tokens: int = 0
    ) -> dict[str, Any]:
        """Run CPU-bound symbolic parsing outside the event loop with hard termination."""
        async with self._parser_semaphore:
            payload = json.dumps(
                {
                    "text": text,
                    "strict": strict,
                    "expected_answer": expected,
                    "output_tokens": output_tokens,
                    "memory_limit_mb": self.config.parser_memory_limit_mb,
                    "timeout_seconds": self.config.parser_timeout_seconds,
                }
            ).encode()
            try:
                process = await asyncio.create_subprocess_exec(
                    str(self._parser_python),
                    str(DIRECTORY / "worker.py"),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except OSError as exc:
                return {"valid": False, "verifier_error": f"parser_launch_failed: {exc}"}
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(payload), timeout=self.config.parser_timeout_seconds
                )
                if process.returncode != 0:
                    reason = stderr.decode(errors="replace")[-512:]
                    return {
                        "valid": False,
                        "verifier_error": f"parser_worker_failed: exit {process.returncode}: {reason}",
                    }
                try:
                    result = json.loads(stdout.decode(errors="replace"))
                    if not isinstance(result, dict) or not isinstance(result.get("valid"), bool):
                        raise ValueError("Missing parser validity flag")
                    if result["valid"]:
                        if (
                            type(result.get("parser_warning")) is not int
                            or result["parser_warning"] not in range(4)
                            or type(result.get("needs_format_retry")) is not bool
                            or "extracted_answer" not in result
                            or not isinstance(result["extracted_answer"], str | type(None))
                        ):
                            raise ValueError("Invalid successful parser result fields")
                        if expected is not None and (
                            type(result.get("reward")) not in {int, float} or result["reward"] not in {0.0, 1.0}
                        ):
                            raise ValueError("Invalid parser reward")
                    elif not isinstance(result.get("verifier_error"), str) or not result["verifier_error"]:
                        raise ValueError("Missing parser error diagnosis")
                    return result
                except (ValueError, UnicodeError) as exc:
                    return {"valid": False, "verifier_error": f"parser_worker_failed: invalid JSON result: {exc}"}
            except TimeoutError:
                return {"valid": False, "verifier_error": "parser_timeout: symbolic parsing exceeded deadline"}
            finally:
                if process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await process.communicate()

    async def needs_format_retry(self, body: FormatRetryRequest) -> FormatRetryResponse:
        """Check strict parsing only; this endpoint never receives a reference answer."""
        result = await self.parse(visible_answer(body.response), strict=True)
        return FormatRetryResponse(**result)

    async def verify(self, body: MathArenaAIMEVerifyRequest) -> MathArenaAIMEVerifyResponse:
        if body.format_retry_check is not None and not body.format_retry_check.valid:
            result = {
                "valid": False,
                "verifier_error": body.format_retry_check.verifier_error or "format_check_failed",
            }
        else:
            result = await self.parse(
                visible_answer(body.turn_responses[-1] if body.turn_responses else body.response),
                strict=False,
                expected=body.expected_answer,
                output_tokens=body.response.usage.output_tokens if body.response.usage else 0,
            )
        valid = result["valid"]
        warning = result.get("parser_warning", 3)
        truncated = any(is_truncated(response) for response in [body.response, *body.turn_responses])
        return MathArenaAIMEVerifyResponse(
            **body.model_dump(
                exclude=set(MathArenaAIMEVerifyResponse.model_fields) - set(MathArenaAIMEVerifyRequest.model_fields)
            ),
            reward=result.get("reward", 0.0),
            extracted_answer=result.get("extracted_answer"),
            parser_warning=warning,
            valid=valid,
            verifier_error=result.get("verifier_error"),
            candidate_truncated=truncated,
            review_required=not valid or warning > 0 or truncated,
            mask_sample=not valid,
            failure_kind="matharena_aime:parser_failed" if not valid else None,
            failure_reason=result.get("verifier_error"),
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Compute pass@k by problem and audit repeat coverage."""
        rows = [row for task in tasks for row in task]
        expected_groups = None
        groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            groups[row["language"]][row["task_id"]].append(row)
            if row.get("expected_groups"):
                if expected_groups is not None and row["expected_groups"] != expected_groups:
                    raise ValueError("Inconsistent expected AIME selections across rollouts")
                expected_groups = row["expected_groups"]
        metrics: dict[str, Any] = {
            "matharena_aime/rollouts": len(rows),
            "matharena_aime/expected_selection_known": bool(expected_groups),
            "matharena_aime/expected_repeats": self.config.expected_num_repeats,
        }
        complete_flags, language_scores, all_problem_scores = [], [], []
        pass_metric = f"pass@{self.config.expected_num_repeats}/accuracy"
        for language in sorted(set(groups) | set(expected_groups or {})):
            problems = groups[language]
            values = [row for repeats in problems.values() for row in repeats]
            scored = [row for row in values if row.get("valid", False) and not row.get("mask_sample", False)]
            prefix = f"matharena_aime/language/{language}"
            problem_scores, duplicate_repeats, missing_repeats = [], 0, 0
            repeat_coverage = True
            for repeats in problems.values():
                indices = [row.get(ROLLOUT_INDEX_KEY_NAME) for row in repeats]
                duplicate_repeats += len(indices) - len(set(indices))
                missing_repeats += len(set(range(self.config.expected_num_repeats)) - set(indices))
                repeat_coverage &= len(indices) == self.config.expected_num_repeats and set(indices) == set(
                    range(self.config.expected_num_repeats)
                )
                valid_repeats = [
                    row for row in repeats if row.get("valid", False) and not row.get("mask_sample", False)
                ]
                if valid_repeats:
                    problem_scores.append(float(any(row["reward"] == 1 for row in valid_repeats)))
            expected = (expected_groups or {}).get(language)
            fingerprint = hashlib.sha256(json.dumps(sorted(problems)).encode()).hexdigest()
            selection_complete = bool(
                expected
                and len(problems) == expected["questions"]
                and fingerprint == expected["ids_sha256"]
                and repeat_coverage
                and len(scored) == len(values)
            )
            complete_flags.append(selection_complete)
            review_count = sum(row.get("review_required", True) for row in values)
            metrics.update(
                {
                    f"{prefix}/questions": len(problems),
                    f"{prefix}/rollouts": len(values),
                    f"{prefix}/valid_rollouts": len(scored),
                    f"{prefix}/invalid_rollouts": len(values) - len(scored),
                    f"{prefix}/duplicate_repeats": duplicate_repeats,
                    f"{prefix}/missing_repeats": missing_repeats,
                    f"{prefix}/parser_warnings": sum(row.get("parser_warning", 3) > 0 for row in values),
                    f"{prefix}/truncated_responses": sum(row.get("candidate_truncated", False) for row in values),
                    f"{prefix}/format_retries": sum(row.get("format_retry_count", 0) for row in values),
                    f"{prefix}/review_required": review_count,
                    f"{prefix}/selection_complete": selection_complete,
                }
            )
            if expected:
                missing_questions = max(0, expected["questions"] - len(problems))
                metrics[f"{prefix}/expected_questions"] = expected["questions"]
                metrics[f"{prefix}/missing_questions"] = missing_questions
                metrics[f"{prefix}/missing_repeats"] += missing_questions * self.config.expected_num_repeats
            if problem_scores:
                pass_at_k = 100.0 * sum(problem_scores) / len(problem_scores)
                language_scores.append(pass_at_k)
                all_problem_scores.extend(problem_scores)
                metrics[f"{prefix}/observed_{pass_metric}"] = pass_at_k
                if selection_complete and not review_count:
                    metrics[f"{prefix}/{pass_metric}"] = pass_at_k
        complete = bool(complete_flags) and all(complete_flags)
        provisional = not complete or any(row.get("review_required", True) for row in rows)
        metrics["matharena_aime/selection_complete"] = complete
        metrics["matharena_aime/provisional"] = provisional
        metrics["matharena_aime/invalid_rollouts"] = sum(
            not row.get("valid", False) or row.get("mask_sample", False) for row in rows
        )
        if all_problem_scores:
            observed_pass_at_k = 100.0 * sum(all_problem_scores) / len(all_problem_scores)
            metrics[f"matharena_aime/observed_{pass_metric}"] = observed_pass_at_k
            if not provisional:
                metrics[pass_metric] = observed_pass_at_k
        if language_scores:
            macro_metric = f"macro_{pass_metric}"
            metrics[f"matharena_aime/observed_{macro_metric}"] = sum(language_scores) / len(language_scores)
            if not provisional:
                metrics[f"matharena_aime/{macro_metric}"] = metrics[f"matharena_aime/observed_{macro_metric}"]
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        pass_metric = f"pass@{self.config.expected_num_repeats}/accuracy"
        return {
            key: value
            for key, value in agent_metrics.items()
            if key.startswith("matharena_aime/") or key == pass_metric
        }

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Keep native masked-quality metrics while coverage includes every attempted rollout."""
        metrics = compute_aggregate_metrics(body.verify_responses)
        grouped = defaultdict(list)
        for row in body.verify_responses:
            grouped[row.get(TASK_INDEX_KEY_NAME, 0)].append(row)
        metrics.agent_metrics.update(self.compute_metrics(list(grouped.values())))
        coverage = {key: value for key, value in metrics.key_metrics.items() if key.startswith("coverage/")}
        metrics.key_metrics = self.get_key_metrics(metrics.agent_metrics) | coverage
        return metrics


if __name__ == "__main__":  # pragma: no cover
    MathArenaAIMEResourcesServer.run_webserver()
