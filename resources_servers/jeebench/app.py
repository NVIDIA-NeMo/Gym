# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JEEBench's published scoring formula, with explicitly provisional answer extraction."""

from __future__ import annotations

import hashlib
import json
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.reward_profile import compute_pass_majority_metrics, compute_subset_metrics, highest_k_metrics
from resources_servers.jeebench.scoring import (
    QuestionType,
    compute_score,
    extract_answer,
    normalize_answer,
    visible_answer,
)


UPSTREAM_REVISION = "a9af8bf354887043bd280a617c8716ef54e7aaec"


def response_fingerprint(response: NeMoGymResponse) -> str:
    """Bind human extraction to the exact normalized native response, including reasoning."""
    payload = json.dumps(response.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JeebenchMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str
    language: str
    subject: str
    question_type: QuestionType
    gold: str
    description: str = ""
    index: int | str | None = None

    @field_validator("task_id", "language", "subject")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("JEEBench task_id, language and subject must be nonempty")
        return value

    @model_validator(mode="after")
    def valid_gold(self) -> JeebenchMetadata:
        normalized = normalize_answer(self.gold, self.question_type)
        if normalized in {None, "None"} or (
            self.question_type in {"MCQ", "MCQ(multiple)"} and normalized != self.gold
        ):
            raise ValueError("JEEBench requires a valid canonical gold answer for its question type")
        return self


class JeebenchResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class JeebenchVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: JeebenchMetadata
    reviewed_answer: str | None = None
    answer_review_id: str | None = None
    reviewed_response_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def valid_review(self) -> JeebenchVerifyRequest:
        if self.reviewed_answer is not None:
            if not self.answer_review_id or not self.answer_review_id.strip():
                raise ValueError("A reviewed answer requires an answer_review_id for provenance")
            if normalize_answer(self.reviewed_answer, self.verifier_metadata.question_type) is None:
                raise ValueError("Reviewed answer must be a type-compatible answer or the string 'None'")
            if self.reviewed_response_sha256 is None:
                raise ValueError("A reviewed answer requires reviewed_response_sha256 to bind it to the response")
            if self.reviewed_response_sha256 != response_fingerprint(self.response):
                raise ValueError("reviewed_response_sha256 does not match the response; repeat human extraction")
        elif self.answer_review_id is not None or self.reviewed_response_sha256 is not None:
            raise ValueError("answer_review_id and reviewed_response_sha256 require reviewed_answer")
        return self


class JeebenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: JeebenchMetadata
    extracted_answer: str
    extraction_source: Literal["automatic", "human_review"]
    extraction_method: str
    extraction_failed: bool
    provisional: bool
    exact_match: bool
    candidate_truncated: bool
    upstream_revision: str = UPSTREAM_REVISION


class JeebenchResourcesServer(SimpleResourcesServer):
    """Single-turn, judge-free JEEBench verification with audit-friendly extraction."""

    ray_enabled = False
    config: JeebenchResourcesServerConfig

    async def verify(self, body: JeebenchVerifyRequest) -> JeebenchVerifyResponse:
        meta = body.verifier_metadata
        human = body.reviewed_answer is not None
        if human:
            answer = normalize_answer(body.reviewed_answer, meta.question_type)
            method = "human_review"
        else:
            answer, method = extract_answer(
                visible_answer(body.response), meta.question_type, question=getattr(body, "question", "")
            )
        score = compute_score(meta.gold, answer, meta.question_type)
        return JeebenchVerifyResponse(
            **body.model_dump(
                exclude=set(JeebenchVerifyResponse.model_fields) - set(JeebenchVerifyRequest.model_fields)
            ),
            reward=score,
            extracted_answer=answer,
            extraction_source="human_review" if human else "automatic",
            extraction_method=method,
            extraction_failed=method == "unparsed",
            provisional=not human,
            exact_match=score == 1.0,
            candidate_truncated=(
                body.response.status == "incomplete"
                or bool(body.response.incomplete_details)
                or any(item.type == "message" and item.status == "incomplete" for item in body.response.output)
            ),
            failure_kind="jeebench:unparsed_answer" if method == "unparsed" else None,
            failure_reason="Automatic answer extraction failed; human review required"
            if method == "unparsed"
            else None,
        )

    @staticmethod
    def _score_fn(row: dict[str, Any]) -> dict[str, float]:
        return {
            "jeebench_score": row["reward"],
            "full_credit": float(row["reward"] == 1.0),
            "extraction_failure": float(row["extraction_failed"]),
        }

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Average partial credit using Gym utilities, with language/type/subject breakdowns."""
        metrics = compute_pass_majority_metrics(tasks, score_fn=self._score_fn)[0]
        # Generic majority voting is not JEEBench's per-option self-consistency rule.
        # Repeats here are independent samples, not the authors' CoT+SC protocol.
        for field in ("language", "subject", "question_type"):
            grouped = [[{**row, field: row["verifier_metadata"][field]} for row in task] for task in tasks]
            metrics.update(
                {
                    f"{field}/{key}": value
                    for key, value in compute_subset_metrics(grouped, field, score_fn=self._score_fn).items()
                }
            )
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        return highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]")


if __name__ == "__main__":  # pragma: no cover
    JeebenchResourcesServer.run_webserver()
