# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit, versioned VoiceBench scoring with shared Responses-format agent/model services."""

from __future__ import annotations

import importlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, cast

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    model_serializer,
    model_validator,
)

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.core import JUDGE_UNPARSEABLE, NeMoGymResponse
from nemo_gym.metrics import compute_aggregate_metrics
from resources_servers.voicebench.legacy import replay_rule_rows
from resources_servers.voicebench.metrics import MetricValue, compute_voicebench_metrics
from resources_servers.voicebench.scorers.base import ScoringResult
from resources_servers.voicebench.scorers.open_judge import JudgeSettings, OpenJudgeScorer
from resources_servers.voicebench.scorers.reference_qa import ReferenceQAScorer
from resources_servers.voicebench.serialization import json_safe
from resources_servers.voicebench.task_data import SUBSET_SCORERS, ScoringType, Subset, VoiceBenchMetadata


class VoiceBenchConfig(BaseResourcesServerConfig, JudgeSettings):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    subset: Subset
    scoring_type: ScoringType
    scoring_version: str = "legacy_v1"
    seed: int | None = None
    expected_num_rows: int = Field(gt=0)
    expected_sample_ids_file: str | None = None
    nltk_data_dir: str | None = None

    @model_validator(mode="after")
    def validate_profile(self) -> "VoiceBenchConfig":
        if self.scoring_type != SUBSET_SCORERS[self.subset]:
            raise ValueError(f"subset {self.subset} requires scoring_type={SUBSET_SCORERS[self.subset]}")
        if self.scoring_version != "legacy_v1":
            raise ValueError("supported scoring_version is legacy_v1")
        if self.num_workers not in (None, 1):
            raise ValueError("legacy scoring requires one worker for the shared random stream")
        if self.scoring_type in ("open", "reference_qa"):
            if self.judge_model_server is None or self.judge_prompt_file is None:
                raise ValueError("judge scoring requires judge_model_server and judge_prompt_file")
        return self


class VoiceBenchVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: VoiceBenchMetadata


class VoiceBenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    subset: Subset
    sample_id: str
    canonical_sample_id: str
    scoring_type: ScoringType
    scoring_version: str
    score_valid: bool
    scoring_metrics: dict[str, float]
    candidate_answer: str
    scoring_answer: str
    candidate_status: str | None
    candidate_valid: bool
    candidate_thinking: list[str]
    candidate_incomplete_details: dict[str, str] | None
    status_safe_reward: float

    @model_serializer(when_used="json")
    def serialize_legacy_numbers(self) -> dict[str, JsonValue]:
        # The normal JSON serializer turns nonfinite extras into null before a
        # wrap handler can preserve them. Python mode retains the original floats.
        return cast(dict[str, JsonValue], json_safe(self.model_dump(mode="python")))


def final_answer(response: NeMoGymResponse) -> str:
    """Match the legacy audio model's assembly and last-think extraction."""
    answer = ""
    for item in reversed(response.output):
        if item.type == "message" and item.role == "assistant":
            answer = "\n".join(part.text for part in item.content if part.type == "output_text")
            break
    reasoning = ""
    for item in reversed(response.output):
        if item.type == "reasoning":
            reasoning = "".join(part.text for part in (item.summary or []))
            break

    # VLMEvalKitMcore's audio model wraps parsed reasoning before stripping the
    # whole prediction. Its empty-think fallback also accepts reasoning-only
    # answers and untagged final text. Keep these stages in order: strip() on
    # the raw final would incorrectly remove spaces after the closing tag.
    if (reasoning and not answer.strip()) or (not reasoning and answer.strip() and "</think>" not in answer):
        prediction = f"<think>\n<think></think>{answer.strip() or reasoning.strip()}"
    elif reasoning:
        prediction = f"<think>\n{reasoning}\n</think>\n{answer}"
    else:
        prediction = answer
    prediction = prediction.strip()
    if "</think>" in prediction:
        return prediction.rsplit("</think>", 1)[-1].lstrip("\n")
    return ""


def _expected_ids(path: str, subset: str) -> set[str]:
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        values = json.loads(text)
    else:
        values = []
        for line in text.splitlines():
            if not line.strip():
                continue
            if line.lstrip().startswith("{"):
                row = json.loads(line)
                metadata = row.get("verifier_metadata", row)
                if metadata.get("subset", subset) != subset:
                    raise ValueError("expected sample IDs file contains a different subset")
                values.append(metadata["sample_id"])
            else:
                values.append(line.strip())
    if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
        raise ValueError("expected sample IDs must be a nonempty string list or JSONL metadata rows")
    if len(set(values)) != len(values):
        raise ValueError("expected sample IDs file contains duplicates")
    return set(values)


def _ensure_nltk(data_dir: str | None) -> None:
    """Provision tokenizers only for the selected IFEval handler, reusing cached resources."""
    import nltk

    if data_dir is None:
        data_dir = os.environ.get("NLTK_DATA", "").split(os.pathsep)[0] or str(Path.home() / ".voicebench_nltk_data")
    Path(data_dir).mkdir(parents=True, exist_ok=True, mode=0o700)
    nltk.data.path.insert(0, data_dir)
    for name in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{name}")
        except LookupError:
            if not nltk.download(name, download_dir=data_dir, quiet=True):
                raise RuntimeError(f"could not provision NLTK tokenizer {name} in {data_dir}")


class VoiceBenchServer(SimpleResourcesServer):
    ray_enabled = False
    config: VoiceBenchConfig
    _judge: OpenJudgeScorer | None = PrivateAttr(default=None)
    _expected_sample_ids: set[str] | None = PrivateAttr(default=None)
    _rule_rng: random.Random = PrivateAttr(default_factory=random.Random)

    def model_post_init(self, context: object) -> None:
        self._rule_rng = random.Random(self.config.seed)
        if self.config.expected_sample_ids_file:
            self._expected_sample_ids = _expected_ids(self.config.expected_sample_ids_file, self.config.subset)
            if len(self._expected_sample_ids) != self.config.expected_num_rows:
                raise ValueError("expected_sample_ids_file count differs from expected_num_rows")
        if self.config.scoring_type == "open":
            self._judge = OpenJudgeScorer(self.config, self.server_client)
        elif self.config.scoring_type == "reference_qa":
            self._judge = ReferenceQAScorer(self.config, self.server_client)
        elif self.config.scoring_type == "ifeval":
            _ensure_nltk(self.config.nltk_data_dir)
        super().model_post_init(context)

    async def verify(self, body: VoiceBenchVerifyRequest) -> VoiceBenchVerifyResponse:
        if body.verifier_metadata.subset != self.config.subset:
            raise ValueError(f"resource {self.config.subset} cannot score subset {body.verifier_metadata.subset}")
        answer = final_answer(body.response)
        metadata = body.verifier_metadata.model_dump(mode="json")
        identity = f"{self.config.subset}/{body.verifier_metadata.sample_id}"
        # pandas.read_excel converts a blank prediction to NaN, then the old
        # wrapper sends str(NaN) to every evaluator and judge.
        scoring_answer = answer if answer != "" else "nan"
        if self._judge is not None:
            scored = await self._judge.score(scoring_answer, metadata)
        else:
            module = importlib.import_module(f"resources_servers.voicebench.scorers.{self.config.scoring_type}")
            scored: ScoringResult = module.score(scoring_answer, metadata, rng=self._rule_rng)
        candidate_valid = (
            body.response.status == "completed" and body.response.incomplete_details is None and bool(answer.strip())
        )
        thinking = [
            part.text for item in body.response.output if item.type == "reasoning" for part in (item.summary or [])
        ]
        return VoiceBenchVerifyResponse.model_validate(
            body.model_dump()
            | scored.details
            | {
                "reward": scored.reward,
                "subset": self.config.subset,
                "sample_id": body.verifier_metadata.sample_id,
                "canonical_sample_id": identity,
                "scoring_type": self.config.scoring_type,
                "scoring_version": self.config.scoring_version,
                "scoring_seed": self.config.seed,
                "score_valid": scored.valid,
                "scoring_metrics": scored.metrics,
                "candidate_answer": answer,
                "scoring_answer": scoring_answer,
                "candidate_status": body.response.status,
                "candidate_valid": candidate_valid,
                "candidate_thinking": thinking,
                "candidate_incomplete_details": body.response.incomplete_details.model_dump()
                if body.response.incomplete_details is not None
                else None,
                "status_safe_reward": scored.reward if candidate_valid and scored.valid else 0.0,
                "failure_kind": JUDGE_UNPARSEABLE
                if not scored.valid and self._judge
                else "voicebench:invalid_score"
                if not scored.valid
                else None,
                "failure_reason": "; ".join(scored.details.get("judge_parse_errors", []))
                if not scored.valid
                else None,
            }
        )

    def compute_metrics(self, tasks: Sequence[Sequence[Mapping[str, object]]]) -> dict[str, MetricValue]:
        """Aggregate subset scores, sample coverage, and judge diagnostics."""
        return compute_voicebench_metrics(
            replay_rule_rows(tasks, scoring_type=self.config.scoring_type, seed=self.config.seed),
            scoring_type=self.config.scoring_type,
            expected_num_rows=self.config.expected_num_rows,
            expected_sample_ids=self._expected_sample_ids,
            judge_enabled=self._judge is not None,
            judge_max_concurrency=self.config.judge_max_concurrency,
            judge_peak_inflight=self._judge.peak_inflight if self._judge is not None else 0,
        )

    def get_key_metrics(self, agent_metrics: dict[str, MetricValue]) -> dict[str, MetricValue]:
        return {key: value for key, value in agent_metrics.items() if key.startswith("voicebench/")}

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        # Avoid replaying the rule RNG twice through the parent custom-metrics hook.
        aggregated = compute_aggregate_metrics(body.verify_responses, get_key_metrics_fn=self.get_key_metrics)
        # Coverage describes received rows; collector sidecar failures are not supplied.
        metrics = self.compute_metrics([body.verify_responses])
        aggregated.agent_metrics.update(metrics)
        aggregated.key_metrics.update(self.get_key_metrics(metrics))
        return aggregated


if __name__ == "__main__":
    VoiceBenchServer.run_webserver()
