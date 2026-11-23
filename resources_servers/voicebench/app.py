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
from typing import ClassVar, Literal, cast
from unittest.mock import MagicMock

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    model_serializer,
    model_validator,
)

from nemo_gym import CACHE_DIR
from nemo_gym.client import ServerClient
from nemo_gym.config import (
    CACHE_DIR_KEY_NAME,
    AggregateMetrics,
    AggregateMetricsRequest,
    get_global_config_dict,
    maybe_get_global_config_dict,
)
from nemo_gym.core import JUDGE_UNPARSEABLE, NeMoGymResponse
from nemo_gym.metrics import compute_aggregate_metrics
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.verification import VerifierFixture
from resources_servers.voicebench.legacy import replay_rule_rows
from resources_servers.voicebench.metrics import MetricValue, compute_voicebench_metrics
from resources_servers.voicebench.scorers.base import ScoringResult
from resources_servers.voicebench.scorers.open_judge import JudgeLimiter, JudgeSettings, OpenJudgeScorer
from resources_servers.voicebench.scorers.reference_qa import ReferenceQAScorer
from resources_servers.voicebench.serialization import json_safe
from resources_servers.voicebench.task_data import SUBSET_SCORERS, ScoringType, Subset, VoiceBenchMetadata


ALL_SUBSETS = "all"


class VoiceBenchConfig(BaseResourcesServerConfig, JudgeSettings):
    """One runtime scores one subset, or every subset in ``expected_rows`` when ``subset`` is ``all``.

    In ``all`` mode each row's ``verifier_metadata.subset`` selects its scorer, the open and reference-QA judges take
    ``judge_prompt_files`` and share one concurrency budget, and metrics are reported per subset under
    ``voicebench/<subset>/``.
    """

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    subset: Subset | Literal["all"]
    scoring_type: ScoringType | None = None
    scoring_version: str = "legacy_v1"
    seed: int | None = None
    expected_num_rows: int | None = Field(default=None, gt=0)
    expected_rows: dict[Subset, int] | None = None
    expected_sample_ids_file: str | None = None
    nltk_data_dir: str | None = None
    judge_prompt_files: dict[ScoringType, str] | None = None
    # "vlmeval": reference audio adapter answer (last message, empty answer falls back to reasoning, blank is "nan").
    # "message": the last assistant message's text as returned, so a response without one is scored as empty.
    answer_extraction: Literal["vlmeval", "message"] = "vlmeval"

    @property
    def subsets(self) -> list[str]:
        return list(self.expected_rows or {}) if self.subset == ALL_SUBSETS else [self.subset]

    @model_validator(mode="after")
    def validate_profile(self) -> "VoiceBenchConfig":
        if self.scoring_version != "legacy_v1":
            raise ValueError("supported scoring_version is legacy_v1")
        if self.num_workers not in (None, 1):
            raise ValueError("legacy scoring requires one worker for the shared random stream")
        if self.subset == ALL_SUBSETS:
            if self.scoring_type is not None or self.expected_num_rows is not None or self.expected_sample_ids_file:
                raise ValueError("subset=all takes expected_rows, not scoring_type/expected_num_rows/sample IDs")
            if not self.expected_rows or any(count <= 0 for count in self.expected_rows.values()):
                raise ValueError("subset=all requires positive expected_rows per subset")
            judged = {SUBSET_SCORERS[name] for name in self.expected_rows} & {"open", "reference_qa"}
            if judged:
                if self.judge_model_server is None or not judged <= set(self.judge_prompt_files or {}):
                    raise ValueError(f"judge scoring requires judge_model_server and judge_prompt_files for {judged}")
            return self
        if self.scoring_type is None or self.expected_num_rows is None:
            raise ValueError("a single subset requires scoring_type and expected_num_rows")
        if self.scoring_type != SUBSET_SCORERS[self.subset]:
            raise ValueError(f"subset {self.subset} requires scoring_type={SUBSET_SCORERS[self.subset]}")
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


def message_text(response: NeMoGymResponse) -> str:
    """The last assistant message's text, as returned; a response that never reached its final message is empty.

    Like ``final_answer`` this reads only the last assistant message, so an earlier commentary message is not fused
    into the answer. Its ``output_text`` parts are joined without a separator, as returned.
    """
    for item in reversed(response.output):
        if item.type == "message" and item.role == "assistant":
            return "".join(part.text for part in item.content if part.type == "output_text")
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
        # Consult the global config only when this process already has one: a bare process never parses a CLI.
        cache_dir = (maybe_get_global_config_dict() or {}).get(CACHE_DIR_KEY_NAME) or CACHE_DIR
        data_dir = os.environ.get("NLTK_DATA", "").split(os.pathsep)[0] or str(
            Path(cache_dir) / "voicebench" / "nltk_data"
        )
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
    _judges: dict[str, OpenJudgeScorer] = PrivateAttr(default_factory=dict)
    _expected_sample_ids: set[str] | None = PrivateAttr(default=None)
    _rule_rng: random.Random = PrivateAttr(default_factory=random.Random)

    def model_post_init(self, context: object) -> None:
        self._rule_rng = random.Random(self.config.seed)
        if self.config.expected_sample_ids_file:
            self._expected_sample_ids = _expected_ids(self.config.expected_sample_ids_file, self.config.subset)
            if len(self._expected_sample_ids) != self.config.expected_num_rows:
                raise ValueError("expected_sample_ids_file count differs from expected_num_rows")
        scoring_types = {SUBSET_SCORERS[name] for name in self.config.subsets}
        if scoring_types & {"open", "reference_qa"}:
            self._require_judge_endpoint()
        limiter = JudgeLimiter(self.config.judge_max_concurrency)
        for scoring_type, scorer_class in (("open", OpenJudgeScorer), ("reference_qa", ReferenceQAScorer)):
            if scoring_type in scoring_types:
                settings = self.config
                if self.config.subset == ALL_SUBSETS:
                    settings = self.config.model_copy(
                        update={"judge_prompt_file": self.config.judge_prompt_files[scoring_type]}
                    )
                self._judges[scoring_type] = scorer_class(settings, self.server_client, limiter)
        if self.config.subset != ALL_SUBSETS:
            self._judge = self._judges.get(self.config.scoring_type)
        if "ifeval" in scoring_types:
            _ensure_nltk(self.config.nltk_data_dir)
        super().model_post_init(context)

    def _require_judge_endpoint(self) -> None:
        """Fail at startup when the judge endpoint is still the judge config's ``unset.local`` placeholder."""
        server = get_global_config_dict().get(self.config.judge_model_server.name) or {}
        for instance in (server.get(self.config.judge_model_server.type) or {}).values():
            if isinstance(instance, Mapping) and "unset.local" in str(instance.get("openai_base_url", "")):
                raise ValueError(
                    f"The judge model server {self.config.judge_model_server.name!r} has no endpoint: "
                    "set judge_base_url, judge_api_key and judge_model_name in env.yaml"
                )

    async def verify(self, body: VoiceBenchVerifyRequest) -> VoiceBenchVerifyResponse:
        subset = body.verifier_metadata.subset
        if subset not in self.config.subsets:
            raise ValueError(f"resource {self.config.subset} cannot score subset {subset}")
        scoring_type = SUBSET_SCORERS[subset]
        if self.config.answer_extraction == "message":
            answer = scoring_answer = message_text(body.response)
        else:
            answer = final_answer(body.response)
            # pandas.read_excel converts a blank prediction to NaN, then the
            # reference wrapper sends str(NaN) to every evaluator and judge.
            scoring_answer = answer if answer != "" else "nan"
        metadata = body.verifier_metadata.model_dump(mode="json")
        identity = f"{subset}/{body.verifier_metadata.sample_id}"
        judge = self._judges.get(scoring_type)
        if judge is not None:
            scored = await judge.score(scoring_answer, metadata)
        else:
            module = importlib.import_module(f"resources_servers.voicebench.scorers.{scoring_type}")
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
                "subset": subset,
                "sample_id": body.verifier_metadata.sample_id,
                "canonical_sample_id": identity,
                "scoring_type": scoring_type,
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
                if not scored.valid and judge
                else "voicebench:invalid_score"
                if not scored.valid
                else None,
                "failure_reason": "; ".join(scored.details.get("judge_parse_errors", []))
                if not scored.valid
                else None,
            }
        )

    def _subset_metrics(self, tasks: Sequence[Sequence[Mapping[str, object]]], subset: str) -> dict[str, MetricValue]:
        scoring_type = SUBSET_SCORERS[subset]
        judge = self._judges.get(scoring_type)
        multi = self.config.subset == ALL_SUBSETS
        metrics = compute_voicebench_metrics(
            replay_rule_rows(tasks, scoring_type=scoring_type, seed=self.config.seed),
            scoring_type=scoring_type,
            expected_num_rows=self.config.expected_rows[subset] if multi else self.config.expected_num_rows,
            expected_sample_ids=self._expected_sample_ids,
            judge_enabled=judge is not None,
            judge_max_concurrency=self.config.judge_max_concurrency,
            judge_peak_inflight=judge.peak_inflight if judge is not None else 0,
        )
        if not multi:
            return metrics
        return {key.replace("voicebench/", f"voicebench/{subset}/", 1): value for key, value in metrics.items()}

    def compute_metrics(self, tasks: Sequence[Sequence[Mapping[str, object]]]) -> dict[str, MetricValue]:
        """Aggregate subset scores, sample coverage, and judge diagnostics (per subset in ``all`` mode)."""
        if self.config.subset != ALL_SUBSETS:
            return self._subset_metrics(tasks, self.config.subset)
        by_subset: dict[str, list[Mapping[str, object]]] = {name: [] for name in self.config.subsets}
        for row in (row for task in tasks for row in task):
            name = row.get("subset") or (row.get("verifier_metadata") or {}).get("subset")
            if name not in by_subset:
                raise ValueError(f"cannot aggregate a row of subset {name!r}")
            by_subset[name].append(row)
        metrics: dict[str, MetricValue] = {}
        for name, rows in by_subset.items():
            try:
                metrics |= self._subset_metrics([rows], name)
            except ValueError as error:
                # A subset refuses to aggregate when one of its rows could not be scored (for example an unparseable
                # judge grade). That is that subset's failure, not the whole benchmark's: keep every other subset.
                metrics[f"voicebench/{name}/aggregation_error"] = str(error)
        return metrics

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


def _fixture_server() -> VoiceBenchServer:
    """A rule-scored (MCQ) server: the fixture needs no judge or network."""
    return VoiceBenchServer(
        config=VoiceBenchConfig(
            name="voicebench",
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            subset="openbookqa",
            scoring_type="mcq",
            expected_num_rows=1,
            seed=0,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=_fixture_server,
    request_model=VoiceBenchVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    VoiceBenchServer.run_webserver()
