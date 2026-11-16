# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym verifier for audio benchmarks with corpus-level scoring."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Optional

from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, ModelServerRef, get_global_config_dict
from nemo_gym.core import NeMoGymChatCompletion
from nemo_gym.metrics import compute_aggregate_metrics
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.verification import VerifierFixture, call_judge
from resources_servers.audio_eval._scoring.text import cleanup_generation_loops
from resources_servers.audio_eval.judges import (
    TEMPLATE_JUDGE,
    JudgeHooks,
    judge_kind,
    template_judgement,
    template_request,
)
from resources_servers.audio_eval.profiles import load_profiles, resolve_repo_path, source_hash
from resources_servers.audio_eval.score import (
    assistant_text,
    evaluator_settings,
    flatten_metrics,
    generation_failure,
    locate,
    score_rows,
)
from resources_servers.audio_eval.verifier_fixture import create_server


GENERATION_FAILED = "audio_eval:generation_failed"


class AudioEvalConfig(BaseResourcesServerConfig):
    """Server settings: the benchmark's profile file and, for judged profiles, the judge model server."""

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    profile_contracts: str
    # Chat-completions judge for the judged profiles (a judge_prompt_template or an evaluator's JUDGE hooks).
    judge_model_server: Optional[ModelServerRef] = None
    judge_max_concurrency: int = Field(default=40, gt=0)


class AudioEvalVerifyRequest(BaseVerifyRequest):
    """A rollout of one prepared row; the profile, sample identity and source row come from the row."""

    model_config = ConfigDict(extra="allow")
    benchmark_profile: str
    sample_id: str
    ng_source_row: dict
    ng_source_row_sha256: str


class AudioEvalVerifyResponse(BaseVerifyResponse):
    """The request plus the extracted ``generation``, the judge fields and the placeholder reward."""

    model_config = ConfigDict(extra="allow")


@dataclass(frozen=True)
class Profile:
    """A profile as the server uses it: its contract, whether to clean repeated output, and its judge."""

    contract: dict
    cleanup_loops: bool
    judge: str | None
    hooks: JudgeHooks | None = None


class AudioEvalResourcesServer(SimpleResourcesServer):
    """Verifies each rollout and scores whole profiles at corpus level in ``aggregate_metrics``."""

    config: AudioEvalConfig
    _profiles: dict[str, Profile] = PrivateAttr(default_factory=dict)
    _judge_slots: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        """Load and validate the profile contracts; fail at startup for a judged profile without a usable judge."""
        if not self.config.profile_contracts:
            raise ValueError("profile_contracts must name a benchmark's profiles.json")
        contracts = load_profiles(resolve_repo_path(self.config.profile_contracts))["profiles"]
        for name, contract in contracts.items():
            evaluator, settings = evaluator_settings(contract)
            hooks = None
            if evaluator is not None and "judge_prompt_template" not in contract:
                hooks = getattr(locate("evaluators", evaluator)[0], "JUDGE", None)
            judge = judge_kind(contract, hooks, settings)
            if judge is not None and self.config.judge_model_server is None:
                raise ValueError(f"Profile {name} is judged: set judge_model_server")
            if judge is not None:
                self._require_judge_endpoint(name)
            self._profiles[name] = Profile(contract, bool(settings.get("cleanup_generation_loops")), judge, hooks)
        self._judge_slots = asyncio.Semaphore(self.config.judge_max_concurrency)
        super().model_post_init(context)

    def _require_judge_endpoint(self, profile_name: str) -> None:
        """Fail at startup when the benchmark's judge endpoint is still the config's ``unset.local`` placeholder."""
        server = get_global_config_dict().get(self.config.judge_model_server.name) or {}
        for instance in (server.get(self.config.judge_model_server.type) or {}).values():
            if isinstance(instance, Mapping) and "unset.local" in str(instance.get("openai_base_url", "")):
                raise ValueError(
                    f"Profile {profile_name} is judged but the judge model server "
                    f"{self.config.judge_model_server.name!r} has no endpoint: set judge_base_url, "
                    "judge_api_key and judge_model_name in env.yaml"
                )

    async def _judge(self, profile: Profile, row: dict) -> dict:
        """The row fields a judge call contributes (empty for judge-free profiles)."""
        if profile.judge is None:
            return {}
        if profile.judge == TEMPLATE_JUDGE:
            request = template_request(row, profile.contract)
        else:
            request = profile.hooks.request(row)
        async with self._judge_slots:
            completion = await call_judge(
                self.server_client,
                server_name=self.config.judge_model_server.name,
                url_path="/v1/chat/completions",
                json=request,
                response_model=NeMoGymChatCompletion,
            )
        if profile.judge == TEMPLATE_JUDGE:
            return {"judgement": template_judgement(completion)}
        return profile.hooks.fields(completion, row)

    async def verify(self, body: AudioEvalVerifyRequest) -> AudioEvalVerifyResponse:
        """Check the request against its source row, extract the generation and run the judge call, if any.

        The reward is a placeholder of 0.0; a generation that is neither complete nor truncated at the output limit
        is masked. The corpus metrics come from :meth:`compute_metrics`.
        """
        profile = self._profiles.get(body.benchmark_profile)
        if profile is None:
            raise ValueError(f"Profile is not configured for this benchmark: {body.benchmark_profile}")
        if source_hash(body.ng_source_row) != body.ng_source_row_sha256:
            raise ValueError("Immutable source row hash mismatch")
        result = body.model_dump(mode="json")
        response = result["response"]
        failure = generation_failure(response)
        if failure is not None:
            return AudioEvalVerifyResponse.model_validate(
                result
                | {"reward": 0.0, "mask_sample": True, "failure_kind": GENERATION_FAILED, "failure_reason": failure}
            )
        generation = assistant_text(response)
        row = dict(body.ng_source_row)
        row.update(
            sample_id=body.sample_id,
            benchmark_profile=body.benchmark_profile,
            generation=cleanup_generation_loops(generation) if profile.cleanup_loops else generation,
        )
        judged = await self._judge(profile, row)
        # Corpus metrics are computed in aggregate_metrics. The reward is a placeholder
        # (a completed-generation indicator), not a task-quality score.
        return AudioEvalVerifyResponse.model_validate(
            result | {"generation": generation, "audio_judge": judged, "reward": 0.0}
        )

    def compute_metrics(self, tasks: list[list[dict]]) -> dict:
        """Corpus metrics per profile as flat ``audio/<profile>/pass@1/<metric>`` keys.

        Masked rows are skipped; every task must have exactly one rollout.
        """
        grouped: dict[str, list[dict]] = defaultdict(list)
        identities: set[str] = set()
        for task in tasks:
            if len(task) != 1:
                raise ValueError("Audio corpus scoring requires exactly one rollout per source task (num_repeats=1)")
            verified = task[0]
            if verified.get("mask_sample"):
                continue
            profile = verified["benchmark_profile"]
            if profile not in self._profiles:
                raise ValueError(f"Unknown audio benchmark profile: {profile}")
            identity = verified["sample_id"]
            if identity in identities or source_hash(verified["ng_source_row"]) != verified["ng_source_row_sha256"]:
                raise ValueError("Duplicate identity or changed audio source row")
            identities.add(identity)
            row = dict(verified["ng_source_row"])
            row.update(sample_id=identity, benchmark_profile=profile, generation=verified["generation"])
            row.update(verified.get("audio_judge") or {})
            grouped[profile].append(row)
        metrics = {}
        for profile, rows in grouped.items():
            _, results = score_rows(rows, profile=profile, contract=self._profiles[profile].contract)
            metrics.update(flatten_metrics(results))
        return metrics

    def get_key_metrics(self, agent_metrics: dict) -> dict:
        """The headline metrics: every ``audio/...`` entry."""
        return {name: value for name, value in agent_metrics.items() if name.startswith("audio/")}

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Score the verified rollouts in a worker thread."""
        # Corpus BLEU and session merging can take minutes; keep the event loop free.
        return await asyncio.to_thread(
            compute_aggregate_metrics,
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
        )


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_server,
    request_model=AudioEvalVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "fixture_cases.jsonl",
)


if __name__ == "__main__":
    AudioEvalResourcesServer.run_webserver()
