# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decision Index (DI 0.3) resources server.

``verify()`` turns one model output into a Decision Index answer and gives the row a binary reward;
``compute_metrics()`` scores the whole run with the Decision Index kit (public index, chance-corrected, coverage-
adjusted, official area weights). See ``verifier.py`` for the two answer modes and README.md for the contract.
"""

from pathlib import Path
from typing import Any, ClassVar, Dict, Literal, Optional

from pydantic import ConfigDict, Field

from nemo_gym.config.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.server_runtime.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture
from resources_servers.decision_index.verifier import (
    ENGINE_ENVELOPE_KEY,
    OK,
    engine_answers,
    flatten_scores,
    llm_answers,
    row_reward,
    score_index,
    scoring_row,
)


class DecisionIndexResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    answer_mode: Literal["decision", "llm", "auto"] = Field(
        default="auto",
        description=(
            "decision: read the decision_engine_model envelope (option probabilities); llm: parse a chat reply; "
            "auto: decision when the reply is an envelope, else llm. Benchmark configs set decision or llm, so the "
            "two modes are run and reported separately."
        ),
    )


class DecisionIndexVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Dict[str, Any]


class DecisionIndexVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    di_mode: str
    di_status: str
    di_error: Optional[str] = None
    di_scored_questions: int = 0
    di_score_row: Dict[str, Any]
    di_result: Dict[str, Any]


class DecisionIndexVerifier:
    """Stateless verification; the server class below adds the run-level Decision Index."""

    async def verify(self, body: DecisionIndexVerifyRequest) -> DecisionIndexVerifyResponse:
        di_row = body.verifier_metadata["di_row"]
        questions = di_row["questions"]
        text = body.response.output_text or ""
        # The fixture builds this class without a config; served instances always have one.
        mode = getattr(getattr(self, "config", None), "answer_mode", "auto")
        if mode == "auto":
            mode = "decision" if text.lstrip().startswith('{"' + ENGINE_ENVELOPE_KEY + '"') else "llm"
        wall_ms, engine_failed = None, False
        if mode == "decision":
            status, response, error, wall_ms, engine_failed = engine_answers(text, questions)
        else:
            status, response, error = llm_answers(text, questions)
        reward, scored = row_reward(di_row, response if status == OK else None)
        e = di_row["_evaluation"]
        result = {"run_id": e["run_id"], "catalog_id": e["catalog_id"], "status": status, "total_wall_ms": float(wall_ms or 0.0)}
        if status == OK:
            result["response"] = response
        extra: Dict[str, Any] = {}
        if engine_failed:
            # The engine call failed (transport or server error), not the policy's answer. The kit still counts the
            # request as wrong (di_result keeps status "error"), but the row leaves Gym's mean reward.
            extra = {"mask_sample": True, "failure_kind": "decision_index:engine_error", "failure_reason": error}
        payload = body.model_dump(exclude={"verifier_metadata"})
        return DecisionIndexVerifyResponse(
            **payload,
            **extra,
            reward=reward if status == OK else 0.0,
            di_mode=mode,
            di_status=status,
            di_error=error,
            di_scored_questions=scored,
            di_score_row=scoring_row(di_row),
            di_result=result,
        )


class DecisionIndexResourcesServer(DecisionIndexVerifier, SimpleResourcesServer):
    config: DecisionIndexResourcesServerConfig

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Gym's reward statistics, plus the Decision Index scored by the kit over every rollout.

        The index is computed here rather than in ``compute_metrics`` because that hook sees only unmasked rollouts,
        and the kit counts a request whose engine call failed as answered wrong (coverage), not as absent.
        """
        result = await super().aggregate_metrics(body)
        index = flatten_scores(score_index(body.verify_responses, engine=f"gym-{self.config.answer_mode}"))
        result.agent_metrics.update(index)
        result.key_metrics.update(self.get_key_metrics(index))
        return result

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        keys = ("decision_index", "decision_index_raw", "decision_index_requests", "decision_index_full_suite")
        out = {k: agent_metrics[k] for k in keys if k in agent_metrics}
        out.update({k: v for k, v in agent_metrics.items() if k.startswith("decision_index_area/")})
        if "mean/reward" in agent_metrics:
            out["mean/reward"] = agent_metrics["mean/reward"]
        return out


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=DecisionIndexVerifier,
    request_model=DecisionIndexVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    DecisionIndexResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = DecisionIndexResourcesServer.run_webserver()  # noqa: F401
