# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EQ-Bench 4 resources server: judges a policy's roleplay conversation against published transcripts.

The ``eqbench4_agent`` runs the conversation between the policy and the simulated persona and sends the transcript
here. Each row names published leaderboard transcripts of the same scenario. For each one the pairwise judge reads
both transcripts twice, once in each order, and the comparison's fraction for the policy follows from the weighted
margins. The reward is the policy's mean fraction. ``compute_metrics`` adds the policy's comparisons to the published
comparison graph and fits Bradley-Terry ratings, which places the policy on the EQ-Bench 4 leaderboard scale.

A conversation the policy or the persona could not finish is left out of the score, as in EQ-Bench 4, and
``aggregate_metrics`` reports how many conversations were left out and why.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Literal

from aiohttp import ClientTimeout
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, ModelServerRef
from nemo_gym.core import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
    observe_llm_judge_grading,
)
from nemo_gym.verification import (
    JUDGE_FAILURES,
    JudgeError,
    VerifierFixture,
    call_judge,
    record_parsed_judge_verdict,
    record_rejected_judge_reply,
)
from resources_servers.eqbench4.judging import (
    NO_FINAL_STATE,
    PromptParts,
    combine_readings,
    pairwise_transcript,
    parse_json_object,
    parse_trait_scores,
    respondent_codes,
    strip_thinking,
    trait_transcript,
    weighted_margins,
)
from resources_servers.eqbench4.metrics import (
    dropped_conversation_alerts,
    dropped_conversation_metrics,
    eqbench4_metrics,
    eqbench4_repeat_metrics,
    key_metrics,
)
from resources_servers.eqbench4.rating import Outcome
from resources_servers.eqbench4.task_data import Opponent, SlottedPrompt, TranscriptMessage
from resources_servers.eqbench4.verifier_fixture import create_fixture_server, invoke_with_fixture_judge


logger = logging.getLogger(__name__)

GYM_ROOT = Path(__file__).resolve().parents[2]
ConversationStatus = Literal["completed", "persona_ended", "persona_failed", "policy_failed"]
Direction = Literal["policy_first", "opponent_first"]
PolicyStopReason = Literal["context_limit", "output_budget", "content_filter", "empty_reply"]
# How a conversation the policy stopped is labelled; a context-window overflow uses Gym's shared kind.
POLICY_STOP_FAILURE_KINDS: dict[str, str] = {
    "context_limit": "model_context_limit",
    "output_budget": "eqbench4:policy_output_budget",
    "content_filter": "eqbench4:policy_content_filter",
    "empty_reply": "eqbench4:policy_empty_reply",
}


class EQBench4ResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    # The pairwise judge panel. Comparison n of the benchmark goes to judge n modulo the panel size.
    judge_model_servers: list[ModelServerRef] = Field(min_length=1)
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    # Readings of one comparison order that return no parseable verdict are asked again up to this many times in all.
    judge_max_attempts: int = Field(default=3, ge=1)
    # The informational behavioural-trait judge; null skips it.
    trait_judge_model_server: ModelServerRef | None = None
    trait_judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    trait_judge_system_message: str
    judge_concurrency: int = Field(default=16, ge=1)
    judge_timeout_secs: float = Field(default=1800, gt=0)
    # Published comparisons written by the benchmark's prepare script, relative to the Gym root unless absolute.
    reference_comparisons_path: str
    rating_bootstrap_resamples: int = Field(default=200, ge=1)


class EQBench4RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    scenario_id: str
    scenario_index: int = Field(ge=0)
    source_type: str = "generated"
    num_turns: int = Field(ge=1)
    pairwise_prompt: SlottedPrompt
    pairwise_dimensions: dict[str, float] = Field(min_length=1)
    trait_prompt: SlottedPrompt
    trait_assistant_label: str
    opponents: list[Opponent] = Field(min_length=1)


class EQBench4VerifyRequest(EQBench4RunRequest, BaseVerifyRequest):
    # Written by eqbench4_agent.
    transcript: list[TranscriptMessage] = Field(default_factory=list)
    conversation_status: ConversationStatus
    conversation_error: str | None = None
    # Why the policy gave no visible reply, for a conversation it stopped.
    policy_stop_reason: PolicyStopReason | None = None


class JudgeReading(BaseModel):
    """One judge reading of a comparison in one presentation order."""

    direction: Direction
    judge_model_server: str
    first_code: str
    second_code: str
    verdicts: dict[str, Any]
    policy_margin: float
    opponent_margin: float
    judge_text: str
    attempts: int


class OpponentComparison(BaseModel):
    opponent: str
    comparison_index: int
    # The policy's share of the comparison, in [0, 1].
    fraction: float
    policy_margin: float
    opponent_margin: float
    readings: list[JudgeReading]
    # A presentation order whose replies held no verdict, and why; the comparison then rests on the other order.
    unreadable_orders: dict[str, str] = Field(default_factory=dict)


class EQBench4VerifyResponse(EQBench4VerifyRequest, BaseVerifyResponse):
    # Whether the conversation was judged. A conversation left out of the score was never judged.
    evaluation_completed: bool = True
    comparisons: list[OpponentComparison] = Field(default_factory=list)
    policy_turns: int = 0
    trait_scores: dict[str, float] | None = None
    trait_judge_text: str | None = None
    trait_judge_error: str | None = None


def response_text(response: NeMoGymResponse) -> str:
    """The visible text of a model response, without reasoning."""
    parts = [
        content.text
        for item in response.output
        if item.type == "message"
        for content in item.content
        if content.type == "output_text"
    ]
    return strip_thinking("".join(parts))


class EQBench4ResourcesServer(SimpleResourcesServer):
    config: EQBench4ResourcesServerConfig
    _judge_semaphore: asyncio.Semaphore = PrivateAttr()
    _reference_outcomes: list[Outcome] | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_concurrency)

    async def verify(self, body: EQBench4VerifyRequest) -> EQBench4VerifyResponse:
        fields = body.model_dump()
        policy_turns = sum(message.role == "assistant" for message in body.transcript)
        if body.conversation_status == "persona_failed":
            # The simulator, not the policy, broke the conversation: leave the row out of the score.
            return EQBench4VerifyResponse(
                **fields,
                reward=0.0,
                policy_turns=policy_turns,
                mask_sample=True,
                evaluation_completed=False,
                failure_kind="eqbench4:persona_failed",
                failure_reason=body.conversation_error or "the persona simulator gave no valid reply",
            )
        if body.conversation_status == "policy_failed" or policy_turns == 0:
            # As in EQ-Bench 4, an unfinished conversation is left out of the score; aggregate_metrics reports it.
            reason = body.policy_stop_reason or "empty_reply"
            return EQBench4VerifyResponse(
                **(fields | {"policy_stop_reason": reason}),
                reward=0.0,
                policy_turns=policy_turns,
                mask_sample=True,
                evaluation_completed=False,
                failure_kind=POLICY_STOP_FAILURE_KINDS[reason],
                failure_reason=body.conversation_error or "the policy gave no visible reply",
            )

        async def grade() -> EQBench4VerifyResponse:
            # Every judge call of the verify belongs to its one grading operation, as Gym's trial gate requires, the
            # informational trait judge included: its reply is part of the graded result, though never of the reward.
            traits = (
                asyncio.ensure_future(self._judge_traits(body))
                if self.config.trait_judge_model_server is not None
                else None
            )
            tasks = [asyncio.ensure_future(self._compare(body, opponent)) for opponent in body.opponents]
            try:
                comparisons = await asyncio.gather(*tasks)
                trait_fields = await traits if traits is not None else {}
            except BaseException:
                pending = tasks + ([traits] if traits is not None else [])
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise
            return EQBench4VerifyResponse(
                **fields,
                reward=sum(comparison.fraction for comparison in comparisons) / len(comparisons),
                policy_turns=policy_turns,
                comparisons=comparisons,
                **trait_fields,
            )

        return await observe_llm_judge_grading(grade)

    async def _compare(self, body: EQBench4VerifyRequest, opponent: Opponent) -> OpponentComparison:
        judges = self.config.judge_model_servers
        judge = judges[opponent.comparison_index % len(judges)].name
        policy_side = (pairwise_transcript([m.model_dump() for m in body.transcript]), NO_FINAL_STATE)
        opponent_side = (pairwise_transcript([m.model_dump() for m in opponent.transcript]), opponent.final_state)
        results = await asyncio.gather(
            self._read(body, opponent, judge, "policy_first", policy_side, opponent_side),
            self._read(body, opponent, judge, "opponent_first", opponent_side, policy_side),
        )
        readings = [result for result in results if isinstance(result, JudgeReading)]
        unreadable: dict[str, str] = {}
        for direction, result in zip(("policy_first", "opponent_first"), results):
            if isinstance(result, str):
                unreadable[direction] = result
        if not readings:
            # As in EQ-Bench 4, one readable order is enough; with neither the comparison has no outcome.
            raise JudgeError(
                f"The judge gave no pairwise verdict against {opponent.model} in either order after "
                f"{self.config.judge_max_attempts} attempt(s) each: "
                + "; ".join(f"{direction}: {problem}" for direction, problem in unreadable.items())
            )
        outcome = combine_readings(
            [(reading.policy_margin, reading.opponent_margin) for reading in readings], body.pairwise_dimensions
        )
        return OpponentComparison(
            opponent=opponent.model,
            comparison_index=opponent.comparison_index,
            fraction=outcome.fraction,
            policy_margin=outcome.policy_margin,
            opponent_margin=outcome.opponent_margin,
            readings=readings,
            unreadable_orders=unreadable,
        )

    async def _read(
        self,
        body: EQBench4VerifyRequest,
        opponent: Opponent,
        judge: str,
        direction: Direction,
        first: tuple[str, str],
        second: tuple[str, str],
    ) -> JudgeReading | str:
        """One order's reading, or why none of its attempts held a verdict."""
        first_code, second_code = respondent_codes(f"{body.scenario_id}|{opponent.model}|{direction}")
        prompt = PromptParts(body.pairwise_prompt.literals, body.pairwise_prompt.slots).fill(
            {
                "first_code": first_code,
                "second_code": second_code,
                "first_transcript": first[0],
                "first_final_state": first[1],
                "second_transcript": second[0],
                "second_final_state": second[1],
            }
        )
        params = self.config.judge_responses_create_params.model_copy(deep=True)
        params.input = [NeMoGymEasyInputMessage(role="user", content=prompt)]
        problem = "no reply"
        for attempt in range(1, self.config.judge_max_attempts + 1):
            async with self._judge_semaphore:
                reply = await call_judge(
                    self.server_client,
                    server_name=judge,
                    url_path="/v1/responses",
                    json=params,
                    response_model=NeMoGymResponse,
                    timeout=ClientTimeout(total=self.config.judge_timeout_secs),
                )
            text = response_text(reply)
            verdicts = parse_json_object(text)
            if verdicts:
                record_parsed_judge_verdict()
                first_margin, second_margin = weighted_margins(
                    verdicts, first_code, second_code, body.pairwise_dimensions
                )
                policy_margin, opponent_margin = (
                    (first_margin, second_margin) if direction == "policy_first" else (second_margin, first_margin)
                )
                return JudgeReading(
                    direction=direction,
                    judge_model_server=judge,
                    first_code=first_code,
                    second_code=second_code,
                    verdicts=verdicts,
                    policy_margin=policy_margin,
                    opponent_margin=opponent_margin,
                    judge_text=text,
                    attempts=attempt,
                )
            problem = (
                f"judge output incomplete: {reply.incomplete_details.reason}"
                if reply.incomplete_details and reply.incomplete_details.reason
                else "no JSON verdict in the judge output"
            )
            if attempt < self.config.judge_max_attempts:
                # A reply rejected before a retry is accounted for; an order that never yields a verdict is not, so
                # its grading stays unknown.
                record_rejected_judge_reply()
        logger.warning(
            "No pairwise verdict for scenario %s against %s (%s) after %d attempt(s): %s",
            body.scenario_id,
            opponent.model,
            direction,
            self.config.judge_max_attempts,
            problem,
        )
        return problem

    async def _judge_traits(self, body: EQBench4VerifyRequest) -> dict[str, Any]:
        """Score the informational behavioural traits; a judge failure is recorded and never affects the reward."""
        prompt = PromptParts(body.trait_prompt.literals, body.trait_prompt.slots).fill(
            {"transcript": trait_transcript([m.model_dump() for m in body.transcript], body.trait_assistant_label)}
        )
        params = self.config.trait_judge_responses_create_params.model_copy(deep=True)
        params.input = [
            NeMoGymEasyInputMessage(role="system", content=self.config.trait_judge_system_message),
            NeMoGymEasyInputMessage(role="user", content=prompt),
        ]
        try:
            async with self._judge_semaphore:
                reply = await call_judge(
                    self.server_client,
                    server_name=self.config.trait_judge_model_server.name,
                    url_path="/v1/responses",
                    json=params,
                    response_model=NeMoGymResponse,
                    timeout=ClientTimeout(total=self.config.judge_timeout_secs),
                )
        except JUDGE_FAILURES as error:
            logger.warning("Trait judge failed for scenario %s: %s", body.scenario_id, error)
            return {"trait_judge_error": str(error)[:2000]}
        text = response_text(reply)
        scores = parse_trait_scores(text)
        # The reply is a dependency of the grading: parsed, or rejected by the parser. The reward never rests on it,
        # so a rejected trait reply leaves the pairwise grade complete.
        if scores is not None:
            record_parsed_judge_verdict()
        else:
            record_rejected_judge_reply()
        return {
            "trait_scores": scores,
            "trait_judge_text": text,
            "trait_judge_error": None if scores is not None else "no trait scores in the trait judge output",
        }

    def _reference(self) -> list[Outcome]:
        if self._reference_outcomes is None:
            path = Path(self.config.reference_comparisons_path)
            path = path if path.is_absolute() else GYM_ROOT / path
            if not path.is_file():
                raise FileNotFoundError(
                    f"EQ-Bench 4 reference comparisons not found at {path}; run `gym eval prepare --benchmark eqbench4`"
                )
            with path.open(encoding="utf-8") as stream:
                self._reference_outcomes = [
                    Outcome(item["model_a"], item["model_b"], float(item["fraction_a"]))
                    for item in map(json.loads, stream)
                ]
        return self._reference_outcomes

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Gym's aggregation, plus how many conversations were left out of the score and why.

        ``compute_metrics`` sees only scored rows, so the left-out conversations are counted here, over every row,
        and each kind that occurred is logged as a warning.
        """
        result = await super().aggregate_metrics(body)
        dropped = dropped_conversation_metrics(body.verify_responses)
        result.agent_metrics.update(dropped)
        result.key_metrics.update(
            {name: dropped[name] for name in ("eqbench4_policy_stopped_rate", "eqbench4_persona_failed_rate")}
        )
        for alert in dropped_conversation_alerts(dropped):
            logger.warning(alert)
        return result

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return eqbench4_metrics(tasks, self._reference(), resamples=self.config.rating_bootstrap_resamples)

    def compute_repeat_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return eqbench4_repeat_metrics(tasks, self._reference())

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        return key_metrics(agent_metrics)


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=create_fixture_server,
    request_model=EQBench4VerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
    invoke=invoke_with_fixture_judge,
)


if __name__ == "__main__":
    EQBench4ResourcesServer.run_webserver()
