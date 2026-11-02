# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VIALS: free-form answers to life-sciences image questions, graded by the upstream LLM judge.

One rollout is one attempt. ``reward`` is 1.0 when the judge finds the extracted answer
equivalent to the ground-truth final answer (GTFA). Accuracy is the mean reward over all
attempts; pass@k and pass^k come from the attempts of each task (see ``compute_metrics``).
"""

import asyncio
from pathlib import Path
from typing import Any, ClassVar, Optional

from pydantic import Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import call_judge
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.reward_profile import compute_pass_majority_metrics, compute_subset_metrics, highest_k_metrics
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.vials.task_data import TaskData
from resources_servers.vials.upstream_protocol import (
    JUDGE_SYSTEM,
    MULTIMODAL_JUDGE_SYSTEM,
    JudgeVerdict,
    build_judge_user_message,
    extract_answer_block,
    parse_judge_verdict,
)


UNVERIFIABLE_PREFIX = "UNVERIFIABLE:"


def _default_judge_params() -> NeMoGymResponseCreateParamsNonStreaming:
    # Upstream: max_tokens=512, structured output, no temperature for gpt-5 judges.
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[],
        max_output_tokens=512,
        text={
            "format": {
                "type": "json_schema",
                "name": "FreeFormJudgeResponse",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["reasoning", "equivalent"],
                    "properties": {"reasoning": {"type": "string"}, "equivalent": {"type": "boolean"}},
                },
            }
        },
    )


class VialsResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=_default_judge_params
    )
    # Upstream JUDGE_MAX_RETRIES: an empty or unparseable verdict is asked again, then scored incorrect.
    judge_attempts: int = Field(default=3, ge=1)
    judge_max_concurrency: int = Field(default=64, ge=1)
    # Upstream default: an UNVERIFIABLE text-only verdict is re-graded with the task images.
    multimodal_judge_fallback: bool = True


class VialsVerifyRequest(BaseVerifyRequest):
    verifier_metadata: TaskData


class VialsVerifyResponse(VialsVerifyRequest, BaseVerifyResponse):
    # Top level so compute_subset_metrics can group by it.
    domain: str
    extracted_answer: Optional[str] = None
    answer_recovered_from_reasoning: bool = False
    judge_reasoning: Optional[str] = None
    judge_unverifiable: bool = False
    judge_multimodal: bool = False
    judge_parse_ok: Optional[bool] = None


def _policy_text(response: NeMoGymResponse) -> tuple[str, bool]:
    """Visible answer text, else the reasoning text (upstream falls back to reasoning_content)."""
    text = (response.output_text or "").strip()
    if text:
        return text, False
    parts = []
    for item in response.output:
        if getattr(item, "type", None) != "reasoning":
            continue
        parts.extend(s.text for s in item.summary)
        parts.extend(c.text for c in item.content or [])
    reasoning = "\n".join(parts).strip()
    return reasoning, bool(reasoning)


def _task_images(params: NeMoGymResponseCreateParamsNonStreaming) -> list[dict[str, Any]]:
    """The input_image parts of the policy prompt, in the order the model saw them."""
    messages = params.model_dump(exclude_none=True)["input"]
    if isinstance(messages, str):
        return []
    images = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, list):
            images.extend(part for part in content if part.get("type") == "input_image")
    return images


def _rollout_domain(rollout: dict[str, Any]) -> Optional[str]:
    metadata = rollout.get("verifier_metadata")
    return rollout.get("domain") or (metadata.get("domain") if isinstance(metadata, dict) else None)


def _pass_power_k(tasks: list[list[dict[str, Any]]]) -> dict[str, float]:
    """Upstream ``pass_power_k``: the share of tasks whose first k attempts are all correct."""
    metrics = {}
    for k in range(1, max(len(rollouts) for rollouts in tasks) + 1):
        full = [rollouts[:k] for rollouts in tasks if len(rollouts) >= k]
        metrics[f"pass^{k}/accuracy"] = 100.0 * sum(all(r["reward"] == 1.0 for r in rs) for rs in full) / len(full)
    return metrics


class VialsResourcesServer(SimpleResourcesServer):
    config: VialsResourcesServerConfig
    _judge_semaphore: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        self._judge_semaphore = asyncio.Semaphore(self.config.judge_max_concurrency)
        super().model_post_init(context)

    async def _judge(self, system: str, user_content: Any) -> tuple[Optional[JudgeVerdict], str]:
        """Ask the judge up to ``judge_attempts`` times; return the verdict (None if never parsed) and last text.

        A failed call raises JudgeError (call_judge), which masks the rollout for a retry:
        upstream scored it incorrect, but it measures the infrastructure, not the model.
        """
        params = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            self.config.judge_responses_create_params.model_dump(exclude_unset=True)
            | {"input": [{"role": "system", "content": system}, {"role": "user", "content": user_content}]}
        )
        raw = ""
        for _ in range(self.config.judge_attempts):
            async with self._judge_semaphore:
                judged = await call_judge(
                    self.server_client,
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=params,
                    response_model=NeMoGymResponse,
                )
            raw = (judged.output_text or "").strip()
            verdict = parse_judge_verdict(raw) if raw else None
            if verdict is not None:
                return verdict, raw
        return None, raw

    async def verify(self, body: VialsVerifyRequest) -> VialsVerifyResponse:
        task = body.verifier_metadata
        text, recovered = _policy_text(body.response)
        answer = extract_answer_block(text)
        result = {
            **body.model_dump(),
            "domain": task.domain,
            "extracted_answer": answer,
            "answer_recovered_from_reasoning": recovered,
        }
        if not answer:
            return VialsVerifyResponse(**result, reward=0.0, judge_reasoning="No final answer was extracted.")

        message_fields = {
            "question": task.question,
            "gtfa": task.gtfa,
            "answer": answer,
            "lower": task.answer_lower_bound,
            "upper": task.answer_upper_bound,
        }
        verdict, raw = await self._judge(JUDGE_SYSTEM, build_judge_user_message(**message_fields, multimodal=False))
        if verdict is None:
            return VialsVerifyResponse(**result, reward=0.0, judge_reasoning=raw, judge_parse_ok=False)

        unverifiable = verdict.reasoning.lstrip().upper().startswith(UNVERIFIABLE_PREFIX)
        images = _task_images(body.responses_create_params)
        if not (unverifiable and self.config.multimodal_judge_fallback and images):
            return VialsVerifyResponse(
                **result,
                reward=float(verdict.equivalent),
                judge_reasoning=verdict.reasoning,
                judge_unverifiable=unverifiable,
                judge_parse_ok=True,
            )

        user_content = [
            *images,
            {"type": "input_text", "text": build_judge_user_message(**message_fields, multimodal=True)},
        ]
        verdict, raw = await self._judge(MULTIMODAL_JUDGE_SYSTEM, user_content)
        return VialsVerifyResponse(
            **result,
            reward=float(verdict.equivalent) if verdict else 0.0,
            judge_reasoning=verdict.reasoning if verdict else raw,
            judge_multimodal=True,
            judge_parse_ok=verdict is not None,
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Upstream's summary on Gym's 0-100 scale, overall and per domain.

        ``accuracy`` is upstream ``accuracy``: the mean over every measured attempt, pooled
        across tasks. ``pass@1[avg-of-k]/accuracy`` averages per task first; the two agree
        when no attempt is masked. ``pass@k/accuracy`` is upstream ``pass_at_k`` (any attempt
        correct) and ``pass^k/accuracy`` is upstream ``pass_power_k`` (every attempt correct).
        """
        if not tasks:
            return {}

        def score(rollout: dict[str, Any]) -> dict[str, float]:
            return {"accuracy": rollout["reward"]}

        metrics = compute_pass_majority_metrics(tasks, score_fn=score)[0]
        attempts = [r["reward"] for rollouts in tasks for r in rollouts]
        metrics["accuracy"] = 100.0 * sum(attempts) / len(attempts)
        metrics.update(_pass_power_k(tasks))
        # A rollout that never ran is counted as a zero-reward row that may lack the verify fields,
        # so each task's domain comes from any of its rollouts or their task metadata.
        by_domain: dict[str, list[list[dict[str, Any]]]] = {}
        for rollouts in tasks:
            domain = next(filter(None, (_rollout_domain(r) for r in rollouts)), None)
            if domain is not None:
                by_domain.setdefault(domain, []).append([r | {"domain": domain} for r in rollouts])
        metrics.update(
            compute_subset_metrics(
                [rs for domain_tasks in by_domain.values() for rs in domain_tasks], subset_key="domain", score_fn=score
            )
        )
        for domain, domain_tasks in by_domain.items():
            rewards = [r["reward"] for rollouts in domain_tasks for r in rollouts]
            metrics[f"{domain}/accuracy"] = 100.0 * sum(rewards) / len(rewards)
            metrics.update({f"{domain}/{name}": value for name, value in _pass_power_k(domain_tasks).items()})

        rollouts = [r for rs in tasks for r in rs]
        for name, field in (
            ("no_answer", "extracted_answer"),
            ("judge_unverifiable", "judge_unverifiable"),
            ("judge_multimodal", "judge_multimodal"),
            ("answer_recovered_from_reasoning", "answer_recovered_from_reasoning"),
        ):
            hits = sum(1 for r in rollouts if (not r.get(field) if name == "no_answer" else r.get(field)))
            metrics[f"rate/{name}"] = 100.0 * hits / len(rollouts)
        metrics["rate/judge_parse_failure"] = (
            100.0 * sum(r.get("judge_parse_ok") is False for r in rollouts) / len(rollouts)
        )
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        """Headline: accuracy, pass@k and pass^k at the highest k, and the token counts."""
        key = {
            name: agent_metrics[name]
            for name in ("accuracy", "mean/input_tokens", "mean/output_tokens")
            if name in agent_metrics
        }
        for pattern in ("pass@1[avg-of-{k}]", "pass@{k}", "pass^{k}"):
            key.update(highest_k_metrics(agent_metrics, pattern, score_names=["accuracy"]))
        return key


def _fixture_server() -> VialsResourcesServer:
    from resources_servers.vials.fixture_judge import create_fixture_server

    return create_fixture_server()


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=_fixture_server,
    request_model=VialsVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    VialsResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = VialsResourcesServer.run_webserver()  # noqa: F401
