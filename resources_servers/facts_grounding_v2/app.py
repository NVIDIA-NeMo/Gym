# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FACTS Grounding v2 resources server.

Long-form grounded generation from the FACTS Benchmark Suite (Cheng et al., 2025, arXiv:2512.10791, section 6):
the model answers a user request using only a supplied context document, and two judges decide whether the
response is (1) eligible, i.e. it actually addresses the request, and (2) grounded, i.e. every information-bearing
sentence is supported by the document. This server reproduces the official reference implementation (Kaggle
starter notebook ``prathameshbang/facts-grounding-v2-benchmark-starter``, version 4):

- judges are tried in a fixed order (official: Gemini 2.5 Flash, then GPT-5 ``gpt-5-2025-08-07``);
- eligibility: each judge first writes its own *baseline* answer to ``full_prompt``, then rates the policy response
  against that baseline with the v2 "Instruction Following" prompt (context redacted); a response is eligible as
  soon as one judge does not report ``Major Issue(s)`` (an unparseable verdict counts as not major, as in the
  starter), and later judges are not asked;
- grounding: every judge classifies each sentence of the response with the v2 ``UFG_REV21`` prompt; a judge's
  verdict is "grounded" only when every parsed sentence is ``supported``/``no_rad``/``unknown`` and at least one
  sentence parsed; the grounding score is the mean of the judges' verdicts;
- the final score is the grounding score for eligible responses and 0 for ineligible ones.

``reward`` is that final (adjusted) score in {0, 0.5, 1}. The unadjusted score, per-judge verdicts, eligibility
ratings, baselines, sentence labels, parse validity and every judge receipt are returned so aggregate metrics
keep eligibility, groundedness, judge disagreement and judge invalidity separate.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import call_judge
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)


PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPT_SHA256 = {
    "eligibility_prompt.txt": "2fd54d7e5cce06f538785cb61af828374219ccf78f3fc18bf515ca6e12f6f521",  # pragma: allowlist secret
    "eligibility_system_prompt.txt": "5920e4e3d3686f93eb2e78617d6d2729215ccd2da76e80555f0a2b755cebcce9",  # pragma: allowlist secret
    "grounding_prompt.txt": "15ab94895c6095a90d9ccc4001923fb58f29c439891436238e335ab373e1ad68",  # pragma: allowlist secret
    "grounding_system_prompt.txt": "a501fca7a0087dd000d351006e070d2803f9074166942c3d487732e287a90a64",  # pragma: allowlist secret
}
PROMPT_VARIANT = "kaggle-starter-prathameshbang-facts-grounding-v2-benchmark-starter-v4"
ELIGIBILITY_RATINGS = ("No Issues", "Minor Issue(s)", "Major Issue(s)", "Invalid")
GROUNDING_LABELS = ("supported", "not_supported", "no_rad", "unknown")

_INSTRUCTION_JSON = re.compile(
    r'{\s*"Instruction Following"\s*:\s*"(No Issues|Minor Issue\(s\)|Major Issue\(s\))"\s*}'
)


def load_prompts(prompts_dir: Path = PROMPTS_DIR) -> Dict[str, str]:
    prompts: Dict[str, str] = {}
    for name, expected in PROMPT_SHA256.items():
        text = (prompts_dir / name).read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if prompts_dir == PROMPTS_DIR and digest != expected:
            raise RuntimeError(f"{name} drifted from the pinned starter prompt: sha256 {digest}")
        prompts[name.removesuffix(".txt")] = text
    return prompts


def extract_text_from_response(response: NeMoGymResponse) -> str:
    """Return the last assistant message text; reasoning items are split off by the model server."""
    for output in reversed(response.output):
        if getattr(output, "type", None) != "message" or getattr(output, "role", None) != "assistant":
            continue
        content = getattr(output, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = [item.text for item in content if isinstance(getattr(item, "text", None), str)]
            if texts:
                return "\n".join(texts).strip()
    return ""


def extract_instruction_rating(text: str) -> str:
    """The starter's ``extract_instruction_json``: the first well-formed verdict object, else ``Invalid``."""
    match = _INSTRUCTION_JSON.search(text)
    if not match:
        return "Invalid"
    try:
        return json.loads(match.group(0))["Instruction Following"]
    except (ValueError, KeyError):  # pragma: no cover - the regex guarantees a parseable object
        return "Invalid"


def is_eligible(rating: str) -> bool:
    """The starter treats every rating except ``Major Issue(s)`` (including ``Invalid``) as eligible."""
    return "Major Issue(s)" not in rating


def _clean_json_content(text: str, start_field: str, end_field: Optional[str] = None) -> str:
    """The starter's ``clean_json_content_8``: strip stray quotes/escapes from one field's value."""
    if end_field:
        trailer_pattern = rf'(,\s*"{re.escape(end_field)}":)'
    else:
        trailer_pattern = r"(\s*})"
    pattern = re.compile(rf'"{re.escape(start_field)}":(.+?){trailer_pattern}', re.DOTALL)

    def _clean_match(match: re.Match[str]) -> str:
        content = match.group(1)
        trailer = match.group(2)
        content = content.replace('"', "").replace("\\", "").replace("\x02", "").strip()
        cleaned = f'"{content}"' if content != "null" else "null"
        return f'"{start_field}":{cleaned}{trailer}'

    cleaned_text, substitutions = pattern.subn(_clean_match, text, count=1)
    return cleaned_text if substitutions > 0 else text


def parse_grounding_verdict(answer: str) -> tuple[bool, List[Dict[str, Any]], int, int]:
    """The starter's ``parse_ufg_rev21_verdict``.

    Returns ``(grounded, parsed_sentences, parsed_count, unparseable_lines)``. ``grounded`` is true only when at
    least one sentence parsed and every parsed label is ``supported``/``no_rad``/``unknown``; an unparseable line
    that still contains ``"not_supported"`` is kept as a ``not_supported`` sentence.
    """
    if "```json" in answer:
        chunks = [part.split("```")[0] for part in answer.split("```json")[1:]]
        answer = "\n".join(chunks)
    answer = answer.strip()
    for marker in ("}\n", "} \n", "}  \n"):
        answer = answer.replace(marker, "}\n@\n@\n")
    answer = answer.replace("<ctrl75>", "")
    parsed: List[Dict[str, Any]] = []
    unparseable = 0
    for raw_line in answer.split("\n@\n@\n"):
        line = raw_line.replace("\n", " ").replace("\\'", "'").replace("<ctrl75>", "").lstrip(",")
        line = _clean_json_content(line, "sentence", "label")
        line = _clean_json_content(line, "label", "rationale")
        line = _clean_json_content(line, "rationale", "excerpt")
        line = _clean_json_content(line, "excerpt")
        try:
            item = json.loads(line)
        except (json.JSONDecodeError, ValueError, TypeError):
            item = None
        if not isinstance(item, dict):
            unparseable += 1
            if '"not_supported"' in raw_line:
                parsed.append({"sentence": "", "rationale": "", "excerpt": "", "label": "not_supported"})
            continue
        item.setdefault("label", "unknown")
        parsed.append(item)
    if parsed:
        grounded = all(item["label"] in ("supported", "no_rad", "unknown") for item in parsed)
    else:
        grounded = False
    return grounded, parsed, len(parsed), unparseable


class FACTSGroundingV2Config(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    judge_model_servers: List[ModelServerRef] = Field(
        description="Judges in the official order; the first judge that finds no major issue settles eligibility."
    )
    judge_labels: Optional[List[str]] = Field(
        default=None, description="Short label per judge for metric keys (defaults to the server names)."
    )
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[]),
        description="Overrides for judge rating calls; unset fields use provider defaults, like the starter.",
    )
    baseline_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[]),
        description="Overrides for the judge's own baseline answer to full_prompt; unset fields use provider defaults.",
    )
    grade_ineligible_responses: bool = Field(
        default=True,
        description=(
            "Also run the grounding judges on ineligible responses so the paper's unadjusted factuality score "
            "can be reported; the adjusted score (reward) is unaffected."
        ),
    )
    prompts_dir: str = Field(default=str(PROMPTS_DIR))


class FACTSGroundingV2RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    id: Optional[Union[int, str]] = None
    system_instruction: Optional[str] = None
    user_request: Optional[str] = None
    context_document: Optional[str] = None
    full_prompt: Optional[str] = None
    domain: Optional[str] = None
    type: Optional[str] = None
    high_level_type: Optional[str] = None


class FACTSGroundingV2VerifyRequest(FACTSGroundingV2RunRequest, BaseVerifyRequest):
    pass


class FACTSGroundingV2VerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    generation: str = ""
    generation_empty: bool = False
    generation_truncated: bool = False
    generation_chars: int = 0
    policy_input_tokens: Optional[int] = None
    policy_output_tokens: Optional[int] = None
    eligible: bool = False
    eligibility_deciding_judge: Optional[str] = None
    eligibility_ratings: Dict[str, str] = Field(default_factory=dict)
    eligibility_invalid_count: int = 0
    grounding_verdicts: Dict[str, bool] = Field(default_factory=dict)
    grounding_parsed_sentences: Dict[str, int] = Field(default_factory=dict)
    grounding_unparseable_lines: Dict[str, int] = Field(default_factory=dict)
    grounding_label_counts: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    grounding_score: Optional[float] = None
    unadjusted_score: Optional[float] = None
    judges_agree: Optional[bool] = None
    judge_receipts: List[Dict[str, Any]] = Field(default_factory=list)
    prompt_variant: str = PROMPT_VARIANT
    is_eligible: float = 0.0
    is_grounded_all_judges: float = 0.0


class FACTSGroundingV2ResourcesServer(SimpleResourcesServer):
    config: FACTSGroundingV2Config

    def model_post_init(self, context: Any) -> None:
        self._prompts = load_prompts(Path(self.config.prompts_dir))
        labels = self.config.judge_labels or [ref.name for ref in self.config.judge_model_servers]
        if len(labels) != len(self.config.judge_model_servers) or len(set(labels)) != len(labels):
            raise ValueError("judge_labels must be unique and match judge_model_servers")
        self._judges = list(zip(labels, self.config.judge_model_servers))
        return super().model_post_init(context)

    @staticmethod
    def _chat_params(
        messages: List[Dict[str, str]], base: NeMoGymResponseCreateParamsNonStreaming
    ) -> NeMoGymChatCompletionCreateParamsNonStreaming:
        params: Dict[str, Any] = {"messages": messages}
        if base.temperature is not None:
            params["temperature"] = base.temperature
        if base.top_p is not None:
            params["top_p"] = base.top_p
        if base.max_output_tokens is not None:
            params["max_tokens"] = base.max_output_tokens
        return NeMoGymChatCompletionCreateParamsNonStreaming(**params)

    async def _call(self, judge_label: str, server_name: str, stage: str, params) -> Dict[str, Any]:
        completion = await call_judge(
            self.server_client,
            server_name=server_name,
            url_path="/v1/chat/completions",
            json=params,
            response_model=NeMoGymChatCompletion,
        )
        choice = completion.choices[0] if completion.choices else None
        content = (choice.message.content if choice and choice.message else None) or ""
        prompt_text = "\n".join(message["content"] for message in params.messages)
        return {
            "judge": judge_label,
            "stage": stage,
            "judge_model": completion.model,
            "response_id": completion.id,
            "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
            "usage": completion.usage.model_dump(mode="json") if completion.usage else None,
            "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            "content": content,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }

    async def _baseline(self, judge_label: str, server_name: str, full_prompt: str) -> Dict[str, Any]:
        params = self._chat_params(
            [{"role": "user", "content": full_prompt}], self.config.baseline_responses_create_params
        )
        return await self._call(judge_label, server_name, "baseline", params)

    async def _eligibility(
        self, judge_label: str, server_name: str, user_request: str, response: str, baseline: str
    ) -> Dict[str, Any]:
        prompt = self._prompts["eligibility_prompt"].format(
            user_request=user_request, response_a=response, response_b=baseline
        )
        params = self._chat_params(
            [
                {"role": "system", "content": self._prompts["eligibility_system_prompt"]},
                {"role": "user", "content": prompt},
            ],
            self.config.judge_responses_create_params,
        )
        receipt = await self._call(judge_label, server_name, "eligibility", params)
        receipt["rating"] = extract_instruction_rating(receipt["content"])
        receipt["eligible"] = is_eligible(receipt["rating"])
        return receipt

    async def _grounding(
        self, judge_label: str, server_name: str, user_request: str, context: str, response: str
    ) -> Dict[str, Any]:
        prompt = self._prompts["grounding_prompt"].format(user_query=user_request, context=context, response=response)
        params = self._chat_params(
            [
                {"role": "system", "content": self._prompts["grounding_system_prompt"]},
                {"role": "user", "content": prompt},
            ],
            self.config.judge_responses_create_params,
        )
        receipt = await self._call(judge_label, server_name, "grounding", params)
        grounded, sentences, parsed_count, unparseable = parse_grounding_verdict(receipt["content"])
        receipt["grounded"] = grounded
        receipt["sentences"] = sentences
        receipt["parsed_sentences"] = parsed_count
        receipt["unparseable_lines"] = unparseable
        receipt["label_counts"] = {
            label: sum(1 for item in sentences if item.get("label") == label) for label in GROUNDING_LABELS
        }
        return receipt

    async def verify(self, body: FACTSGroundingV2VerifyRequest) -> FACTSGroundingV2VerifyResponse:
        generation = extract_text_from_response(body.response)
        incomplete = body.response.incomplete_details
        truncated = bool(incomplete and getattr(incomplete, "reason", None) == "max_output_tokens")
        usage = body.response.usage
        user_request = body.user_request or ""
        context = body.context_document or ""
        full_prompt = body.full_prompt or ""
        receipts: List[Dict[str, Any]] = []

        # Eligibility: judges in order, the first one that finds no major issue settles it (starter behaviour).
        eligible = False
        deciding: Optional[str] = None
        ratings: Dict[str, str] = {}
        for judge_label, ref in self._judges:
            baseline = await self._baseline(judge_label, ref.name, full_prompt)
            receipts.append(baseline)
            rating = await self._eligibility(judge_label, ref.name, user_request, generation, baseline["content"])
            receipts.append(rating)
            ratings[judge_label] = rating["rating"]
            if rating["eligible"]:
                eligible = True
                deciding = judge_label
                break

        verdicts: Dict[str, bool] = {}
        parsed_counts: Dict[str, int] = {}
        unparseable: Dict[str, int] = {}
        label_counts: Dict[str, Dict[str, int]] = {}
        grounding_score: Optional[float] = None
        if eligible or self.config.grade_ineligible_responses:
            grounding = await asyncio.gather(
                *(self._grounding(label, ref.name, user_request, context, generation) for label, ref in self._judges)
            )
            for receipt in grounding:
                receipts.append(receipt)
                verdicts[receipt["judge"]] = receipt["grounded"]
                parsed_counts[receipt["judge"]] = receipt["parsed_sentences"]
                unparseable[receipt["judge"]] = receipt["unparseable_lines"]
                label_counts[receipt["judge"]] = receipt["label_counts"]
            grounding_score = sum(verdicts.values()) / len(verdicts)

        reward = grounding_score if (eligible and grounding_score is not None) else 0.0
        return FACTSGroundingV2VerifyResponse(
            **body.model_dump(),
            reward=reward,
            generation=generation,
            generation_empty=not generation,
            generation_truncated=truncated,
            generation_chars=len(generation),
            policy_input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            policy_output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            eligible=eligible,
            eligibility_deciding_judge=deciding,
            eligibility_ratings=ratings,
            eligibility_invalid_count=sum(1 for rating in ratings.values() if rating == "Invalid"),
            grounding_verdicts=verdicts,
            grounding_parsed_sentences=parsed_counts,
            grounding_unparseable_lines=unparseable,
            grounding_label_counts=label_counts,
            grounding_score=grounding_score,
            unadjusted_score=grounding_score,
            judges_agree=(len(set(verdicts.values())) == 1) if verdicts else None,
            judge_receipts=receipts,
            is_eligible=1.0 if eligible else 0.0,
            is_grounded_all_judges=1.0 if verdicts and all(verdicts.values()) else 0.0,
        )

    @staticmethod
    def _bootstrap_ci(values: List[float], *, resamples: int = 2000, seed: int = 20251211) -> tuple[float, float]:
        if len(values) < 2:
            return (float("nan"), float("nan"))
        rng = random.Random(seed)
        count = len(values)
        means = sorted(sum(rng.choice(values) for _ in range(count)) / count for _ in range(resamples))
        return (means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1])

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        rollouts = [rollout for task in tasks for rollout in task]
        if not rollouts:
            return {}
        judge_labels = [label for label, _ in self._judges]

        def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
            count = len(rows)
            eligible_rows = [row for row in rows if row.get("eligible")]
            graded = [row for row in rows if row.get("unadjusted_score") is not None]
            out: Dict[str, Any] = {
                "factuality_score": sum(float(row.get("reward", 0.0)) for row in rows) / count,
                "eligibility_rate": len(eligible_rows) / count,
                "num_rollouts": count,
                "num_eligible": len(eligible_rows),
                "num_graded_for_unadjusted": len(graded),
                "unadjusted_factuality_score": (
                    sum(float(row["unadjusted_score"]) for row in graded) / len(graded) if graded else 0.0
                ),
                "grounded_all_judges_rate_eligible": (
                    sum(float(row.get("is_grounded_all_judges", 0.0)) for row in eligible_rows) / len(eligible_rows)
                    if eligible_rows
                    else 0.0
                ),
            }
            for label in judge_labels:
                verdict_rows = [row for row in eligible_rows if label in (row.get("grounding_verdicts") or {})]
                out[f"grounded_rate_eligible/{label}"] = (
                    sum(1 for row in verdict_rows if row["grounding_verdicts"][label]) / len(verdict_rows)
                    if verdict_rows
                    else 0.0
                )
            disagreements = [row for row in eligible_rows if row.get("judges_agree") is False]
            out["judge_disagreement_rate_eligible"] = len(disagreements) / len(eligible_rows) if eligible_rows else 0.0
            return out

        metrics = _summary(rollouts)
        rewards = [float(row.get("reward", 0.0)) for row in rollouts]
        low, high = self._bootstrap_ci(rewards)
        metrics["factuality_score_ci95_low"] = low
        metrics["factuality_score_ci95_high"] = high
        count = len(rollouts)
        for label in judge_labels:
            decided = sum(1 for row in rollouts if row.get("eligibility_deciding_judge") == label)
            metrics[f"eligibility_decided_by/{label}"] = decided / count
            rated = [row for row in rollouts if label in (row.get("eligibility_ratings") or {})]
            for rating in ELIGIBILITY_RATINGS:
                metrics[f"eligibility_rating/{label}/{rating}"] = (
                    sum(1 for row in rated if row["eligibility_ratings"][label] == rating) / len(rated)
                    if rated
                    else 0.0
                )
            graded = [row for row in rollouts if label in (row.get("grounding_parsed_sentences") or {})]
            metrics[f"grounding_parse_empty_rate/{label}"] = (
                sum(1 for row in graded if row["grounding_parsed_sentences"][label] == 0) / len(graded)
                if graded
                else 0.0
            )
            metrics[f"grounding_unparseable_line_rate/{label}"] = (
                sum(1 for row in graded if row["grounding_unparseable_lines"][label] > 0) / len(graded)
                if graded
                else 0.0
            )
        metrics["eligibility_invalid_rate"] = (
            sum(int(row.get("eligibility_invalid_count", 0)) > 0 for row in rollouts) / count
        )
        metrics["generation_empty_rate"] = sum(1 for row in rollouts if row.get("generation_empty")) / count
        metrics["generation_truncated_rate"] = sum(1 for row in rollouts if row.get("generation_truncated")) / count
        input_tokens = [row["policy_input_tokens"] for row in rollouts if row.get("policy_input_tokens") is not None]
        if input_tokens:
            metrics["policy_input_tokens_max"] = max(input_tokens)
            metrics["policy_input_tokens_mean"] = sum(input_tokens) / len(input_tokens)
        for slice_key in ("domain", "high_level_type", "type"):
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for row in rollouts:
                groups.setdefault(str(row.get(slice_key) or "unknown"), []).append(row)
            for value, rows in sorted(groups.items()):
                summary = _summary(rows)
                metrics[f"factuality_score/{slice_key}/{value}"] = summary["factuality_score"]
                metrics[f"eligibility_rate/{slice_key}/{value}"] = summary["eligibility_rate"]
                metrics[f"num_rollouts/{slice_key}/{value}"] = summary["num_rollouts"]
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            "factuality_score",
            "factuality_score_ci95_low",
            "factuality_score_ci95_high",
            "unadjusted_factuality_score",
            "eligibility_rate",
            "judge_disagreement_rate_eligible",
            "generation_truncated_rate",
            "mean/input_tokens",
            "mean/output_tokens",
        ] + [f"grounded_rate_eligible/{label}" for label, _ in self._judges]
        return {key: agent_metrics[key] for key in keys if key in agent_metrics}


if __name__ == "__main__":
    FACTSGroundingV2ResourcesServer.run_webserver()
