# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FACTS Search V2 Search-On tool and verifier.

The sole tool is the canonical Brave Web Search request/format from Google's
public implementation.  Final answers are graded once with the public A/B/C
prompt and the current benchmark-page judge, Gemini 3.5 Flash.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional

from fastapi import FastAPI
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
    RATE_LIMIT_ERROR_CODES,
    RETRY_ERROR_CODES,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import raise_for_status, request


BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
GRADER_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "grader_template.txt"
GRADER_TEMPLATE_SHA256 = "4bfbea8aeb6726cf8c23a663a8c3f43fd11f2a072c24229507f3a6d7c4890317"  # pragma: allowlist secret
GRADER_VARIANT = "kaggle-aminmohamedmohami-facts-search-on-implementation-v3-current-gemini-3.5-flash"
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.*)", re.IGNORECASE | re.DOTALL)
_STRICT_GRADE_RE = re.compile(r"^\s*([ABC])(?:\s*[:.)-].*)?\s*$", re.IGNORECASE | re.DOTALL)


class FACTSSearchChatCompletion(NeMoGymChatCompletion):
    """Judge response shape with OpenRouter's reserved-capacity tier.

    OpenRouter can return ``provisioned`` when a request uses reserved provider
    capacity. The value is response metadata; it does not change the public
    FACTS grader or the score.
    """

    service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority", "provisioned"]] = None


def load_grader_template() -> str:
    template = GRADER_TEMPLATE_PATH.read_text(encoding="utf-8").strip()
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if digest != GRADER_TEMPLATE_SHA256:
        raise RuntimeError(f"FACTS Search grader prompt drifted: expected {GRADER_TEMPLATE_SHA256}, got {digest}")
    return template


def extract_final_answer(response: NeMoGymResponse) -> str:
    for output in reversed(response.output):
        if getattr(output, "type", None) != "message" or getattr(output, "role", None) != "assistant":
            continue
        content = getattr(output, "content", None)
        if isinstance(content, str):
            text = content
        else:
            text = "\n".join(item.text for item in (content or []) if isinstance(getattr(item, "text", None), str))
        text = text.strip()
        match = _FINAL_ANSWER_RE.search(text)
        return match.group(1).strip() if match else text
    return ""


def parse_grade(text: str) -> str:
    """The public notebook's exact parser: first uppercase A/B/C, else C."""
    match = re.search(r"(A|B|C)", text or "")
    return match.group(1) if match else "C"


def strict_grade(text: str) -> str:
    """Diagnostic parse of the grader's requested one-letter output contract."""
    match = _STRICT_GRADE_RE.match(text or "")
    return match.group(1).upper() if match else "UNKNOWN"


def format_brave_results(payload: Dict[str, Any]) -> str:
    """Byte-equivalent field order and separators to the public notebook."""
    parsed = []
    for item in payload.get("web", {}).get("results", []):
        parsed.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "extra_snippets": item.get("extra_snippets", []),
                "description": item.get("description", ""),
            }
        )
    result = ""
    for item in parsed:
        result += f"Title: {item['title']}\n"
        result += f"URL: {item['url']}\n"
        result += f"Description: {item['description']}\n"
        result += "Extra Snippets:\n"
        for snippet in item["extra_snippets"]:
            result += f"  - {snippet}\n"
        result += "\n"
    return result or "No results returned."


class FACTSSearchConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    brave_api_key: str
    brave_num_results: int = Field(default=5, ge=1, le=20)
    brave_timeout_seconds: float = Field(default=30.0, gt=0)
    brave_max_retries: int = Field(default=3, ge=1)
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )
    grader_template_path: str = str(GRADER_TEMPLATE_PATH)


class FACTSSearchRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    id: Optional[str] = None
    problem: Optional[str] = None
    gold_answer: Optional[str] = None
    row_sha256: Optional[str] = None


class FACTSSearchVerifyRequest(FACTSSearchRunRequest, BaseVerifyRequest):
    pass


class FACTSSearchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    final_answer: str = ""
    final_answer_empty: bool = False
    final_answer_truncated: bool = False
    grade_letter: str = "UNKNOWN"
    is_correct: float = 0.0
    is_incorrect: float = 0.0
    is_not_attempted: float = 0.0
    is_unknown: float = 0.0
    search_hops: int = 0
    search_queries: int = 0
    forced_final: bool = False
    judge_valid: bool = True
    judge_prompt_sha256: str = ""
    judge_receipt: Dict[str, Any] = Field(default_factory=dict)
    grader_variant: str = GRADER_VARIANT


class FACTSSearchResourcesServer(SimpleResourcesServer):
    config: FACTSSearchConfig

    def model_post_init(self, context: Any) -> None:
        path = Path(self.config.grader_template_path)
        self._grader_template = (
            load_grader_template() if path == GRADER_TEMPLATE_PATH else path.read_text(encoding="utf-8").strip()
        )
        return super().model_post_init(context)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/brave_search")(self.brave_search)
        return app

    async def brave_search(self, body: Dict[str, Any]) -> Dict[str, str]:
        query = body.get("query") if isinstance(body, dict) else None
        if not isinstance(query, str) or not query.strip():
            return {"result": "Error: query must be a non-empty string."}
        if not self.config.brave_api_key:
            raise RuntimeError("FACTS_SEARCH_BRAVE_API_KEY is required; no substitute search provider is permitted")
        headers = {"Accept": "application/json", "X-Subscription-Token": self.config.brave_api_key}
        params = {"q": query, "count": self.config.brave_num_results}
        last_response = None
        for attempt in range(self.config.brave_max_retries):
            last_response = await request(
                method="GET",
                url=BRAVE_ENDPOINT,
                headers=headers,
                params=params,
                timeout=self.config.brave_timeout_seconds,
            )
            if last_response.status not in RETRY_ERROR_CODES:
                await raise_for_status(last_response)
                return {"result": format_brave_results(await last_response.json())}
            if attempt + 1 < self.config.brave_max_retries:
                delay = (2**attempt) if last_response.status not in RATE_LIMIT_ERROR_CODES else 2 * (2**attempt)
                await asyncio.sleep(delay)
        assert last_response is not None
        await raise_for_status(last_response)
        raise RuntimeError("unreachable")

    def _judge_params(self, prompt: str) -> NeMoGymChatCompletionCreateParamsNonStreaming:
        base = self.config.judge_responses_create_params
        params: Dict[str, Any] = {"messages": [{"role": "user", "content": prompt}]}
        if base.temperature is not None:
            params["temperature"] = base.temperature
        if base.top_p is not None:
            params["top_p"] = base.top_p
        if base.max_output_tokens is not None:
            params["max_tokens"] = base.max_output_tokens
        return NeMoGymChatCompletionCreateParamsNonStreaming(**params)

    async def verify(self, body: FACTSSearchVerifyRequest) -> FACTSSearchVerifyResponse:
        answer = extract_final_answer(body.response)
        prompt = self._grader_template.format(
            question=body.problem or "",
            target=body.gold_answer or "",
            predicted_answer=answer,
        )
        completion = await call_judge(
            self.server_client,
            server_name=self.config.judge_model_server.name,
            url_path="/v1/chat/completions",
            json=self._judge_params(prompt),
            response_model=FACTSSearchChatCompletion,
        )
        choice = completion.choices[0] if completion.choices else None
        judge_text = (choice.message.content if choice and choice.message else None) or ""
        grade = parse_grade(judge_text)
        strict = strict_grade(judge_text)
        metadata = body.response.metadata or {}
        hops = int(metadata.get("facts_search_hops", 0))
        queries = int(metadata.get("facts_search_queries", 0))
        valid = strict in {"A", "B", "C"}
        incomplete = body.response.incomplete_details
        truncated = bool(incomplete and getattr(incomplete, "reason", None) == "max_output_tokens")
        receipt = {
            "judge_model": completion.model,
            "response_id": completion.id,
            "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
            "usage": completion.usage.model_dump(mode="json") if completion.usage else None,
            "content": judge_text,
            "content_sha256": hashlib.sha256(judge_text.encode("utf-8")).hexdigest(),
            "parsed_grade": grade,
            "strict_grade": strict,
        }
        return FACTSSearchVerifyResponse(
            **body.model_dump(),
            reward=1.0 if grade == "A" else 0.0,
            failure_reason=None
            if valid
            else "Judge output violated the one-letter contract; the public first-uppercase-A/B/C parser applied",
            final_answer=answer,
            final_answer_empty=not answer,
            final_answer_truncated=truncated,
            grade_letter=grade,
            is_correct=float(grade == "A"),
            is_incorrect=float(grade == "B"),
            is_not_attempted=float(grade == "C"),
            is_unknown=float(not valid),
            search_hops=hops,
            search_queries=queries,
            forced_final=str(metadata.get("facts_search_forced_final", "false")).lower() == "true",
            judge_valid=valid,
            judge_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            judge_receipt=receipt,
        )

    @staticmethod
    def _score(labels: List[str]) -> Dict[str, float]:
        n = len(labels)
        correct = labels.count("A")
        incorrect = labels.count("B")
        not_attempted = labels.count("C")
        accuracy = correct / n if n else 0.0
        attempted_accuracy = correct / (correct + incorrect) if correct + incorrect else 0.0
        f1 = (
            2 * accuracy * attempted_accuracy / (accuracy + attempted_accuracy)
            if accuracy + attempted_accuracy
            else 0.0
        )
        return {
            "f1": f1,
            "accuracy": accuracy,
            "attempted_accuracy": attempted_accuracy,
            "hedging_rate": not_attempted / n if n else 0.0,
        }

    @classmethod
    def _bootstrap_f1(cls, labels: List[str], resamples: int = 1000, seed: int = 0) -> tuple[float, float, float]:
        if len(labels) < 2:
            return (float("nan"), float("nan"), float("nan"))
        rng = random.Random(seed)
        values = sorted(cls._score([rng.choice(labels) for _ in labels])["f1"] for _ in range(resamples))
        observed = cls._score(labels)["f1"]
        low = values[int(0.025 * resamples)]
        high = values[int(0.975 * resamples) - 1]
        return low, high, max(observed - low, high - observed)

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        rows = [row for task in tasks for row in task]
        labels = [str(row["grade_letter"]) for row in rows if row.get("grade_letter") in {"A", "B", "C"}]
        metrics: Dict[str, Any] = self._score(labels)
        low, high, radius = self._bootstrap_f1(labels)
        metrics.update(
            {
                "f1_ci95_low": low,
                "f1_ci95_high": high,
                "f1_ci95_radius": radius,
                "num_rollouts": len(rows),
                "num_valid_judgments": sum(bool(row.get("judge_valid", True)) for row in rows),
                "num_correct": labels.count("A"),
                "num_incorrect": labels.count("B"),
                "num_not_attempted": labels.count("C"),
                "num_unknown": sum(not bool(row.get("judge_valid", True)) for row in rows),
                "judge_valid_rate": sum(bool(row.get("judge_valid", True)) for row in rows) / len(rows)
                if rows
                else 0.0,
                "average_searches": sum(int(row.get("search_hops", 0)) for row in rows) / len(rows) if rows else 0.0,
                "average_search_queries": sum(int(row.get("search_queries", 0)) for row in rows) / len(rows)
                if rows
                else 0.0,
                "forced_final_rate": sum(bool(row.get("forced_final")) for row in rows) / len(rows) if rows else 0.0,
                "final_answer_empty_rate": sum(bool(row.get("final_answer_empty")) for row in rows) / len(rows)
                if rows
                else 0.0,
                "final_answer_truncated_rate": sum(bool(row.get("final_answer_truncated")) for row in rows) / len(rows)
                if rows
                else 0.0,
            }
        )
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        keys = (
            "f1",
            "f1_ci95_low",
            "f1_ci95_high",
            "accuracy",
            "attempted_accuracy",
            "hedging_rate",
            "average_searches",
            "judge_valid_rate",
        )
        return {key: agent_metrics[key] for key in keys if key in agent_metrics}


if __name__ == "__main__":
    FACTSSearchResourcesServer.run_webserver()
