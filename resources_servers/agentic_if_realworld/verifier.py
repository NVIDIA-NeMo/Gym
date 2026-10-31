# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen verification and synthetic fixtures, independent of the HTTP server."""

import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.agentic_if_realworld.benchmark.core import (
    DEFAULT_JUDGE_MODEL,
    Dataset,
    digest,
    judge,
    read,
    source_preflight,
    unscored,
)
from resources_servers.agentic_if_realworld.benchmark.scoring import score
from resources_servers.agentic_if_realworld.benchmark.transport import delivered, verified_response


class VerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    task_id: str
    verifier_metadata: dict
    cached_judge_only: bool = False
    validate_judge_cache: bool = False


class VerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    scoring_status: str
    item_verdict: str | None = None
    item_id: str
    evaluation_completed: bool


def project_gym_response(response: NeMoGymResponse) -> dict:
    """Make a judging copy; keep the original Gym response and token metadata intact."""
    if response.status != "completed" or response.error or response.incomplete_details:
        raise ValueError("Incomplete Gym response")
    message = {"role": "assistant", "content": "", "refusal": ""}
    reasoning, messages = [], 0
    for item in response.output:
        if item.type == "reasoning":
            # Gym's Chat adapter puts parsed reasoning in summary; Responses
            # providers may expose full reasoning in content instead.
            reasoning.extend(part.text for part in (item.content or item.summary))
        elif item.type == "message" and item.status == "completed":
            messages += 1
            for part in item.content:
                field = "content" if part.type == "output_text" else "refusal"
                message[field] += part.text if field == "content" else part.refusal
        else:
            raise ValueError("Incomplete message or native action in Gym response")
    if messages != 1:
        raise ValueError("Expected one assistant message")
    if reasoning:
        message["reasoning_content"] = "\n\n".join(reasoning)
    return delivered({"choices": [{"finish_reason": "stop", "message": message}]})


class Verifier:
    def __init__(
        self,
        *,
        dataset: Dataset,
        receipt_dir: Path,
        endpoint: str = "",
        model: str = DEFAULT_JUDGE_MODEL,
        key_env: str = "AGENTIC_IF_JUDGE_API_KEY",
        reasoning: str = "medium",
        response_source: Literal["raw_receipt", "gym_response"] = "raw_receipt",
    ):
        self.dataset, self.receipt_dir = dataset, receipt_dir
        self.response_source = response_source
        self.endpoint, self.model, self.key_env, self.reasoning = (
            endpoint,
            model,
            key_env,
            reasoning,
        )

    async def evaluate(self, payload: dict, *, cached_only: bool = False, validate_only: bool = False) -> dict:
        return await score(
            payload,
            directory=self.receipt_dir,
            endpoint=self.endpoint,
            policy_prompt=self.dataset.judge_prompt(),
            model=self.model,
            key_env=self.key_env,
            reasoning=self.reasoning,
            cached_only=cached_only,
            validate_only=validate_only,
        )

    async def verify(self, body: VerifyRequest) -> VerifyResponse:
        identifier = body.task_id
        metadata = body.verifier_metadata
        expected = self.dataset.request(identifier)
        if metadata != {
            "item_id": identifier,
            "dataset_sha256": self.dataset.manifest_sha256,
            "input_sha256": judge.sha(expected),
        }:
            raise ValueError("Verifier input provenance mismatch")
        actual = body.responses_create_params.model_dump()["input"]
        # Pydantic may represent the one text content as an input_text block.
        if len(actual) != 1 or actual[0]["role"] != "user":
            raise ValueError("Model-visible wrapper changed")
        content = actual[0]["content"]
        if isinstance(content, list):
            if len(content) != 1 or content[0].get("type") != "input_text":
                raise ValueError("Model-visible content projection changed")
            content = content[0]["text"]
        if (
            content != expected["messages"][0]["content"]
            or body.responses_create_params.tools
            or body.responses_create_params.instructions
            or body.responses_create_params.previous_response_id
            or body.responses_create_params.prompt
            or body.responses_create_params.conversation is not None
        ):
            raise ValueError("Model-visible input changed or native tools enabled")
        if self.response_source == "gym_response":
            try:
                response = project_gym_response(body.response)
            except ValueError:
                if body.cached_judge_only or body.validate_judge_cache:
                    raise
                result = unscored(identifier, "incomplete_or_unsupported_gym_response")
            else:
                result = await self.score_response(
                    identifier,
                    expected,
                    response,
                    cached_only=body.cached_judge_only,
                    validate_only=body.validate_judge_cache,
                )
            return VerifyResponse(
                **{**body.model_dump(), **result, "evaluation_completed": result["scoring_status"] == "scored"}
            )

        response = self.receipt_response(body, expected)
        if response is None:
            if body.cached_judge_only or body.validate_judge_cache:
                raise ValueError("Saved selected model delivery failed integrity validation")
            result = unscored(identifier, "incomplete_model_delivery")
        else:
            result = await self.score_response(
                identifier,
                expected,
                response,
                cached_only=body.cached_judge_only,
                validate_only=body.validate_judge_cache,
            )
        return VerifyResponse(
            **{**body.model_dump(), **result, "evaluation_completed": result["scoring_status"] == "scored"}
        )

    async def score_response(
        self,
        identifier: str,
        expected: dict,
        response: dict,
        *,
        cached_only: bool = False,
        validate_only: bool = False,
    ) -> dict:
        binding = {
            "id": identifier,
            "input_sha256": judge.sha(expected),
            "response_sha256": judge.sha(response),
            "benchmark_manifest_sha256": self.dataset.manifest_sha256,
        }
        payload = self.dataset.payload(identifier, response, binding)
        if cached_only:
            return await self.evaluate(payload, cached_only=True)
        if validate_only:
            return await self.evaluate(payload, validate_only=True)
        return await self.evaluate(payload)

    def receipt_response(self, body: VerifyRequest, expected: dict) -> dict | None:
        receipt_id = (body.response.metadata or {}).get("receipt_id", "")
        if not re.fullmatch(r"[a-f0-9]{64}", receipt_id):
            raise ValueError("Missing trusted model delivery receipt")
        path = self.receipt_dir / "model" / (receipt_id + ".json")
        delivery = read(path)
        if {k: delivery["request_body"][k] for k in ["messages"]} != expected:
            raise ValueError("Actual upstream prompt differs")
        if delivery["request_sha256"] != judge.sha(delivery["request_body"]):
            raise ValueError("Actual request hash differs")
        for raw in delivery["raw_attempts"]:
            if digest(Path(raw["path"])) != raw["sha256"]:
                raise ValueError("Raw model delivery changed")
        try:
            response = delivered(verified_response(delivery))
        except (KeyError, ValueError, TypeError):
            return None
        if body.response.output_text != (
            response["message"].get("content") or response["message"].get("refusal") or ""
        ):
            raise ValueError("Gym response differs from selected raw answer")
        return response


@lru_cache(maxsize=2)
def cached_dataset(directory: str, synthetic: bool) -> Dataset:
    return Dataset(directory, synthetic=synthetic)


@lru_cache(maxsize=2)
def cached_preflight(split_registry: str, ground_truth: str, directory: str) -> dict:
    return source_preflight(split_registry, ground_truth, directory=directory)


class FixtureVerifier(Verifier):
    """Use production verification and synthetic receipts, replacing only the judge."""

    def __init__(self):
        from resources_servers.agentic_if_realworld.benchmark.synthetic import create, delivery

        self.work = tempfile.TemporaryDirectory(prefix="agentic-if-verifier-")
        root = Path(self.work.name)
        create(root / "data")
        super().__init__(
            dataset=Dataset(root / "data", synthetic=True),
            receipt_dir=root / "receipts",
        )
        for text, finish in [("OK.", "stop"), ("OK..", "stop"), ("partial", "length")]:
            delivery(self.dataset, self.receipt_dir, text, finish)

    async def evaluate(self, payload: dict, *, cached_only: bool = False, validate_only: bool = False) -> dict:
        from resources_servers.agentic_if_realworld.benchmark.synthetic import review

        checked = judge.validate_review(review(payload), payload)
        return {
            **checked,
            "reward": float(checked["item_verdict"] == "pass"),
            "mask_sample": False,
        }


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=FixtureVerifier,
    request_model=VerifyRequest,
    cases_path=Path(__file__).parent / "tests/verifier_cases.jsonl",
)
