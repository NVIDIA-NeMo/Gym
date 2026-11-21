# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline onboarding cases with scripted judge and retrieval transports."""

from __future__ import annotations

import json as jsonlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.verification import VerifierFixture
from resources_servers.longfact.evaluator import JsonEvaluator
from resources_servers.longfact.tavily_provider import TavilyProvider


if TYPE_CHECKING:
    from resources_servers.longfact.app import LongFactSafeServer


FixtureLabel = Literal["SUPPORTED", "CONTRADICTED", "UNSURE", "IRRELEVANT"]


class LongFactFixtureRequest(BaseModel):
    prompt: str
    response_text: str = Field(pattern=r"\S")
    judge_label: Literal["SUPPORTED", "CONTRADICTED", "UNSURE"] = "SUPPORTED"
    claim_labels: list[FixtureLabel] | None = None


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="offline-response",
        created_at=0,
        model="offline-fixture",
        object="response",
        status="completed",
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        output=[
            {
                "id": "offline-message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    )


@dataclass(frozen=True)
class _JudgeResponse:
    payload: bytes
    ok: bool = True

    async def read(self) -> bytes:
        return self.payload


class _JudgeTransport:
    def __init__(self) -> None:
        self.label: Literal["SUPPORTED", "CONTRADICTED", "UNSURE"] | None = None
        self.claim_labels: list[FixtureLabel] | None = None
        self.requests: list[dict[str, object]] = []
        self.searches: list[str] = []
        self.global_config_dict = {"observability_enabled": True}

    def _label_for(self, fact: str) -> FixtureLabel:
        if self.claim_labels is not None:
            return self.claim_labels[int(fact.removeprefix("Fixture fact ").removesuffix("."))]
        if self.label is None:
            raise RuntimeError("Offline judge case must be configured before verification")
        return self.label

    async def post(self, *, json: NeMoGymResponseCreateParamsNonStreaming, **kwargs: object) -> _JudgeResponse:
        if self.label is None and self.claim_labels is None:
            raise RuntimeError("Offline judge case must be configured before verification")
        schema_name = json.text["format"]["name"]
        prompt = json.input[-1].content
        if schema_name == "AtomicFacts":
            count = len(self.claim_labels) if self.claim_labels is not None else 100
            payload = [f"Fixture fact {index}." for index in range(count)]
        elif schema_name == "PreparedFact":
            fact = prompt.rsplit("ATOMIC FACT:\n", 1)[1].strip()
            payload = {
                "self_contained_fact": fact,
                "relevant": self._label_for(fact) != "IRRELEVANT",
                "queries": [fact],
            }
        elif schema_name == "ThreeWayClaimJudgment":
            fact = prompt.split("ATOMIC FACT:\n", 1)[1].split("\n\nRETRIEVED EVIDENCE:", 1)[0]
            payload = {"label": self._label_for(fact), "explanation": "Scripted external judge response."}
        elif schema_name in {"BatchSearchPlan", "CardBatchVerdicts"}:
            claims = jsonlib.loads(prompt.rsplit("Claims:\n", 1)[1].split("\n\nSearch results:", 1)[0])
            if schema_name == "BatchSearchPlan":
                payload = {"claims": [{"claim_id": item["claim_id"], "queries": [item["claim"]]} for item in claims]}
            else:
                answers = {"SUPPORTED": "true", "CONTRADICTED": "false", "UNSURE": "unsure"}
                payload = {
                    "verdicts": [
                        {
                            "claim_id": item["claim_id"],
                            "claim": item["claim"],
                            "answer": answers[self._label_for(item["claim"])],
                            "reasoning": "Scripted external card verdict.",
                            "supporting_evidence": [],
                        }
                        for item in claims
                    ]
                }
        else:
            raise ValueError(f"Unexpected offline judge schema: {schema_name}")
        response = _response(jsonlib.dumps(payload))
        self.requests.append({"schema": schema_name, **kwargs})
        response.id = f"offline-judge-{len(self.requests)}"
        return _JudgeResponse(response.model_dump_json().encode())


class _SearchTransport:
    def __init__(self, searches: list[str]) -> None:
        self.searches = searches

    async def search(self, *, query: str, **_kwargs: object) -> dict[str, list[dict[str, str]]]:
        self.searches.append(query)
        return {
            "results": [
                {
                    "title": "Offline fixture evidence",
                    "url": "https://example.test/longfact-fixture",
                    "content": "Scripted retrieval response for verifier contract checks.",
                }
            ]
        }


def _server_factory() -> LongFactSafeServer:
    from resources_servers.longfact.app import LongFactConfig, LongFactSafeServer

    class OfflineLongFactServer(LongFactSafeServer):
        def model_post_init(self, context: object) -> None:
            # Only external I/O setup changes; verification, parsing and scoring use the production methods.
            self._evaluator = JsonEvaluator(
                server_client=self.server_client,
                model_server=self.config.evaluator_model_server,
                create_params=self.config.evaluator_responses_create_params,
                max_concurrency=self.config.evaluator_max_concurrency,
                max_attempts=self.config.evaluator_max_attempts,
                timeout_seconds=self.config.evaluator_timeout_seconds,
            )
            self._search_provider = TavilyProvider(
                api_key="offline-fixture",
                search_depth=self.config.tavily_search_depth,
                max_results=self.config.max_search_results,
                timeout_seconds=self.config.tavily_timeout_seconds,
                max_concurrency=self.config.tavily_max_concurrency,
                search_max_attempts=self.config.tavily_search_max_attempts,
                cache_mode="live",
                client=_SearchTransport(self.server_client.searches),
            )

    config = LongFactConfig(
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
        name="longfact_offline_fixture",
        evaluator_model_server={"type": "responses_api_models", "name": "offline_judge"},
        evaluator_responses_create_params={"input": []},
        search_backend="tavily",
    )
    return OfflineLongFactServer.model_construct(config=config, server_client=_JudgeTransport())


async def invoke_fixture(server: object, body: BaseModel) -> dict[str, object]:
    from resources_servers.longfact.app import BENCHMARK_VARIANT, LongFactSafeServer, LongFactVerifyRequest

    if not isinstance(server, LongFactSafeServer) or not isinstance(body, LongFactFixtureRequest):
        raise TypeError("LongFact fixture requires its resources server and typed request")
    transport = server.server_client
    if not isinstance(transport, _JudgeTransport):
        raise TypeError("LongFact fixture requires the offline judge transport")
    transport.label = body.judge_label
    transport.claim_labels = body.claim_labels
    result = await server.verify(
        LongFactVerifyRequest(
            responses_create_params={"input": []},
            response=_response(body.response_text),
            id="longfact-offline-fixture",
            benchmark_variant=BENCHMARK_VARIANT,
            prompt=body.prompt,
            source_provenance={"source": "offline-verifier-fixture"},
        )
    )
    return result.model_dump(mode="json")


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=_server_factory,
    request_model=LongFactFixtureRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
    invoke=invoke_fixture,
)
