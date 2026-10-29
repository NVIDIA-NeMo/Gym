# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-turn LongFact evaluation using SAFE-style claim verification.

LongFact SAFE used Serper and free-form parsing; this adapter supports Tantivy,
Tavily, or Milvus retrieval and uses NeMo Gym model transports and JSON schemas
while retaining the published F1@K aggregation formula (K = f1_max_claims, 100 by default).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any, Literal, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, PrivateAttr, RootModel, field_validator

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.gymnasium import GymnasiumServer, extract_text
from resources_servers.longfact.claim_prompts import ClaimExtractionPrompt, build_claim_extraction_request
from resources_servers.longfact.evaluator import (
    EvaluatorIdentityError,
    EvaluatorParseError,
    JsonEvaluator,
)
from resources_servers.longfact.milvus_provider import MilvusProvider
from resources_servers.longfact.tantivy_provider import TantivyProvider
from resources_servers.longfact.tavily_provider import TavilyProvider


BENCHMARK_VARIANT = "longfact-safe-v1"
F1_MAX_CLAIMS = 100


class AtomicFacts(RootModel[list[str]]):
    """The evaluator returns a bare JSON array, matching the proven runner."""


class PreparedFact(BaseModel):
    self_contained_fact: str
    relevant: bool = True
    queries: list[str] = Field(default_factory=list)


class BinaryClaimJudgment(BaseModel):
    label: Literal["SUPPORTED", "NOT_SUPPORTED"]
    explanation: str = ""


class ThreeWayClaimJudgment(BaseModel):
    label: Literal["SUPPORTED", "CONTRADICTED", "UNSURE"]
    explanation: str = ""

    @field_validator("label", mode="before")
    @classmethod
    def normalize_legacy_unsupported_label(cls, value: Any) -> Any:
        # Some otherwise well-formed judges fall back to SAFE's legacy binary
        # vocabulary even when explicitly given the three-way contract. A
        # generic unsupported verdict does not establish a contradiction, so
        # preserve the conservative three-way semantics by treating it as
        # UNSURE instead of discarding the claim as an evaluator error.
        if isinstance(value, str):
            normalized = "_".join(value.strip().upper().replace("-", " ").split())
            if normalized in {"UNSUPPORTED", "NOT_SUPPORTED"}:
                return "UNSURE"
            if normalized in {"SUPPORTED", "CONTRADICTED", "UNSURE"}:
                return normalized
        return value


ClaimJudgment = BinaryClaimJudgment | ThreeWayClaimJudgment


class LongFactClaimCheckError(RuntimeError):
    """At least one extracted claim has no verdict after every configured attempt."""


class LongFactConfig(BaseResourcesServerConfig):
    evaluator_model_server: ModelServerRef
    evaluator_model_identity: str = "unknown"
    evaluator_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    claim_extraction_prompt: ClaimExtractionPrompt = "legacy_v1"
    claim_verdict_mode: Literal["binary", "three_way"] = "three_way"
    search_backend: Literal["tantivy", "tavily", "milvus"] = "tantivy"
    tavily_api_key: str = "MISSING"
    tantivy_index_dir: str = "MISSING"
    tantivy_corpus_db_path: str = "MISSING"
    tantivy_candidate_pool: int = Field(default=32, ge=1)
    tantivy_max_result_chars: int = Field(default=6000, ge=1)
    milvus_uri: str = "MISSING"
    milvus_collection_name: str = "MISSING"
    milvus_token: str = "MISSING"
    milvus_embedding_model: str = "google/embeddinggemma-300m"
    milvus_embedding_device: str = "cpu"
    milvus_anns_field: str = "emb"
    milvus_search_list: int = Field(default=256, ge=1)
    milvus_timeout_seconds: float = Field(default=30.0, gt=0)
    milvus_candidate_multiplier: int = Field(default=4, ge=1)
    milvus_max_result_chars: int = Field(default=8000, ge=1)
    milvus_cache_size: int = Field(default=10000, ge=0)
    benchmark_variant: Literal["longfact-safe-v1"] = BENCHMARK_VARIANT
    max_claims: int = Field(default=0, ge=0)
    max_queries: int = Field(default=3, ge=1)
    max_search_results: int = Field(default=3, ge=1)
    f1_max_claims: int = Field(default=F1_MAX_CLAIMS, ge=1)
    # Guided decoding with the Pydantic schemas works on vLLM; OpenAI strict mode rejects them (array root).
    evaluator_constrain_json_schema: bool = True
    evaluator_max_concurrency: int = Field(default=128, ge=1)
    evaluator_max_attempts: int = Field(default=6, ge=1)
    evaluator_timeout_seconds: float = Field(default=180.0, gt=0)
    tavily_max_concurrency: int = Field(default=15, ge=1)
    tavily_timeout_seconds: float = Field(default=90.0, gt=0)
    tavily_search_max_attempts: int = Field(default=4, ge=1)
    tavily_search_depth: Literal["basic", "advanced"] = "advanced"
    tavily_cache_mode: Literal["live", "read_write", "replay"] = "live"
    tavily_cache_dir: str | None = "/tmp/longfact-safe-tavily-cache"


class LongFactSafeServer(GymnasiumServer):
    config: LongFactConfig

    _evaluator: JsonEvaluator = PrivateAttr()
    _search_provider: TantivyProvider | TavilyProvider | MilvusProvider = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        self._evaluator = JsonEvaluator(
            server_client=self.server_client,
            model_server=self.config.evaluator_model_server,
            create_params=self.config.evaluator_responses_create_params,
            max_concurrency=self.config.evaluator_max_concurrency,
            max_attempts=self.config.evaluator_max_attempts,
            timeout_seconds=self.config.evaluator_timeout_seconds,
            expected_model_identity=self.config.evaluator_model_identity,
        )
        if self.config.search_backend == "tantivy":
            self._search_provider = TantivyProvider(
                index_dir=self.config.tantivy_index_dir,
                corpus_db_path=self.config.tantivy_corpus_db_path,
                max_results=self.config.max_search_results,
                candidate_pool=self.config.tantivy_candidate_pool,
                max_result_chars=self.config.tantivy_max_result_chars,
            )
        elif self.config.search_backend == "tavily":
            key = self.config.tavily_api_key.strip()
            if self.config.tavily_cache_mode != "replay" and (not key or key.upper() in {"MISSING", "NONE", "NULL"}):
                raise ValueError("TAVILY_API_KEY must be set for search_backend=tavily")
            self._search_provider = TavilyProvider(
                api_key="" if self.config.tavily_cache_mode == "replay" else key,
                search_depth=self.config.tavily_search_depth,
                max_results=self.config.max_search_results,
                timeout_seconds=self.config.tavily_timeout_seconds,
                max_concurrency=self.config.tavily_max_concurrency,
                search_max_attempts=self.config.tavily_search_max_attempts,
                cache_mode=self.config.tavily_cache_mode,
                cache_dir=self.config.tavily_cache_dir,
            )
        else:
            uri = self.config.milvus_uri.strip()
            collection_name = self.config.milvus_collection_name.strip()
            missing_values = {"", "MISSING", "NONE", "NULL"}
            if uri.upper() in missing_values:
                raise ValueError("LONGFACT_MILVUS_URI must be set for search_backend=milvus")
            if collection_name.upper() in missing_values:
                raise ValueError("LONGFACT_MILVUS_COLLECTION_NAME must be set for search_backend=milvus")
            self._search_provider = MilvusProvider(
                uri=uri,
                collection_name=collection_name,
                token=(
                    None
                    if self.config.milvus_token.strip().upper() in missing_values
                    else self.config.milvus_token.strip()
                ),
                embedding_model=self.config.milvus_embedding_model,
                embedding_device=self.config.milvus_embedding_device,
                anns_field=self.config.milvus_anns_field,
                search_list=self.config.milvus_search_list,
                timeout_seconds=self.config.milvus_timeout_seconds,
                candidate_multiplier=self.config.milvus_candidate_multiplier,
                max_results=self.config.max_search_results,
                max_result_chars=self.config.milvus_max_result_chars,
                cache_size=self.config.milvus_cache_size,
            )
        super().model_post_init(context)

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        prompt = str(metadata.get("prompt") or "").strip()
        if metadata.get("benchmark_variant") != self.config.benchmark_variant:
            raise ValueError("LongFact benchmark_variant does not match this resource server")
        if not prompt:
            raise ValueError("LongFact row is missing prompt")
        provenance = metadata.get("source_provenance")
        if not isinstance(provenance, dict):
            raise ValueError("LongFact row is missing source_provenance")
        key = self._session_key(session_id, metadata)
        state = {
            "task_id": str(metadata.get("id") or key),
            "subset": str(metadata.get("subset") or "unknown"),
            "topic": str(metadata.get("topic") or "unknown"),
            "prompt": prompt,
            "source_provenance": dict(provenance),
        }
        self.session_state[key] = state
        # A failed step raises before the step route closes the session, so the agent closes it via /close.
        return None, self._info(state) | {"supports_explicit_close": True}

    async def _close_endpoint(self, request: Request) -> dict[str, bool]:
        await self.close_session(request.session.get(SESSION_ID_KEY))
        self.checkpoint_session_ended(request)
        return {"ok": True}

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/close")(self._close_endpoint)
        return app

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        key = self._session_key(session_id, metadata)
        state = self.session_state.get(key)
        if state is None:
            raise RuntimeError("LongFact session is missing; /reset must precede /step")
        response = extract_text(action)
        # Judge and search failures propagate and fail the rollout, so resume runs it again.
        evaluation = await self._evaluate(state["prompt"], response)
        if evaluation["error_claims"]:
            # An unchecked claim counts in neither precision nor recall, so the score is not a measurement.
            first_error = next(trace["fact_error"] for trace in evaluation["traces"] if trace.get("label") == "ERROR")
            raise LongFactClaimCheckError(
                f"{evaluation['error_claims']} of {evaluation['num_extracted_claims']} claim(s) could not be "
                f"checked; first: {first_error}"
            )
        state["response"] = response
        state.update(evaluation)
        return None, float(evaluation["f1_at_k"]), True, False, self._info(state)

    async def _evaluate(self, prompt: str, response: str) -> dict[str, Any]:
        extracted = await self._generate_json(
            AtomicFacts,
            self._claim_extraction_request(prompt, response),
        )
        facts = [fact.strip() for fact in extracted.root if isinstance(fact, str) and fact.strip()]
        if self.config.max_claims:
            facts = facts[: self.config.max_claims]

        async def evaluate_fact_safely(fact: str) -> dict[str, Any]:
            try:
                return await self._evaluate_fact(prompt, response, fact)
            except EvaluatorIdentityError:
                raise
            except Exception as exc:  # noqa: BLE001 - retain per-claim failure provenance.
                return {
                    "atomic_fact": fact,
                    "label": "ERROR",
                    "fact_error": repr(exc),
                    "evidence": [],
                }

        results = await asyncio.gather(*(evaluate_fact_safely(fact) for fact in facts))
        labels = Counter(result["label"] for result in results)
        supported = labels["SUPPORTED"]
        contradicted = labels["CONTRADICTED"]
        unsure = labels["UNSURE"]
        # Preserve the published SAFE metrics by collapsing the two three-way
        # non-supported verdicts.
        not_supported = labels["NOT_SUPPORTED"] + contradicted + unsure
        denominator = supported + not_supported
        precision = supported / denominator if denominator else 1.0
        f1 = self._f1_at_k(supported, not_supported, self.config.f1_max_claims)
        evaluation = {
            "claim_extraction_prompt": self.config.claim_extraction_prompt,
            "claim_verdict_mode": self.config.claim_verdict_mode,
            "num_extracted_claims": len(facts),
            "supported_claims": supported,
            "not_supported_claims": not_supported,
            "irrelevant_claims": labels["IRRELEVANT"],
            "error_claims": labels["ERROR"],
            "precision": precision,
            "f1_at_k": f1,
            "traces": results,
        }
        if self.config.claim_verdict_mode == "three_way":
            classified = supported + contradicted + unsure
            decisive = supported + contradicted
            evaluation.update(
                {
                    "contradicted_claims": contradicted,
                    "unsure_claims": unsure,
                    "decisive_precision": supported / decisive if decisive else 1.0,
                    "contradiction_rate": contradicted / classified if classified else 0.0,
                    "unsure_rate": unsure / classified if classified else 0.0,
                }
            )
        return evaluation

    def _claim_extraction_request(self, prompt: str, response: str) -> str:
        return build_claim_extraction_request(self.config.claim_extraction_prompt, prompt, response)

    async def _evaluate_fact(self, prompt: str, response: str, fact: str) -> dict[str, Any]:
        prepared = await self._prepare_fact(prompt, response, fact)
        trace: dict[str, Any] = {
            "atomic_fact": fact,
            "self_contained_fact": prepared.self_contained_fact,
            "relevant": prepared.relevant,
            "queries": prepared.queries,
            "searches": [],
        }
        if not prepared.relevant:
            trace["label"] = "IRRELEVANT"
            trace["evidence"] = []
            return trace

        queries = prepared.queries or [prepared.self_contained_fact]
        query_results = await asyncio.gather(*(self._search_provider.search(query) for query in queries))
        evidence: list[dict[str, Any]] = []
        for query, sources in zip(queries, query_results, strict=True):
            evidence = [
                {
                    "title": source.title,
                    "url": source.url,
                    "text": source.snippet,
                    "score": source.search_score,
                }
                for source in sources
            ]
            trace["searches"].append({"query": query, "results": evidence})
            trace.setdefault("evidence", []).extend(evidence)

        judgment = await self._judge_fact(
            prepared.self_contained_fact,
            trace["evidence"],
        )
        trace.update(
            {
                "label": judgment.label,
                "explanation": judgment.explanation,
            }
        )
        return trace

    async def _prepare_fact(self, prompt: str, response: str, fact: str) -> PreparedFact:
        request = f"""For the factuality evaluation below:
1. Rewrite the ATOMIC FACT to replace vague pronouns or references using RESPONSE context, without adding claims.
2. Decide whether the fact is relevant to answering QUESTION.
3. If relevant, propose 1 to 3 concise web-search queries that would verify the fact.

Return only JSON:
{{"self_contained_fact":"...", "relevant":true, "queries":["..."]}}

QUESTION:
{prompt}

RESPONSE:
{response}

ATOMIC FACT:
{fact}
"""
        # A failure here propagates: this stage also decides relevance, so guessing it would change the score.
        prepared = await self._generate_json(PreparedFact, request)
        prepared.self_contained_fact = prepared.self_contained_fact.strip() or fact
        prepared.queries = [query.strip() for query in prepared.queries if isinstance(query, str) and query.strip()][
            : self.config.max_queries
        ]
        return prepared

    async def _judge_fact(self, fact: str, evidence: list[dict[str, Any]]) -> ClaimJudgment:
        evidence_text = "\n\n".join(
            f"Source: {item.get('url', '')}\nExcerpt: {item.get('text', '')}" for item in evidence
        )
        if self.config.claim_verdict_mode == "three_way":
            schema = ThreeWayClaimJudgment
            instruction = """Classify the ATOMIC FACT using only the retrieved evidence.
- SUPPORTED: the evidence establishes every material part of the fact and does not contradict it.
- CONTRADICTED: the evidence directly establishes that at least one material
  part of the fact is false. Missing evidence is not a contradiction.
- UNSURE: the evidence is missing, insufficient, ambiguous, mixed, or
  conflicting, so neither support nor contradiction is established.

Return only JSON:
{"label":"SUPPORTED" or "CONTRADICTED" or "UNSURE", "explanation":"brief evidence-grounded explanation"}"""
        else:
            schema = BinaryClaimJudgment
            instruction = """Determine whether the ATOMIC FACT is supported by the retrieved evidence.
Use only the evidence. If evidence is missing, irrelevant, or insufficient, choose NOT_SUPPORTED.
Return only JSON:
{"label":"SUPPORTED" or "NOT_SUPPORTED", "explanation":"brief evidence-grounded explanation"}"""
        request = f"""{instruction}

ATOMIC FACT:
{fact}

RETRIEVED EVIDENCE:
{evidence_text or "No evidence retrieved."}
"""
        # No verdict is a judge failure, not UNSURE: the claim becomes ERROR and the rollout fails.
        return await self._generate_json(schema, request)

    async def _generate_json(self, schema: type[BaseModel], prompt: str) -> Any:
        last_error: ValueError | None = None
        for _ in range(self.config.evaluator_max_attempts):
            try:
                return await self._evaluator.generate_json(
                    schema,
                    None,
                    prompt,
                    reasoning_effort=None,
                    include_schema_instruction=False,
                    salvage_embedded_json=True,
                    constrain_json_schema=self.config.evaluator_constrain_json_schema,
                )
            except ValueError as exc:
                last_error = exc
        # Preserve a distinct failure type so verdict callers can map an
        # exhausted judge refusal to UNSURE without hiding transport or model
        # identity failures.
        raise EvaluatorParseError(
            f"Evaluator returned malformed JSON {self.config.evaluator_max_attempts} times: {last_error}"
        )

    @staticmethod
    def _f1_at_k(supported: int, not_supported: int, max_claims: int) -> float:
        if supported <= 0:
            return 0.0
        precision = supported / (supported + not_supported)
        recall = min(1.0, supported / max_claims)
        return 2 * precision * recall / (precision + recall)

    def _info(self, state: dict[str, Any]) -> dict[str, Any]:
        info = {
            "benchmark_variant": self.config.benchmark_variant,
            "task_id": state["task_id"],
            "subset": state["subset"],
            "topic": state["topic"],
            "source_provenance": dict(state["source_provenance"]),
            "safe_variant": {
                "search_provider": self.config.search_backend,
                "claim_extraction_prompt": self.config.claim_extraction_prompt,
                "claim_verdict_mode": self.config.claim_verdict_mode,
                "original_search_provider": "serper",
                "f1_max_claims": self.config.f1_max_claims,
            },
        }
        for key in (
            "claim_extraction_prompt",
            "claim_verdict_mode",
            "num_extracted_claims",
            "supported_claims",
            "not_supported_claims",
            "contradicted_claims",
            "unsure_claims",
            "irrelevant_claims",
            "error_claims",
            "precision",
            "decisive_precision",
            "contradiction_rate",
            "unsure_rate",
            "f1_at_k",
            "traces",
        ):
            if key in state:
                info[key] = state[key]
        if getattr(self.config, "search_backend", None) == "tavily":
            info["tavily_cache_stats"] = self._search_provider.cache_stats
        return info

    @staticmethod
    def _session_key(session_id: Optional[str], metadata: dict[str, Any]) -> str:
        return session_id or f"local:{metadata.get('id', 'unknown')}"

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        records = [rollout.get("info", {}) for task in tasks for rollout in task]
        cache_snapshots = [
            record["tavily_cache_stats"] for record in records if isinstance(record.get("tavily_cache_stats"), dict)
        ]
        cache_metrics = {}
        if cache_snapshots:
            latest = max(cache_snapshots, key=lambda snapshot: int(snapshot.get("completed_lookups", 0)))
            for key in (
                "lookups",
                "completed_lookups",
                "persistent_hits",
                "cross_process_hits",
                "inflight_hits",
                "initial_misses",
                "live_fetches",
                "writes",
                "corruptions",
                "effective_hits",
                "persistent_hit_rate",
                "effective_hit_rate",
            ):
                if key in latest:
                    cache_metrics[f"longfact_safe/tavily_cache_{key}"] = latest[key]
        k = self.config.f1_max_claims
        scored = [record for record in records if "f1_at_k" in record]
        if not scored:
            return {
                "longfact_safe/episodes": len(records),
                "longfact_safe/mean_precision": 0.0,
                f"longfact_safe/mean_f1_at_{k}": 0.0,
                f"longfact_safe/micro_f1_at_{k}": 0.0,
                **cache_metrics,
            }
        total = lambda key: sum(int(record.get(key, 0)) for record in scored)
        supported, unsupported = total("supported_claims"), total("not_supported_claims")
        metrics = {
            "longfact_safe/episodes": len(records),
            "longfact_safe/scored_episodes": len(scored),
            "longfact_safe/supported_claims": supported,
            "longfact_safe/not_supported_claims": unsupported,
            "longfact_safe/irrelevant_claims": total("irrelevant_claims"),
            "longfact_safe/error_claims": total("error_claims"),
            "longfact_safe/mean_precision": (sum(float(record["precision"]) for record in scored) / len(scored)),
            # The published LongFact runner averages F1@K per prompt. Keep
            # that headline comparable at the aggregation level and expose a
            # claim-weighted diagnostic separately.
            f"longfact_safe/mean_f1_at_{k}": (sum(float(record["f1_at_k"]) for record in scored) / len(scored)),
            # Claim-weighted diagnostic: recall pools each scored episode's K, so it matches a single episode.
            f"longfact_safe/micro_f1_at_{k}": self._f1_at_k(supported, unsupported, k * len(scored)),
            **cache_metrics,
        }
        three_way = [record for record in scored if record.get("claim_verdict_mode") == "three_way"]
        if three_way:
            three_way_supported = sum(int(record.get("supported_claims", 0)) for record in three_way)
            contradicted = sum(int(record.get("contradicted_claims", 0)) for record in three_way)
            unsure = sum(int(record.get("unsure_claims", 0)) for record in three_way)
            classified = three_way_supported + contradicted + unsure
            decisive = three_way_supported + contradicted
            metrics.update(
                {
                    "longfact_safe/contradicted_claims": contradicted,
                    "longfact_safe/unsure_claims": unsure,
                    "longfact_safe/mean_decisive_precision": sum(
                        float(record["decisive_precision"]) for record in three_way
                    )
                    / len(three_way),
                    "longfact_safe/mean_contradiction_rate": sum(
                        float(record["contradiction_rate"]) for record in three_way
                    )
                    / len(three_way),
                    "longfact_safe/mean_unsure_rate": sum(float(record["unsure_rate"]) for record in three_way)
                    / len(three_way),
                    "longfact_safe/micro_supported_rate": (three_way_supported / classified if classified else 1.0),
                    "longfact_safe/micro_decisive_precision": (three_way_supported / decisive if decisive else 1.0),
                }
            )
        return metrics


if __name__ == "__main__":
    LongFactSafeServer.run_webserver()
