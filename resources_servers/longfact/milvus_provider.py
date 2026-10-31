# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU query embedding and Milvus retrieval for LongFact SAFE."""

from __future__ import annotations

import asyncio
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


class MilvusProviderError(RuntimeError):
    """Raised when the configured Milvus collection cannot serve a query."""


@dataclass(frozen=True)
class MilvusEvidence:
    title: str
    url: str
    snippet: str
    search_score: float


class MilvusProvider:
    """Search passages in a compatible Milvus vector collection."""

    def __init__(
        self,
        *,
        uri: str,
        collection_name: str,
        token: str | None = None,
        embedding_model: str,
        embedding_device: str,
        anns_field: str,
        search_list: int,
        timeout_seconds: float,
        candidate_multiplier: int,
        max_results: int,
        max_result_chars: int,
        cache_size: int = 10000,
        client: Any | None = None,
        encoder: Any | None = None,
    ) -> None:
        try:
            if client is None:
                from pymilvus import MilvusClient

                client_kwargs: dict[str, Any] = {"uri": uri, "timeout": timeout_seconds}
                if token:
                    client_kwargs["token"] = token
                client = MilvusClient(**client_kwargs)
            if encoder is None:
                from sentence_transformers import SentenceTransformer

                encoder = SentenceTransformer(embedding_model, device=embedding_device)
        except Exception as exc:  # pragma: no cover - depends on external packages and model access.
            raise MilvusProviderError(
                "pymilvus, sentence-transformers, and access to the query encoder are required "
                "for search_backend=milvus"
            ) from exc

        try:
            if not client.has_collection(collection_name=collection_name):
                raise MilvusProviderError(f"Milvus collection {collection_name!r} does not exist at {uri}.")
            schema = client.describe_collection(collection_name=collection_name)
            vector_dim = self._vector_dimension(schema, anns_field)
            encoder_dim = encoder.get_sentence_embedding_dimension()
            if encoder_dim is not None and int(encoder_dim) != vector_dim:
                raise MilvusProviderError(
                    f"Milvus field dimension is {vector_dim}, but query encoder "
                    f"{embedding_model!r} produces {encoder_dim} dimensions."
                )
        except MilvusProviderError:
            raise
        except Exception as exc:
            raise MilvusProviderError(f"Failed to initialize Milvus collection {collection_name!r}: {exc}") from exc

        self._client = client
        self._encoder = encoder
        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self._anns_field = anns_field
        self._search_list = search_list
        self._candidate_multiplier = candidate_multiplier
        self._max_results = max_results
        self._max_result_chars = max_result_chars
        self._encoder_lock = threading.Lock()
        self._cache_size = cache_size
        self._cache: OrderedDict[str, list[MilvusEvidence]] = OrderedDict()
        self._cache_lock = threading.Lock()

    @staticmethod
    def _vector_dimension(schema: dict[str, Any], anns_field: str) -> int:
        for field in schema.get("fields", []):
            if field.get("name") == anns_field:
                raw_dimension = field.get("params", {}).get("dim")
                if raw_dimension is not None:
                    return int(raw_dimension)
        raise MilvusProviderError(f"Milvus vector field {anns_field!r} is missing its dimension.")

    async def search(self, query: str) -> list[MilvusEvidence]:
        return await asyncio.to_thread(self._search, query)

    def _search(self, query: str) -> list[MilvusEvidence]:
        query = query.strip()
        if not query:
            return []
        cached = self._cache_get(query)
        if cached is not None:
            return cached

        try:
            with self._encoder_lock:
                encode_query = getattr(self._encoder, "encode_query", None)
                if encode_query is not None:
                    query_vector = encode_query(query, convert_to_numpy=True)
                else:
                    query_vector = self._encoder.encode(query, convert_to_numpy=True)
                vector = query_vector.tolist()
            if not vector or not all(math.isfinite(float(value)) for value in vector):
                raise MilvusProviderError("Milvus query encoder returned a non-finite vector.")

            candidate_count = max(self._max_results, self._max_results * self._candidate_multiplier)
            results = self._client.search(
                collection_name=self._collection_name,
                data=[vector],
                limit=candidate_count,
                anns_field=self._anns_field,
                output_fields=["orig_id", "text"],
                search_params={"search_list": self._search_list},
            )
        except MilvusProviderError:
            raise
        except Exception as exc:
            raise MilvusProviderError(f"Milvus search failed for {query!r}: {exc}") from exc

        evidence: list[MilvusEvidence] = []
        seen_texts: set[str] = set()
        raw_hits = results[0] if results else []
        for raw_hit in raw_hits:
            entity = raw_hit.get("entity") or {}
            text = str(entity.get("text") or "").strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            result_id = raw_hit.get("id")
            original_id = str(entity.get("orig_id") or result_id)
            distance = float(raw_hit.get("distance", 0.0))
            evidence.append(
                MilvusEvidence(
                    title=original_id,
                    url=f"milvus://{self._collection_name}/{original_id}",
                    snippet=text[: self._max_result_chars],
                    search_score=-distance,
                )
            )
            if len(evidence) == self._max_results:
                break

        self._cache_put(query, evidence)
        return list(evidence)

    def _cache_get(self, query: str) -> list[MilvusEvidence] | None:
        if self._cache_size <= 0:
            return None
        with self._cache_lock:
            value = self._cache.get(query)
            if value is None:
                return None
            self._cache.move_to_end(query)
            return list(value)

    def _cache_put(self, query: str, evidence: list[MilvusEvidence]) -> None:
        if self._cache_size <= 0:
            return
        with self._cache_lock:
            self._cache[query] = list(evidence)
            self._cache.move_to_end(query)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
