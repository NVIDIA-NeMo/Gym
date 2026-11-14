# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from resources_servers.longfact.milvus_provider import MilvusProvider, MilvusProviderError


class _Vector:
    def tolist(self):
        return [0.1, 0.2, 0.3]


class _Encoder:
    def __init__(self, dimension=3):
        self.dimension = dimension
        self.queries = []

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode_query(self, query, *, convert_to_numpy):
        assert convert_to_numpy is True
        self.queries.append(query)
        return _Vector()


class _Client:
    def __init__(self):
        self.searches = []

    def has_collection(self, *, collection_name):
        return collection_name == "fineweb"

    def describe_collection(self, *, collection_name):
        assert collection_name == "fineweb"
        return {"fields": [{"name": "emb", "params": {"dim": 3}}]}

    def search(self, **kwargs):
        self.searches.append(kwargs)
        return [
            [
                {"id": 1, "distance": 0.1, "entity": {"orig_id": "a", "text": "first passage"}},
                {"id": 2, "distance": 0.2, "entity": {"orig_id": "b", "text": "first passage"}},
                {"id": 3, "distance": 0.3, "entity": {"orig_id": "c", "text": "second passage"}},
            ]
        ]


def _provider(*, client=None, encoder=None):
    return MilvusProvider(
        uri="http://milvus:19530",
        collection_name="fineweb",
        token=None,
        embedding_model="embedding-model",
        embedding_device="cpu",
        anns_field="emb",
        search_list=256,
        timeout_seconds=30.0,
        candidate_multiplier=4,
        max_results=2,
        max_result_chars=100,
        client=client or _Client(),
        encoder=encoder or _Encoder(),
    )


def test_milvus_search_matches_training_query_and_deduplication_contract():
    client = _Client()
    encoder = _Encoder()
    provider = _provider(client=client, encoder=encoder)

    evidence = provider._search("  query text  ")

    assert encoder.queries == ["query text"]
    assert [item.snippet for item in evidence] == ["first passage", "second passage"]
    assert [item.url for item in evidence] == ["milvus://fineweb/a", "milvus://fineweb/c"]
    assert [item.search_score for item in evidence] == [-0.1, -0.3]
    assert client.searches == [
        {
            "collection_name": "fineweb",
            "data": [[0.1, 0.2, 0.3]],
            "limit": 8,
            "anns_field": "emb",
            "output_fields": ["orig_id", "text"],
            "search_params": {"search_list": 256},
        }
    ]

    assert provider._search("query text") == evidence
    assert len(client.searches) == 1


def test_milvus_provider_rejects_encoder_dimension_mismatch():
    with pytest.raises(MilvusProviderError, match="produces 4 dimensions"):
        _provider(encoder=_Encoder(dimension=4))
