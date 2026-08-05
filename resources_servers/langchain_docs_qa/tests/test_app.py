# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

from resources_servers.langchain_docs_qa.retriever import BM25
from resources_servers.langchain_docs_qa.scoring import (
    citation_match,
    mcqa_match,
    parse_answer,
    parse_choice,
)


CHUNKS = [
    {
        "page": "oss/streaming",
        "title": "Stream subgraph outputs",
        "link": "https://docs.langchain.com/oss/streaming",
        "content": "Set subgraphs=True to stream outputs from nested graphs.",
    },
    {
        "page": "langsmith/abac",
        "title": "ABAC",
        "link": "https://docs.langchain.com/langsmith/abac",
        "content": "ABAC supports resource_tag_key as an attribute_name in policies.",
    },
]


@pytest.mark.parametrize(
    "generation,expected",
    [
        ('{"answer": "C"}', "C"),
        ("the answer is B", "B"),
        ("I will go with D", "D"),
        ("no letter here at all", ""),
    ],
)
def test_parse_choice(generation, expected):
    assert parse_choice(generation) == expected


def test_mcqa_match_requires_exact_letter():
    assert mcqa_match("C", '{"answer": "C"}') == 1.0
    assert mcqa_match("C", '{"answer": "D"}') == 0.0
    assert mcqa_match("", '{"answer": "C"}') == 0.0


def test_parse_answer_reads_json_then_falls_back():
    assert parse_answer('{"answer": "42", "cited_pages": ["a/b"]}') == ("42", ["a/b"])
    assert parse_answer("just prose") == ("just prose", [])


def test_citation_match():
    assert citation_match("langsmith/abac", ["langsmith/abac"]) == 1.0
    assert citation_match("langsmith/abac", ["oss/streaming"]) == 0.0
    assert citation_match("", ["langsmith/abac"]) == 0.0


def test_bm25_ranks_the_relevant_chunk_first():
    hits = BM25(CHUNKS).search("how do I stream subgraph outputs", k=2)
    assert hits[0]["page"] == "oss/streaming"
