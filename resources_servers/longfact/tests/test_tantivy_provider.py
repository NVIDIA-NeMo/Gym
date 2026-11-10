# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
from contextlib import closing

import pytest
import tantivy

from resources_servers.longfact.tantivy_provider import TantivyProvider, TantivyProviderError


@pytest.fixture
def tantivy_index_and_corpus(tmp_path):
    schema = (
        tantivy.SchemaBuilder()
        .add_text_field("docid", stored=True, tokenizer_name="raw")
        .add_text_field("text")
        .add_text_field("url", stored=True)
        .add_text_field("title", stored=True)
        .build()
    )
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    index = tantivy.Index(schema, path=str(index_dir))
    writer = index.writer()
    writer.add_document(tantivy.Document(docid="1", text="Ada Lovelace"))
    writer.add_document(
        tantivy.Document(
            docid="2",
            text="Grace Hopper",
            title="Stored Grace title",
            url="https://index.example.test/displayoverride",
        )
    )
    writer.commit()
    writer.wait_merging_threads()

    corpus_path = tmp_path / "corpus.sqlite"
    with closing(sqlite3.connect(corpus_path)) as corpus:
        corpus.execute("CREATE TABLE corpus (docid TEXT PRIMARY KEY, url TEXT, text TEXT)")
        corpus.executemany(
            "INSERT INTO corpus VALUES (?, ?, ?)",
            [
                ("1", "https://example.test/ada", "Ada was a mathematician."),
                ("2", "https://example.test/grace", "Grace wrote a compiler."),
            ],
        )
        corpus.commit()
    return index_dir, corpus_path


@pytest.mark.parametrize(
    ("max_result_chars", "expected_snippet"), [(1000, "Ada was a mathematician."), (7, "Ada was")]
)
@pytest.mark.asyncio
async def test_search_hydrates_corpus_and_uses_its_url_for_missing_display_fields(
    tantivy_index_and_corpus, max_result_chars, expected_snippet
):
    index_dir, corpus_path = tantivy_index_and_corpus
    provider = TantivyProvider(
        index_dir=str(index_dir),
        corpus_db_path=str(corpus_path),
        max_results=2,
        candidate_pool=2,
        max_result_chars=max_result_chars,
    )

    evidence = await provider.search("Ada")

    assert [(item.title, item.url, item.snippet) for item in evidence] == [
        ("https://example.test/ada", "https://example.test/ada", expected_snippet)
    ]


@pytest.mark.parametrize("query", ["Grace", "displayoverride"])
@pytest.mark.asyncio
async def test_search_matches_text_and_url_and_preserves_stored_display_fields(tantivy_index_and_corpus, query):
    index_dir, corpus_path = tantivy_index_and_corpus
    provider = TantivyProvider(
        index_dir=str(index_dir),
        corpus_db_path=str(corpus_path),
        max_results=2,
        candidate_pool=2,
        max_result_chars=1000,
    )

    evidence = await provider.search(query)

    assert [(item.title, item.url, item.snippet) for item in evidence] == [
        ("Stored Grace title", "https://index.example.test/displayoverride", "Grace wrote a compiler.")
    ]


@pytest.mark.parametrize("missing_docids", [("1",), ("1", "2")])
def test_startup_probe_rejects_partly_or_fully_mismatched_corpus(tantivy_index_and_corpus, missing_docids):
    index_dir, corpus_path = tantivy_index_and_corpus
    with closing(sqlite3.connect(corpus_path)) as corpus:
        corpus.executemany("DELETE FROM corpus WHERE docid = ?", [(docid,) for docid in missing_docids])
        corpus.commit()

    with pytest.raises(TantivyProviderError, match="startup hydration probe failed.*no corpus rows"):
        TantivyProvider(
            index_dir=str(index_dir),
            corpus_db_path=str(corpus_path),
            max_results=1,
            candidate_pool=2,
            max_result_chars=1000,
        )


@pytest.mark.parametrize("missing_docids", [("1",), ("2",), ("1", "2")])
@pytest.mark.asyncio
async def test_search_rejects_any_missing_corpus_row_even_beyond_result_limit(
    tantivy_index_and_corpus, missing_docids
):
    index_dir, corpus_path = tantivy_index_and_corpus
    provider = TantivyProvider(
        index_dir=str(index_dir),
        corpus_db_path=str(corpus_path),
        max_results=1,
        candidate_pool=2,
        max_result_chars=1000,
    )
    with closing(sqlite3.connect(corpus_path)) as corpus:
        corpus.executemany("DELETE FROM corpus WHERE docid = ?", [(docid,) for docid in missing_docids])
        corpus.commit()

    with pytest.raises(TantivyProviderError, match="Tantivy search failed.*no corpus rows"):
        await provider.search("Ada OR Grace")


@pytest.mark.asyncio
async def test_no_matching_index_hits_is_a_valid_empty_search(tantivy_index_and_corpus):
    index_dir, corpus_path = tantivy_index_and_corpus
    provider = TantivyProvider(
        index_dir=str(index_dir),
        corpus_db_path=str(corpus_path),
        max_results=2,
        candidate_pool=2,
        max_result_chars=1000,
    )

    assert await provider.search("unmatchedkeyword") == []


def test_document_value_normalizes_tantivy_scalar_and_list_fields():
    assert TantivyProvider._value({"docid": [123]}, "docid") == "123"
    assert TantivyProvider._value({"docid": "abc"}, "docid") == "abc"
    assert TantivyProvider._value({"docid": []}, "docid") == ""
    assert TantivyProvider._value({}, "docid") == ""
