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
"""Benchmark-contamination guard: drop provider results that reference BrowseComp.

A search or browse result that quotes the benchmark itself (a dataset mirror, a
leaderboard, simple-evals) hands the model the answer. Training on those rollouts
teaches retrieval of the answer key rather than research, and it surfaces later as
an unearned score on the benchmark being measured.

PER-ITEM, not per-response: one poisoned hit out of five costs that hit, not the
whole tool call, on a 60-turn rollout. The whole output is withheld only when
EVERY item is contaminated.

FOUR EXITS. The returned string is not the only way a result reaches the model --
in terminal mode `_search_one_to_disk` and `browse` write each raw page to
`pages/*.txt`, which the model later reads with grep/cat through the bash tool. A
filter applied only to the returned string would leave the contamination on disk
and fully readable. The four paths a provider result can leave by are:

    _exa_search_one · _postprocess_search_results (inline tavily)
    _search_one_to_disk (incl. page writes) · browse (incl. page writes)

Every test below drives the real public entry point (`search` / `browse`) with a
mocked provider client, so a new exit added later that bypasses the guard fails
here rather than leaking silently.
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import fixture

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    CONTAMINATED_URL_SUBSTRINGS,
    CONTAMINATION_PATTERNS,
    BrowseRequest,
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
    _drop_contaminated,
    _is_contaminated,
    _is_contaminated_url,
)


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_DUMMY_EXCLUDE_DOMAINS_FILE = os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json")


def _results() -> list[dict]:
    """Three tavily-shaped results, the middle one contaminated."""
    return [
        {"title": "clean one", "url": "https://a.example/1", "content": "alpha", "raw_content": "alpha body"},
        {
            "title": "BrowseComp dataset",
            "url": "https://hf.co/datasets/openai/BrowseComp",
            "content": "leak",
            "raw_content": "leak body",
        },
        {"title": "clean two", "url": "https://b.example/2", "content": "beta", "raw_content": "beta body"},
    ]


def _dirty_only() -> list[dict]:
    return [r for r in _results() if "Browse" in r["title"]]


def _req() -> MagicMock:
    m = MagicMock()
    m.session = {SESSION_ID_KEY: "test_session_id"}
    return m


_TELLTALES = ("withheld", "benchmark", "browsecomp", "contaminat", "answer key")


def _assert_looks_like_an_ordinary_empty_search(s: str, query: str = "q") -> None:
    """An all-dropped search must be indistinguishable from a provider that returned
    nothing: the bare header, no entries, and no hint of WHY it is empty."""
    assert s.startswith(f"[Search Query]: {query}")
    assert "[Title]" not in s and "[URL]" not in s
    assert not any(t in s.lower() for t in _TELLTALES), s


def _server(provider: str, workspace_root: str = None) -> TavilySearchResourcesServer:
    kwargs = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        search_provider=provider,
        exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
    )
    if provider == "exa":
        kwargs["exa_api_key"] = "test_exa_key"  # pragma: allowlist secret
    else:
        kwargs["tavily_api_key"] = "test_tavily_key"  # pragma: allowlist secret
    if workspace_root is not None:
        kwargs["workspace"] = "per_session"
        kwargs["workspace_root"] = workspace_root
    config = TavilySearchResourcesServerConfig(**kwargs)
    return TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestPatterns:
    def test_the_agreed_patterns(self) -> None:
        # Widened 2026-09-09 after an internal contamination audit. Pinned so a
        # future edit has to be deliberate -- this list is evidence-backed, not a guess.
        assert set(CONTAMINATION_PATTERNS) == {
            "browsecomp",
            "browse_comp",
            "browse-comp",
            "simple-eval",
            "bcplus",
            "bc-plus",
            "bc_plus",
        }

    def test_measured_rejects_are_absent_from_the_pattern_list(self) -> None:
        """The audit measured these against five 400-sample runs and said DO NOT ADOPT:
        deep-research is legitimate subject matter, GAIA/HLE are swamped by false
        positives. Assert they never creep in."""
        for rejected in ("deep-research", "deepresearch", "gaia", "hle"):
            assert rejected not in CONTAMINATION_PATTERNS

    def test_url_substrings_are_the_two_dataset_hosts(self) -> None:
        assert set(CONTAMINATED_URL_SUBSTRINGS) == {
            "huggingface.co/datasets",
            "datasets-server.huggingface.co",
        }

    @pytest.mark.parametrize(
        "text",
        [
            "BrowseComp",
            "openai/browsecomp",  # 'browsecomp' already covers the org-prefixed form
            "huggingface.co/datasets/foo/browse_comp",
            "openai/simple-evals",  # 'simple-eval' covers the plural
            "SIMPLE-EVAL",
        ],
    )
    def test_matches_are_case_insensitive_substrings(self, text: str) -> None:
        assert _is_contaminated(text)

    @pytest.mark.parametrize(
        "text",
        ["browse the comp", "comprehensive browsing", "simple evaluation", "", None],
    )
    def test_innocuous_text_is_not_flagged(self, text) -> None:
        assert not _is_contaminated(text)

    def test_scans_every_field_including_nested_highlights(self) -> None:
        # Exa results carry their text in a `highlights` LIST, so a field-by-field
        # check that only looked at title/url/content would miss them entirely.
        assert _is_contaminated(json.dumps({"title": "x", "highlights": ["see browsecomp"]}))


class TestDropContaminated:
    def test_drops_only_the_offending_item(self) -> None:
        kept, dropped = _drop_contaminated(_results())

        assert dropped == 1
        assert [r["title"] for r in kept] == ["clean one", "clean two"]

    def test_all_clean_is_a_passthrough(self) -> None:
        clean = [r for r in _results() if "Browse" not in r["title"]]

        kept, dropped = _drop_contaminated(clean)

        assert dropped == 0 and kept == clean

    def test_empty_list_is_safe(self) -> None:
        assert _drop_contaminated([]) == ([], 0)


class TestInlineTavilySearchFilters:
    """Exit 1: `_postprocess_search_results`, the inline (no workspace) tavily path."""

    @fixture
    def server(self) -> TavilySearchResourcesServer:
        return _server("tavily")

    async def test_contaminated_result_is_absent_from_the_output(self, server) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": _results()})
        server._async_tavily_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))

        assert "clean one" in resp.results_string and "clean two" in resp.results_string
        assert "BrowseComp" not in resp.results_string and "leak" not in resp.results_string

    async def test_all_contaminated_withholds_the_whole_output(self, server) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": _dirty_only()})
        server._async_tavily_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))

        _assert_looks_like_an_ordinary_empty_search(resp.results_string)


class TestExaSearchFilters:
    """Exit 2: `_exa_search_one`, highlights returned inline."""

    @fixture
    def server(self) -> TavilySearchResourcesServer:
        return _server("exa")

    async def test_contaminated_highlight_is_dropped(self, server) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={
                "results": [
                    {"title": "clean", "url": "https://a.example/1", "highlights": ["fine"]},
                    {"title": "x", "url": "https://b.example/2", "highlights": ["from the simple-evals repo"]},
                ]
            }
        )
        server._exa_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))

        assert "clean" in resp.results_string
        assert "simple-evals" not in resp.results_string

    async def test_all_contaminated_withholds_the_whole_output(self, server) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={"results": [{"title": "x", "url": "https://b/2", "highlights": ["browsecomp mirror"]}]}
        )
        server._exa_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))

        _assert_looks_like_an_ordinary_empty_search(resp.results_string)


class TestDiskSearchNeverWritesAContaminatedPage:
    """Exit 3: `_search_one_to_disk`. The one that matters most -- in terminal mode
    the model reads pages/*.txt with the bash tool, so a contaminated page written
    to disk is readable no matter what the tool output said."""

    async def test_page_is_not_written_for_the_contaminated_result(self, tmp_path) -> None:
        server = _server("tavily", str(tmp_path))
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": _results()})
        server._async_tavily_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))

        pages = sorted((Path(tmp_path) / "test_session_id" / "pages").iterdir())
        assert len(pages) == 2, f"expected 2 page writes, got {len(pages)}"
        on_disk = "\n".join(p.read_text() for p in pages).lower()
        assert "browsecomp" not in on_disk and "leak body" not in on_disk
        assert "clean one" in resp.results_string and "BrowseComp" not in resp.results_string

    async def test_all_contaminated_writes_nothing_and_withholds(self, tmp_path) -> None:
        server = _server("tavily", str(tmp_path))
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": _dirty_only()})
        server._async_tavily_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))

        assert list((Path(tmp_path) / "test_session_id" / "pages").iterdir()) == []
        _assert_looks_like_an_ordinary_empty_search(resp.results_string)


class TestBrowseFilters:
    """Exit 4: `browse` has its own write loop and does not route through any of
    the search paths, so it is filtered separately -- above its page writes."""

    async def test_inline_browse_drops_the_contaminated_page(self) -> None:
        server = _server("tavily")
        mock = MagicMock()
        mock.extract = AsyncMock(
            return_value={
                "results": [
                    {"url": "https://a.example/1", "raw_content": "clean body"},
                    {"url": "https://hf.co/datasets/openai/BrowseComp", "raw_content": "leak body"},
                ]
            }
        )
        server._async_tavily_clients = [mock]

        resp = await server.browse(
            _req(), BrowseRequest(urls=["https://a.example/1", "https://hf.co/datasets/openai/BrowseComp"])
        )

        assert "clean body" in resp.results_string
        assert "BrowseComp" not in resp.results_string and "leak body" not in resp.results_string

    async def test_disk_browse_never_writes_the_contaminated_page(self, tmp_path) -> None:
        server = _server("tavily", str(tmp_path))
        mock = MagicMock()
        mock.extract = AsyncMock(
            return_value={
                "results": [
                    {"url": "https://a.example/1", "raw_content": "clean body"},
                    {"url": "https://b.example/2", "raw_content": "answers from openai/simple-evals"},
                ]
            }
        )
        server._async_tavily_clients = [mock]

        await server.browse(_req(), BrowseRequest(urls=["https://a.example/1", "https://b.example/2"]))

        pages = sorted((Path(tmp_path) / "test_session_id" / "pages").iterdir())
        assert len(pages) == 1
        assert "simple-evals" not in pages[0].read_text()

    async def test_all_contaminated_withholds_the_whole_output(self) -> None:
        server = _server("exa")
        mock = MagicMock()
        mock.get_contents = AsyncMock(
            return_value={"results": [{"url": "https://b.example/2", "text": "a browsecomp mirror"}]}
        )
        server._exa_clients = [mock]

        resp = await server.browse(_req(), BrowseRequest(urls=["https://b.example/2"]))

        # Indistinguishable from a page the extractor simply could not read.
        assert resp.results_string == "No content extracted."
        assert "browsecomp" not in resp.results_string.lower()


class TestWidenedPatterns:
    """An internal audit measured the original three patterns
    against five full 400-sample runs: they caught 2,023 of 2,594 mirror-URL blocks and
    MISSED 571 (22%) across 34 URLs. Every string below is from that evader table."""

    @pytest.mark.parametrize(
        "text",
        [
            "huggingface.co/datasets/Nithish2410/benchmark-bcplus",  # 359 blocks, serves test.jsonl
            "Nithish2410/benchmark-bcplus_agent",  # 60 blocks
            "metatext.io/datasets/nithish2410/benchmark-bcplus",  # 59 blocks
            "Yuqi-Zhou/BC-Plus-Eval-Results",  # 14 blocks, hyphen form
            "ZhuofengLi/bcplus-eval-100",  # 9 blocks
            "Yuqi-Zhou/BC-Plus-Leaderboard",  # 8 blocks, an HF *space* not a dataset
            "BC_Plus_results",  # underscore form
            "browse-comp",  # hyphen spelling of the benchmark itself
        ],
    )
    def test_evading_spellings_are_now_caught(self, text: str) -> None:
        assert _is_contaminated(text)

    @pytest.mark.parametrize(
        "text",
        [
            # The audit measured these and said DO NOT ADOPT. They must stay unflagged,
            # or the guard starts eating legitimate subject matter.
            "a deep-research agent for the web",
            "deepresearch pipeline",
            "GAIA benchmark results",
            "HLE score",
            "simple evaluation of the method",
        ],
    )
    def test_rejected_patterns_stay_unflagged(self, text: str) -> None:
        assert not _is_contaminated(text)


class TestUrlDomainBlock:
    """HuggingFace hosts 2,163 of the leaked blocks -- every long-tail mirror that matches
    no name pattern. A dataset-viewer page is never a legitimate primary source for a
    BrowseComp question. Scoped to the `url` FIELD, never the page text."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://huggingface.co/datasets/RUC-AIBOX/Evo-Bench",
            "https://huggingface.co/datasets/Halcyon-Zhang/BrowseComp-V3",
            "https://huggingface.co/datasets/Forival/LiveBrowseComp",
            "https://datasets-server.huggingface.co/rows?dataset=foo&config=default",
            "https://HuggingFace.co/DATASETS/Some/Mirror",  # case-insensitive
        ],
    )
    def test_dataset_hosts_are_dropped(self, url: str) -> None:
        assert _is_contaminated_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://huggingface.co/blog/some-post",  # HF, but not a dataset page
            "https://huggingface.co/models/bert-base",
            "https://en.wikipedia.org/wiki/Mosquito",
            None,
            "",
        ],
    )
    def test_other_urls_are_kept(self, url) -> None:
        assert not _is_contaminated_url(url)

    def test_url_block_is_field_scoped_not_text_scoped(self) -> None:
        """A page that merely MENTIONS a HF dataset URL in its body must survive --
        folding the domains into the text patterns would drop it, which is a far larger
        blast radius than intended."""
        r = {
            "title": "A blog about ML datasets",
            "url": "https://example.com/post",
            "content": "see https://huggingface.co/datasets/squad for the data",
        }

        kept, dropped = _drop_contaminated([r])

        assert dropped == 0 and kept == [r]

    def test_dataset_url_drops_even_with_innocuous_text(self) -> None:
        r = {"title": "rows", "url": "https://huggingface.co/datasets/x/y", "content": "nothing notable"}

        kept, dropped = _drop_contaminated([r])

        assert dropped == 1 and kept == []


class TestWidenedGuardStillFiltersEveryExit:
    """The widened predicate must reach all four exits, not just the helpers."""

    async def test_disk_search_never_writes_a_bcplus_page(self, tmp_path) -> None:
        server = _server("tavily", str(tmp_path))
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={
                "results": [
                    {"title": "clean", "url": "https://a.example/1", "content": "x", "raw_content": "clean body"},
                    {
                        "title": "rows",
                        "url": "https://huggingface.co/datasets/Nithish2410/benchmark-bcplus",
                        "content": "y",
                        "raw_content": "row 865 answer Vera Nunning",
                    },
                ]
            }
        )
        server._async_tavily_clients = [mock]

        await server.search(_req(), TavilySearchRequest(queries=["q"]))

        pages = sorted((Path(tmp_path) / "test_session_id" / "pages").iterdir())
        assert len(pages) == 1
        on_disk = pages[0].read_text()
        assert "Vera Nunning" not in on_disk and "bcplus" not in on_disk.lower()

    async def test_browse_drops_a_dataset_viewer_page(self) -> None:
        server = _server("tavily")
        mock = MagicMock()
        mock.extract = AsyncMock(
            return_value={
                "results": [
                    {"url": "https://a.example/1", "raw_content": "clean body"},
                    {"url": "https://datasets-server.huggingface.co/rows?dataset=z", "raw_content": "answer key"},
                ]
            }
        )
        server._async_tavily_clients = [mock]

        resp = await server.browse(_req(), BrowseRequest(urls=["https://a.example/1", "https://x/2"]))

        assert "clean body" in resp.results_string
        assert "answer key" not in resp.results_string


class TestExaDeepAnswerIsGuarded:
    """This branch renders exa deep-search's synthesized `output.content` as a [Deep Answer]
    block ahead of the per-URL entries. A synthesis built from a benchmark mirror is the
    answer key in prose, so it goes through the same text guard as the results."""

    async def test_contaminated_deep_answer_is_dropped_but_clean_results_stay(self) -> None:
        server = _server("exa")
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={
                "output": {"content": "Per the BrowseComp answer key the yacht is Seeker 1"},
                "results": [{"title": "clean", "url": "https://a.example/1", "highlights": ["fine"]}],
            }
        )
        server._exa_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))

        assert "[Deep Answer]" not in resp.results_string
        assert "Seeker 1" not in resp.results_string
        assert "clean" in resp.results_string

    async def test_clean_deep_answer_still_renders(self) -> None:
        server = _server("exa")
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={
                "output": {"content": "a legitimate synthesis"},
                "results": [{"title": "clean", "url": "https://a.example/1", "highlights": ["fine"]}],
            }
        )
        server._exa_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))

        assert "[Deep Answer]: a legitimate synthesis" in resp.results_string
