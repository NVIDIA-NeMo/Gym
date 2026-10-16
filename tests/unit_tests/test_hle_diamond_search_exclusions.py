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
import json
import re
import sys
import types
from pathlib import Path

import pytest

from benchmarks.hle_diamond import build_search_exclusions as module


REPO_ROOT = Path(__file__).resolve().parents[2]
DIAMOND_POLICY = REPO_ROOT / "benchmarks" / "hle_diamond" / "search_exclusions.json"
HLE_POLICY = REPO_ROOT / "benchmarks" / "hle" / "search_exclusions.json"

TOO_LONG_LABEL = "x" * 64 + ".com"
SYNTHETIC_UPSTREAM = {
    "domains": [
        "Example.COM.",
        "example.com",
        "Sub.Example.com",
        "bücher.de",
        "",
        ".leading.dot",
        "has/path.com",
        TOO_LONG_LABEL,
        "single",
        42,
    ],
    "url_prefixes": [
        "HTTPS://WWW.Example.com/Path",
        "example.com/path",
        "www.other.org/x",
        "http://news.site/a?b=1",
        "",
        "   ",
    ],
    "serp_url_patterns": ["^google\\.com/search", r"^([a-z0-9-]+\.)*bing\.com/search"],
    "url_patterns": ["hle", "last exam"],
    "new_family": ["x", "y", "z"],
}
SYNTHETIC_HLE = {
    "notices": [
        {
            "properties": [
                {"type": "domain", "value": "extra.net"},
                {"type": "domain", "value": "example.com"},
                {"type": "url_substring", "value": "example.com/path"},
                {"type": "url_substring", "value": "://scale.com"},
                {"type": "url_substring", "value": "hle-exam"},
                {"type": "url_substring", "value": ""},
                {"type": "author_name", "value": "Someone"},
            ]
        }
    ]
}
EXPECTED_DOMAINS = ["example.com", "sub.example.com", "xn--bcher-kva.de", "extra.net"]
EXPECTED_SUBSTRINGS = [
    "example.com/path",
    "other.org/x",
    "news.site/a?b=1",
    "bing.com/search",
    "://scale.com",
    "hle-exam",
]
# One search-engine result page per emitted SERP substring, as the with-tools pilot saw agents fetch them.
SERP_URLS = [
    "https://www.google.com/search?q=humanity%27s+last+exam",
    "https://scholar.google.com/scholar?q=hle+diamond",
    "https://www.bing.com/search?q=hle",
    "https://duckduckgo.com/html/?q=hle",
    "https://duckduckgo.com/lite/?q=hle",
    "https://duckduckgo.com/?q=hle&t=h_",
    "https://html.duckduckgo.com/html/?q=hle",
    "https://search.yahoo.com/search?p=hle",
    "https://www.baidu.com/s?wd=hle",
    "https://yandex.ru/search/?text=hle",
    "https://www.startpage.com/sp/search?query=hle",
    "https://search.brave.com/search?q=hle",
    "https://www.ecosia.org/search?q=hle",
    "https://www.mojeek.com/search?q=hle",
]
NON_SERP_URLS = [
    "https://www.google.com/maps/place/Berlin",
    "https://duckduckgo.com/about",
    "https://www.ecosia.org/",
    "https://en.wikipedia.org/wiki/Yandex",
]


@pytest.fixture
def tavily_app(monkeypatch):
    """The real Tavily server module; the tavily SDK is a resource-server extra the policy code never uses."""
    try:
        import tavily  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("tavily")
        stub.AsyncTavilyClient = type("AsyncTavilyClient", (), {})
        monkeypatch.setitem(sys.modules, "tavily", stub)
    from resources_servers.tavily_search import app

    return app


@pytest.fixture
def url_exclusion_policy(tavily_app):
    return tavily_app.URLExclusionPolicy


def _properties(policy_json: dict) -> list[dict]:
    return [prop for notice in policy_json["notices"] for prop in notice["properties"]]


def test_pinned_upstream_revision_is_a_full_sha1_that_the_download_url_uses():
    """A truncated pin resolves nowhere (GitHub answers 404 for a 39-character revision), so --download breaks."""
    assert re.fullmatch(r"[0-9a-f]{40}", module.UPSTREAM_COMMIT)
    assert re.fullmatch(r"[0-9a-f]{64}", module.UPSTREAM_BLOCKLIST_SHA256)
    assert module.UPSTREAM_BLOCKLIST_URL == (
        f"https://raw.githubusercontent.com/centerforaisafety/hle/{module.UPSTREAM_COMMIT}/docs/blocklist.json"
    )


def test_build_policy_normalises_deduplicates_and_unions_with_hle_policy():
    policy = module.build_policy(SYNTHETIC_UPSTREAM, SYNTHETIC_HLE)
    assert policy.domains == EXPECTED_DOMAINS
    assert policy.url_substrings == EXPECTED_SUBSTRINGS


def test_build_policy_drops_and_reports_entries_the_parser_rejects_or_never_matches():
    policy = module.build_policy(SYNTHETIC_UPSTREAM, SYNTHETIC_HLE)
    assert [(source, value) for source, value, _ in policy.dropped] == [
        ("upstream domains", ""),
        ("upstream domains", ".leading.dot"),
        ("upstream domains", "has/path.com"),
        ("upstream domains", TOO_LONG_LABEL),
        ("upstream domains", "single"),
        ("upstream domains", 42),
        ("upstream url_prefixes", ""),
        ("upstream url_prefixes", "   "),
        ("upstream serp_url_patterns", "^google\\.com/search"),
        ("benchmarks/hle/search_exclusions.json", ""),
        ("benchmarks/hle/search_exclusions.json", "Someone"),
    ]
    reasons = {value: reason for _, value, reason in policy.dropped}
    assert "IDNA" in reasons[".leading.dot"]
    assert reasons["has/path.com"] == "not a hostname"
    assert reasons[""] == "empty"
    assert "author_name" in reasons["Someone"]
    assert "SERP_PATTERN_SUBSTRINGS" in reasons["^google\\.com/search"]


def test_unrepresented_upstream_rule_families_are_counted_including_unknown_ones():
    policy = module.build_policy(SYNTHETIC_UPSTREAM)
    assert policy.unrepresented == {"url_patterns": 2, "new_family": 3}
    assert policy.approximated == {"serp_url_patterns": {"upstream_rules": 2, "url_substring_rules": 1}}
    rendered = module.render(policy)
    assert rendered["unrepresented_upstream_rules"] == policy.unrepresented
    assert rendered["approximated_upstream_rules"] == policy.approximated
    assert (
        "approximated_upstream_rules" not in module.render(module.build_policy({"domains": ["a.test"]}))["notices"][0]
    )
    assert module.render(module.build_policy({"domains": ["a.test"]}))["approximated_upstream_rules"] == {}


@pytest.mark.parametrize("url", SERP_URLS)
def test_serp_approximation_is_sound_on_result_page_urls_and_the_committed_policy_blocks_them(
    url, url_exclusion_policy
):
    """Upstream matches its anchored regexes on the lower-cased URL without scheme or leading www.; every such
    page must contain one of the substrings emitted for a regex that matches it, and the committed list blocks it."""
    stripped = re.sub(r"^https?://(www\.)?", "", url.lower())
    matching = [regex for regex in module.SERP_PATTERN_SUBSTRINGS if re.match(regex, stripped)]
    assert matching, url
    emitted = {substring for regex in matching for substring in module.SERP_PATTERN_SUBSTRINGS[regex]}
    assert any(substring in url.lower() for substring in emitted), (url, emitted)
    assert url_exclusion_policy(DIAMOND_POLICY).blocked(url)


@pytest.mark.parametrize("url", NON_SERP_URLS)
def test_committed_policy_leaves_non_result_pages_of_search_engines_alone(url, url_exclusion_policy):
    assert not url_exclusion_policy(DIAMOND_POLICY).blocked(url)


def test_rendered_policy_loads_through_gym_parser_which_ignores_metadata_keys(tmp_path, url_exclusion_policy):
    rendered = module.render(module.build_policy(SYNTHETIC_UPSTREAM, SYNTHETIC_HLE))
    assert set(rendered) == {"source", "unrepresented_upstream_rules", "approximated_upstream_rules", "notices"}
    assert module.UPSTREAM_COMMIT[:8] in rendered["source"]["upstream_commit"]
    path = tmp_path / "search_exclusions.json"
    path.write_text(json.dumps(rendered))

    parsed = url_exclusion_policy(path)
    assert parsed.domains == EXPECTED_DOMAINS
    assert parsed.substrings == EXPECTED_SUBSTRINGS
    assert parsed.blocked("https://deep.Sub.Example.com/anything")
    assert parsed.blocked("https://mirror.test/?u=https://www.other.org/x")
    assert not parsed.blocked("https://unrelated.test/")


@pytest.fixture(scope="module")
def diamond_policy_json() -> dict:
    return json.loads(DIAMOND_POLICY.read_text())


def test_committed_policy_contains_only_non_empty_normalised_rules(diamond_policy_json):
    properties = _properties(diamond_policy_json)
    assert {prop["type"] for prop in properties} == {"domain", "url_substring"}
    assert all(isinstance(prop["value"], str) and prop["value"].strip() for prop in properties)
    rules = [(prop["type"], prop["value"]) for prop in properties]
    assert len(rules) == len(set(rules))
    assert all(value == value.lower() for _, value in rules)
    assert set(diamond_policy_json["unrepresented_upstream_rules"]) == set(module.UNREPRESENTED_RULE_FAMILIES)
    assert diamond_policy_json["approximated_upstream_rules"] == {
        "serp_url_patterns": {"upstream_rules": 12, "url_substring_rules": 14}
    }
    assert module.UPSTREAM_COMMIT[:8] in diamond_policy_json["source"]["upstream_commit"]


def test_committed_policy_loads_through_gym_parser_and_blocks_load_bearing_hosts(url_exclusion_policy):
    policy = url_exclusion_policy(DIAMOND_POLICY)
    for domain in (
        "huggingface.co",
        "hf.co",
        "lastexam.ai",
        "scale.com",
        "github.com",
        "githubusercontent.com",
        "archive.today",
        "archive.ph",
        "archive.is",
    ):
        assert domain in policy.domains, domain
    for url in (
        "https://huggingface.co/datasets/cais/hle-diamond",
        "https://LastExam.ai/blog/hle-diamond",
        "https://raw.githubusercontent.com/centerforaisafety/hle/main/docs/blocklist.json",
        "https://archive.ph/abc12",
        "https://arxiv.org/abs/2510.08959",
        "https://www.nature.com/articles/s41586-025-09962-4",
    ):
        assert policy.blocked(url), url
    assert not policy.blocked("https://en.wikipedia.org/wiki/Riemann_hypothesis")
    assert not policy.blocked("https://arxiv.org/abs/1706.03762")


def test_committed_policy_provider_subset_is_capped_priority_first_and_the_tail_stays_local(tavily_app):
    """Tavily's /search accepts at most 150 exclude_domains (HTTP 400 beyond; with-tools pilot P-T1). The server
    sends the priority block then file order; the 70 domains it cannot send are enforced locally only."""
    policy = tavily_app.URLExclusionPolicy(DIAMOND_POLICY)
    assert len(policy.domains) == 220
    sent = tavily_app.select_provider_domains(
        policy.domains, tavily_app.TAVILY_EXCLUDE_DOMAINS_LIMIT, tavily_app.DEFAULT_EXCLUDE_DOMAINS_PRIORITY
    )
    assert len(sent) == 150
    assert sent[:15] == [
        "huggingface.co",
        "hf.co",
        "lastexam.ai",
        "agi.safe.ai",
        "safe.ai",
        "scale.com",
        "github.com",
        "githubusercontent.com",
        "archive.today",
        "archive.ph",
        "archive.is",
        "archive.md",
        "archive.li",
        "archive.vn",
        "archive.fo",
    ]
    assert sent[15:] == [domain for domain in policy.domains if domain not in sent[:15]][:135]
    not_sent = [domain for domain in policy.domains if domain not in sent]
    assert len(not_sent) == 70
    local_only = {"facebook.com", "x.com", "reddit.com", "stackexchange.com", "stackoverflow.com", "quora.com"}
    assert local_only <= set(not_sent)
    for url in (
        "https://www.reddit.com/r/singularity/comments/abc/hle_answers/",
        "https://x.com/someone/status/1",
        "https://math.stackexchange.com/questions/1",
    ):
        assert policy.blocked(url), url


def test_committed_policy_is_at_least_as_strict_as_the_hle_policy(diamond_policy_json):
    diamond_rules = {(prop["type"], prop["value"]) for prop in _properties(diamond_policy_json)}
    for prop in _properties(json.loads(HLE_POLICY.read_text())):
        normalizer = module.normalize_domain if prop["type"] == "domain" else module.normalize_url_substring
        value, reason = normalizer(prop["value"])
        assert reason is None, prop
        assert (prop["type"], value) in diamond_rules, prop
