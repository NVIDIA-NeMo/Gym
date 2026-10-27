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
"""HLE web tools: upstream's ``web_search`` and ``web_fetch`` behind its blocklist, served over MCP.

The HLE authors' with-tools protocol (``docs/evaluation-with-tools.md`` in centerforaisafety/hle) exposes
exactly two web tools to the agent and loads ``docs/blocklist.json`` in the wrapper around them. This
server is that wrapper: it holds the search provider's credentials (Tavily), applies every rule family of
the blocklist with the semantics the protocol describes, refuses leaky queries and URLs, re-checks fetched
and cached pages, and logs each blocked call with the family and rule that matched. The question of the
rollout arrives through ``/seed_session`` and drives the verbatim-span refusal.

The sandbox is meant to reach the tools over MCP with the per-rollout session token, but the plain
``POST /web_search`` and ``POST /web_fetch`` routes stay served, and a call without the rollout's cookie
or token lands in a fresh session that carries no question. The policy therefore does not depend on the
transport: every family applies on both routes, and ``web_search`` fails closed on a session that was never
seeded with a question (``require_seeded_question``, default on) rather than skipping the span rule.
"""

import hashlib
import json
import logging
import re
from asyncio import sleep
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import parse_qsl, unquote, urlsplit

from aiohttp import ClientConnectionError, ClientPayloadError, ClientTimeout
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY, request


logger = logging.getLogger(__name__)

# The protocol this server implements, pinned to one upstream revision. The three files were fetched from
# raw.githubusercontent.com at this commit; their SHA-256 digests are recorded here and ``blocklist.json``
# next to this module is a byte-exact copy whose digest the server verifies at start.
UPSTREAM_REPOSITORY = "https://github.com/centerforaisafety/hle"
UPSTREAM_COMMIT = "22ed3074b1e7b134bcbc09028d0ba320839b0655"  # pragma: allowlist secret
UPSTREAM_PROTOCOL_PATH = "docs/evaluation-with-tools.md"
UPSTREAM_PROTOCOL_SHA256 = (
    "3852a7f7008ceb27e9aec97f223436acbecdc9a4bc22cd63de54e5f0d49d9cab"  # pragma: allowlist secret
)
UPSTREAM_BLOCKLIST_PATH = "docs/blocklist.json"
UPSTREAM_BLOCKLIST_SHA256 = (
    "3abaf35e03d6d5d361d2611746683a32315fe73ef851ec5f33c54b3c6a526575"  # pragma: allowlist secret
)
UPSTREAM_REQUIREMENTS_PATH = "docs/requirements-tools.txt"
UPSTREAM_REQUIREMENTS_SHA256 = (
    "c1c626beb8f2fab11609f6ed41a2721d9acd139554d6ce3c8c83f52a49797068"  # pragma: allowlist secret
)
DEFAULT_BLOCKLIST_FILE_PATH = "resources_servers/hle_web_tools/blocklist.json"

# The protocol's two tools, and nothing else, are advertised over MCP.
MCP_TOOLS = ("web_search", "web_fetch")
# Tavily's /search returns at most 20 results per call, which is also the protocol's cap.
TAVILY_MAX_RESULTS = 20
# The protocol's cap on fetched Markdown.
PROTOCOL_MAX_FETCH_CHARS = 60_000
# Tavily rejects longer queries with HTTP 400.
TAVILY_MAX_QUERY_CHARS = 400

BLOCKLIST_FAMILIES = (
    "domains",
    "url_prefixes",
    "serp_url_patterns",
    "url_patterns",
    "patterns_case_sensitive",
    "answer_title_patterns",
    "answer_title_allow_hosts",
)
# Protocol rules that are not blocklist families but refuse a call all the same.
QUESTION_SPAN_FAMILY = "question_span"
MALFORMED_URL_FAMILY = "malformed_url"
# This server's own fail-closed rule: a search from a session with no seeded question cannot be span-checked.
UNSEEDED_SESSION_FAMILY = "unseeded_session"

# Keys Tavily rejects outright; the pool moves on to the next key for the same call.
TAVILY_KEY_REJECTED_CODES = frozenset({401, 432, 433})
TAVILY_RETRY_CODES = frozenset({408, 429, 500, 502, 503, 504})

_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_WWW_RE = re.compile(r"^www\.", re.IGNORECASE)
# A lookahead, so a URL nested inside another (``https://reader.example/https://blocked.example``) is found
# at its own position rather than swallowed by the outer match.
_EMBEDDED_URL_RE = re.compile(r"(?=(https?://[^\s\"'<>]+))", re.IGNORECASE)
# A query value or path remainder that names a host, with or without a scheme: ``hf.co/datasets/x``.
_BARE_URL_RE = re.compile(r"^(?:https?:)?(?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:[:/?#]|$)", re.IGNORECASE)
_SITE_OPERATOR_RE = re.compile(r"\bsite:\s*[\"']?([^\s\"']+)", re.IGNORECASE)
_WAYBACK_PATH_RE = re.compile(r"^/web/[^/]*/(.+)$", re.IGNORECASE | re.DOTALL)
_GOOGLE_CACHE_RE = re.compile(r"cache:(?:[A-Za-z0-9_-]+:)?(.+)$", re.DOTALL)
_WORD_RE = re.compile(r"\w+")
_DATA_URI_RE = re.compile(r"data:[a-z]+/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", re.IGNORECASE)
_TITLE_SEPARATOR_RE = re.compile(r"\s+(?:\||-|–|—|·|•|»|::)\s+")
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
# Hosts whose job is to serve another page: their path remainder is treated as the wrapped URL. Hosts the
# blocklist names outright (archive.ph and friends, r.jina.ai, translate.goog, translate.google.com,
# 12ft.io, corsproxy.io, allorigins, codetabs, thingproxy, whateverorigin) are refused by the ``domains``
# family before any unwrapping is needed.
WRAPPER_HOSTS = frozenset({"web.archive.org", "archive.org", "webcache.googleusercontent.com", "r.jina.ai"})
MAX_URL_CANDIDATES = 64
MAX_URL_DECODE_ROUNDS = 4
MAX_SESSION_QUESTIONS = 10_000


def normalize_hostname(value: str) -> str:
    """IDNA-encode, lower-case and strip the trailing dot: the one form every hostname comparison uses."""
    return value.encode("idna").decode().lower().rstrip(".")


def schemeless(url: str) -> str:
    """The URL without its scheme: what ``serp_url_patterns`` are anchored against."""
    return _SCHEME_RE.sub("", url.strip(), count=1)


def prefix_form(url: str) -> str:
    """The URL as ``url_prefixes`` see it: scheme and leading ``www.`` removed, lower-cased."""
    return _WWW_RE.sub("", schemeless(url), count=1).lower()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Block:
    """Why a call or result was refused: the rule family, the rule, what it matched and where."""

    family: str
    rule: str
    subject: str
    where: str

    def refusal(self) -> str:
        reason = {
            "domains": "the source is a blocked site",
            "url_prefixes": "the URL is a blocked page",
            "serp_url_patterns": "search-engine result pages cannot be fetched",
            "url_patterns": f"the {self.where} matches a blocked pattern",
            "patterns_case_sensitive": f"the {self.where} matches a blocked pattern",
            "answer_title_patterns": "the page title matches an answer-key pattern",
            QUESTION_SPAN_FAMILY: f"the query repeats {self.rule} verbatim; rephrase it in your own words",
            MALFORMED_URL_FAMILY: "the URL is not a valid http(s) URL",
            UNSEEDED_SESSION_FAMILY: "search is served only to a rollout session seeded with its question",
        }[self.family]
        return f"Refused by the HLE web-tool policy: {reason}."


class Blocklist:
    """Upstream's ``docs/blocklist.json`` with the semantics of ``docs/evaluation-with-tools.md``.

    - ``domains``: exact hostnames and all their subdomains.
    - ``url_prefixes``: prefixes of the URL after removing the scheme and a leading ``www.``, ignoring case.
    - ``serp_url_patterns``: regexes anchored at the start of the scheme-less URL of a search-engine result page.
    - ``url_patterns`` / ``patterns_case_sensitive``: regexes applied to queries, URLs, result titles and
      snippets, and fetched titles and Markdown; the first family case-insensitively, the second as written.
    - ``answer_title_patterns``: case-insensitive regexes applied only to result titles, also with trailing
      site names removed; hosts in ``answer_title_allow_hosts`` (and their subdomains) are exempt from this
      family alone.
    """

    def __init__(self, data: Dict[str, Any]):
        unknown = set(data) - set(BLOCKLIST_FAMILIES)
        if unknown:
            raise ValueError(f"Unsupported blocklist families: {sorted(unknown)}")
        self.domains: List[str] = [normalize_hostname(d) for d in data.get("domains", [])]
        self.url_prefixes: List[str] = list(data.get("url_prefixes", []))
        self.serp_url_patterns = [re.compile(p, re.IGNORECASE) for p in data.get("serp_url_patterns", [])]
        self.url_patterns = [re.compile(p, re.IGNORECASE) for p in data.get("url_patterns", [])]
        self.patterns_case_sensitive = [re.compile(p) for p in data.get("patterns_case_sensitive", [])]
        self.answer_title_patterns = [re.compile(p, re.IGNORECASE) for p in data.get("answer_title_patterns", [])]
        self.answer_title_allow_hosts: List[str] = [
            normalize_hostname(h) for h in data.get("answer_title_allow_hosts", [])
        ]

    @classmethod
    def load(cls, path: Path, expected_sha256: Optional[str]) -> "Blocklist":
        if expected_sha256 is not None:
            actual = sha256_file(path)
            if actual != expected_sha256.lower():
                raise ValueError(
                    f"Blocklist {path} has SHA-256 {actual}, expected {expected_sha256}: it is not the pinned "
                    f"upstream snapshot ({UPSTREAM_REPOSITORY} {UPSTREAM_BLOCKLIST_PATH} at {UPSTREAM_COMMIT}). "
                    "Set blocklist_sha256 to null to run a deliberately different policy file."
                )
        return cls(json.loads(path.read_text(encoding="utf-8")))

    # ---- single-family checks ----

    def host_block(self, host: str, subject: str, where: str) -> Optional[Block]:
        for domain in self.domains:
            if host == domain or host.endswith("." + domain):
                return Block("domains", domain, subject, where)
        return None

    def _host_allowed_for_titles(self, host: str) -> bool:
        return any(host == allow or host.endswith("." + allow) for allow in self.answer_title_allow_hosts)

    def text_block(self, text: str, where: str) -> Optional[Block]:
        """``url_patterns`` and ``patterns_case_sensitive`` against any text."""
        for pattern in self.url_patterns:
            if pattern.search(text):
                return Block("url_patterns", pattern.pattern, text, where)
        for pattern in self.patterns_case_sensitive:
            if pattern.search(text):
                return Block("patterns_case_sensitive", pattern.pattern, text, where)
        return None

    def answer_title_block(self, title: str, host: str) -> Optional[Block]:
        if self._host_allowed_for_titles(host):
            return None
        for variant in title_variants(title):
            for pattern in self.answer_title_patterns:
                if pattern.search(variant):
                    return Block("answer_title_patterns", pattern.pattern, variant, "result title")
        return None

    def _single_url_block(self, url: str, where: str) -> Optional[Block]:
        """The URL-shaped families against one URL string, plus the text families against the string itself."""
        try:
            parts = urlsplit(url)
            host = normalize_hostname(parts.hostname) if parts.hostname else ""
        except (ValueError, UnicodeError):
            return None  # the outer URL's parse failure is handled by url_block
        if host:
            block = self.host_block(host, url, where)
            if block is not None:
                return block
        form = prefix_form(url)
        for prefix in self.url_prefixes:
            if form.startswith(prefix.lower()):
                return Block("url_prefixes", prefix, url, where)
        bare = schemeless(url)
        for pattern in self.serp_url_patterns:
            if pattern.search(bare):
                return Block("serp_url_patterns", pattern.pattern, url, where)
        return self.text_block(url, where)

    def url_block(self, url: str, where: str = "URL") -> Optional[Block]:
        """Every URL-shaped rule against the URL, its percent-decoded forms and the URLs it wraps."""
        try:
            parts = urlsplit(url)
            if parts.scheme.lower() not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
                return Block(MALFORMED_URL_FAMILY, "http(s) URL without credentials", url, where)
            normalize_hostname(parts.hostname)
        except (ValueError, UnicodeError):
            return Block(MALFORMED_URL_FAMILY, "parseable URL", url, where)
        for candidate in url_candidates(url):
            block = self._single_url_block(candidate, where)
            if block is not None:
                return block
        return None

    def query_block(self, query: str, question_spans: Optional["QuestionSpans"]) -> Optional[Block]:
        """The protocol's query refusals: blocked patterns, a blocked ``site:`` domain, a verbatim question span."""
        block = self.text_block(query, "query")
        if block is not None:
            return block
        for match in _SITE_OPERATOR_RE.finditer(query):
            host = schemeless(match.group(1)).split("/", 1)[0].split(":", 1)[0]
            try:
                host = normalize_hostname(_WWW_RE.sub("", host, count=1))
            except UnicodeError:
                continue
            block = self.host_block(host, match.group(0), "site: operator")
            if block is not None:
                return block
        if question_spans is not None:
            span = question_spans.longest_shared_span(query)
            if span is not None:
                return Block(
                    QUESTION_SPAN_FAMILY,
                    f"{span} consecutive words of the question",
                    f"{span} words",  # never the words themselves: the question is gated data
                    "query",
                )
        return None


def title_variants(title: str) -> List[str]:
    """The title, then the title with each trailing ``| Site``-style segment removed, longest first."""
    variants = [title]
    for match in _TITLE_SEPARATOR_RE.finditer(title):
        variants.append(title[: match.start()])
    return variants


def _wrapped_urls(url: str) -> List[str]:
    """URLs a proxy or archive wrapper carries: the path remainder of known wrappers, ``cache:`` targets and
    every query value that names a host."""
    found: List[str] = []
    try:
        parts = urlsplit(url)
        host = normalize_hostname(parts.hostname) if parts.hostname else ""
    except (ValueError, UnicodeError):
        return found
    if host in WRAPPER_HOSTS:
        remainder = parts.path
        wayback = _WAYBACK_PATH_RE.match(remainder)
        if wayback:
            remainder = wayback.group(1)
        remainder = remainder.lstrip("/")
        if parts.query and host != "webcache.googleusercontent.com":
            remainder += "?" + parts.query
        if remainder:
            found.append(remainder)
    for _, value in parse_qsl(parts.query, keep_blank_values=False):
        cache = _GOOGLE_CACHE_RE.search(value)
        if cache:
            found.append(cache.group(1))
        found.append(value)
    found.extend(match.group(1) for match in _EMBEDDED_URL_RE.finditer(url) if match.start() > 0)
    urls: List[str] = []
    for item in found:
        item = item.strip()
        if not item:
            continue
        if _SCHEME_RE.match(item):
            urls.append(item)
        elif _BARE_URL_RE.match(item):
            urls.append("https://" + item.lstrip("/"))
    return urls


def url_candidates(url: str) -> List[str]:
    """The URL, its percent-decoded forms (several rounds) and every URL wrapped inside any of them."""
    seen: List[str] = []
    frontier = [url]
    for _ in range(MAX_URL_DECODE_ROUNDS):
        next_frontier: List[str] = []
        for candidate in frontier:
            if candidate in seen or len(seen) >= MAX_URL_CANDIDATES:
                continue
            seen.append(candidate)
            next_frontier.extend(_wrapped_urls(candidate))
            decoded = unquote(candidate)
            if decoded != candidate:
                next_frontier.append(decoded)
        if not next_frontier:
            break
        frontier = next_frontier
    return seen


def question_words(text: str) -> Tuple[str, ...]:
    """Case-folded maximal runs of word characters: the unit of the verbatim-span rule."""
    return tuple(_WORD_RE.findall(text.casefold()))


class QuestionSpans:
    """The word n-grams of one rollout's question, for the verbatim-span refusal.

    ``span_words`` is the configured threshold; a question with fewer words than that is protected as a whole.
    """

    def __init__(self, question: str, span_words: int):
        words = question_words(question)
        # Identifies the question without keeping its text, so a retried seed can be told from a re-seed.
        self.digest = hashlib.sha256("\x1f".join(words).encode()).hexdigest()
        self.span_words = min(span_words, len(words)) if words else 0
        n = self.span_words
        self.ngrams = frozenset(words[i : i + n] for i in range(len(words) - n + 1)) if n else frozenset()

    def longest_shared_span(self, query: str) -> Optional[int]:
        """The span length when the query repeats that many consecutive question words, else ``None``."""
        n = self.span_words
        if not n:
            return None
        words = question_words(query)
        if any(words[i : i + n] in self.ngrams for i in range(len(words) - n + 1)):
            return n
        return None


def extract_question(seed_body: Dict[str, Any], field: str) -> Optional[str]:
    """The question text of a seed: the task row's ``field``, else the last user message of its create params.

    An Agent seeds a tool server with its run request, whose extra fields are the task row (``question`` for
    the HLE presets); an Environment Server seeds with ``task_data`` holding the same row.
    """
    data = seed_body.get("task_data") if isinstance(seed_body.get("task_data"), dict) else seed_body
    value = data.get(field)
    if isinstance(value, str) and value.strip():
        return value
    params = data.get("responses_create_params")
    messages = params.get("input") if isinstance(params, dict) else None
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [part.get("text") for part in content if isinstance(part, dict)]
                joined = "\n".join(text for text in texts if isinstance(text, str))
                if joined.strip():
                    return joined
    return None


class TavilyBackend(BaseModel):
    """Tavily's ``/search`` and ``/extract`` over Gym's shared aiohttp client, with a key pool.

    A key Tavily rejects (401, 432, 433) is skipped for the rest of the call; transient failures retry with
    backoff on the next key. Provider bodies are never surfaced: they may describe credentials.
    """

    api_keys: List[str] = Field(repr=False, min_length=1)
    base_url: str = "https://api.tavily.com"
    max_attempts: int = Field(default=3, ge=1)
    timeout_s: float = Field(default=60, gt=0)
    _next_key: int = PrivateAttr(default=0)

    async def search(self, query: str, *, max_results: int, search_depth: str) -> Dict[str, Any]:
        return await self._post(
            "/search",
            {
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                # No generated answer: it could summarise a page the policy drops. No page bodies or images.
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
            },
        )

    async def extract(self, url: str, *, extract_depth: str) -> Dict[str, Any]:
        return await self._post(
            "/extract",
            {"urls": [url], "extract_depth": extract_depth, "format": "markdown", "include_images": False},
        )

    async def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        keys = self.api_keys[self._next_key :] + self.api_keys[: self._next_key]
        self._next_key = (self._next_key + 1) % len(self.api_keys)
        rejected: set = set()
        key_index = 0
        for attempt in range(self.max_attempts):
            while keys[key_index] in rejected:
                key_index = (key_index + 1) % len(keys)
            key = keys[key_index]
            response = None
            try:
                response = await request(
                    "POST",
                    url=f"{self.base_url}{endpoint}",
                    headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
                    data=json.dumps(payload),
                    timeout=ClientTimeout(total=self.timeout_s),
                    _max_connection_retries=0,
                )
                if response.status == 200:
                    return await response.json()
                if response.status in TAVILY_KEY_REJECTED_CODES:
                    rejected.add(key)
                    if len(rejected) < len(keys):
                        key_index = (key_index + 1) % len(keys)
                        continue  # another key may work: no backoff, same attempt budget
                elif response.status not in TAVILY_RETRY_CODES:
                    raise RuntimeError(f"Tavily HTTP {response.status} on {endpoint}")
                if attempt + 1 == self.max_attempts:
                    raise RuntimeError(f"Tavily HTTP {response.status} on {endpoint} after {attempt + 1} attempts")
            except (ClientConnectionError, ClientPayloadError, TimeoutError):
                if attempt + 1 == self.max_attempts:
                    raise RuntimeError(
                        f"Tavily transport failure on {endpoint} after {attempt + 1} attempts"
                    ) from None
            finally:
                if response is not None:
                    response.release()
            key_index = (key_index + 1) % len(keys)
            await sleep(min(2**attempt, 8))
        raise RuntimeError(f"Tavily retry budget exhausted on {endpoint}")


class HLEWebToolsResourcesServerConfig(BaseResourcesServerConfig):
    tavily_api_key: str | List[str]
    blocklist_file_path: str = DEFAULT_BLOCKLIST_FILE_PATH
    # The digest the blocklist file must have; null accepts any file (a deliberately different policy).
    blocklist_sha256: Optional[str] = UPSTREAM_BLOCKLIST_SHA256
    # A query is refused when it repeats this many consecutive words of the question. Upstream states no
    # threshold ("a long verbatim span of the question"); 8 is this server's choice.
    verbatim_span_words: int = Field(default=8, ge=2)
    # The seed-body field that carries the question; the last user message is the fallback.
    question_field: str = "question"
    # Fail closed: refuse web_search from any session that was not seeded with a question (a plain-HTTP call
    # without the rollout's cookie or MCP token, or a rollout whose seed carried no question). Off serves such
    # sessions without the span rule.
    require_seeded_question: bool = True
    max_results: int = Field(default=TAVILY_MAX_RESULTS, ge=1, le=TAVILY_MAX_RESULTS)
    max_fetch_chars: int = Field(default=PROTOCOL_MAX_FETCH_CHARS, ge=1)
    max_query_chars: int = Field(default=TAVILY_MAX_QUERY_CHARS, ge=1)
    search_depth: Literal["basic", "advanced"] = "advanced"
    extract_depth: Literal["basic", "advanced"] = "basic"
    max_cached_pages: int = Field(default=128, ge=0)
    max_cached_page_chars: int = Field(default=500_000, ge=1)
    max_http_attempts: int = Field(default=3, ge=1)
    http_timeout_s: float = Field(default=60, gt=0)
    tavily_base_url: str = "https://api.tavily.com"


class HLEWebToolsSeedSessionRequest(BaseSeedSessionRequest):
    # An Agent's run request (its extra fields are the task row) or an Environment Server's typed seed.
    model_config = ConfigDict(extra="allow")

    resources_session_id: Optional[str] = None


class WebSearchRequest(BaseModel):
    query: str = Field(description="The search query.")


class WebSearchResponse(BaseModel):
    text: str
    result_count: int
    refused: bool


class WebFetchRequest(BaseModel):
    url: str = Field(description="The http(s) URL of the web page or PDF to fetch.")


class WebFetchResponse(BaseModel):
    text: str
    truncated: bool
    refused: bool


class HLEWebToolsVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


@dataclass
class CachedPage:
    final_url: str
    title: str
    markdown: str


class HLEWebToolsResourcesServer(SimpleResourcesServer):
    config: HLEWebToolsResourcesServerConfig

    _backend: Optional[TavilyBackend] = PrivateAttr(default=None)
    _blocklist: Optional[Blocklist] = PrivateAttr(default=None)
    _session_spans: "OrderedDict[str, QuestionSpans]" = PrivateAttr(default_factory=OrderedDict)
    _page_cache: "OrderedDict[str, CachedPage]" = PrivateAttr(default_factory=OrderedDict)

    def model_post_init(self, __context) -> None:
        keys = self.config.tavily_api_key
        if isinstance(keys, str):
            keys = [keys]
        keys = [key.strip() for group in keys for key in group.split(",") if key.strip()]
        if not keys:
            raise ValueError("At least one Tavily API key is required")
        self._backend = TavilyBackend(
            api_keys=keys,
            base_url=self.config.tavily_base_url,
            max_attempts=self.config.max_http_attempts,
            timeout_s=self.config.http_timeout_s,
        )
        path = _resolve_under_cwd_or_install(self.config.blocklist_file_path)
        self._blocklist = Blocklist.load(path, self.config.blocklist_sha256)
        logger.info(
            "hle_web_tools policy %s: %d domains, %d url_prefixes, %d serp_url_patterns, %d url_patterns, "
            "%d patterns_case_sensitive, %d answer_title_patterns, %d answer_title_allow_hosts; "
            "verbatim_span_words=%d",
            path,
            len(self._blocklist.domains),
            len(self._blocklist.url_prefixes),
            len(self._blocklist.serp_url_patterns),
            len(self._blocklist.url_patterns),
            len(self._blocklist.patterns_case_sensitive),
            len(self._blocklist.answer_title_patterns),
            len(self._blocklist.answer_title_allow_hosts),
            self.config.verbatim_span_words,
        )

    # ---- routes and MCP exposure ----

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/web_search")(self.web_search)
        app.post("/web_fetch")(self.web_fetch)
        return app

    def mcp_tools(self, harvested, catchall):
        tools = [tool for tool in harvested if tool.name in MCP_TOOLS]
        if {tool.name for tool in tools} != set(MCP_TOOLS):
            raise ValueError(f"hle_web_tools must expose exactly {MCP_TOOLS}, harvested {[t.name for t in harvested]}")
        return tools

    def mcp_allowed_tools_for_session(self, seed_body: Dict[str, Any]) -> List[str]:
        return list(MCP_TOOLS)

    # ---- sessions ----

    async def seed_session(
        self, request: Request, body: HLEWebToolsSeedSessionRequest
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        question = extract_question(body.model_dump(), self.config.question_field)
        current = self._session_spans.get(session_id)
        if current is not None:
            # A session carries one question for its whole life. A retried seed with the same question is served
            # again; a seed that would change or drop the question is a conflict, so a caller who holds a session
            # (a sandbox reaching the plain route with the rollout's credentials) cannot swap the span check
            # out from under it. A fresh session seeded from inside the sandbox is NOT caught here: that needs
            # an authenticated seed channel (see the README).
            if question is None or QuestionSpans(question, self.config.verbatim_span_words).digest != current.digest:
                logger.warning(
                    "hle_web_tools refused to re-seed session %s with %s",
                    session_id,
                    "no question" if question is None else "a different question",
                )
                raise ResourcesSessionConflictError(
                    "This session was already seeded with a question; a session cannot be re-seeded with another."
                )
            self._session_spans.move_to_end(session_id)
        elif question is None:
            logger.warning(
                "hle_web_tools session %s seeded without a question (field %r or a user message): %s",
                session_id,
                self.config.question_field,
                "its web_search calls will be refused (require_seeded_question)"
                if self.config.require_seeded_question
                else "the verbatim-span rule cannot apply to its queries",
            )
        else:
            self._session_spans[session_id] = QuestionSpans(question, self.config.verbatim_span_words)
            self._session_spans.move_to_end(session_id)
            while len(self._session_spans) > MAX_SESSION_QUESTIONS:
                self._session_spans.popitem(last=False)
        if body.resources_session_id is not None:
            return ResourcesSeedSessionResponse(resources_session_id=body.resources_session_id)
        return BaseSeedSessionResponse()

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        self._session_spans.pop(body.resources_session_id, None)
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    async def retire_session_state(self, session_id: str) -> None:
        self._session_spans.pop(session_id, None)

    # ---- tools ----

    async def web_search(self, request: Request, body: WebSearchRequest) -> WebSearchResponse:
        """Search the web. Returns at most 20 results, each with a URL, title and snippet. Sources on the
        evaluation blocklist are removed, and a query that restricts search to a blocked site or quotes a long
        span of the question verbatim is refused."""
        session_id = request.session[SESSION_ID_KEY]
        spans = self._session_spans.get(session_id)
        if spans is None and self.config.require_seeded_question:
            block = Block(UNSEEDED_SESSION_FAMILY, "require_seeded_question", session_id, "session")
            self._log_block("web_search", session_id, block)
            return WebSearchResponse(text=block.refusal(), result_count=0, refused=True)
        query = body.query.strip()
        if not query:
            return WebSearchResponse(text="The query is empty.", result_count=0, refused=True)
        if len(query) > self.config.max_query_chars:
            return WebSearchResponse(
                text=f"The query is longer than {self.config.max_query_chars} characters.",
                result_count=0,
                refused=True,
            )
        block = self._blocklist.query_block(query, spans)
        if block is not None:
            self._log_block("web_search", session_id, block)
            return WebSearchResponse(text=block.refusal(), result_count=0, refused=True)

        response = await self._backend.search(
            query, max_results=self.config.max_results, search_depth=self.config.search_depth
        )
        kept: List[Dict[str, Any]] = []
        for result in response.get("results", []):
            url, title, snippet = result.get("url"), result.get("title") or "", result.get("content") or ""
            if not isinstance(url, str):
                continue
            block = self._result_block(url, title, snippet)
            if block is not None:
                self._log_block("web_search", session_id, block, dropped_result=True)
                continue
            kept.append({"url": url, "title": title, "snippet": snippet})
            if len(kept) == self.config.max_results:
                break
        if not kept:
            return WebSearchResponse(text="No results.", result_count=0, refused=False)
        lines = [
            f"[{i}] {r['title']}\n    URL: {r['url']}\n    Snippet: {r['snippet']}" for i, r in enumerate(kept, 1)
        ]
        return WebSearchResponse(text="\n\n".join(lines), result_count=len(kept), refused=False)

    async def web_fetch(self, request: Request, body: WebFetchRequest) -> WebFetchResponse:
        """Fetch a web page or PDF and return its main content as Markdown, capped at 60,000 characters with a
        truncation indicator. Image, audio and video payloads are not returned. Pages on the evaluation
        blocklist, including through archive or proxy links, are refused."""
        session_id = request.session[SESSION_ID_KEY]
        url = body.url.strip()
        block = self._blocklist.url_block(url)
        if block is not None:
            self._log_block("web_fetch", session_id, block)
            return WebFetchResponse(text=block.refusal(), truncated=False, refused=True)

        page = self._page_cache.get(url)
        if page is not None:
            self._page_cache.move_to_end(url)
        else:
            response = await self._backend.extract(url, extract_depth=self.config.extract_depth)
            results = [r for r in response.get("results", []) if isinstance(r, dict)]
            if not results or not results[0].get("raw_content"):
                failed = response.get("failed_results") or []
                error = failed[0].get("error") if failed and isinstance(failed[0], dict) else None
                detail = f": {str(error)[:200]}" if error else ""
                return WebFetchResponse(text=f"Fetch failed for {url}{detail}", truncated=False, refused=False)
            result = results[0]
            markdown = _DATA_URI_RE.sub("[data URI removed]", str(result["raw_content"]))
            markdown = markdown[: self.config.max_cached_page_chars]
            title = result.get("title")
            if not isinstance(title, str) or not title.strip():
                heading = _MARKDOWN_HEADING_RE.search(markdown)
                title = heading.group(1) if heading else ""
            final_url = result.get("url") if isinstance(result.get("url"), str) else url
            page = CachedPage(final_url=final_url, title=title.strip(), markdown=markdown)
            if self.config.max_cached_pages:
                self._page_cache[url] = page
                while len(self._page_cache) > self.config.max_cached_pages:
                    self._page_cache.popitem(last=False)

        # Cached or fresh, every response is checked: the reported final URL, then the title and the Markdown.
        block = None
        if page.final_url != url:
            block = self._blocklist.url_block(page.final_url, "fetched URL")
        block = block or self._blocklist.text_block(page.title, "fetched title")
        block = block or self._blocklist.text_block(page.markdown, "fetched content")
        if block is not None:
            self._page_cache.pop(url, None)
            self._log_block("web_fetch", session_id, block)
            return WebFetchResponse(text=block.refusal(), truncated=False, refused=True)

        content = page.markdown
        truncated = len(content) > self.config.max_fetch_chars
        if truncated:
            omitted = len(content) - self.config.max_fetch_chars
            content = (
                content[: self.config.max_fetch_chars]
                + f"\n\n[Content truncated at {self.config.max_fetch_chars:,} characters; {omitted:,} characters omitted]"
            )
        header = f"Title: {page.title}\nURL: {page.final_url}\n\n" if page.title else f"URL: {page.final_url}\n\n"
        return WebFetchResponse(text=header + content, truncated=truncated, refused=False)

    async def verify(self, request: Request, body: BaseVerifyRequest) -> HLEWebToolsVerifyResponse:
        """A tool server does not grade: pair it through ``tool_servers`` and grade with a judge resources server."""
        return HLEWebToolsVerifyResponse(
            **body.model_dump(),
            reward=0.0,
            mask_sample=True,
            failure_reason="hle_web_tools serves tools only; the rollout must be graded by its resources_server",
        )

    # ---- policy helpers ----

    def _result_block(self, url: str, title: str, snippet: str) -> Optional[Block]:
        block = self._blocklist.url_block(url, "result URL")
        if block is not None:
            return block
        try:
            host = normalize_hostname(urlsplit(url).hostname or "")
        except (ValueError, UnicodeError):
            host = ""
        block = self._blocklist.answer_title_block(title, host)
        block = block or self._blocklist.text_block(title, "result title")
        return block or self._blocklist.text_block(snippet, "result snippet")

    def _log_block(self, tool: str, session_id: str, block: Block, dropped_result: bool = False) -> None:
        subject = block.subject if block.family == QUESTION_SPAN_FAMILY else block.subject[:200]
        if dropped_result:
            logger.info(
                "hle_web_tools dropped a %s result (session %s): family=%s rule=%r where=%s subject=%r",
                tool,
                session_id,
                block.family,
                block.rule,
                block.where,
                subject,
            )
        else:
            logger.warning(
                "hle_web_tools blocked a %s call (session %s): family=%s rule=%r where=%s subject=%r",
                tool,
                session_id,
                block.family,
                block.rule,
                block.where,
                subject,
            )


if __name__ == "__main__":
    HLEWebToolsResourcesServer.run_webserver()
