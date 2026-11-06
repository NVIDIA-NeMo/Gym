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
seeded with a question (``require_seeded_question``, default on) rather than skipping the span rule. It does
not cover a fresh session the sandbox seeds itself with a question of its choice: ``/seed_session`` is not
authenticated (README, residual: sandbox-seeded sessions).
"""

import hashlib
import json
import logging
import multiprocessing
import re
import unicodedata
from asyncio import get_running_loop, sleep
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
from urllib.parse import parse_qsl, unquote, urlsplit

import idna
from aiohttp import ClientConnectionError, ClientPayloadError, ClientTimeout, ContentTypeError
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.client import SESSION_ID_KEY, request
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFence,
    SimpleResourcesServer,
)


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
# This server's bound on the whole-page check: a fetched page longer than ``max_page_chars`` is refused unread.
OVERSIZED_PAGE_FAMILY = "oversized_page"

# Keys Tavily rejects outright; the pool moves on to the next key for the same call.
TAVILY_KEY_REJECTED_CODES = frozenset({401, 432, 433})
TAVILY_RETRY_CODES = frozenset({408, 429, 500, 502, 503, 504})

_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_WWW_RE = re.compile(r"^www\.", re.IGNORECASE)
# Characters on which WHATWG fetchers and ``urllib.parse`` disagree about where a URL's parts are: a
# backslash is a path separator to WHATWG and part of the host to ``urlsplit`` (``https://a.example\.b.example``),
# and whitespace and control characters are removed or rejected by one and kept by the other.
_AMBIGUOUS_URL_CHAR_RE = re.compile(r"[\\\s\x00-\x1f\x7f-\x9f]")
# Tab and newlines, which WHATWG parsers delete wherever they appear in a URL.
_WHATWG_REMOVED_RE = re.compile(r"[\t\n\r]")
# Where an embedded URL starts, and where it ends: at a delimiter or at the start of the next embedded URL,
# so a URL nested inside another (``https://reader.example/https://blocked.example``) is found at its own
# position and the substrings taken from one URL never add up to more than its length.
_EMBEDDED_URL_START_RE = re.compile(r"https?:/", re.IGNORECASE)
_EMBEDDED_URL_END_RE = re.compile(r"[\s\"'<>]|https?:/", re.IGNORECASE)
# ``https:/host``, ``https:///host`` and ``https:host``: WHATWG parsers (and archive.org) read all of them as
# ``https://host``.
_LOOSE_SCHEME_RE = re.compile(r"^(https?):/*", re.IGNORECASE)
# A query value that names a host, with or without a scheme: ``hf.co/datasets/x``, ``scale.com./x``.
_BARE_URL_RE = re.compile(r"^(?:https?:)?(?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}\.?(?:[:/?#]|$)", re.IGNORECASE)
# The operand is host characters only, so ``(site:a.org OR site:b.org)``, ``site:a.org|site:b.org`` and
# ``site:b.org,`` yield the bare hosts; one leading dot (``site:.b.org``) is dropped. Typographic quotes and
# guillemets (``site:“b.org”``, ``site:«b.org»``) are read like ASCII quotes, and openers may be stacked
# (``site:((b.org))``).
_SITE_QUOTES = "\"'‘’‚‛“”„‟«»‹›"
_SITE_OPERATOR_RE = re.compile(
    r"\bsite:\s*[" + _SITE_QUOTES + r"(\[]*\.?([^\s" + _SITE_QUOTES + r"()\[\],;|]+)", re.IGNORECASE
)
_SLASH_RUN_RE = re.compile(r"/{2,}")
# A ``site:`` operand after ``normalize_hostname`` and with trailing punctuation dropped: labels of letters, digits,
# ``-`` and ``_``, optionally after ``*.``.
_SITE_HOST_RE = re.compile(r"^(?:\*\.)?[a-z0-9_-]+(?:\.[a-z0-9_-]+)*$")
_TRAILING_NON_HOST_RE = re.compile(r"[^a-z0-9]+$")
_WAYBACK_PATH_RE = re.compile(r"^/web/(?:[^/]*/)?(.+)$", re.IGNORECASE | re.DOTALL)
_GOOGLE_CACHE_RE = re.compile(r"cache:(?:[A-Za-z0-9_-]+:)?(.+)$", re.DOTALL)
# Word characters without the underscore: ``odd_spin_structures`` is three words, as search engines read it.
_WORD_RE = re.compile(r"[^\W_]+")
# A data URI with or without a media type (RFC 2397 lets ``data:;base64,`` omit it) and media-type parameters,
# base64 or percent-encoded, up to the end of its Markdown destination: the payload stops at whitespace, ``)``, a
# quote or ``>``, so following prose is kept. Every repetition is bounded, so the scan stays linear on
# adversarial pages.
_DATA_URI_RE = re.compile(
    r"\bdata:(?:[\w.+-]{1,127}/[\w.+-]{1,127})?(?:;[\w.+-]{1,127}=[^;,\s)]{0,256}){0,16}(?:;base64)?,[^\s)\"'>]*",
    re.IGNORECASE,
)
# Starts only at the first blank of a run, so a long run of blanks is scanned once, not once per blank.
_TITLE_SEPARATOR_RE = re.compile(r"(?<!\s)\s+(?:\||-|–|—|·|•|»|::)\s+")
# Linear on every line: the capture starts at a non-blank character and runs to the end of the line, and the
# closing ``#`` sequence and trailing blanks are stripped afterwards, outside the regex.
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}[ \t]+(\S.*)$", re.MULTILINE)
# Hosts whose job is to serve another page: their path remainder is treated as the wrapped URL. Every host
# under ``WRAPPER_DOMAINS`` counts (``web.archive.org``, ``wayback.archive.org``, ...). Hosts the blocklist
# names outright (archive.ph and friends, r.jina.ai, translate.goog, translate.google.com, 12ft.io,
# corsproxy.io, allorigins, codetabs, thingproxy, whateverorigin) are refused by the ``domains`` family
# before any unwrapping is needed.
WRAPPER_HOSTS = frozenset({"webcache.googleusercontent.com", "r.jina.ai"})
WRAPPER_DOMAINS = ("archive.org",)
DEFAULT_PORTS = {"http": 80, "https": 443}
# A longer URL is refused as ``malformed_url`` before any parsing.
MAX_URL_CHARS = 4096
# The URL, its percent-decoded forms and the URLs wrapped in them are expanded breadth first for the original
# plus this many rounds and up to this many distinct candidates; a URL that still has unexplored candidates
# when either bound is reached is refused as ``malformed_url`` rather than certified on a partial check.
MAX_URL_CANDIDATES = 64
MAX_URL_DECODE_ROUNDS = 4
MAX_SESSION_QUESTIONS = 10_000
# A result or page title longer than this is cut, with an ellipsis, before it is shown or checked against
# ``answer_title_patterns``; providers put no bound on titles.
MAX_TITLE_CHARS = 1_000


def cap_title(title: str) -> str:
    return title if len(title) <= MAX_TITLE_CHARS else title[:MAX_TITLE_CHARS] + "…"


def normalize_hostname(value: str) -> str:
    """The one form every hostname comparison uses: mapped as UTS #46 fetchers map it, IDNA-encoded, lower-cased,
    trailing dot stripped.

    Browsers (WHATWG URL) and ``yarl`` map a host with UTS #46 before IDNA, so ``sc\U0001f130le.com`` and
    ``sc\u1d43le.com`` reach ``scale.com``; Python's IDNA 2003 codec alone leaves such letters unmapped.
    """
    if not value.isascii():
        value = idna.uts46_remap(value, std3_rules=False, transitional=False)
    return value.encode("idna").decode().lower().rstrip(".")


def schemeless(url: str) -> str:
    """The URL without its scheme: what ``serp_url_patterns`` are anchored against."""
    return _SCHEME_RE.sub("", url.strip(), count=1)


def prefix_form(url: str) -> str:
    """The URL as ``url_prefixes`` see it: scheme and leading ``www.`` removed, lower-cased."""
    return _WWW_RE.sub("", schemeless(url), count=1).lower()


def remove_dot_segments(path: str) -> str:
    """RFC 3986 section 5.2.4 on an absolute path: ``.`` and ``..`` segments resolved, ``//`` runs kept."""
    segments = path.split("/")
    out: List[str] = []
    for i, segment in enumerate(segments[1:], 1):
        if segment in (".", ".."):
            if segment == ".." and out:
                out.pop()
            if i == len(segments) - 1:
                out.append("")
        else:
            out.append(segment)
    return "/" + "/".join(out)


def canonical_schemeless(url: str) -> Optional[str]:
    """The scheme-less URL as a fetcher resolves it: IDNA, lower-cased host without a trailing dot, userinfo
    and a default port dropped, dot segments resolved, runs of ``/`` collapsed (as nginx's ``merge_slashes``
    serves them), fragment dropped. None when it has no parseable host. Used for matching only, never fetched."""
    try:
        parts = urlsplit(url)
        if not parts.hostname:
            return None
        host = normalize_hostname(parts.hostname)
        port = parts.port
    except (ValueError, UnicodeError):
        return None
    netloc = host if port in (None, DEFAULT_PORTS.get(parts.scheme.lower())) else f"{host}:{port}"
    path = _SLASH_RUN_RE.sub("/", remove_dot_segments(parts.path or "/"))
    return netloc + path + (f"?{parts.query}" if parts.query else "")


def url_shape_problem(url: str) -> Optional[str]:
    """Why a URL is refused before any expansion (``malformed_url``), else None."""
    if len(url) > MAX_URL_CHARS:
        return f"URL of at most {MAX_URL_CHARS} characters"
    if _AMBIGUOUS_URL_CHAR_RE.search(url):
        return "URL without backslashes, whitespace or control characters"
    try:
        parts = urlsplit(url)
        if parts.scheme.lower() not in DEFAULT_PORTS or not parts.hostname:
            return "http(s) URL with a host"
        if "@" in parts.netloc:
            return "http(s) URL without userinfo"
        normalize_hostname(parts.hostname)
        parts.port  # noqa: B018 - raises ValueError on a port that is not a number in range
    except (ValueError, UnicodeError):
        return "parseable URL"
    return None


class URLExpansionLimitError(ValueError):
    """Candidate expansion stopped with candidates unexplored: the URL cannot be certified."""


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
            MALFORMED_URL_FAMILY: "the site: operand is not a valid hostname"
            if self.where == "site: operator"
            else "the URL is not a valid http(s) URL the policy can check",
            UNSEEDED_SESSION_FAMILY: "search is served only to a rollout session seeded with its question",
            OVERSIZED_PAGE_FAMILY: "the page is larger than the policy can check",
        }[self.family]
        return f"Refused by the HLE web-tool policy: {reason}."


class Blocklist:
    """Upstream's ``docs/blocklist.json`` with the semantics of ``docs/evaluation-with-tools.md``.

    - ``domains``: exact hostnames and all their subdomains.
    - ``url_prefixes``: prefixes of the URL after removing the scheme and a leading ``www.``, ignoring case; a
      prefix with a query also matches with its ``key=value`` pairs in any order among the URL's.
    - ``serp_url_patterns``: regexes anchored at the start of the scheme-less URL of a search-engine result page.
      Both URL families are tried on the URL as written and on its canonical form (``canonical_schemeless``).
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
        # A prefix with a query (``openreview.net/pdf?id=X``) also matches its path as a prefix of the URL's path
        # with each of its ``key=value`` pairs among the URL's query pairs, in any order: a server reads
        # ``pdf?&id=X`` and ``pdf?x=1&id=X`` as ``pdf?id=X``.
        self.query_prefixes: List[Tuple[str, str, List[Tuple[str, str]]]] = []
        for prefix in self.url_prefixes:
            path, separator, query = prefix.lower().partition("?")
            if separator:
                self.query_prefixes.append((prefix, path, parse_qsl(query, keep_blank_values=True)))
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
        """The URL-shaped families against one URL string, plus the text families against the string itself.

        ``url_prefixes`` and ``serp_url_patterns`` see both the string as written and its canonical form, so
        ``arxiv.org:443/abs/...``, ``arxiv.org./abs/...`` and ``arxiv.org/x/../abs/...`` match ``arxiv.org/abs/...``.
        """
        try:
            parts = urlsplit(url)
            host = normalize_hostname(parts.hostname) if parts.hostname else ""
        except (ValueError, UnicodeError):
            # A wrapped or decoded URL Python cannot parse may still be one a fetcher reads (``https://[::1]@scale.com``
            # is ``scale.com`` to a browser and to httpx): refuse it rather than skip its checks.
            return Block(MALFORMED_URL_FAMILY, "parseable URL", url, where)
        if host:
            block = self.host_block(host, url, where)
            if block is not None:
                return block
        serp_forms = [schemeless(url)]
        canonical = canonical_schemeless(url)
        if canonical is not None and canonical != serp_forms[0]:
            serp_forms.append(canonical)
        prefix_forms = {_WWW_RE.sub("", form, count=1).lower() for form in serp_forms}
        for prefix in self.url_prefixes:
            if any(form.startswith(prefix.lower()) for form in prefix_forms):
                return Block("url_prefixes", prefix, url, where)
        for prefix, path, pairs in self.query_prefixes:
            if any(_query_prefix_matches(form, path, pairs) for form in prefix_forms):
                return Block("url_prefixes", prefix, url, where)
        for pattern in self.serp_url_patterns:
            if any(pattern.search(form) for form in serp_forms):
                return Block("serp_url_patterns", pattern.pattern, url, where)
        return self.text_block(url, where)

    def url_block(self, url: str, where: str = "URL") -> Optional[Block]:
        """Every URL-shaped rule against the URL, its percent-decoded forms and the URLs it wraps.

        Fails closed: a URL whose shape fetchers could read differently, or whose expansion would need more
        candidates or rounds than are checked, is refused as ``malformed_url``.
        """
        problem = url_shape_problem(url)
        if problem is not None:
            return Block(MALFORMED_URL_FAMILY, problem, url, where)
        try:
            candidates = url_candidates(url)
        except URLExpansionLimitError as error:
            return Block(MALFORMED_URL_FAMILY, str(error), url, where)
        for candidate in candidates:
            block = self._single_url_block(candidate, where)
            if block is not None:
                return block
        return None

    def query_block(self, query: str, question_spans: Optional["QuestionSpans"]) -> Optional[Block]:
        """The protocol's query refusals: blocked patterns, a blocked ``site:`` domain, a verbatim question span.

        The patterns and ``site:`` operands are checked on the query as written and as ``fold_text`` reads it.
        """
        forms = list(dict.fromkeys((query, fold_text(query), fold_text(query, " "))))
        for form in forms:
            block = self.text_block(form, "query")
            if block is not None:
                return block
        for match in (match for form in forms for match in _SITE_OPERATOR_RE.finditer(form)):
            host = schemeless(match.group(1)).split("/", 1)[0].split(":", 1)[0]
            try:
                # Trailing punctuation (``site:huggingface.co!``) is dropped, as search engines drop it.
                host = _TRAILING_NON_HOST_RE.sub("", normalize_hostname(_WWW_RE.sub("", host, count=1)))
            except UnicodeError:
                host = ""
            if not _SITE_HOST_RE.match(host):
                # An operand that is no hostname cannot be checked against ``domains``: refuse, do not skip it.
                return Block(
                    MALFORMED_URL_FAMILY, "site: operand that is a hostname", match.group(0), "site: operator"
                )
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


def _query_prefix_matches(form: str, path: str, pairs: List[Tuple[str, str]]) -> bool:
    """Whether a prefix form's path starts with ``path`` and its query holds every pair (a value as a prefix)."""
    form_path, _, query = form.partition("?")
    if not form_path.startswith(path):
        return False
    form_pairs = parse_qsl(query.partition("#")[0], keep_blank_values=True)
    return all(any(key == k and value.startswith(v) for key, value in form_pairs) for k, v in pairs)


def title_variants(title: str) -> List[str]:
    """The title, then the title with each trailing ``| Site``-style segment removed, longest first."""
    variants = [title]
    for match in _TITLE_SEPARATOR_RE.finditer(title):
        variants.append(title[: match.start()])
    return variants


def _is_wrapper_host(host: str) -> bool:
    return host in WRAPPER_HOSTS or any(host == d or host.endswith("." + d) for d in WRAPPER_DOMAINS)


def _embedded_urls(url: str) -> List[str]:
    """Every ``http(s):/`` inside the URL (not at its start), each up to a delimiter or the next one: linear."""
    found: List[str] = []
    for match in _EMBEDDED_URL_START_RE.finditer(url):
        if match.start() == 0:
            continue
        end = _EMBEDDED_URL_END_RE.search(url, match.end())
        found.append(url[match.start() : end.start() if end else len(url)])
    return found


def _wrapped_urls(url: str) -> List[str]:
    """URLs a proxy or archive wrapper carries: the path remainder of known wrappers, ``cache:`` targets,
    every query value that names a host and every URL embedded in the string.

    A wrapper fetches its remainder (and Google's cache its ``cache:`` target) whatever its spelling, so each is
    read as a URL even without a scheme, and ``urlsplit`` and ``normalize_hostname`` decide its host:
    ``scale.com./x``, ``x@scale.com/x`` and ``ｓｃａｌｅ.com/x`` all name ``scale.com``. Any other query value is
    read as a URL only when it has a scheme or starts with a hostname.
    """
    targets: List[str] = []
    found: List[str] = []
    try:
        parts = urlsplit(url)
        host = normalize_hostname(parts.hostname) if parts.hostname else ""
    except (ValueError, UnicodeError):
        return found
    if _is_wrapper_host(host):
        wayback = _WAYBACK_PATH_RE.match(parts.path)
        # ``/web/<timestamp>/<url>``, or ``/web/<url>`` with no timestamp, which archive.org redirects to a snapshot
        # of ``<url>``: the first segment is read both ways.
        remainders = dict.fromkeys((wayback.group(1), parts.path[len("/web/") :])) if wayback else [parts.path]
        for remainder in remainders:
            if not _LOOSE_SCHEME_RE.match(remainder):
                remainder = remainder.lstrip("/")
            if parts.query and host != "webcache.googleusercontent.com":
                remainder += "?" + parts.query
            if remainder:
                targets.append(remainder)
    for _, value in parse_qsl(parts.query, keep_blank_values=False):
        cache = _GOOGLE_CACHE_RE.search(value)
        if cache:
            targets.append(cache.group(1))
        found.append(value)
    found.extend(_embedded_urls(url))
    urls: List[str] = []
    for index, item in enumerate(targets + found):
        is_target = index < len(targets)
        # ``https:/host`` and ``https:///host`` reach the same page as ``https://host``.
        item = _LOOSE_SCHEME_RE.sub(lambda m: m.group(1) + "://", item.strip(), count=1)
        if not item:
            continue
        if _SCHEME_RE.match(item):
            urls.append(item)
        elif is_target or _BARE_URL_RE.match(item):
            urls.append("https://" + item.lstrip("/"))
    return urls


def url_candidates(url: str) -> List[str]:
    """The URL, its percent-decoded forms and every URL wrapped inside any of them, breadth first.

    A wrapped or decoded URL is also read the way a WHATWG fetcher reads it (backslashes as ``/``, tabs and
    newlines deleted). Raises ``URLExpansionLimitError`` rather than return a partial list: when more than
    ``MAX_URL_CANDIDATES`` distinct candidates exist, or candidates remain after the original and
    ``MAX_URL_DECODE_ROUNDS`` rounds.
    """
    seen: List[str] = []
    seen_set: Set[str] = set()
    frontier = [url]
    for _ in range(MAX_URL_DECODE_ROUNDS + 1):
        next_frontier: List[str] = []
        for candidate in frontier:
            if candidate in seen_set:
                continue
            if len(seen) == MAX_URL_CANDIDATES:
                raise URLExpansionLimitError(f"URL expanding to at most {MAX_URL_CANDIDATES} candidates")
            seen.append(candidate)
            seen_set.add(candidate)
            next_frontier.extend(_wrapped_urls(candidate))
            for form in (unquote(candidate), _WHATWG_REMOVED_RE.sub("", candidate).replace("\\", "/")):
                if form != candidate:
                    next_frontier.append(form)
        frontier = [candidate for candidate in next_frontier if candidate not in seen_set]
        if not frontier:
            return seen
    raise URLExpansionLimitError(f"URL nested or encoded at most {MAX_URL_DECODE_ROUNDS} levels deep")


def markdown_heading_title(markdown: str) -> str:
    """The text of the first non-empty ATX heading (``# Title``, closing ``#`` sequence removed), else ``""``."""
    for match in _MARKDOWN_HEADING_RE.finditer(markdown):
        title = match.group(1).rstrip()
        body = title.rstrip("#")
        if body != title and (not body or body[-1] in " \t"):
            title = body.rstrip()
        if title:
            return title
    return ""


def fold_text(text: str, format_chars: str = "") -> str:
    """NFKC with each format character (category Cf: zero-width space, soft hyphen, word joiner) replaced by
    ``format_chars``: the spelling search engines index, so full-width, mathematical and zero-width variants read
    as the plain text. Whether an engine drops a format character or splits words at it is not known, so the
    checks read text both ways: removed (``sp<ZWSP>in`` is ``spin``) and as a space (``How<WJ>many`` is two words).
    """
    if text.isascii():
        return text
    return "".join(
        format_chars if unicodedata.category(ch) == "Cf" else ch for ch in unicodedata.normalize("NFKC", text)
    )


def question_words(text: str, format_chars: str = "") -> Tuple[str, ...]:
    """Case-folded maximal runs of letters and digits after ``fold_text`` and with combining marks removed
    (``holomorphíc`` is ``holomorphic``): the unit of the verbatim-span rule."""
    text = fold_text(text, format_chars)
    if not text.isascii():
        decomposed = unicodedata.normalize("NFD", text)
        text = unicodedata.normalize("NFC", "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn"))
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
        readings = (words, question_words(question, " ")) if n else ()
        self.ngrams = frozenset(reading[i : i + n] for reading in readings for i in range(len(reading) - n + 1))

    def longest_shared_span(self, query: str) -> Optional[int]:
        """The span length when the query repeats that many consecutive question words, else ``None``."""
        n = self.span_words
        if not n:
            return None
        for words in dict.fromkeys((question_words(query), question_words(query, " "))):
            if any(words[i : i + n] in self.ngrams for i in range(len(words) - n + 1)):
                return n
        return None


def extract_question(seed_body: Dict[str, Any], field: str) -> Optional[str]:
    """The question text of a seed: the task row's ``field``, else the last user message of its create params.

    An Agent seeds a tool server with its run request, whose extra fields are the task row (``question`` for
    the HLE presets); an Environment Server seeds with ``task_data`` holding the same row. A text with no words
    (``"?!"``) cannot drive the verbatim-span rule and counts as no question, so such a session fails closed.
    """
    question = _question_text(seed_body, field)
    return question if question is not None and question_words(question) else None


def _question_text(seed_body: Dict[str, Any], field: str) -> Optional[str]:
    data = seed_body.get("task_data") if isinstance(seed_body.get("task_data"), dict) else seed_body
    value = data.get(field)
    if isinstance(value, str) and question_words(value):
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

    A key Tavily rejects (401, 432, 433) is skipped for the rest of the call without spending the retry budget,
    and the call fails once every key is rejected; transient failures retry with backoff on the next key that
    is still accepted. Provider bodies are never surfaced: they may describe credentials.
    """

    api_keys: List[str] = Field(repr=False, min_length=1)
    base_url: str = "https://api.tavily.com"
    max_attempts: int = Field(default=3, ge=1)
    timeout_s: float = Field(default=60, gt=0)
    _next_key: int = PrivateAttr(default=0)

    @field_validator("api_keys")
    @classmethod
    def _distinct_keys(cls, keys: List[str]) -> List[str]:
        # A key listed twice is one key: Tavily rejecting it rejects every copy.
        return list(dict.fromkeys(keys))

    async def search(self, query: str, *, max_results: int, search_depth: str) -> Any:
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

    async def extract(self, url: str, *, extract_depth: str) -> Any:
        return await self._post(
            "/extract",
            {"urls": [url], "extract_depth": extract_depth, "format": "markdown", "include_images": False},
        )

    async def _post(self, endpoint: str, payload: Dict[str, Any]) -> Any:
        """The decoded body of Tavily's 200 reply, or None when that body is not JSON."""
        keys = self.api_keys[self._next_key :] + self.api_keys[: self._next_key]
        self._next_key = (self._next_key + 1) % len(self.api_keys)
        rejected: Set[int] = set()
        key_index = 0
        attempt = 0  # spent by transient failures only; a rejected key moves on without spending it
        while True:
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
                    try:
                        return await response.json()
                    except (ContentTypeError, ValueError):
                        # A proxy or gateway page in a 200: a malformed body, which the tools answer as a tool error.
                        return None
                if response.status in TAVILY_KEY_REJECTED_CODES:
                    rejected.add(key_index)
                    next_index = _next_accepted_key(len(keys), key_index, rejected)
                    if next_index is None:
                        raise RuntimeError(
                            f"Tavily HTTP {response.status} on {endpoint}: every configured key was rejected"
                        )
                    key_index = next_index
                    continue  # another key may work: no backoff, same attempt budget
                if response.status not in TAVILY_RETRY_CODES:
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
            # The current key was not rejected, so there is always one to retry on.
            key_index = _next_accepted_key(len(keys), key_index, rejected)
            await sleep(min(2**attempt, 8))
            attempt += 1


def _next_accepted_key(count: int, index: int, rejected: Set[int]) -> Optional[int]:
    """The first key index after ``index`` (wrapping round to ``index`` itself) not in ``rejected``, or None."""
    for step in range(1, count + 1):
        candidate = (index + step) % count
        if candidate not in rejected:
            return candidate
    return None


def _tavily_results(response: Any) -> Optional[List[Dict[str, Any]]]:
    """The objects in a Tavily body's ``results`` list, or None when the body or that list is malformed."""
    results = response.get("results") if isinstance(response, dict) else None
    return [r for r in results if isinstance(r, dict)] if isinstance(results, list) else None


def _text_field(item: Dict[str, Any], key: str) -> str:
    value = item.get(key)
    return value if isinstance(value, str) else ""


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
    # A fetched page (the provider's Markdown, before anything is removed or cut) longer than this is refused
    # unread as ``oversized_page``: the content families are checked on the whole page, and this bounds that scan.
    max_page_chars: int = Field(default=1_000_000, ge=1)
    max_cached_pages: int = Field(default=128, ge=0)
    # A page whose sanitized Markdown is longer than this is checked and served but not cached.
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


# The page work (removing data URIs, the heading fallback for the title, the content families) runs in worker
# processes, not threads: Python's ``re`` keeps the GIL for a whole ``search()`` or ``sub()`` call, so one scan of
# an adversarial page (about 4 s for 1,000,000 characters of ``抱抱脸`` against the pinned blocklist) would stall
# the event loop from a thread just as from the loop itself. One pool per process, shared by every server instance;
# ``spawn`` starts each worker as a fresh interpreter, which imports this module once.
PAGE_CHECK_PROCESSES = 2
_page_check_pool: Optional[ProcessPoolExecutor] = None


async def run_page_check(function, *args):
    """``function(*args)`` in the page-check pool; a pool whose worker died is replaced for the next call."""
    global _page_check_pool
    if _page_check_pool is None:
        _page_check_pool = ProcessPoolExecutor(
            max_workers=PAGE_CHECK_PROCESSES, mp_context=multiprocessing.get_context("spawn")
        )
    pool = _page_check_pool
    try:
        return await get_running_loop().run_in_executor(pool, function, *args)
    except BrokenProcessPool:
        if _page_check_pool is pool:
            _page_check_pool = None
        raise


def build_page(raw: str, title: Any, final_url: str) -> CachedPage:
    """The page as served and cached: data URIs removed, the provider's title or else the first heading."""
    markdown = _DATA_URI_RE.sub("[data URI removed]", raw)
    if not isinstance(title, str) or not title.strip():
        title = markdown_heading_title(markdown)
    return CachedPage(final_url=final_url, title=cap_title(title.strip()), markdown=markdown)


def content_block(blocklist: Blocklist, page: CachedPage, raw: Optional[str]) -> Optional[Block]:
    """The text families against the fetched title, the provider's raw page (fresh fetches) and the sanitized
    Markdown (unless it is the raw page unchanged)."""
    block = blocklist.text_block(page.title, "fetched title")
    if block is None and raw is not None:
        block = blocklist.text_block(raw, "fetched content")
    if block is None and page.markdown != raw:
        block = blocklist.text_block(page.markdown, "fetched content")
    return block


def check_fetched_page(
    blocklist: Blocklist, raw: str, title: Any, final_url: str
) -> Tuple[CachedPage, Optional[Block]]:
    """A fresh page as served, and the content block that refuses it (if any): one page-check call."""
    page = build_page(raw, title, final_url)
    return page, content_block(blocklist, page, raw)


class HLEWebToolsResourcesServer(SimpleResourcesServer):
    config: HLEWebToolsResourcesServerConfig

    _backend: Optional[TavilyBackend] = PrivateAttr(default=None)
    _blocklist: Optional[Blocklist] = PrivateAttr(default=None)
    _session_spans: "OrderedDict[str, QuestionSpans]" = PrivateAttr(default_factory=OrderedDict)
    # The episode and task each typed (Environment Server) session is bound to, under the same cap as the questions.
    _session_identity: "OrderedDict[str, Tuple[EpisodeId, TaskId]]" = PrivateAttr(default_factory=OrderedDict)
    _fence: ResourcesSessionFence = PrivateAttr()
    _page_cache: "OrderedDict[str, CachedPage]" = PrivateAttr(default_factory=OrderedDict)

    def model_post_init(self, __context) -> None:
        self._fence = ResourcesSessionFence(self.config.session_close_retry_window_seconds)
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

    def multiple_workers_unsupported_reason(self) -> Optional[str]:
        # A rollout's seed and its searches could reach different workers, and the module defines no worker app.
        return "seeded questions, session bindings and the page cache live in one process"

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
        raw = body.model_dump()
        question = extract_question(raw, self.config.question_field)
        if body.resources_session_id is None:
            # An Agent's run body: the session is the cookie's, and carries no episode identity.
            self._install_question(request.session[SESSION_ID_KEY], question)
            return BaseSeedSessionResponse()
        try:
            typed = ResourcesSeedSessionRequest.model_validate(raw)
        except ValueError as error:
            raise ResourcesSeedRejectedError(f"invalid typed seed: {error}") from error
        resources_session_id = typed.resources_session_id
        identity = (typed.episode_id, typed.task_id)
        async with self._fence.locked(resources_session_id):
            # A seed that arrives after its session closed (a retry, or a reply that raced the close) must not
            # recreate it.
            self._fence.raise_if_closed(resources_session_id)
            bound = self._session_identity.get(resources_session_id)
            if bound is not None and bound != identity:
                raise ResourcesSessionConflictError("resources_session_id is already bound to another episode or task")
            # Session routing makes the cookie's session ID equal resources_session_id for a typed seed.
            self._install_question(resources_session_id, question)
            self._session_identity[resources_session_id] = identity
            self._session_identity.move_to_end(resources_session_id)
            self._forget_sessions_over_cap()
        return ResourcesSeedSessionResponse(resources_session_id=resources_session_id)

    def _install_question(self, session_id: str, question: Optional[str]) -> None:
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
            self._forget_sessions_over_cap()

    def _forget_sessions_over_cap(self) -> None:
        """Forget the least recently seeded sessions, question and identity together, beyond the cap.

        A forgotten typed session is fenced as if closed by its episode, so a later seed of its ID for any episode
        gets 409 instead of binding a session whose question is gone.
        """
        for table in (self._session_spans, self._session_identity):
            while len(table) > MAX_SESSION_QUESTIONS:
                session_id = next(iter(table))
                identity = self._session_identity.pop(session_id, None)
                if identity is not None:
                    self._fence.mark_closed(session_id, identity[0])
                if self._session_spans.pop(session_id, None) is not None:
                    logger.warning(
                        "hle_web_tools forgot the question of session %s: more than %d seeded sessions are held; "
                        "its web_search calls will be refused as %s",
                        session_id,
                        MAX_SESSION_QUESTIONS,
                        UNSEEDED_SESSION_FAMILY,
                    )

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        # Released by resources_session_id: a close after a lost seed reply carries no cookie.
        resources_session_id = body.resources_session_id
        async with self._fence.locked(resources_session_id):
            closed_episode = self._fence.closed_episode(resources_session_id)
            if closed_episode is not None and closed_episode != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
            bound = self._session_identity.get(resources_session_id)
            if bound is not None and bound[0] != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the seeded resources session")
            self._session_spans.pop(resources_session_id, None)
            self._session_identity.pop(resources_session_id, None)
            self._fence.mark_closed(resources_session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=resources_session_id)

    async def retire_session_state(self, session_id: str) -> None:
        self._session_spans.pop(session_id, None)
        self._session_identity.pop(session_id, None)

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
        results = _tavily_results(response)
        if results is None:
            logger.warning("hle_web_tools web_search got a malformed Tavily body (session %s)", session_id)
            return WebSearchResponse(
                text="Search failed: the search provider returned a malformed response.", result_count=0, refused=False
            )
        kept: List[Dict[str, Any]] = []
        for result in results:
            url, title, snippet = result.get("url"), _text_field(result, "title"), _text_field(result, "content")
            if not isinstance(url, str):
                continue
            block = self._result_block(url, title, snippet)
            if block is not None:
                self._log_block("web_search", session_id, block, dropped_result=True)
                continue
            kept.append({"url": url, "title": cap_title(title), "snippet": snippet})
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
        raw: Optional[str] = None
        if page is not None:
            self._page_cache.move_to_end(url)
        else:
            response = await self._backend.extract(url, extract_depth=self.config.extract_depth)
            results = _tavily_results(response) or []
            if not results or not _text_field(results[0], "raw_content"):
                failed = response.get("failed_results") if isinstance(response, dict) else None
                failed = failed if isinstance(failed, list) else []
                error = failed[0].get("error") if failed and isinstance(failed[0], dict) else None
                detail = f": {str(error)[:200]}" if error else ""
                return WebFetchResponse(text=f"Fetch failed for {url}{detail}", truncated=False, refused=False)
            result = results[0]
            raw = result["raw_content"]
            if len(raw) > self.config.max_page_chars:
                block = Block(
                    OVERSIZED_PAGE_FAMILY,
                    f"max_page_chars={self.config.max_page_chars}",
                    f"{len(raw)} characters",
                    "fetched content",
                )
                self._log_block("web_fetch", session_id, block)
                return WebFetchResponse(text=block.refusal(), truncated=False, refused=True)
            final_url = result.get("url") if isinstance(result.get("url"), str) else url

        # Cached or fresh, every response is checked: the reported final URL, then the title and the content.
        # A fresh page's content is the provider's whole page before any data URI is removed or anything is cut;
        # a cached page's is its sanitized Markdown. The page work runs in the page-check processes.
        if page is None:
            block = None if final_url == url else self._blocklist.url_block(final_url, "fetched URL")
            if block is None:
                page, block = await run_page_check(
                    check_fetched_page, self._blocklist, raw, result.get("title"), final_url
                )
        else:
            block = None if page.final_url == url else self._blocklist.url_block(page.final_url, "fetched URL")
            block = block or await run_page_check(content_block, self._blocklist, page, None)
        if block is not None:
            self._page_cache.pop(url, None)
            self._log_block("web_fetch", session_id, block)
            return WebFetchResponse(text=block.refusal(), truncated=False, refused=True)
        if (
            raw is not None
            and self.config.max_cached_pages
            and len(page.markdown) <= self.config.max_cached_page_chars
        ):
            self._page_cache[url] = page
            while len(self._page_cache) > self.config.max_cached_pages:
                self._page_cache.popitem(last=False)

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
        # The text families see the whole title (linear); the answer-title variants see the title as shown.
        block = self._blocklist.answer_title_block(cap_title(title), host)
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
