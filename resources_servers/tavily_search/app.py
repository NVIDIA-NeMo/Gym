# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging
import re
from asyncio import sleep
from collections import OrderedDict, defaultdict
from contextlib import asynccontextmanager
from fnmatch import fnmatchcase
from pathlib import Path
from time import time
from typing import Any, ClassVar, Dict, List, Literal, Optional, Sequence
from urllib.parse import unquote, urlparse, urlsplit, urlunsplit

from aiohttp import ClientConnectionError, ClientPayloadError, ClientTimeout, ContentTypeError
from fastapi import FastAPI, Request
from httpx import AsyncClient
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from tavily import AsyncTavilyClient

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.client import RETRY_ERROR_CODES, SESSION_ID_KEY, request
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    SimpleResourcesServer,
)
from nemo_gym.verification import JUDGE_FAILURES, call_judge, is_judge_failure
from resources_servers.tavily_search.judge_prompt import JUDGE_PROMPT_TEMPLATE


logger = logging.getLogger(__name__)

# Tavily's /search rejects more than 150 ``exclude_domains`` with HTTP 400 ("Max 150 domains are allowed";
# https://docs.tavily.com/documentation/api-reference/endpoint/search, confirmed by direct probes on 2026-10-16:
# 150 domains accepted, 151 rejected). A longer policy is pre-filtered provider-side with this many domains
# and enforced in full locally; see ``select_provider_domains``.
TAVILY_EXCLUDE_DOMAINS_LIMIT = 150
# Policy domains sent to the provider ahead of the rest when the policy exceeds the limit: dataset hubs,
# benchmark publishers, code hosts and archive mirrors, the hosts most likely to surface an answer key.
# Shell-style patterns matched against the policy's normalised hostnames; a bare hostname matches only itself.
# Patterns are normalised the same way (IDNA, lower-case, trailing dot removed), so their case does not matter.
DEFAULT_EXCLUDE_DOMAINS_PRIORITY = (
    "huggingface.co",
    "hf.co",
    "lastexam.ai",
    "agi.safe.ai",
    "safe.ai",
    "scale.com",
    "*.scale.com",
    "github.com",
    "githubusercontent.com",
    "archive.*",
)


class TavilySearchResourcesServerConfig(BaseResourcesServerConfig):
    # The search backend of web_search, find_in_page and scroll_page. The exclusion policy, page cache, output
    # limits and retries apply to both; only the chosen provider's key is required.
    search_provider: Literal["tavily", "exa"] = "tavily"
    tavily_api_key: Optional[str | List[str]] = None
    exa_api_key: Optional[str | List[str]] = None
    exa_base_url: str = "https://api.exa.ai"
    exclude_domains_file_path: str
    use_judge: bool = True  # If False, use regex matching instead of LLM judge
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    debug: bool = False
    dump_session_id_to_metrics_on_exit: bool = False
    max_results: int = Field(default=10, ge=1, le=20)
    max_result_chars: int = Field(default=2000, ge=1)
    search_depth: Literal["basic", "advanced"] = "advanced"
    max_http_attempts: int = Field(default=3, ge=1)
    http_timeout_s: float = Field(default=60, gt=0)
    max_cached_pages: int = Field(default=128, ge=0)
    max_cached_page_chars: Optional[int] = Field(default=None, ge=1)
    max_scroll_words: Optional[int] = Field(default=None, ge=1)
    # Provider-side pre-filtering only; 0 or null sends no exclude_domains to Tavily. A value above Tavily's limit
    # is rejected here because every /search would then fail with HTTP 400. The full domain and url_substring
    # policy is enforced locally on requested and returned URLs whatever this is set to.
    tavily_exclude_domains_limit: Optional[int] = Field(
        default=TAVILY_EXCLUDE_DOMAINS_LIMIT, ge=0, le=TAVILY_EXCLUDE_DOMAINS_LIMIT
    )
    tavily_exclude_domains_priority: List[str] = Field(default_factory=lambda: list(DEFAULT_EXCLUDE_DOMAINS_PRIORITY))


class TavilySearchRequest(BaseModel):
    query: Optional[str] = None  # Make optional to handle missing args gracefully


class TavilySearchResponse(BaseModel):
    results_string: str


class FindInPageRequest(BaseModel):
    url: Optional[str] = None
    query: Optional[str] = None


class FindInPageResponse(BaseModel):
    results_string: str


class ScrollPageRequest(BaseModel):
    url: Optional[str] = None
    start_index: int = 0
    n: int = 2000


class ScrollPageResponse(BaseModel):
    results_string: str
    total_words: int


class TavilySearchRunRequest(BaseRunRequest):
    ground_truth: str
    question: str


class TavilySearchVerifyRequest(TavilySearchRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    judge_response_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    reasoning: str
    extracted_final_answer: str
    reward: float
    judge_response: Optional[NeMoGymResponse] = None


class TavilySearchSingleAsyncTavilyMetrics(BaseModel):
    function: str
    status: str
    start_time: float
    end_time: float
    time_taken: Optional[float] = None

    @model_validator(mode="after")
    def compute_time_taken(self):
        self.time_taken = self.end_time - self.start_time
        return self


class TavilySearchMetrics(BaseModel):
    async_tavily_calls: List[TavilySearchSingleAsyncTavilyMetrics] = Field(default_factory=list)


class TavilySearchVerifyResponse(TavilySearchVerifyRequest, JudgeEvaluation):
    num_tool_calls: int
    metrics: TavilySearchMetrics


class TavilySearchAIOHTTPClientResponse(BaseModel):
    status_code: int
    data: Dict[str, Any]

    def json(self) -> Dict[str, Any]:
        return self.data


class TavilySearchAIOHTTPClient(BaseModel):
    headers: Dict[str, str] = Field(repr=False)
    base_url: str
    debug: bool
    retry_api_keys: List[str] = Field(default_factory=list, repr=False)
    max_attempts: int = Field(default=3, ge=1)
    timeout_s: float = Field(default=60, gt=0)
    # The lower-case header that carries the API key, the prefix of its value, and the provider named in errors.
    auth_header: str = "authorization"
    auth_prefix: str = "Bearer "
    provider_name: str = "Tavily"
    # Statuses that mean this key cannot serve the call (rejected or out of credits), so the next key is tried:
    # Tavily's 401 and its 432/433 plan and pay-as-you-go limits by default.
    rejected_key_statuses: frozenset[int] = frozenset({401, 432, 433})

    async def post(self, endpoint: str, content: str, timeout: float) -> TavilySearchAIOHTTPClientResponse:
        headers = {key.lower(): value for key, value in self.headers.items()}
        authorizations = list(
            dict.fromkeys(
                [headers.get(self.auth_header, ""), *(self.auth_prefix + key for key in self.retry_api_keys)]
            )
        )
        rejected = set()
        key_index = 0
        for attempt in range(self.max_attempts):
            # Skip keys rejected during this call; transient failures still rotate the pool.
            while authorizations[key_index] in rejected:
                key_index = (key_index + 1) % len(authorizations)
            headers[self.auth_header] = authorizations[key_index]
            response = None
            try:
                response = await request(
                    "POST",
                    headers=headers.copy(),
                    url=f"{self.base_url}{endpoint}",
                    data=content,
                    timeout=ClientTimeout(total=min(timeout, self.timeout_s)),
                    _max_connection_retries=0,
                )
                if response.status == 200:
                    try:
                        data = await response.json()
                    except (ContentTypeError, ValueError):
                        data = None
                    if not isinstance(data, dict):
                        # The decode error's repr carries the request headers, key included, and a validation
                        # error would quote the body; name only the provider.
                        raise RuntimeError(f"{self.provider_name} HTTP 200 with an unreadable body")
                    return TavilySearchAIOHTTPClientResponse(status_code=200, data=data)
                key_rejected = response.status in self.rejected_key_statuses
                if key_rejected:
                    rejected.add(headers[self.auth_header])
                retryable = response.status in RETRY_ERROR_CODES or (
                    key_rejected and len(rejected) < len(authorizations)
                )
                if not retryable or attempt + 1 == self.max_attempts:
                    # Provider error bodies may contain credentials; never forward them to the agent.
                    raise RuntimeError(f"{self.provider_name} HTTP {response.status} after {attempt + 1} attempts")
            except (ClientConnectionError, ClientPayloadError, TimeoutError):
                if attempt + 1 == self.max_attempts:
                    raise RuntimeError(
                        f"{self.provider_name} transport failure after {attempt + 1} attempts"
                    ) from None
            finally:
                if response is not None:
                    response.release()
            key_index = (key_index + 1) % len(authorizations)
            await sleep(min(2**attempt, 8))
        raise RuntimeError(f"{self.provider_name} retry budget exhausted")

    @classmethod
    def from_httpx_AsyncClient(cls, client: AsyncClient, debug: bool, **kwargs) -> "TavilySearchAIOHTTPClient":
        return cls(headers=client.headers, base_url=str(client.base_url), debug=debug, **kwargs)


class ExaSearchClient(BaseModel):
    """Exa's /search and /contents, answered in the result shape of the Tavily SDK's ``search`` and ``extract``.

    The tools read only ``url``, ``title`` and ``content`` (search) or ``raw_content`` (extract) of each result, so
    the exclusion policy, formatting and page cache stay the same for both providers.
    """

    http: TavilySearchAIOHTTPClient

    async def search(self, query: str, *, max_results: int, exclude_domains: Optional[List[str]]) -> Dict[str, Any]:
        # Exa's highlights are the passages of each page that best match the query, as Tavily's content is.
        body: Dict[str, Any] = {"query": query, "numResults": max_results, "contents": {"highlights": True}}
        if exclude_domains:
            body["excludeDomains"] = exclude_domains
        payload = await self._post("/search", body)
        return {
            "results": [
                {
                    "url": result.get("url"),
                    "title": result.get("title") or "",
                    "content": " ... ".join(_strings(result.get("highlights"))),
                }
                for result in payload.get("results") or []
                if isinstance(result, dict)
            ]
        }

    async def extract(self, url: str, *, query: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"urls": [url], "text": True}
        if query is not None:
            # Like Tavily's query-ranked extract: the passages that match the query, else the page text.
            body["highlights"] = {"query": query}
        payload = await self._post("/contents", body)
        return {
            "results": [
                {
                    "url": result.get("url"),
                    "raw_content": "\n\n".join(_strings(result.get("highlights"))) or result.get("text") or "",
                }
                for result in payload.get("results") or []
                if isinstance(result, dict)
            ]
        }

    async def _post(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.http.post(endpoint, json.dumps(body), timeout=self.http.timeout_s)
        return response.json()


def _strings(values: Any) -> List[str]:
    return [value for value in values or [] if isinstance(value, str) and value]


def _api_keys(configured: Optional[str | List[str]]) -> List[str]:
    """A key pool from one key, a comma-separated list, or a list of either."""
    groups = [configured] if isinstance(configured, str) else configured or []
    return [key.strip() for group in groups for key in group.split(",") if key.strip()]


def normalize_hostname(value: str) -> str:
    """IDNA-encode, lower-case and strip the trailing dot: the one form every hostname comparison here uses."""
    return value.encode("idna").decode().lower().rstrip(".")


class URLExclusionPolicy:
    """Domain and URL-pattern checks shared by search results and page extraction."""

    def __init__(self, path: Path):
        properties = [p for n in json.loads(path.read_text())["notices"] for p in n["properties"]]
        self.domains = []
        for prop in properties:
            if prop["type"] == "domain":
                try:
                    self.domains.append(normalize_hostname(prop["value"]))
                except UnicodeError as exc:
                    raise ValueError(f"Invalid exclusion domain in {path}: {prop['value']!r}") from exc
        self.substrings = [p["value"].lower() for p in properties if p["type"] == "url_substring"]
        unknown = {p["type"] for p in properties} - {"domain", "url_substring", "author_name", "publisher_name"}
        if unknown:
            raise ValueError(f"Unsupported exclusion types: {sorted(unknown)}")

    def blocked(self, url: str) -> bool:
        for _ in range(4):
            try:
                parts = urlsplit(url)
                if (
                    parts.scheme.lower() not in {"https", "http"}
                    or not parts.hostname
                    or parts.username
                    or parts.password
                ):
                    return True
                host = normalize_hostname(parts.hostname)
                canonical = urlunsplit((parts.scheme.lower(), host, parts.path, parts.query, "")).lower()
                if any(host == domain or host.endswith("." + domain) for domain in self.domains):
                    return True
                if any(pattern in canonical or pattern in url.lower() for pattern in self.substrings):
                    return True
                decoded = unquote(url)
                if decoded == url:
                    return False
                url = decoded
            except (ValueError, UnicodeError):
                return True
        return True


def select_provider_domains(
    domains: Sequence[str], limit: Optional[int], priority: Sequence[str]
) -> Optional[List[str]]:
    """Deterministic subset of a policy's domains for a provider that caps ``exclude_domains``.

    Returns ``None`` when ``limit`` is ``None`` or 0 (no provider-side exclusion). Otherwise the domains matching
    the ``priority`` patterns come first, in pattern order and then policy order, followed by the remaining
    domains in policy order, truncated to ``limit``. Patterns are normalised like the policy's hostnames before
    matching (an invalid pattern raises ``ValueError``). A policy within ``limit`` is sent as the same set of
    domains in this order with duplicates removed: the set is preserved, the file order is not. Every domain
    stays enforced locally regardless.
    """
    if not limit:
        return None
    ordered: List[str] = []
    for pattern in priority:
        try:
            normalized_pattern = normalize_hostname(pattern)
        except UnicodeError as exc:
            raise ValueError(f"Invalid tavily_exclude_domains_priority pattern: {pattern!r}") from exc
        ordered.extend(
            domain for domain in domains if fnmatchcase(domain, normalized_pattern) and domain not in ordered
        )
    ordered.extend(domain for domain in domains if domain not in ordered)
    return ordered[:limit]


class TavilySearchResourcesServer(SimpleResourcesServer):
    config: TavilySearchResourcesServerConfig

    _async_tavily_clients: Optional[List[AsyncTavilyClient]] = PrivateAttr(default=None)
    _exa_clients: Optional[List[ExaSearchClient]] = PrivateAttr(default=None)
    _num_requests: int = 0
    _session_id_to_metrics: Optional[Dict[str, TavilySearchMetrics]] = PrivateAttr(default=None)

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = JUDGE_PROMPT_TEMPLATE

    def model_post_init(self, __context) -> None:
        if self.config.search_provider == "exa":
            exa_api_keys = _api_keys(self.config.exa_api_key)
            if not exa_api_keys:
                raise ValueError("At least one Exa API key is required when search_provider is exa")
            # One client per key, each falling back to the others when Exa rejects its key, as for Tavily.
            self._exa_clients = [
                ExaSearchClient(
                    http=TavilySearchAIOHTTPClient(
                        headers={"content-type": "application/json", "x-api-key": key},
                        base_url=self.config.exa_base_url.rstrip("/"),
                        debug=self.config.debug,
                        retry_api_keys=exa_api_keys[index + 1 :] + exa_api_keys[: index + 1],
                        max_attempts=self.config.max_http_attempts,
                        timeout_s=self.config.http_timeout_s,
                        auth_header="x-api-key",
                        auth_prefix="",
                        provider_name="Exa",
                        # 402: the key's credits or spending budget are exhausted.
                        rejected_key_statuses=frozenset({401, 402}),
                    )
                )
                for index, key in enumerate(exa_api_keys)
            ]
        else:
            tavily_api_keys = _api_keys(self.config.tavily_api_key)
            if not tavily_api_keys:
                raise ValueError("At least one Tavily API key is required")

            self._async_tavily_clients = [AsyncTavilyClient(api_key=k) for k in tavily_api_keys]
            for index, async_tavily_client in enumerate(self._async_tavily_clients):
                async_tavily_client._client = TavilySearchAIOHTTPClient.from_httpx_AsyncClient(
                    async_tavily_client._client,
                    self.config.debug,
                    retry_api_keys=tavily_api_keys[index + 1 :] + tavily_api_keys[: index + 1],
                    max_attempts=self.config.max_http_attempts,
                    timeout_s=self.config.http_timeout_s,
                )

        self._session_id_to_metrics = defaultdict(TavilySearchMetrics)

        policy_path = _resolve_under_cwd_or_install(self.config.exclude_domains_file_path)
        self._url_policy = URLExclusionPolicy(policy_path)
        self._provider_exclude_domains = select_provider_domains(
            self._url_policy.domains,
            self.config.tavily_exclude_domains_limit,
            self.config.tavily_exclude_domains_priority,
        )
        if self._provider_exclude_domains is not None:
            sent = set(self._provider_exclude_domains)
            not_sent = [domain for domain in dict.fromkeys(self._url_policy.domains) if domain not in sent]
            if not_sent:
                # Once per server start: the subset is fixed by the policy file and the config.
                logger.warning(
                    "Exclusion policy %s has %d domain rules but tavily_exclude_domains_limit is %d: %d domains are "
                    "not sent to %s /search (%s, ...) and are enforced only locally; every domain and "
                    "url_substring rule is still applied to requested and returned URLs.",
                    policy_path,
                    len(sent) + len(not_sent),
                    self.config.tavily_exclude_domains_limit,
                    len(not_sent),
                    "Exa" if self.config.search_provider == "exa" else "Tavily",
                    ", ".join(not_sent[:5]),
                )
        self._page_cache: OrderedDict[str, str] = OrderedDict()
        if self.config.debug:
            print("Debug mode enabled")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/web_search")(self.web_search)
        app.post("/find_in_page")(self.find_in_page)
        app.post("/scroll_page")(self.scroll_page)

        main_app_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan_wrapper(app):
            async with main_app_lifespan(app) as maybe_state:
                yield maybe_state

            if self.config.dump_session_id_to_metrics_on_exit:
                out_file = Path(__file__).parent / "session_id_metrics.json"
                print(f"Dumping session_id metrics to {out_file}")

                to_dump = {k: v.model_dump(mode="json") for k, v in self._session_id_to_metrics.items()}
                with out_file.open("w") as f:
                    json.dump(to_dump, f)

        app.router.lifespan_context = lifespan_wrapper

        return app

    def mcp_allowed_tools_for_session(self, seed_body: dict[str, Any]) -> list[str]:
        return ["web_search", "find_in_page", "scroll_page"]

    def _select_tavily_client(self) -> AsyncTavilyClient:
        client = self._async_tavily_clients[self._num_requests % len(self._async_tavily_clients)]
        self._num_requests += 1
        return client

    def _select_exa_client(self) -> ExaSearchClient:
        client = self._exa_clients[self._num_requests % len(self._exa_clients)]
        self._num_requests += 1
        return client

    async def _search(self, query: str) -> Dict[str, Any]:
        # At most tavily_exclude_domains_limit domains go to the provider (Tavily rejects longer lists); the full
        # domain and url_substring policy is applied locally to every returned URL by _postprocess_search_results.
        if self.config.search_provider == "exa":
            return await self._select_exa_client().search(
                query, max_results=self.config.max_results, exclude_domains=self._provider_exclude_domains
            )
        return await self._select_tavily_client().search(
            query,
            max_results=self.config.max_results,
            exclude_domains=self._provider_exclude_domains,
            search_depth=self.config.search_depth,
            # Its LLM-generated answer could summarize a page we later discard, so return
            # source results only, including for domain-only policies for consistent behavior.
            include_answer=False,
            include_raw_content=False,
        )

    async def _extract(self, url: str, query: Optional[str] = None) -> Dict[str, Any]:
        if self.config.search_provider == "exa":
            return await self._select_exa_client().extract(url, query=query)
        if query is None:
            return await self._select_tavily_client().extract(urls=url)
        return await self._select_tavily_client().extract(urls=url, query=query)

    async def web_search(self, request: Request, body: TavilySearchRequest) -> TavilySearchResponse:
        metrics = self._session_id_to_metrics[request.session[SESSION_ID_KEY]]

        if self.config.debug:
            print("\n\n body.query: ", body.query)
        if body.query is None:
            return TavilySearchResponse(results_string="Query is none")

        if len(body.query) > 400:
            return TavilySearchResponse(results_string="Query is too long")

        start_time = time()
        results = await self._search(body.query)
        metrics.async_tavily_calls.append(
            TavilySearchSingleAsyncTavilyMetrics(
                function="search", status="success", start_time=start_time, end_time=time()
            )
        )

        postprocessed_results = self._postprocess_search_results(results)
        return TavilySearchResponse(results_string="".join(postprocessed_results))

    async def find_in_page(self, request: Request, body: FindInPageRequest) -> FindInPageResponse:
        metrics = self._session_id_to_metrics[request.session[SESSION_ID_KEY]]

        if self.config.debug:
            print("\n\n find_in_page ")
            print(f"url={body.url}, query={body.query}")

        if body.url is None:
            return FindInPageResponse(results_string="URL is none")
        if body.query is None:
            return FindInPageResponse(results_string="Query is none")

        if self._is_url_excluded(body.url):
            return FindInPageResponse(results_string="URL is in excluded domains")

        start_time = time()
        results = await self._extract(body.url, query=body.query)
        metrics.async_tavily_calls.append(
            TavilySearchSingleAsyncTavilyMetrics(
                function="extract", status="success", start_time=start_time, end_time=time()
            )
        )

        # Recheck provider-returned URLs, including any reported redirect destination.
        allowed_results = self._allowed_results(results)
        if allowed_results:
            raw_content = allowed_results[0].get("raw_content", "")
        else:
            raw_content = ""

        if not raw_content:
            return FindInPageResponse(results_string="No content found.")

        # Format: header + clean + truncate + line numbers
        domain = self._extract_domain(body.url)
        cleaned = self._clean_text(raw_content)
        truncated, was_truncated = self._truncate_text(cleaned)
        numbered = self._add_line_numbers(truncated)

        header = (
            f"Content from: {domain}\n"
            f"URL: {body.url}\n"
            f'Query: "{body.query}"\n'
            f"========================================\n"
        )
        footer = ""
        if was_truncated:
            footer = "\n[...truncated, use scroll_page for full content]"

        return FindInPageResponse(results_string=header + numbered + footer)

    async def scroll_page(self, request: Request, body: ScrollPageRequest) -> ScrollPageResponse:
        metrics = self._session_id_to_metrics[request.session[SESSION_ID_KEY]]

        if self.config.debug:
            print("\n\n scroll_page ")
            print(f"url={body.url}, start_index={body.start_index}, n={body.n}")

        if body.url is None:
            return ScrollPageResponse(results_string="URL is none", total_words=0)

        if self._is_url_excluded(body.url):
            return ScrollPageResponse(results_string="URL is in excluded domains", total_words=0)

        # Check cache first
        if body.url in self._page_cache:
            if self.config.debug:
                print(f"Cache hit for {body.url}")
            page_content = self._page_cache[body.url]
            self._page_cache.move_to_end(body.url)
        else:
            if self.config.debug:
                print(f"Cache miss for {body.url}, fetching with {self.config.search_provider} extract")

            start_time = time()
            results = await self._extract(body.url)
            metrics.async_tavily_calls.append(
                TavilySearchSingleAsyncTavilyMetrics(
                    function="extract", status="success", start_time=start_time, end_time=time()
                )
            )

            allowed_results = self._allowed_results(results)
            if allowed_results:
                page_content = allowed_results[0].get("raw_content", "")
            else:
                page_content = ""

            # An empty page is usually a failed fetch (Tavily's failed_results, an Exa crawl error in statuses);
            # caching it would hide the page from every later session.
            if page_content and self.config.max_cached_pages:
                if self.config.max_cached_page_chars is not None:
                    page_content = page_content[: self.config.max_cached_page_chars]
                self._page_cache[body.url] = page_content
                self._page_cache.move_to_end(body.url)
                while len(self._page_cache) > self.config.max_cached_pages:
                    self._page_cache.popitem(last=False)

        words = page_content.split()
        total_words = len(words)
        start_index = max(0, body.start_index)
        n = max(0, body.n)
        if self.config.max_scroll_words is not None:
            n = min(n, self.config.max_scroll_words)
        sliced_words = words[start_index : start_index + n]
        chunk_text = " ".join(sliced_words)

        # Format: header + clean + line numbers
        domain = self._extract_domain(body.url)
        cleaned = self._clean_text(chunk_text)
        numbered = self._add_line_numbers(cleaned)

        end_index = min(start_index + n, total_words)
        header = (
            f"Page content from: {domain}\n"
            f"URL: {body.url}\n"
            f"Showing words [{start_index}-{end_index}] of {total_words}\n"
            f"========================================\n"
        )

        return ScrollPageResponse(
            results_string=header + numbered,
            total_words=total_words,
        )

    async def verify(self, request: Request, body: TavilySearchVerifyRequest) -> TavilySearchVerifyResponse:
        question = body.question
        ground_truth = body.ground_truth
        last_assistant_response = body.response.output_text

        judge_error = None
        if self.config.use_judge:
            judge_evaluation, judge_error = await self._verify_answer_with_judge(
                question, ground_truth, last_assistant_response
            )
        else:
            judge_evaluation = self._verify_answer_with_regex(ground_truth, last_assistant_response)
        response = TavilySearchVerifyResponse(
            **body.model_dump(),
            **judge_evaluation.model_dump(),
            num_tool_calls=sum(o.type == "function_call" for o in body.response.output),
            metrics=self._session_id_to_metrics[request.session[SESSION_ID_KEY]],
        )
        if judge_error is not None:
            raise judge_error
        return response

    ###### UTILITY FUNCTIONS ######

    def _is_url_excluded(self, url: str) -> bool:
        return self._url_policy.blocked(url)

    def _allowed_results(self, response: dict) -> list[dict]:
        return [
            result
            for result in response.get("results", [])
            if isinstance(result.get("url"), str) and not self._is_url_excluded(result["url"])
        ]

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        return urlparse(url).hostname or url

    def _clean_text(self, text: str) -> str:
        """Remove wiki/web navigation artifacts and normalize whitespace."""
        # Strip [edit] markers
        text = re.sub(r"\[edit\]", "", text)
        # Strip wiki navigation chrome lines: [Jump to content], [Search...], [Read], [View history], etc.
        text = re.sub(r"^\[(?:Jump to content|Search|Read|Edit|View history)[^\]]*\].*$", "", text, flags=re.MULTILINE)
        # Strip wiki language sidebar links: [LangName](https://xx.wikipedia.org/...)
        text = re.sub(r"\[[^\]]+\]\(https?://[a-z]{2,3}\.wikipedia\.org/[^\)]*\)", "", text)
        # Strip table-of-contents anchor links: * [(Top)](#) etc.
        text = re.sub(r"^\s*\*\s*\[[^\]]*\]\(#[^\)]*\)\s*$", "", text, flags=re.MULTILINE)
        # Strip zero-width spaces and special unicode
        text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
        text = text.replace("\u3010", "[").replace("\u3011", "]")
        # Strip trailing whitespace per line
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        # Collapse 3+ consecutive newlines to 2 (one blank line)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _add_line_numbers(self, text: str) -> str:
        """Add L0:, L1:, ... prefix per line."""
        lines = text.split("\n")
        return "\n".join(f"L{i}: {line}" for i, line in enumerate(lines))

    def _truncate_text(self, text: str, max_chars: int = None) -> tuple:
        """Truncate text to max_chars, snapping to last full line boundary.
        Returns (truncated_text, was_truncated).
        """
        if max_chars is None:
            max_chars = self.config.max_result_chars
        if len(text) <= max_chars:
            return text, False
        # Find the last newline within max_chars
        cut = text.rfind("\n", 0, max_chars)
        if cut == -1:
            cut = max_chars
        return text[:cut], True

    def _postprocess_search_results(self, results: dict) -> list[str]:
        # Ignore any aggregate answer even if returned despite include_answer=False:
        # filtering source URLs cannot remove blocked content from a generated summary.
        formatted_results = ["Search Results\n==============\n"]
        for i, result in enumerate(self._allowed_results(results)[: self.config.max_results], 1):
            domain = self._extract_domain(result["url"])
            snippet = self._clean_text(result.get("content", ""))
            snippet, _ = self._truncate_text(snippet)
            formatted_results.append(
                f"[{i}] {result['title']} ({domain})\n    URL: {result['url']}\n    Summary: {snippet}\n\n"
            )
        return formatted_results

    async def _verify_answer_with_judge(
        self, question: str, ground_truth: str, response: str
    ) -> tuple[JudgeEvaluation, Optional[Exception]]:
        async def _get_judge_response(
            question: str, ground_truth: str, response: str
        ) -> tuple[NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponse]:
            judge_create_params = self.config.judge_responses_create_params.model_copy(deep=True)
            judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
                question=question, correct_answer=ground_truth, response=response
            )
            judge_create_params.input = [
                NeMoGymEasyInputMessage(
                    role="user",
                    content=judge_prompt,
                ),
            ]
            judge_response = await call_judge(
                self.server_client,
                server_name=self.config.judge_model_server.name,
                url_path="/v1/responses",
                json=judge_create_params,
                response_model=NeMoGymResponse,
            )
            return judge_create_params, judge_response

        def _grade_sample(
            judge_create_params: NeMoGymResponseCreateParamsNonStreaming, judge_response: NeMoGymResponse
        ) -> JudgeEvaluation:
            # Taken from: https://github.com/openai/simple-evals/blob/5e623c2b400af62a1278e23595f95b0853d7fe8a/browsecomp_eval.py#L79-L93
            grading_response = judge_response.output[-1].content[-1].text
            if self.config.debug:
                print("\n\n grading_response \n\n")
                print(grading_response)
            match = re.search(r"correct: (yes|no)", grading_response)
            extracted_final_answer = match.group(1) if match else ""
            reward = 1.0 if extracted_final_answer == "yes" else 0.0
            return JudgeEvaluation(
                judge_response_create_params=judge_create_params,
                reasoning=grading_response,
                extracted_final_answer=extracted_final_answer,
                reward=reward,
                judge_response=judge_response,
            )

        try:
            judge_create_params, judge_response = await _get_judge_response(question, ground_truth, response)
        except JUDGE_FAILURES as e:
            if not is_judge_failure(e):
                raise
            return JudgeEvaluation(reasoning="", extracted_final_answer="", reward=0.0), e
        judge_evaluation = _grade_sample(judge_create_params, judge_response)
        return judge_evaluation, None

    def _verify_answer_with_regex(self, ground_truth: str, response: str) -> JudgeEvaluation:
        """Verify answer by checking if ground_truth (as regex) matches in response."""
        matches = re.findall(r"Answer:\s*(.*)\s*Confidence:", response, re.IGNORECASE)

        if matches:
            answer = matches[-1].strip()  # Get the last item in the list
        else:
            answer = ""
        if self.config.debug:
            print(answer)
        reward = 1.0 if answer == ground_truth else 0.0
        return JudgeEvaluation(
            judge_response_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            reasoning=f"Regex match for '{ground_truth}': {'found' if answer == ground_truth else 'not found'}",
            extracted_final_answer=answer,
            reward=reward,
            judge_response=None,
        )


if __name__ == "__main__":
    TavilySearchResourcesServer.run_webserver()
