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
"""RL environment which allows access to web search (Search Provider: You.com).

Mirrors ``resources_servers/tavily_search`` tool-for-tool (``web_search`` /
``find_in_page`` / ``scroll_page``) so the two can be swapped under an otherwise
identical harness and the resulting numbers compared directly.

You.com exposes two REST endpoints and no SDK is needed:

  - ``POST /v1/search``   — ranked web (and optionally news) results. The
    ``extraction`` object selects how much of each page comes back, which is the
    single knob that separates the three ``search_mode`` arms below.
  - ``POST /v1/contents`` — page content for explicit URLs, backing
    ``find_in_page`` and ``scroll_page``.

Because there is no SDK there is no httpx transport to adapt (contrast
``TavilySearchAIOHTTPClient``); every call goes through NeMo Gym's global
aiohttp client directly.
"""

import json
import re
from asyncio import sleep
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from time import time
from typing import Any, ClassVar, Dict, List, Literal, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import (
    RATE_LIMIT_ERROR_CODES,
    RETRY_ERROR_CODES,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import SESSION_ID_KEY, raise_for_status, request
from resources_servers.you_search.judge_prompt import JUDGE_PROMPT_TEMPLATE


# You.com caps include/exclude/boost domain lists at 500 entries per request. Larger
# opt-out registries are truncated for the wire call and enforced client-side instead.
MAX_EXCLUDE_DOMAINS = 500

SearchMode = Literal["snippets", "highlights", "full_page"]


class YouSearchResourcesServerConfig(BaseResourcesServerConfig):
    you_api_key: str | List[str]
    # How much of each page the search call returns:
    #   snippets   — keyword excerpts + description only (cheapest, fewest tokens)
    #   highlights — query-relevant passages extracted per page
    #   full_page  — the whole page as markdown (most tokens, billed per page crawled)
    search_mode: SearchMode = "snippets"
    base_url: str = "https://ydc-index.io"
    num_results: int = 10
    include_news: bool = False
    # full_page only: seconds You.com may spend crawling a page before giving up.
    crawl_timeout: int = 10
    # Per-result character cap. 2000 matches the Tavily server; raise it when running
    # full_page, where the default truncates away most of what you paid to crawl.
    max_result_chars: int = 2000
    # Optional. Path to a domain opt-out registry; see _parse_exclude_domains.
    exclude_domains_file_path: Optional[str] = None
    use_judge: bool = True  # If False, use regex matching instead of LLM judge
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    debug: bool = False
    dump_session_id_to_metrics_on_exit: bool = False


class YouSearchRequest(BaseModel):
    query: Optional[str] = None  # Make optional to handle missing args gracefully


class YouSearchResponse(BaseModel):
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


class YouSearchRunRequest(BaseRunRequest):
    ground_truth: str
    question: str


class YouSearchVerifyRequest(YouSearchRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    judge_response_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    reasoning: str
    extracted_final_answer: str
    reward: float
    judge_response: Optional[NeMoGymResponse] = None


class YouSearchSingleAPICallMetrics(BaseModel):
    function: str
    status: str
    start_time: float
    end_time: float
    time_taken: Optional[float] = None

    @model_validator(mode="after")
    def compute_time_taken(self):
        self.time_taken = self.end_time - self.start_time
        return self


class YouSearchMetrics(BaseModel):
    you_api_calls: List[YouSearchSingleAPICallMetrics] = Field(default_factory=list)


class YouSearchVerifyResponse(YouSearchVerifyRequest, JudgeEvaluation):
    num_tool_calls: int
    metrics: YouSearchMetrics


class YouSearchResourcesServer(SimpleResourcesServer):
    config: YouSearchResourcesServerConfig

    _num_requests: int = 0
    _session_id_to_metrics: Optional[Dict[str, YouSearchMetrics]] = PrivateAttr(default=None)

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = JUDGE_PROMPT_TEMPLATE

    def model_post_init(self, __context) -> None:
        you_api_keys = self.config.you_api_key
        if isinstance(you_api_keys, str):
            you_api_keys = [you_api_keys]
        self._api_keys = you_api_keys

        self._session_id_to_metrics = defaultdict(YouSearchMetrics)

        self._exclude_domains = self._parse_exclude_domains()
        self._page_cache: dict[str, str] = {}
        print(f"Excluded domains: {self._exclude_domains}")
        print(f"Search mode: {self.config.search_mode}")
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

    ###### YOU.COM TRANSPORT ######

    def _select_api_key(self) -> str:
        key = self._api_keys[self._num_requests % len(self._api_keys)]
        self._num_requests += 1
        return key

    async def _post(self, endpoint: str, payload: Dict[str, Any]) -> Any:
        """POST to a You.com endpoint, retrying on transient status codes.

        Retries do not count against the try budget when the failure is a rate
        limit, matching the Tavily server's behaviour.
        """
        headers = {"X-API-Key": self._select_api_key()}
        url = f"{self.config.base_url}{endpoint}"

        MAX_NUM_TRIES = 3  # Hardcode for now
        max_num_tries = MAX_NUM_TRIES
        tries = 0
        response = None
        while tries < max_num_tries:
            tries += 1
            response = await request(method="POST", url=url, headers=headers, json=payload)

            if response.status in RETRY_ERROR_CODES:
                if response.status in RATE_LIMIT_ERROR_CODES:
                    max_num_tries += 1

                content = (await response.content.read()).decode()
                print(
                    f"Hit a {response.status} trying to query a You.com endpoint (try {tries}). "
                    f"Sleeping 0.5s. Error message: {content}"
                )
                await sleep(0.5)
                continue

            data = await response.json()
            if self.config.debug:
                print(f"Received the following You.com response: {data}")
            return data

        # We've exited the loop
        await raise_for_status(response)

    def _search_payload(self, query: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query": query, "count": self.config.num_results}

        if self.config.search_mode == "highlights":
            payload["extraction"] = {"extraction_mode": "highlights"}
        elif self.config.search_mode == "full_page":
            payload["extraction"] = {
                "extraction_mode": "full_page",
                "full_page": {"extraction_formats": ["markdown"]},
            }
            payload["crawl_timeout"] = self.config.crawl_timeout

        if self._exclude_domains:
            payload["exclude_domains"] = self._exclude_domains[:MAX_EXCLUDE_DOMAINS]

        return payload

    ###### TOOLS ######

    async def web_search(self, request: Request, body: YouSearchRequest) -> YouSearchResponse:
        metrics = self._session_id_to_metrics[request.session[SESSION_ID_KEY]]

        if self.config.debug:
            print("\n\n body.query: ", body.query)
        if body.query is None:
            return YouSearchResponse(results_string="Query is none")

        if len(body.query) > 400:
            return YouSearchResponse(results_string="Query is too long")

        start_time = time()
        results = await self._post("/v1/search", self._search_payload(body.query))
        metrics.you_api_calls.append(
            YouSearchSingleAPICallMetrics(function="search", status="success", start_time=start_time, end_time=time())
        )

        postprocessed_results = self._postprocess_search_results(results)
        return YouSearchResponse(results_string="".join(postprocessed_results))

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

        raw_content = await self._fetch_page(body.url, metrics)

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

        page_content = await self._fetch_page(body.url, metrics)

        words = page_content.split()
        total_words = len(words)
        sliced_words = words[body.start_index : body.start_index + body.n]
        chunk_text = " ".join(sliced_words)

        # Format: header + clean + line numbers
        domain = self._extract_domain(body.url)
        cleaned = self._clean_text(chunk_text)
        numbered = self._add_line_numbers(cleaned)

        end_index = min(body.start_index + body.n, total_words)
        header = (
            f"Page content from: {domain}\n"
            f"URL: {body.url}\n"
            f"Showing words [{body.start_index}-{end_index}] of {total_words}\n"
            f"========================================\n"
        )

        return ScrollPageResponse(
            results_string=header + numbered,
            total_words=total_words,
        )

    async def _fetch_page(self, url: str, metrics: YouSearchMetrics) -> str:
        """Markdown for a single URL via the Contents API, cached per server process."""
        if url in self._page_cache:
            if self.config.debug:
                print(f"Cache hit for {url}")
            return self._page_cache[url]

        if self.config.debug:
            print(f"Cache miss for {url}, fetching with You.com contents")

        start_time = time()
        results = await self._post(
            "/v1/contents",
            {"urls": [url], "formats": ["markdown"], "crawl_timeout": self.config.crawl_timeout},
        )
        metrics.you_api_calls.append(
            YouSearchSingleAPICallMetrics(
                function="contents", status="success", start_time=start_time, end_time=time()
            )
        )

        # /v1/contents answers with a bare array, one entry per requested URL.
        page_content = ""
        if results:
            page_content = results[0].get("markdown") or ""

        self._page_cache[url] = page_content
        return page_content

    async def verify(self, request: Request, body: YouSearchVerifyRequest) -> YouSearchVerifyResponse:
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
        response = YouSearchVerifyResponse(
            **body.model_dump(),
            **judge_evaluation.model_dump(),
            num_tool_calls=sum(o.type == "function_call" for o in body.response.output),
            metrics=self._session_id_to_metrics[request.session[SESSION_ID_KEY]],
        )
        if judge_error is not None:
            raise JudgeError(judge_error)
        return response

    ###### UTILITY FUNCTIONS ######

    def _is_url_excluded(self, url: str) -> bool:
        """Check if the URL's domain is in the excluded domains list."""
        hostname = urlparse(url).hostname or ""
        return any(hostname == domain or hostname.endswith("." + domain) for domain in self._exclude_domains)

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
        text = text.replace("​", "").replace("‌", "").replace("‍", "").replace("﻿", "")
        text = text.replace("【", "[").replace("】", "]")
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

    def _result_body(self, result: dict) -> str:
        """The per-result text the model sees, by search_mode precedence.

        full_page and highlights both ride in ``contents``; either can come back
        empty for a given result (a crawl that timed out, a page with no
        query-relevant span), in which case we fall back to the snippet and
        description that every result carries.
        """
        contents = result.get("contents") or {}

        markdown = contents.get("markdown")
        if markdown:
            return self._truncate_text(self._clean_text(markdown))[0]

        highlights = contents.get("highlights") or []
        if highlights:
            return self._truncate_text(self._clean_text("\n".join(highlights)))[0]

        snippets = result.get("snippets") or ""
        if isinstance(snippets, list):
            snippets = " ".join(snippets)
        body = "\n".join(self._dedupe_texts([snippets, result.get("description", "")]))
        return self._truncate_text(self._clean_text(body))[0]

    @staticmethod
    def _dedupe_texts(texts: list[str]) -> list[str]:
        """Drop entries already contained in a longer entry, preserving order.

        You.com's ``description`` is very often a truncated copy of ``snippets``
        (or the reverse). Emitting both roughly doubles the per-result token cost
        for no extra information, and tokens-per-result is precisely what this
        environment exists to measure.
        """

        def core(text: str) -> str:
            # Normalize the trailing ellipsis off truncated copies so containment holds.
            return re.sub(r"[\s.…]+$", "", text)

        candidates = [(i, t.strip()) for i, t in enumerate(texts) if t and t.strip()]
        kept: list[tuple[int, str]] = []
        for index, text in sorted(candidates, key=lambda pair: len(pair[1]), reverse=True):
            if any(core(text) in core(other) for _, other in kept):
                continue
            kept.append((index, text))
        return [text for _, text in sorted(kept)]

    def _postprocess_search_results(self, results: dict) -> list[str]:
        sections = results.get("results") or {}
        web_results = sections.get("web") or []
        news_results = sections.get("news") or [] if self.config.include_news else []

        # News first: it is the freshness-sensitive half of the corpus, and burying it
        # under ten web results is what makes models miss it on recency questions.
        all_results = news_results + web_results

        # The API-side exclude_domains list is capped at MAX_EXCLUDE_DOMAINS, so a large
        # registry still needs enforcing here.
        all_results = [r for r in all_results if not self._is_url_excluded(r.get("url", ""))]

        if not all_results:
            return ["Search Results\n==============\nNo results found.\n"]

        formatted_results = ["Search Results\n==============\n"]
        for i, result in enumerate(all_results, 1):
            domain = self._extract_domain(result.get("url", ""))
            formatted_results.append(
                f"[{i}] {result.get('title', '')} ({domain})\n"
                f"    URL: {result.get('url', '')}\n"
                f"    Summary: {self._result_body(result)}\n\n"
            )
        return formatted_results

    def _parse_exclude_domains(self) -> list[str]:
        """Domains we must not return, read from an opt-out registry file.

        Optional: unset means no exclusions, which is the normal case outside of
        deployments carrying their own legal opt-out list.
        """
        if not self.config.exclude_domains_file_path:
            return []

        with open(self.config.exclude_domains_file_path, "r") as f:
            exclude_config = json.load(f)
        exclude_domains = []
        # this is pretty hard-coded so we ensure the file structure is correct
        notices = exclude_config["notices"]
        for notice in notices:
            for prop in notice["properties"]:
                if prop.get("type") == "domain":
                    exclude_domains.append(prop["value"])
        return exclude_domains

    async def _verify_answer_with_judge(
        self, question: str, ground_truth: str, response: str
    ) -> tuple[JudgeEvaluation, Optional[str]]:
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
        except JudgeError as e:
            return JudgeEvaluation(reasoning="", extracted_final_answer="", reward=0.0), str(e)
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
    YouSearchResourcesServer.run_webserver()
