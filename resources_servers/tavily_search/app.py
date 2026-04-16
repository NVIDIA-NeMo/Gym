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
import re
from asyncio import sleep
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from time import time
from typing import Any, ClassVar, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from httpx import AsyncClient
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from tavily import AsyncTavilyClient

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    RATE_LIMIT_ERROR_CODES,
    RETRY_ERROR_CODES,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import SESSION_ID_KEY, raise_for_status, request
from resources_servers.tavily_search.judge_prompt import JUDGE_PROMPT_TEMPLATE


class TavilySearchResourcesServerConfig(BaseResourcesServerConfig):
    tavily_api_key: str | List[str]
    exclude_domains_file_path: str
    use_judge: bool = True  # If False, use regex matching instead of LLM judge
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    debug: bool = False
    dump_session_id_to_metrics_on_exit: bool = False


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
    headers: Dict[str, str]
    base_url: str

    debug: bool

    async def post(self, endpoint: str, content: str, timeout: float) -> TavilySearchAIOHTTPClientResponse:
        """
        endpoint: str e.g. "/search" or "/extract"
        timeout: float is not used
        """
        request_kwargs = {
            "method": "POST",
            "headers": self.headers,
            "url": f"{self.base_url}{endpoint}",
            "data": content,
        }

        MAX_NUM_TRIES = 3  # Hardcode for now
        max_num_tries = MAX_NUM_TRIES
        tries = 0
        while tries < max_num_tries:
            tries += 1
            response = await request(**request_kwargs)

            if response.status in RETRY_ERROR_CODES:
                # If we hit a rate limit, we don't want to hit max num tries, so we increment both.
                if response.status in RATE_LIMIT_ERROR_CODES:
                    max_num_tries += 1

                content = (await response.content.read()).decode()
                print(
                    f"Hit a {response.status} trying to query an Tavily endpoint (try {tries}). Sleeping 0.5s. Error message: {content}"
                )
                await sleep(0.5)
                continue
            else:
                tavily_response = TavilySearchAIOHTTPClientResponse(
                    status_code=response.status,
                    data=await response.json(),
                )
                if self.debug:
                    print(f"Received the following Tavily response: {tavily_response}")

                return tavily_response

        # We've exited the loop
        await raise_for_status(response)

    @classmethod
    def from_httpx_AsyncClient(cls, client: AsyncClient, debug: bool) -> "TavilySearchAIOHTTPClient":
        return cls(
            headers=client.headers,
            base_url=str(client.base_url),
            debug=debug,
        )


class TavilySearchResourcesServer(SimpleResourcesServer):
    config: TavilySearchResourcesServerConfig
    MAX_RESULTS: int = 10
    MAX_RESULT_CHARS: int = 2000

    _async_tavily_clients: Optional[List[AsyncTavilyClient]] = PrivateAttr(default=None)
    _num_requests: int = 0
    _session_id_to_metrics: Optional[Dict[str, TavilySearchMetrics]] = PrivateAttr(default=None)

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = JUDGE_PROMPT_TEMPLATE

    def model_post_init(self, __context) -> None:
        tavily_api_keys = self.config.tavily_api_key
        if isinstance(tavily_api_keys, str):
            tavily_api_keys = [tavily_api_keys]

        self._async_tavily_clients = [AsyncTavilyClient(api_key=k) for k in tavily_api_keys]
        for async_tavily_client in self._async_tavily_clients:
            async_tavily_client._client = TavilySearchAIOHTTPClient.from_httpx_AsyncClient(
                async_tavily_client._client, self.config.debug
            )

        self._session_id_to_metrics = defaultdict(TavilySearchMetrics)

        self._exclude_domains = self._parse_exclude_domains()
        self._page_cache: dict[str, str] = {}
        print(f"Excluded domains: {self._exclude_domains}")
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

    def _select_tavily_client(self) -> AsyncTavilyClient:
        client = self._async_tavily_clients[self._num_requests % len(self._async_tavily_clients)]
        self._num_requests += 1
        return client

    async def web_search(self, request: Request, body: TavilySearchRequest) -> TavilySearchResponse:
        metrics = self._session_id_to_metrics[request.session[SESSION_ID_KEY]]

        if self.config.debug:
            print("\n\n body.query: ", body.query)
        if body.query is None:
            return TavilySearchResponse(results_string="Query is none")

        if len(body.query) > 400:
            return TavilySearchResponse(results_string="Query is too long")

        async_tavily_client = self._select_tavily_client()
        start_time = time()
        results = await async_tavily_client.search(
            body.query,
            max_results=self.MAX_RESULTS,
            exclude_domains=self._exclude_domains,
            search_depth="advanced",
        )
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

        async_tavily_client = self._select_tavily_client()
        start_time = time()
        results = await async_tavily_client.extract(
            urls=body.url,
            query=body.query,
        )
        metrics.async_tavily_calls.append(
            TavilySearchSingleAsyncTavilyMetrics(
                function="extract", status="success", start_time=start_time, end_time=time()
            )
        )

        # Extract raw_content from the first successful result
        if results.get("results"):
            raw_content = results["results"][0].get("raw_content", "")
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
        else:
            if self.config.debug:
                print(f"Cache miss for {body.url}, fetching with tavily extract")

            async_tavily_client = self._select_tavily_client()
            start_time = time()
            results = await async_tavily_client.extract(
                urls=body.url,
            )
            metrics.async_tavily_calls.append(
                TavilySearchSingleAsyncTavilyMetrics(
                    function="extract", status="success", start_time=start_time, end_time=time()
                )
            )

            if results.get("results"):
                page_content = results["results"][0].get("raw_content", "")
            else:
                page_content = ""

            # Store in cache
            self._page_cache[body.url] = page_content

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

    async def verify(self, request: Request, body: TavilySearchVerifyRequest) -> TavilySearchVerifyResponse:
        question = body.question
        ground_truth = body.ground_truth
        last_assistant_response = body.response.output_text

        if self.config.use_judge:
            judge_evaluation = await self._verify_answer_with_judge(question, ground_truth, last_assistant_response)
        else:
            judge_evaluation = self._verify_answer_with_regex(ground_truth, last_assistant_response)
        return TavilySearchVerifyResponse(
            **body.model_dump(),
            **judge_evaluation.model_dump(),
            num_tool_calls=sum(o.type == "function_call" for o in body.response.output),
            metrics=self._session_id_to_metrics[request.session[SESSION_ID_KEY]],
        )

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
            max_chars = self.MAX_RESULT_CHARS
        if len(text) <= max_chars:
            return text, False
        # Find the last newline within max_chars
        cut = text.rfind("\n", 0, max_chars)
        if cut == -1:
            cut = max_chars
        return text[:cut], True

    def _postprocess_search_results(self, results: dict) -> list[str]:
        # If an answer is present, return ONLY the answer (no individual search results)
        answer = results.get("answer")
        if answer is not None:
            return [f"Search Answer\n==============\n{answer}\n"]

        formatted_results = ["Search Results\n==============\n"]
        for i, result in enumerate(results["results"], 1):
            domain = self._extract_domain(result["url"])
            snippet = self._clean_text(result.get("content", ""))
            snippet, _ = self._truncate_text(snippet)
            formatted_results.append(
                f"[{i}] {result['title']} ({domain})\n    URL: {result['url']}\n    Summary: {snippet}\n\n"
            )
        return formatted_results

    def _parse_exclude_domains(self) -> list[str]:
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

    async def _verify_answer_with_judge(self, question: str, ground_truth: str, response: str) -> JudgeEvaluation:
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
            http_response = await self.server_client.post(
                server_name=self.config.judge_model_server.name,
                url_path="/v1/responses",
                json=judge_create_params,
            )
            judge_response = NeMoGymResponse.model_validate(await http_response.json())
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

        judge_create_params, judge_response = await _get_judge_response(question, ground_truth, response)
        judge_evaluation = _grade_sample(judge_create_params, judge_response)
        return judge_evaluation

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