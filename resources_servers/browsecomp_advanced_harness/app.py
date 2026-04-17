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
import asyncio
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
from tavily.errors import BadRequestError

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
from resources_servers.browsecomp_advanced_harness.judge_prompt import JUDGE_PROMPT_TEMPLATE


class TavilySearchResourcesServerConfig(BaseResourcesServerConfig):
    tavily_api_key: str | List[str]
    exclude_domains_file_path: str
    use_judge: bool = True  # If False, use regex matching instead of LLM judge
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    debug: bool = False
    dump_session_id_to_metrics_on_exit: bool = False


class TavilySearchRequest(BaseModel):
    queries: Optional[List[str]] = None  # Make optional to handle missing args gracefully
    max_total_length: int = 30000

    @model_validator(mode="before")
    @classmethod
    def coerce_queries(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        queries = data.get("queries")
        if queries is None:
            return data

        # Case 1: JSON-encoded string → parse it
        if isinstance(queries, str):
            try:
                queries = json.loads(queries)
            except (json.JSONDecodeError, ValueError):
                queries = [queries]

        # Case 2: nested list e.g. [["q1", "q2"]] → flatten one level
        if isinstance(queries, list) and queries and isinstance(queries[0], list):
            queries = [q for sublist in queries for q in sublist if isinstance(q, str)]

        data = dict(data)
        data["queries"] = queries
        return data


class TavilySearchResponse(BaseModel):
    results_string: str


class BrowseRequest(BaseModel):
    urls: List[str]
    goal: Optional[str] = None
    max_total_length: int = 30000

    @model_validator(mode="before")
    @classmethod
    def coerce_urls(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        urls = data.get("querurlsies")
        if urls is None:
            return data

        # Case 1: JSON-encoded string → parse it
        if isinstance(urls, str):
            try:
                urls = json.loads(urls)
            except (json.JSONDecodeError, ValueError):
                urls = [urls]

        # Case 2: nested list e.g. [["q1", "q2"]] → flatten one level
        if isinstance(urls, list) and urls and isinstance(urls[0], list):
            urls = [q for sublist in urls for q in sublist if isinstance(q, str)]

        data = dict(data)
        data["urls"] = urls
        return data


class BrowseResponse(BaseModel):
    results_string: str


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
    MAX_RESULTS: int = 5

    _async_tavily_clients: Optional[List[AsyncTavilyClient]] = PrivateAttr(default=None)
    _num_requests: int = 0
    _session_id_to_metrics: Optional[Dict[str, TavilySearchMetrics]] = PrivateAttr(default=None)

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = JUDGE_PROMPT_TEMPLATE
    JUDGE_MAX_ATTEMPTS: int = 10

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

        app.post("/search")(self.search)
        app.post("/browse")(self.browse)

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

    async def _search_one(self, query: str, max_length: int) -> str:
        if len(query) > 400:
            return "Query is too long"

        client = self._select_tavily_client()
        try:
            results = await client.search(
                query,
                max_results=self.MAX_RESULTS,
                exclude_domains=self._exclude_domains,
                search_depth="advanced",
                include_raw_content=True,
            )
        except BadRequestError as e:
            return f"Search failed: {e}"

        postprocessed_results = self._postprocess_search_results(query, results, max_length)
        return postprocessed_results

    async def search(self, request: Request, body: TavilySearchRequest) -> TavilySearchResponse:
        metrics = self._session_id_to_metrics[request.session[SESSION_ID_KEY]]

        if self.config.debug:
            print("\n\n body.queries: ", body.queries)

        if body.queries is None or len(body.queries) == 0:
            return TavilySearchResponse(results_string="Query is none or empty")

        # set max length per query
        max_per_query_length = body.max_total_length // len(body.queries)

        # search for queries
        start_time = time()
        results = await asyncio.gather(*[self._search_one(q, max_per_query_length) for q in body.queries])
        metrics.async_tavily_calls.append(
            TavilySearchSingleAsyncTavilyMetrics(
                function="search", status="success", start_time=start_time, end_time=time()
            )
        )

        # concat results
        postprocessed_results = "\n\n".join(results)
        return TavilySearchResponse(results_string=postprocessed_results)

    async def browse(self, request: Request, body: BrowseRequest) -> BrowseResponse:
        metrics = self._session_id_to_metrics[request.session[SESSION_ID_KEY]]

        if self.config.debug:
            print("\n\n browse urls: ", body.urls)
            print(f"goal={body.goal}")

        urls = [u for u in body.urls if not self._is_url_excluded(u)]
        if not urls:
            return BrowseResponse(results_string="Error: no URLs provided.")
        urls = urls[:5]

        # set max length per url
        max_per_url_length = body.max_total_length // len(urls)

        # search for urls
        async_tavily_client = self._select_tavily_client()
        start_time = time()
        try:
            results = await async_tavily_client.extract(
                urls=urls,
                query=body.goal,  # optional hint to prioritize relevant content
            )
            metrics.async_tavily_calls.append(
                TavilySearchSingleAsyncTavilyMetrics(
                    function="extract", status="success", start_time=start_time, end_time=time()
                )
            )
        except Exception as e:
            # return if failed to call api
            metrics.async_tavily_calls.append(
                TavilySearchSingleAsyncTavilyMetrics(
                    function="extract", status="error", start_time=start_time, end_time=time()
                )
            )
            return BrowseResponse(results_string=f"Failed to extract content: {e}")

        # return if no results
        result_list = results.get("results", [])
        if not result_list:
            return BrowseResponse(results_string="No content extracted.")

        # concat results
        blocks = []
        for result in result_list:
            url = result.get("url", "")
            content = result.get("raw_content", "")
            if len(content) > max_per_url_length:
                content = content[:max_per_url_length] + "\n... [truncated]"
            blocks.append(f"[URL]: {url}\n[Content]:\n{content}\n")

        results_string = "\n\n".join(blocks)
        return BrowseResponse(results_string=results_string)

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


    def _postprocess_search_results(self, query: str, results: dict, max_length: int) -> str:
        blocks = [f"[Search Query]: {query}"]
        running_len = len(blocks[0])

        for result in results["results"]:
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("raw_content") or result.get("content", "")
            if len(content) > 5000:
                content = content[:5000] + "\n... [truncated]"
            entry = (
                f"[Title]: {title}\n"
                f"[URL]: {url}\n"
                f"[Content]:\n{content}\n"
            )

            if running_len + len(entry) > max_length:
                break
            blocks.append(entry)
            running_len += len(entry)

        formatted_results = "\n".join(blocks)
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
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            correct_answer=ground_truth,
            response=response
        )

        judge_create_params = self.config.judge_responses_create_params.model_copy(deep=True)
        judge_create_params.max_output_tokens = 2048
        judge_create_params.input = [
            NeMoGymEasyInputMessage(role="user", content=judge_prompt),
        ]

        judge_response = None
        for attempt in range(self.JUDGE_MAX_ATTEMPTS):
            try:
                temp = 0.0 if attempt == 0 else min(0.3 + 0.1 * attempt, 1.0)
                judge_create_params.temperature = temp

                http_response = await self.server_client.post(
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=judge_create_params,
                )
                judge_response = NeMoGymResponse.model_validate(await http_response.json())
                text = judge_response.output[-1].content[-1].text

                is_correct, extracted, parsed_ok = self._parse_judge(text)
                if parsed_ok:
                    return JudgeEvaluation(
                        judge_response_create_params=judge_create_params,
                        reasoning=text,
                        extracted_final_answer=extracted,
                        reward=1.0 if is_correct else 0.0,
                        judge_response=judge_response,
                    )

            except Exception:
                await sleep(min(2**attempt, 30))

        return JudgeEvaluation(
            judge_response_create_params=judge_create_params,
            reasoning="",
            extracted_final_answer="",
            reward=0.0,
            judge_response=judge_response,
        )

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

    def _parse_judge(self, text: str) -> tuple[bool, str, bool]:
        """Parse grading output. Returns (is_correct, extracted, parsed_ok).

        Uses the LAST 'correct: yes/no' match to avoid picking up template
        echoes or reasoning inside <think> blocks.
        """
        matches = list(re.finditer(r"correct:\s*(yes|no)\b", text, re.IGNORECASE))
        if not matches:
            return False, "", False

        is_correct = matches[-1].group(1).lower() == "yes"

        ans_matches = list(re.finditer(
            r"extracted_final_answer:\s*(.+?)(?:\n|$)", text
        ))
        extracted = ans_matches[-1].group(1).strip() if ans_matches else ""

        if extracted and "The final exact answer extracted from the [response]" in extracted:
            return False, "", False

        return is_correct, extracted, True


if __name__ == "__main__":
    TavilySearchResourcesServer.run_webserver()
