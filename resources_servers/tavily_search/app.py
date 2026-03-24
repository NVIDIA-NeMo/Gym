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
from typing import ClassVar, List, Optional
from urllib.parse import urlparse

from fastapi import FastAPI
from pydantic import BaseModel, PrivateAttr
from tavily import AsyncTavilyClient, UsageLimitExceededError

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from resources_servers.tavily_search.judge_prompt import JUDGE_PROMPT_TEMPLATE


class TavilySearchResourcesServerConfig(BaseResourcesServerConfig):
    tavily_api_key: str | List[str]
    exclude_domains_file_path: str
    use_judge: bool = True  # If False, use regex matching instead of LLM judge
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    max_retries: int = 5  # Max retries for UsageLimitExceededError
    retry_delay_seconds: int = 30  # Delay between retries in seconds
    debug: bool = False


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


class TavilySearchVerifyResponse(BaseVerifyResponse, JudgeEvaluation):
    num_tool_calls: int


class TavilySearchResourcesServer(SimpleResourcesServer):
    config: TavilySearchResourcesServerConfig
    MAX_RESULTS: int = 10
    MAX_RESULT_CHARS: int = 2000

    _async_tavily_clients: Optional[List[AsyncTavilyClient]] = PrivateAttr(default=None)
    _num_requests: int = 0

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = JUDGE_PROMPT_TEMPLATE

    def model_post_init(self, __context) -> None:
        tavily_api_keys = self.config.tavily_api_key
        if isinstance(tavily_api_keys, str):
            tavily_api_keys = [tavily_api_keys]

        self._async_tavily_clients = [AsyncTavilyClient(api_key=k) for k in tavily_api_keys]

        self._exclude_domains = self._parse_exclude_domains()
        self._page_cache: dict[str, str] = {}
        print(self._exclude_domains)
        if self.config.debug:
            print("Debug mode enabled")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/web_search")(self.web_search)
        app.post("/find_in_page")(self.find_in_page)
        app.post("/scroll_page")(self.scroll_page)

        return app

    def _select_tavily_client(self) -> AsyncTavilyClient:
        client = self._async_tavily_clients[self._num_requests % len(self._async_tavily_clients)]
        self._num_requests += 1
        return client

    async def web_search(self, body: TavilySearchRequest) -> TavilySearchResponse:
        if self.config.debug:
            print("\n\n body.query: ", body.query)
        if body.query is None:
            return TavilySearchResponse(results_string="Query is none")

        if len(body.query) > 400:
            return TavilySearchResponse(results_string="Query is too long")

        max_retries = self.config.max_retries
        retry_delay_seconds = self.config.retry_delay_seconds

        for attempt in range(max_retries):
            # @bxyu-nvidia: We put this in the attempt loop so we have a different client per attempt to better avoid rate limits.
            async_tavily_client = self._select_tavily_client()
            try:
                results = await async_tavily_client.search(
                    body.query,
                    max_results=self.MAX_RESULTS,
                    exclude_domains=self._exclude_domains,
                    search_depth="advanced",
                )
                break  # Success, exit the retry loop
            except UsageLimitExceededError as e:
                if self.config.debug:
                    print(f"UsageLimitExceededError (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    if self.config.debug:
                        print(f"Sleeping for {retry_delay_seconds} seconds before retrying...")
                    await asyncio.sleep(retry_delay_seconds)
                else:
                    if self.config.debug:
                        print("Max retries exceeded. Returning empty results.")
                    return TavilySearchResponse(results_string="[]")
        postprocessed_results = self._postprocess_search_results(results)
        return TavilySearchResponse(results_string="".join(postprocessed_results))

    async def find_in_page(self, body: FindInPageRequest) -> FindInPageResponse:
        if self.config.debug:
            print("\n\n find_in_page ")
            print(f"url={body.url}, query={body.query}")

        if body.url is None:
            return FindInPageResponse(results_string="URL is none")
        if body.query is None:
            return FindInPageResponse(results_string="Query is none")

        if self._is_url_excluded(body.url):
            return FindInPageResponse(results_string="URL is in excluded domains")

        max_retries = self.config.max_retries
        retry_delay_seconds = self.config.retry_delay_seconds

        for attempt in range(max_retries):
            try:
                async_tavily_client = self._select_tavily_client()
                results = await async_tavily_client.extract(
                    urls=body.url,
                    query=body.query,
                )
                break  # Success, exit the retry loop
            except UsageLimitExceededError as e:
                if self.config.debug:
                    print(f"UsageLimitExceededError (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    if self.config.debug:
                        print(f"Sleeping for {retry_delay_seconds} seconds before retrying...")
                    await asyncio.sleep(retry_delay_seconds)
                else:
                    if self.config.debug:
                        print("Max retries exceeded. Returning empty results.")
                    return FindInPageResponse(results_string="[]")

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

    async def scroll_page(self, body: ScrollPageRequest) -> ScrollPageResponse:
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
            max_retries = self.config.max_retries
            retry_delay_seconds = self.config.retry_delay_seconds

            for attempt in range(max_retries):
                async_tavily_client = self._select_tavily_client()
                try:
                    results = await async_tavily_client.extract(
                        urls=body.url,
                    )
                    break  # Success, exit the retry loop
                except UsageLimitExceededError as e:
                    if self.config.debug:
                        print(f"UsageLimitExceededError (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        if self.config.debug:
                            print(f"Sleeping for {retry_delay_seconds} seconds before retrying...")
                        await asyncio.sleep(retry_delay_seconds)
                    else:
                        if self.config.debug:
                            print("Max retries exceeded. Returning empty results.")
                        return ScrollPageResponse(results_string="[]", total_words=0)

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

    async def verify(self, body: TavilySearchVerifyRequest) -> TavilySearchVerifyResponse:
        question = body.question
        ground_truth = body.ground_truth
        last_assistant_response = self._get_last_assistant_response(body.response)

        if self.config.use_judge:
            judge_evaluation = await self._verify_answer_with_judge(question, ground_truth, last_assistant_response)
        else:
            judge_evaluation = self._verify_answer_with_regex(ground_truth, last_assistant_response)
        return TavilySearchVerifyResponse(
            **body.model_dump(),
            **judge_evaluation.model_dump(),
            num_tool_calls=sum(o.type == "function_call" for o in body.response.output),
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

    def _get_last_assistant_response(self, response: NeMoGymResponse) -> str:
        for output_item in response.output[::-1]:
            if output_item.type != "message":
                continue
            # if any content item is of type output_text, then return the text
            for content_item in output_item.content:
                if content_item.type == "output_text":
                    return content_item.text
        return ""


if __name__ == "__main__":
    TavilySearchResourcesServer.run_webserver()
