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
from typing import Optional, ClassVar
from pydantic import BaseModel, PrivateAttr
import re
from fastapi import FastAPI
from nemo_gym.config_types import ModelServerRef

from nemo_gym.base_resources_server import (
    SimpleResourcesServer,
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)

from judge_prompt import JUDGE_PROMPT_TEMPLATE

from tavily import AsyncTavilyClient,UsageLimitExceededError


class TavilySearchResourcesServerConfig(BaseResourcesServerConfig):
    tavily_api_key: str
    exclude_domains_file_path: str
    use_judge: bool = True  # If False, use regex matching instead of LLM judge
    judge_model_server: Optional[ModelServerRef] = None
    judge_responses_create_params: Optional[NeMoGymResponseCreateParamsNonStreaming] = None
    max_retries: int = 5  # Max retries for UsageLimitExceededError
    retry_delay_seconds: int = 30  # Delay between retries in seconds

class TavilySearchRequest(BaseModel):
    query: str

class TavilySearchResponse(BaseModel):
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

class TavilySearchVerifyResponse(BaseVerifyResponse, JudgeEvaluation):
    pass

class TavilySearchResourcesServer(SimpleResourcesServer):
    config: TavilySearchResourcesServerConfig
    MAX_RESULTS: int = 10
    _async_tavily: Optional[AsyncTavilyClient] = PrivateAttr(default=None)

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = JUDGE_PROMPT_TEMPLATE

    def model_post_init(self, __context) -> None:
        self._async_tavily = AsyncTavilyClient(api_key=self.config.tavily_api_key)
        self._exclude_domains = self._parse_exclude_domains()
        print(self._exclude_domains)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/web_search")(self.web_search)

        return app
    
    async def web_search(self, body: TavilySearchRequest) -> TavilySearchResponse:
        print(f"\n\n {body.query=}")
        
        max_retries = self.config.max_retries
        retry_delay_seconds = self.config.retry_delay_seconds
        
        for attempt in range(max_retries):
            try:
                results = await self._async_tavily.search(
                    body.query, 
                    max_results=self.MAX_RESULTS, 
                    exclude_domains=self._exclude_domains,
                    search_depth="advanced"
                )
                break  # Success, exit the retry loop
            except UsageLimitExceededError as e:
                print(f"UsageLimitExceededError (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Sleeping for {retry_delay_seconds} seconds before retrying...")
                    await asyncio.sleep(retry_delay_seconds)
                else:
                    print("Max retries exceeded. Returning empty results.")
                    return TavilySearchResponse(results_string="[]")
        postprocessed_results = self._postprocess_search_results(results)
        postprocessed_results_dump = json.dumps(postprocessed_results)
        return TavilySearchResponse(results_string=postprocessed_results_dump)


    async def verify(self, body: TavilySearchVerifyRequest) -> TavilySearchVerifyResponse:
        question = body.question
        ground_truth = body.ground_truth
        last_assistant_response = self._get_last_assistant_response(body.response)

        if self.config.use_judge:
            judge_evaluation = await self._verify_answer_with_judge(question, ground_truth, last_assistant_response)
        else:
            judge_evaluation = self._verify_answer_with_regex(ground_truth, last_assistant_response)
        return TavilySearchVerifyResponse(**body.model_dump(), **judge_evaluation.model_dump()) 

    ###### UTILITY FUNCTIONS ######

    def _postprocess_search_results(self,results: list[dict]) -> list[dict]:
        formatted_results = []
        for result in results["results"]:
            formatted_results.append({
                "url": result["url"],
                "title": result["title"],
                "content": result["content"]
            })
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

        async def _get_judge_response(question: str, ground_truth: str, response: str) -> JudgeEvaluation:
            judge_create_params = self.config.judge_responses_create_params.model_copy(deep=True)
            judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
                question=question,
                correct_answer=ground_truth,
                response=response
            )
            judge_create_params.input = [
                NeMoGymEasyInputMessage(
                    role="user",
                    content=judge_prompt,
                ),
            ]
            response = await self.server_client.post(
                server_name=self.config.judge_model_server.name,
                url_path="/v1/responses",
                json=judge_create_params,   
            )
            judge_response = NeMoGymResponse.model_validate(await response.json())
            return judge_create_params, judge_response   

        def _grade_sample(judge_create_params: NeMoGymResponseCreateParamsNonStreaming, judge_response: NeMoGymResponse) -> JudgeEvaluation:
            #Taken from: https://github.com/openai/simple-evals/blob/5e623c2b400af62a1278e23595f95b0853d7fe8a/browsecomp_eval.py#L79-L93
            grading_response = judge_response.output[-1].content[-1].text
            print(f"\n\n {grading_response=}")
            match = re.search(r"correct: (yes|no)", grading_response)
            extracted_final_answer = match.group(1) if match else ""
            reward = 1.0 if extracted_final_answer == "yes" else 0.0
            print(f"\n\n {reward=}")
            return JudgeEvaluation(
                judge_response_create_params=judge_create_params,
                reasoning=grading_response,
                extracted_final_answer=extracted_final_answer,
                reward=reward,
                judge_response=judge_response
            )

        judge_create_params, judge_response = await _get_judge_response(question, ground_truth, response)
        judge_evaluation = _grade_sample(judge_create_params, judge_response)
        return judge_evaluation

    def _verify_answer_with_regex(self, ground_truth: str, response: str) -> JudgeEvaluation:
        """Verify answer by checking if ground_truth (as regex) matches in response."""
        matches = re.findall(r"Answer:\s*(.*)\s*Confidence:", response, re.IGNORECASE)

        if matches:
            answer = matches[-1].strip() # Get the last item in the list
        else:
            answer = ""
        print(f"\n\n {answer=}")
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
            #if any content item is of type output_text, then return the text
            for content_item in output_item.content:
                if content_item.type == "output_text":
                    return content_item.text
        return ""


if __name__ == "__main__":
    TavilySearchResourcesServer.run_webserver()
