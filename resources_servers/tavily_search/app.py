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

from tavily import TavilyClient


class TavilySearchResourcesServerConfig(BaseResourcesServerConfig):
    tavily_api_key: str
    exclude_domains_file_path: str
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming

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
    judge_response_create_params: NeMoGymResponseCreateParamsNonStreaming
    reasoning: str
    extracted_final_answer: str
    reward: float
    judge_response: NeMoGymResponse

class TavilySearchVerifyResponse(BaseVerifyResponse, JudgeEvaluation):
    pass

class TavilySearchResourcesServer(SimpleResourcesServer):
    config: TavilySearchResourcesServerConfig
    NUM_RESULTS: int = 10
    _tavily: Optional[TavilyClient] = PrivateAttr(default=None)

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = JUDGE_PROMPT_TEMPLATE

    def model_post_init(self, __context) -> None:
        self._tavily = TavilyClient(api_key=self.config.tavily_api_key)
        self._exclude_domains = self._parse_exclude_domains()
        print(self._exclude_domains)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/web_search")(self.web_search)

        return app
    
    async def web_search(self, body: TavilySearchRequest) -> TavilySearchResponse:
        results = self._tavily.search(
            body.query, 
            num_results=self.NUM_RESULTS, 
            exclude_domains=self._exclude_domains
        )
        postprocessed_results = self._postprocess_search_results(results)
        postprocessed_results_dump = json.dumps(postprocessed_results)
        return TavilySearchResponse(results_string=postprocessed_results_dump)


    async def verify(self, body: TavilySearchVerifyRequest) -> TavilySearchVerifyResponse:
        question = body.question
        ground_truth = body.ground_truth
        response = body.response.output
        judge_evaluation = await self._verify_answer_with_judge(question, ground_truth, response)
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
            match = re.search(r"correct: (yes|no)", grading_response)
            extracted_final_answer = match.group(1) if match else ""
            reward = 1.0 if extracted_final_answer == "yes" else 0.0
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



if __name__ == "__main__":
    TavilySearchResourcesServer.run_webserver()


'''
[done]1. I refactored the functions to not be outside
[done] 2. LLM judge is only half way done
[done it] 3. Async needs to be added in 
[done] 4. config needs to be changed
5. new dataset needs to be created with the question as well.
6. Make questions of the form QUERY_TEMPLATE
7. Currently we are using browsecomp prompt for judge. This may cause baselining issues - please measure.
'''