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
from typing import Optional

import requests
import trafilatura
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class GoogleSearchResourcesServerConfig(BaseResourcesServerConfig):
    google_api_key: str
    google_cx: str


class BaseSearchQueryRequest(BaseModel):
    query: str


class BaseGetPageContentRequest(BaseModel):
    url: str


class BaseGetPageContentResponse(BaseModel):
    page_content: str


class BaseGetSearchQueryResponse(BaseModel):
    search_results: str


class GoogleSearchRunRequest(BaseRunRequest):
    expected_answer: str
    task_difficulty_qwen3_32b_avg_8: float


class GoogleSearchVerifyRequest(GoogleSearchRunRequest, BaseVerifyRequest):
    pass


class GoogleSearchVerifyResponse(BaseVerifyResponse):
    parsed_option: str


def box_parser(output_str: str) -> Optional[str]:
    if not isinstance(output_str, str):
        output_str = str(output_str)

    if output_str is None:
        return None
    try:
        match = re.search(r"\\boxed\{(.*?)\}", output_str)
        parsed_option = match.group(1) if match else None
        return parsed_option
    except Exception as e:
        print(f"Regex error: {e}")
        return None


def _extract_last_assistant_text(body: GoogleSearchVerifyRequest) -> str:
    last_message = body.response.output[-1]
    if last_message.type == "message" and last_message.role == "assistant":
        return last_message.content
    else:
        return None


class GoogleSearchResourcesServer(SimpleResourcesServer):
    config: GoogleSearchResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)
        app.post("/search")(self.search)
        app.post("/browse")(self.browse)
        return app

    async def search(self, body: BaseSearchQueryRequest) -> BaseGetSearchQueryResponse:
        request_params = {
            "key": self.config.google_api_key,
            "cx": self.config.google_cx,
            "q": body.query,
        }
        try:
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params=request_params,
                timeout=10,
            )
            response.raise_for_status()
            json_str = json.dumps(response.json())
            return BaseGetSearchQueryResponse(search_results=json_str)
        except Exception as e:
            return BaseGetSearchQueryResponse(search_results=f"Error: Unexpected error - {str(e)}")

    async def browse(self, body: BaseGetPageContentRequest) -> BaseGetPageContentResponse:
        try:
            html = trafilatura.fetch_url(body.url)
            if html:
                text = trafilatura.extract(html)
                if len(text.split()) > 10000:
                    text = text[:5000] + "..." + text[-5000:]
                if text:
                    return BaseGetPageContentResponse(page_content=text)
                else:
                    return BaseGetPageContentResponse(page_content="No text found")
            else:
                return BaseGetPageContentResponse(page_content="No HTML found")
        except Exception as e:
            return BaseGetPageContentResponse(page_content=f"Error: Unexpected error = {str(e)}")

    async def verify(self, body: GoogleSearchVerifyRequest) -> GoogleSearchVerifyResponse:
        expected_answer = body.expected_answer
        response_text = _extract_last_assistant_text(body)
        parsed_option = box_parser(response_text)
        if parsed_option == expected_answer:
            reward = 1.0
        else:
            reward = 0.0
        return GoogleSearchVerifyResponse(**body.model_dump(), reward=reward, parsed_option=parsed_option)


if __name__ == "__main__":
    GoogleSearchResourcesServer.run_webserver()
