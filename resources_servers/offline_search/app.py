from pydantic import BaseModel
import requests
import json
from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    SimpleResourcesServer,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.search_parsing_utils import box_parser  


class BaseSearchQueryRequest(BaseModel):
    query: str
    topk: int = 10


class OfflineSearchResourcesServerConfig(BaseResourcesServerConfig):
    base_url: str #please spin this up by yourself

class BaseGetSearchQueryResponse(BaseModel):
    search_results: str

class OfflineSearchRunRequest(BaseRunRequest):
    expected_answer: str

class OfflineSearchVerifyRequest(BaseVerifyRequest, OfflineSearchRunRequest):
    pass

class OfflineSearchVerifyResponse(BaseVerifyResponse):
    parsed_option: str


def _extract_last_assistant_text(body: OfflineSearchVerifyRequest) -> str:
    last_message = body.response.output[-1]
    if last_message.type == "message" and last_message.role == "assistant":
        return last_message.content
    else:
        return None

class OfflineSearchResourcesServer(SimpleResourcesServer):
    config: OfflineSearchResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/search")(self.search)

        return app
    
    async def search(self, body: BaseSearchQueryRequest) -> BaseGetSearchQueryResponse:
        url = f"{self.config.base_url}/retrieve"
        payload = {
            "queries": [body.query],
            "topk": body.topk,
            "return_scores": False #FIXME: we keep this as false for now
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            json_str = json.dumps(response.json())
            return BaseGetSearchQueryResponse(search_results=json_str)
        except Exception as e:
            return BaseGetSearchQueryResponse(search_results=f"Error: Unexpected error - {str(e)}")
        

    async def verify(self, body: OfflineSearchVerifyRequest) -> OfflineSearchVerifyResponse:
        expected_answer = body.expected_answer
        response_text = _extract_last_assistant_text(body)
        parsed_option = box_parser(response_text)
        if parsed_option == expected_answer:
            reward = 1.0
        else:
            reward = 0.0
        return OfflineSearchVerifyResponse(**body.model_dump(), reward=reward, parsed_option=parsed_option)


if __name__ == "__main__":
    OfflineSearchResourcesServer.run_webserver()

'''
TODOs:
 - [done] create chatcompletionparams dataset
 - [done]test ng server
 - [done]test with ng collect
 - write README:
    - [done] write tests for server
    - collect examples.json (start from train as we will have two)
    - create updated config with the two datasets
    - run ng_prepare_data
    - run example_rollouts. 
    - check on git commit.
- Get reward profiling numbers
I think this should maybe take an hour to run and create?
'''