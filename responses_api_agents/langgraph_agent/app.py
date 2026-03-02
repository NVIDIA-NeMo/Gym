from abc import abstractmethod
from typing import Any

from fastapi import Body, Request, Response
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status


class LangGraphAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    resources_server: ResourcesServerRef


class LangGraphAgentAdapter(SimpleResponsesAPIAgent):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    config: LangGraphAgentConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = self.build_graph()

    @abstractmethod
    def build_graph(self) -> Any:
        pass

    @abstractmethod
    async def get_initial_state(self, body: NeMoGymResponseCreateParamsNonStreaming, cookies: dict) -> dict:
        pass

    @abstractmethod
    def extract_outputs(self, final_state: dict) -> list:
        pass

    def extract_model_response(self, final_state: dict) -> NeMoGymResponse:
        if "last_model_response" in final_state:
            return final_state["last_model_response"]
        raise NotImplementedError("State must contain 'last_model_response' or override extract_model_response()")

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        initial_state = await self.get_initial_state(body, request.cookies)
        final_state = await self.graph.ainvoke(initial_state)

        if "cookies" in final_state:
            for k, v in final_state["cookies"].items():
                response.set_cookie(k, v)

        model_response = self.extract_model_response(final_state)
        outputs = self.extract_outputs(final_state)
        model_response.output = outputs
        return model_response

    async def run(self, request: Request, body: BaseRunRequest) -> BaseVerifyResponse:
        cookies = request.cookies

        seed = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed)
        cookies = seed.cookies

        resp = await self.server_client.post(
            server_name=self.config.name, url_path="/v1/responses", json=body.responses_create_params, cookies=cookies
        )
        await raise_for_status(resp)

        verify_request_dict = body.model_dump() | {"response": await get_response_json(resp)}

        verify = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request_dict,
            cookies=resp.cookies,
        )
        await raise_for_status(verify)
        return BaseVerifyResponse.model_validate(await get_response_json(verify))
