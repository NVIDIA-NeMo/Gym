import json

from pydantic import ConfigDict

from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    EasyInputMessageParam,
)

from nemo_gym.base_resources_server import (
    BaseVerifyRequest,
    BaseRunRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    SimpleResponsesAPIAgent,
    BaseResponsesAPIAgentConfig,
    Body,
)
from nemo_gym.server_utils import ResourcesServerRef, ModelServerRef

from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponse,
)


class SimpleAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef


class SimpleAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class SimpleAgent(SimpleResponsesAPIAgent):
    config: SimpleAgentConfig

    async def responses(
        self, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        if isinstance(body["input"], str):
            body["input"] = [
                EasyInputMessageParam(
                    content=body["input"],
                    role="user",
                    type="message",
                )
            ]

        new_outputs = []
        while True:
            new_body: NeMoGymResponseCreateParamsNonStreaming = body.copy()
            new_body["input"] = body["input"] + new_outputs

            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
            )
            model_response = NeMoGymResponse.model_validate(model_response.json())

            output = model_response.output
            new_outputs.extend((o.model_dump() for o in output))
            if output[-1].type != "function_call":
                break

            output_function_call: ResponseFunctionToolCall = output[-1]

            api_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path=f"/{output_function_call.name}",
                json=json.loads(output_function_call.arguments),
            )

            tool_response = FunctionCallOutput(
                type="function_call_output",
                call_id=output_function_call.call_id,
                output=json.dumps(api_response.json()),
            )
            new_outputs.append(tool_response)

        final_response_dict = model_response.model_dump()
        final_response_dict["output"] = new_outputs
        return final_response_dict

    async def run(self, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        response = await self.responses(body.responses_create_params)

        verify_request = SimpleAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": response}
        )
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
        )
        return SimpleAgentVerifyResponse.model_validate(verify_response.json())


if __name__ == "__main__":
    SimpleAgent.run_webserver()
