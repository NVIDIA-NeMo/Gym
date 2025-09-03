from typing import List

import json

from pydantic import ConfigDict, ValidationError

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
from nemo_gym.config_types import ResourcesServerRef, ModelServerRef

from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymFunctionCallOutput,
    NeMoGymEasyInputMessage,
)


class SimpleAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_turns: int = None


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
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs = []
        turn = 0

        while True:
            turn += 1
            new_body = body.model_copy(update={"input": body.input + new_outputs})

            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
            )
            model_response_json = model_response.json()
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e

            output = model_response.output
            new_outputs.extend(output)

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [
                o for o in output if o.type == "function_call"
            ]
            if not all_fn_calls:
                break

            for output_function_call in all_fn_calls:
                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=json.loads(output_function_call.arguments),
                )

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=json.dumps(api_response.json()),
                )
                new_outputs.append(tool_response)

            # Check if max turns is not None and if we have exhausted it.
            if self.config.max_turns and turn >= self.config.max_turns:
                break

        model_response.output = new_outputs
        return model_response

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
