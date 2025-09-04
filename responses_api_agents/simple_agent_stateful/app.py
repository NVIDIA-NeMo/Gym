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


class SimpleAgentStatefulConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef


class SimpleAgentStatefulRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    expected_result: str  # Add this field
    expected_code_contains: str = ""


class SimpleAgentStatefulVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentStatefulVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class SimpleAgentStateful(SimpleResponsesAPIAgent):
    config: SimpleAgentStatefulConfig

    async def responses(
        self, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs = []
        session_id = None  # Track session ID for statefulness

        while True:
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
                function_args = json.loads(output_function_call.arguments)
                if session_id:  # Add session_id to subsequent calls
                    function_args["session_id"] = session_id
                api_response = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{output_function_call.name}",
                    json=function_args,
                )

                # Extract session_id from first response for reuse
                if session_id is None:
                    response_data = api_response.json()
                    if "session_id" in response_data:
                        session_id = response_data["session_id"]

                # --- create a compliant FunctionCallOutput --------------------------
                response_data = api_response.json()
                simplified_output = {
                    "stdout": response_data.get("stdout", ""),
                    "stderr": response_data.get("stderr", ""),
                    "result": response_data.get("result", ""),
                }

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=json.dumps(simplified_output),
                )
                new_outputs.append(tool_response)

        model_response.output = new_outputs
        return model_response

    async def run(
        self, body: SimpleAgentStatefulRunRequest
    ) -> SimpleAgentStatefulVerifyResponse:
        response = await self.responses(body.responses_create_params)

        response.expected_answer = body.expected_result

        verify_request = SimpleAgentStatefulVerifyRequest.model_validate(
            body.model_dump() | {"response": response}
        )
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
        )

        return SimpleAgentStatefulVerifyResponse.model_validate(verify_response.json())


if __name__ == "__main__":
    SimpleAgentStateful.run_webserver()
