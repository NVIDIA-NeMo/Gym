import json


from pydantic import ConfigDict

from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.response_input_param import FunctionCallOutput

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
)


class SimpleAgentStatefulConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef


class SimpleAgentStatefulRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentStatefulVerifyRequest(BaseVerifyRequest):
    expected_result: str  # Add this field
    expected_code_contains: str = ""
    model_config = ConfigDict(extra="allow")


class SimpleAgentStatefulVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class SimpleAgentStateful(SimpleResponsesAPIAgent):
    config: SimpleAgentStatefulConfig

    async def responses(
        self, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        new_outputs = []
        session_id = None  # Track session ID for statefulness

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

            # Prepare function call arguments
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

            tool_response = FunctionCallOutput(
                type="function_call_output",
                call_id=output_function_call.call_id,  # REQUIRED by spec
                output=json.dumps(simplified_output),
            )

            new_outputs.append(tool_response)

        final_response_dict = model_response.model_dump()
        final_response_dict["output"] = new_outputs
        return final_response_dict

    async def run(
        self, body: SimpleAgentStatefulRunRequest
    ) -> SimpleAgentStatefulVerifyResponse:
        response = await self.responses(body.responses_create_params)

        response["expected_answer"] = body.expected_result

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
