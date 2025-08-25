from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from nemo_gym.base_responses_api_agent import Body, BaseResponsesAPIAgentConfig

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletion,
)

from responses_api_agents.simple_agent.app import (
    SimpleAgent,
    SimpleAgentRunRequest,
    SimpleAgentVerifyRequest,
    SimpleAgentVerifyResponse,
)


class SimpleAgentChatCompletionsConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef


class SimpleAgentChatCompletions(SimpleAgent):
    """
    This is the same as SimpleAgent, just using Chat Completions rather than Responses API.

    Right now, this agent only support single turn no tools.
    """

    config: SimpleAgentChatCompletionsConfig

    async def responses(
        self, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        cc_body: NeMoGymChatCompletionCreateParamsNonStreaming = body.model_dump(
            exclude_unset=True
        )
        assert not cc_body.get("tools"), "Tools are not supported currently!"
        cc_body["messages"] = cc_body.pop("input")

        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/chat/completions",
            json=cc_body,
        )
        model_response = NeMoGymChatCompletion.model_validate(model_response.json())

        message = model_response.choices[0].message

        return NeMoGymResponse(
            id=model_response.id,
            created_at=0.0,
            model=cc_body.get("model", ""),
            object="response",
            output=[
                ResponseOutputMessage(
                    id=model_response.id,
                    content=[
                        ResponseOutputText(
                            annotations=[],
                            text=message.content,
                            type="output_text",
                        ),
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=cc_body.get("parallel_tool_calls", True),
            temperature=cc_body.get("temperature", None),
            tool_choice="auto",
            tools=[],
        )

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
    SimpleAgentChatCompletions.run_webserver()
