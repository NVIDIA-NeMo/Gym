import json

from pydantic import ConfigDict

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
from nemo_gym.server_utils import ResourcesServerRef, ModelServerRef

from nemo_gym.openai_utils import (
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponse,
)


class SimpleGameAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_moves: int = 50  # Maximum number of moves allowed


class SimpleGameAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")
    # Game-specific parameters will be passed through to get_initial_board
    clues: int = 30
    scale: int = 9


class SimpleGameAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SimpleGameAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class SimpleGameAgent(SimpleResponsesAPIAgent):
    config: SimpleGameAgentConfig

    # Add a class attribute to temporarily store game params
    _current_game_params: dict = {}

    async def responses(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
        game_params: dict = None,
    ) -> NeMoGymResponse:
        """Run the game with direct model-environment communication - no tools needed."""

        if game_params is None:
            game_params = {}

        conversation = body["input"].copy()
        moves_made = 0
        game_state = None
        reward = 0.0
        is_complete = False

        # NEW: will be written back into response["input"]
        input_messages: list[dict] = []

        # Model-side outputs only (function_call, function_call_output, assistant message)
        new_outputs = []

        # Get initial board
        game_params = self._current_game_params
        initial_board_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/get_initial_board",
            json=game_params,
        )
        board_data = initial_board_response.json()
        game_state = board_data.get("game_state")

        # Add initial board to conversation (for the model) ...
        initial_message = {
            "role": "user",
            "content": f"{board_data['instructions']}\n\n{board_data['board_text']}",
        }
        conversation.append(initial_message)

        # ... and also to input_messages (for the saved trajectory)
        input_messages.append(
            {
                "role": "user",
                "type": "message",
                "content": initial_message["content"],
            }
        )

        # Game loop
        while True:
            new_body = body.copy()
            new_body["input"] = conversation

            #! This is so that the model only performs one move at a time. Without this them model can perform multiple moves and then our env would have to be made changed to handle that complexity.
            new_body["parallel_tool_calls"] = False
            new_body["max_tool_calls"] = 1

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

            # Handle tool call
            output_function_call = output[-1]
            function_args = json.loads(output_function_call.arguments)
            function_args["game_state"] = game_state
            api_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path=f"/{output_function_call.name}",
                json=function_args,
            )

            # Record tool result in outputs
            tool_response = FunctionCallOutput(
                type="function_call_output",
                call_id=output_function_call.call_id,
                output=json.dumps(api_response.json()),
            )
            new_outputs.append(tool_response)

            # Env feedback → conversation + input_messages
            move_data = api_response.json()
            game_state = move_data.get("game_state", game_state)
            moves_made += 1
            reward += move_data.get("move_reward", 0.0)

            msg = move_data.get("message", str(move_data))
            board_txt = move_data.get("board_text", "")
            env_text = f"{msg}\n\n{board_txt}"

            conversation.append({"role": "user", "content": env_text})
            # input_messages.append({
            #     "role": "user",
            #     "type": "message",
            #     "content": env_text,
            # })

            if (
                move_data.get("is_complete", False)
                or moves_made >= self.config.max_moves
            ):
                is_complete = move_data.get("is_complete", False)
                break

        # Metrics
        self._reward = reward
        self._total_moves = moves_made
        self._is_complete = is_complete

        # FINAL: AIME-style
        final_response_dict = model_response.model_dump()
        final_response_dict["output"] = new_outputs  # assistant + tools only
        final_response_dict["input"] = (
            input_messages  # initial board + all env feedback, in order
        )
        return final_response_dict

    async def run(
        self, body: SimpleGameAgentRunRequest
    ) -> SimpleGameAgentVerifyResponse:
        """Run a complete game session."""

        # Prepare the conversation
        conversation_body = body.responses_create_params

        # Extract game parameters
        game_params = {
            k: v
            for k, v in body.model_dump().items()
            if k not in ["responses_create_params"]
        }

        # Store in class attribute instead of trying to setattr on the TypedDict
        self._current_game_params = game_params

        # Run the game with game_params passed directly
        response = await self.responses(conversation_body, game_params)

        # Create verify request
        verify_request = SimpleGameAgentVerifyRequest.model_validate(
            body.model_dump()
            | {
                "response": response,  # OpenAI response
                "reward": self._reward,  # numbers we stored
                "total_moves": self._total_moves,
                "is_complete": self._is_complete,
            }
        )

        # Call verify on the resources server
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
        )

        return SimpleGameAgentVerifyResponse.model_validate(verify_response.json())


if __name__ == "__main__":
    SimpleGameAgent.run_webserver()
