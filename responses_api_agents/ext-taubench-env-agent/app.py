# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from os import environ
from pathlib import Path
from typing import Literal, Union

# Set the data dir to the actual data dir as the package is installed. Must be before the main tau2 imports.
import tau2
from fastapi import Request, Response
from pydantic import ConfigDict


path = Path(tau2.__file__).parent / "data"
environ["TAU2_DATA_DIR"] = str(path)

from loguru import logger as loguru_logger
from tau2.api_service.task_service import rollout, verify
from tau2.data_model.rollout import RolloutRequest
from tau2.data_model.verifier import VerifyRequest

from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymChoice,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from responses_api_models.vllm_model.app import VLLMConverter


class ExternalTaubenchEnvironmentAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    user_model_server: ModelServerRef

    # These parameter defaults are the ones recommended by the Ext Taubench env creators.
    max_steps: int = 200
    max_errors: int = 10
    seed: int = 300

    log_level: Union[Literal["DEBUG"], Literal["ERROR"]] = "ERROR"


class ExternalTaubenchEnvironmentAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ExternalTaubenchEnvironmentAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class ExternalTaubenchEnvironmentAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    num_steps: int
    verify_response: dict


class ExternalTaubenchEnvironmentAgent(SimpleResponsesAPIAgent):
    config: ExternalTaubenchEnvironmentAgentConfig

    def model_post_init(self, context):
        # DEBUG is the default
        if self.config.log_level != "DEBUG":
            loguru_logger.remove()
            loguru_logger.add(sys.stderr, level=self.config.log_level)

        return super().model_post_init(context)

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        pass

    def _resolve_model_config(
        self,
        global_config_dict: ConfigDict,
        name: str,
        server_name: str,
        responses_create_params_dict: dict,
    ) -> dict:
        server_config = get_first_server_config_dict(global_config_dict, server_name)
        server_url = f"http://{server_config['host']}:{server_config['port']}/v1"

        sampling_params = dict()
        if responses_create_params_dict.get("temperature") is not None:
            sampling_params["temperature"] = responses_create_params_dict["temperature"]
        if responses_create_params_dict.get("top_p") is not None:
            sampling_params["top_p"] = responses_create_params_dict["top_p"]

        return {
            "name": name,
            "model": {
                "name": "dummy_model",  # Dummy model for Gym model server
                "api_base": server_url,
                "api_key": "dummy_key",  # Dummy API key needed for Gym model server
                "params": sampling_params,
            },
        }

    def _trajectory_to_responses_output_items(self, messages: list[dict]) -> list:
        responses_output_items = []
        for message in messages:
            if message["role"] == "user" and message["content"]:
                """
                For user messages with content None - those messages will mostly be when the user llm issues tool calls and we dont want to train on those. We only train on stuff we get from the assistant.

                We want to set the role = assistant for the user llm generated messages to role = user as that is the right thing to do
                """
                responses_user_message = NeMoGymEasyInputMessage(**(message["raw_data"]["message"] | {"role": "user"}))
                responses_output_items.append(responses_user_message)

            elif message["role"] == "tool" and message["requestor"] == "assistant":
                responses_tool_message_json = {
                    "call_id": message["id"],
                    "output": message["content"],
                    "type": "function_call_output",
                    "id": None,
                    "status": None,
                }
                responses_tool_message = NeMoGymFunctionCallOutput(**responses_tool_message_json)
                responses_output_items.append(responses_tool_message)

            elif message["role"] == "assistant" and message.get("raw_data"):
                return_token_id_information = "prompt_token_ids" in message["raw_data"]["message"]
                converter_return_token_ids = VLLMConverter(return_token_id_information=return_token_id_information)
                responses_assistant_tool_call_input_message = NeMoGymChoice.model_validate(message["raw_data"])
                responses_assistant_tool_call_message = converter_return_token_ids.postprocess_chat_response(
                    responses_assistant_tool_call_input_message
                )
                responses_output_items.extend(responses_assistant_tool_call_message)

        return responses_output_items

    async def run(
        self, body: ExternalTaubenchEnvironmentAgentRunRequest, request: Request = None
    ) -> ExternalTaubenchEnvironmentAgentVerifyResponse:
        body = body.model_dump()

        rollout_request = {
            "task_params": {
                "domain": body["user_scenario"]["instructions"]["domain"],
                "max_steps": self.config.max_steps,
                "max_errors": self.config.max_errors,
                "seed": self.config.seed,
            },
            "task": body,
            "db_seed": None,
            "user_db_seed": None,
        }

        # Resolve model configs
        global_config_dict = self.server_client.global_config_dict
        rollout_request["agent"] = self._resolve_model_config(
            global_config_dict=global_config_dict,
            name="llm_agent",
            server_name=self.config.model_server.name,
            responses_create_params_dict=body["responses_create_params"],
        )
        rollout_request["user"] = self._resolve_model_config(
            global_config_dict=global_config_dict,
            name="user_simulator",
            server_name=self.config.user_model_server.name,
            responses_create_params_dict=body["responses_create_params"],
        )

        payload = RolloutRequest(**rollout_request)
        rollout_response = await rollout(payload)

        verify_request = VerifyRequest(
            evaluation_type="all",
            trajectory=rollout_response.trajectory,
            rollout_params=rollout_response.rollout_context.model_dump(),
        )
        verify_response = await verify(verify_request)

        response_object = NeMoGymResponse(
            id="1",
            created_at="1.0",
            model="dummy_model",
            object="response",
            output=self._trajectory_to_responses_output_items(rollout_response.model_dump()["trajectory"]),
            parallel_tool_calls=False,
            tools=[],
            tool_choice="auto",
        )

        num_steps = len(rollout_response.trajectory)

        raw_rollout = rollout_response.model_dump(mode="json")
        # Remove the prompt token ids from the raw trajectory.
        for message in raw_rollout["trajectory"]:
            if not message.get("raw_data"):
                continue

            raw_message = message["raw_data"]["message"]
            if "prompt_token_ids" in raw_message:
                raw_message.pop("prompt_token_ids")
                raw_message.pop("generation_token_ids")
                raw_message.pop("generation_log_probs")

        return ExternalTaubenchEnvironmentAgentVerifyResponse(
            reward=verify_response.response.reward,
            response=response_object,
            **body,
            verify_response=verify_response.response.model_dump(mode="json"),
            raw_rollout=raw_rollout,
            num_steps=num_steps,
        )


if __name__ == "__main__":
    ExternalTaubenchEnvironmentAgent.run_webserver()
