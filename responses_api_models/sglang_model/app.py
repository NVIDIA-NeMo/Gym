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
import os
from multiprocessing import Process
from time import sleep, time
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import ray
from aiohttp.client_exceptions import ClientResponseError
from fastapi import Request

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.global_config import find_open_port
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionMessage,
    NeMoGymChoice,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.ray_utils import (
    lookup_current_ray_node_ip,
    spinup_single_ray_gpu_node_worker,
)
from nemo_gym.server_utils import SESSION_ID_KEY
from responses_api_models.vllm_model.app import (
    VLLMConverter,
    _vllm_server_heartbeat,
)


class SGLangModelConfig(BaseResponsesAPIModelConfig):
    base_url: Union[str, List[str]]
    api_key: str
    model: str
    return_token_id_information: bool

    uses_reasoning_parser: bool
    replace_developer_role_with_system: bool = False

    spinup_server: bool = False
    server_args: Optional[Dict[str, Any]] = None

    router_dp_size: int = 1

    def model_post_init(self, context):
        if isinstance(self.base_url, str):
            self.base_url = [self.base_url]
        return super().model_post_init(context)


def _start_sglang_server(config: SGLangModelConfig, server_host: str, server_port: int, router_dp_rank: int) -> None:
    import sglang.srt.entrypoints.http_server
    import sglang.srt.server_args

    argv = []
    argv.append("--model-path")
    argv.append(config.model)
    argv.append("--host")
    argv.append(server_host)
    argv.append("--port")
    argv.append(f"{server_port}")
    for k, v in (config.server_args or {}).items():
        k2 = k.replace("_", "-")
        if v is None:
            pass
        elif isinstance(v, bool):
            if not v:
                arg_key = f"--no-{k2}"
            else:
                arg_key = f"--{k2}"
            argv.append(arg_key)
        else:
            arg_key = f"--{k2}"
            argv.append(arg_key)
            argv.append(f"{v}")

    server_args = sglang.srt.server_args.prepare_server_args(argv)
    sglang.srt.entrypoints.http_server.launch_server(server_args)


@ray.remote
class SGLangServerSpinupWorker:
    def __init__(self, config: SGLangModelConfig, working_dir: Optional[str], router_dp_rank: int):
        self.config = config
        self.working_dir = working_dir
        self.router_dp_rank = router_dp_rank
        self._server_host = lookup_current_ray_node_ip()
        self._server_port = find_open_port()

        if self.working_dir is not None:
            os.chdir(self.working_dir)

        server_proc = Process(
            target=_start_sglang_server,
            args=(
                self.config,
                self._server_host,
                self._server_port,
                self.router_dp_rank,
            ),
            daemon=False,
        )
        server_proc.start()
        self._server_proc = server_proc

    def _get_ip(self) -> int:
        return self._server_host

    def _get_port(self) -> int:
        return self._server_port


class SGLangModel(SimpleResponsesAPIModel):
    config: SGLangModelConfig

    def model_post_init(self, context):
        working_dir = os.getcwd()

        if self.config.spinup_server:
            self._server_urls = []
            self._server_workers = []
            self._clients = []

            # TODO: support for other parallel sizes.
            server_tp_size = (self.config.server_args or {}).get("tensor_parallel_size", 1)
            server_dp_size = (self.config.server_args or {}).get("data_parallel_size", 1)

            assert server_dp_size == 1

            router_dp_size = max(1, self.config.router_dp_size)

            for router_dp_rank in range(router_dp_size):
                server_worker = spinup_single_ray_gpu_node_worker(
                    SGLangServerSpinupWorker,
                    server_tp_size,
                    config=self.config,
                    working_dir=working_dir,
                    router_dp_rank=router_dp_rank,
                )

                server_ip = ray.get(server_worker._get_ip.remote())
                server_port = ray.get(server_worker._get_port.remote())
                server_url = f"http://{server_ip}:{server_port}/v1"

                self._server_urls.append(server_url)
                self._server_workers.append(server_worker)

                self._clients.append(
                    NeMoGymAsyncOpenAI(
                        base_url=server_url,
                        api_key=self.config.api_key,
                    )
                )

            for server_url in self._server_urls:
                while True:
                    try:
                        _vllm_server_heartbeat(server_url)
                        break
                    except Exception:
                        sleep(3)
                        continue

        else:
            self._server_urls = None
            self._server_workers = None
            self._clients = [
                NeMoGymAsyncOpenAI(
                    base_url=base_url,
                    api_key=self.config.api_key,
                )
                for base_url in self.config.base_url
            ]

        self._session_id_to_client: Dict[str, NeMoGymAsyncOpenAI] = dict()

        # TODO: sglang converter.
        self._converter = VLLMConverter(
            return_token_id_information=self.config.return_token_id_information,
        )

        return super().model_post_init(context)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        # Response Create Params -> Chat Completion Create Params
        chat_completion_create_params = self._converter.responses_to_chat_completion_create_params(body)
        body.model = self.config.model

        # Chat Completion Create Params -> Chat Completion
        chat_completion_response = await self.chat_completions(request, chat_completion_create_params)

        choice = chat_completion_response.choices[0]

        response_output = self._converter.postprocess_chat_response(choice)
        response_output_dicts = [item.model_dump() for item in response_output]

        # Chat Completion -> Response
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=body.model,
            object="response",
            output=response_output_dicts,
            tool_choice=body.tool_choice if "tool_choice" in body else "auto",
            parallel_tool_calls=body.parallel_tool_calls,
            tools=body.tools,
            temperature=body.temperature,
            top_p=body.top_p,
            background=body.background,
            max_output_tokens=body.max_output_tokens,
            max_tool_calls=body.max_tool_calls,
            previous_response_id=body.previous_response_id,
            prompt=body.prompt,
            reasoning=body.reasoning,
            service_tier=body.service_tier,
            text=body.text,
            top_logprobs=body.top_logprobs,
            truncation=body.truncation,
            metadata=body.metadata,
            instructions=body.instructions,
            user=body.user,
        )

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        if self.config.replace_developer_role_with_system:
            for message in body.messages:
                if message["role"] == "developer":
                    message["role"] = "system"

        body_dict = body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.model

        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self._session_id_to_client:
            # There is probably a better way to select the endpoint for this request. But this will do for now.
            client_idx = len(self._session_id_to_client) % len(self._clients)
            client = self._clients[client_idx]
            self._session_id_to_client[session_id] = client
        client = self._session_id_to_client[session_id]

        create_params = body_dict

        if self.config.return_token_id_information:
            create_params |= dict(
                logprobs=True,
                # Typically passed via OpenAI client extra_body.
                return_tokens_as_token_ids=True,
                # TODO add this when NeMo RL upgrades to vLLM 0.10.2 support for prompt token ids
                # For prompt and generation token IDs
                # return_token_ids=True,
                # For prompt token IDs
                # prompt_logprobs=0,
            )

        if self.config.uses_reasoning_parser:
            for message_dict in body_dict["messages"]:
                if message_dict.get("role") != "assistant" or "content" not in message_dict:
                    continue

                content = message_dict["content"]
                if isinstance(content, str):
                    reasoning_matches, remaining_content = self._converter._extract_reasoning_from_content(content)
                    message_dict["content"] = remaining_content
                    if reasoning_matches:
                        message_dict["reasoning_content"] = reasoning_matches[0]
                elif isinstance(content, list):
                    reasoning_content = None
                    for content_item_dict in content:
                        reasoning_matches, remaining_content = self._converter._extract_reasoning_from_content(
                            content_item_dict["text"]
                        )
                        assert reasoning_content is None or not reasoning_matches, (
                            f"Found multiple reasoning matches in a single assistant message content item list!\nMessage: {message_dict}"
                        )

                        # Even though we set the reasoning content already here, we still loop through all the content item dicts for the assert above.
                        content_item_dict["text"] = remaining_content
                        if reasoning_matches:
                            message_dict["reasoning_content"] = reasoning_matches[0]
                elif not content:
                    # No content or content None is a no-op
                    pass
                else:
                    raise NotImplementedError

        try:
            chat_completion_dict = await client.create_chat_completion(**create_params)
        except ClientResponseError as e:
            """
            Example messages for out of context length:

            1. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/entrypoints/openai/serving_engine.py#L914
            ```json
            {"object":"error","message":"This model\'s maximum context length is 32768 tokens. However, you requested 32818 tokens in the messages, Please reduce the length of the messages. None","type":"BadRequestError","param":null,"code":400}
            ```
            2. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/entrypoints/openai/serving_engine.py#L940
            3. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/entrypoints/openai/serving_engine.py#L948
            4. https://github.com/vllm-project/vllm/blob/685c99ee77b4818dcdd15b30fe0e0eff0d5d22ec/vllm/sampling_params.py#L463
            """
            result_content_str = e.response_content.decode()

            is_out_of_context_length = e.status == 400 and (
                "context length" in result_content_str or "max_tokens" in result_content_str
            )
            if is_out_of_context_length:
                return NeMoGymChatCompletion(
                    id="chtcmpl-123",
                    object="chat.completion",
                    created=int(time()),
                    model=self.config.model,
                    choices=[
                        NeMoGymChoice(
                            index=0,
                            finish_reason="stop",
                            message=NeMoGymChatCompletionMessage(
                                role="assistant",
                                content=None,
                                tool_calls=None,
                            ),
                        )
                    ],
                )
            else:
                raise e

        choice_dict = chat_completion_dict["choices"][0]
        if self.config.uses_reasoning_parser:
            reasoning_content = choice_dict["message"].get("reasoning_content")
            if reasoning_content:
                choice_dict["message"].pop("reasoning_content")

                # We wrap this here in think tags for Gym's sake and to return a valid OpenAI Chat Completions response.
                choice_dict["message"]["content"] = self._converter._wrap_reasoning_in_think_tags(
                    [reasoning_content]
                ) + (choice_dict["message"]["content"] or "")
        else:
            assert not choice_dict["message"].get("reasoning_content"), (
                "Please do not use a reasoning parser in vLLM! There is one source of truth for handling data (including reasoning), which is NeMo Gym!"
            )

        if self.config.return_token_id_information:
            log_probs = choice_dict["logprobs"]["content"]
            generation_log_probs = [log_prob["logprob"] for log_prob in log_probs]

            """
            START TODO remove this when NeMo RL upgrades to vLLM 0.10.2 support for prompt token ids
            """
            # Looks like `"token_id:151667"`
            generation_token_ids = [log_prob["token"].removeprefix("token_id:") for log_prob in log_probs]

            # The tokenize endpoint doesn't accept any sampling parameters
            # The only relevant params are model, messages, and tools.
            tokenize_body_dict = dict()
            for key in ("model", "messages", "tools"):
                if key in body_dict:
                    tokenize_body_dict[key] = body_dict[key]

            # The base url has /v1 at the end but vLLM's tokenize endpoint does not have v1, hence the ..
            # I can't believe the path is resolved correctly LOL
            tokenize_response = await client.create_tokenize(**tokenize_body_dict)
            """
            END
            """

            message_dict = choice_dict["message"]
            message_dict.update(
                dict(
                    # TODO add this when NeMo RL upgrades to vLLM 0.10.2 support for prompt token ids
                    # prompt_token_ids=chat_completion_dict["prompt_token_ids"],
                    prompt_token_ids=tokenize_response["tokens"],
                    # generation_token_ids=choice_dict["token_ids"],
                    generation_token_ids=generation_token_ids,
                    generation_log_probs=generation_log_probs,
                )
            )

            # Clean the duplicated information
            choice_dict.pop("logprobs")
            # TODO add this when NeMo RL upgrades to vLLM 0.10.2 support for prompt token ids
            # chat_completion_dict.pop("prompt_token_ids")
            # choice_dict.pop("token_ids")

        return NeMoGymChatCompletion.model_validate(chat_completion_dict)


if __name__ == "__main__":
    SGLangModel.run_webserver()
