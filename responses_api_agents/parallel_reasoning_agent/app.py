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
import asyncio
import json
import logging
from enum import StrEnum
from typing import Any, List, Literal

from fastapi import Request, Response
from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

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
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from responses_api_agents.parallel_reasoning_agent.utils import ParallelReasoningUtils


class ParallelReasoningConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = None
    num_parallelizer: int
    num_executor: int
    keep_executor_prompt: bool = False
    parallel_type: Literal["planner", "rewriter"] = "planner"
    use_identity_rewrite: bool = False
    use_reducer: bool = False
    return_reducer_only: bool = False
    reducer_type: Literal["genselect"] = "genselect"
    reduce_across: Literal["all"] = "all"


class ParallelReasoningRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ParallelReasoningVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class BaseParallelReasoningVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class ParallelReasoningVerifyResponse(BaseModel):
    responses: List[BaseParallelReasoningVerifyResponse]


class Stage(StrEnum):
    PARALLELIZER = "parallelizer"
    EXECUTOR = "executor"
    REDUCER = "reducer"


class ParallelReasoning(SimpleResponsesAPIAgent):
    config: ParallelReasoningConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._setup_logger()

        if self.config.parallel_type not in ["planner", "rewriter"]:
            raise NotImplementedError(
                f"'parallel_type' must be one of ['planner', 'rewriter'], got {self.config.parallel_type}"
            )

        if self.config.use_reducer:
            if self.config.reducer_type not in ["genselect"]:
                raise NotImplementedError(
                    f"'reducer_type' must be one of ['genselect'], got {self.config.reducer_type}"
                )
            if self.config.reduce_across not in ["all"]:
                raise NotImplementedError(f"'reduce_across' must be one of ['all'], got {self.config.reduce_across}")

    def _setup_logger(self):
        # Install rich traceback handler for better error formatting
        install(show_locals=True)

        # Set up rich logging
        console = Console()
        rich_handler = RichHandler(console=console, rich_tracebacks=True, tracebacks_show_locals=True, markup=True)
        rich_handler.setFormatter(logging.Formatter(fmt="%(message)s", datefmt="[%X]"))

        # Configure the parallel reasoning logger
        logger = logging.getLogger("parallel_reasoning")
        logger.setLevel(logging.INFO)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.addHandler(rich_handler)
        logger.propagate = False

        logger.info("[bold green]🚀 Parallel Reasoning Agent initialized[/bold green]")
        logger.info(
            f"[blue]Configuration:[/blue] Parallelizers={self.config.num_parallelizer}, Executors={self.config.num_executor}"
        )

    @property
    def logger(self):
        """Get the configured rich logger for this agent."""
        return logging.getLogger("parallel_reasoning")

    async def get_reducer_response_genselect(
        self, body, executor_responses: List[NeMoGymResponse], executor_cookies: dict
    ):
        if self.config.reduce_across == "all":
            executor_outputs = [
                executor_response.output[0].content[0].text for executor_response in executor_responses
            ]
            reducer_prompt = ParallelReasoningUtils.construct_prompt_genselect_reducer(
                body.input[0].content, executor_outputs
            )
            reducer_body = body.model_copy(
                update={"input": [NeMoGymEasyInputMessage(role="user", content=reducer_prompt)]}
            )
            reducer_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="v1/responses",
                json=reducer_body,
                cookies=executor_cookies,
            )
            reducer_cookies = reducer_response.cookies
            try:
                reducer_response_obj: NeMoGymResponse = NeMoGymResponse.model_validate(await reducer_response.json())
                reducer_response_obj.metadata = {
                    "executor_resp_ids": [executor_response.id for executor_response in executor_responses],
                    "stage": Stage.REDUCER.value,
                }
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(await reducer_response.json())}"
                ) from e

        return reducer_response_obj, reducer_cookies

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> List[NeMoGymResponse]:
        self.logger.debug("[bold cyan]🔄 Starting parallel reasoning process[/bold cyan]")

        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        model_server_cookies = None  # update the cookies on every model response
        resources_server_cookies = request.cookies  # update the cookies on every resources server response

        # CONFIG
        num_parallelizer = self.config.num_parallelizer
        num_executor = self.config.num_executor

        self.logger.debug(
            f"[yellow]📝 Input query:[/yellow] {body.input[0].content[:100]}{'...' if len(body.input[0].content) > 100 else ''}"
        )
        self.logger.debug(
            f"[blue]⚙️  Configuration:[/blue] {num_parallelizer} parallelizers, {num_executor} executors per parallelizers"
        )

        # ------ STAGE 1: PARALLELIZER ------- #
        self.logger.debug("[bold magenta]🧠 Starting parallelizer stage[/bold magenta]")

        async def get_parallelizer_response(parallelizer_prompt: str):
            new_body = body.model_copy(
                update={"input": [NeMoGymEasyInputMessage(role="user", content=parallelizer_prompt)]}
            )
            parallelizer_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
                cookies=model_server_cookies,
            )
            model_response_json = await parallelizer_response.json()
            parallelizer_cookies = parallelizer_response.cookies
            try:
                parallelizer_response_obj: NeMoGymResponse = NeMoGymResponse.model_validate(model_response_json)
                parallelizer_response_obj.metadata = {"stage": Stage.PARALLELIZER.value}
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e

            return parallelizer_response_obj, parallelizer_cookies

        if self.config.parallel_type == "planner":
            parallelizer_prompts = [
                ParallelReasoningUtils.construct_prompt_planner_parallelize(body.input[0].content)
                for _ in range(num_parallelizer)
            ]
        elif self.config.parallel_type == "rewriter":
            parallelizer_prompts = [
                ParallelReasoningUtils.construct_prompt_rewriter_parallelize(body.input[0].content)
                for _ in range(num_parallelizer)
            ]

        self.logger.debug(
            f"[magenta]🔄 Running {len(parallelizer_prompts)} parallelizer requests concurrently[/magenta]"
        )
        parallelizer_results = await asyncio.gather(
            *(get_parallelizer_response(parallelizer_prompt) for parallelizer_prompt in parallelizer_prompts)
        )

        parallelizer_responses = []
        all_parallelizer_cookies = {}
        for i, (parallelizer_response, parallelizer_cookies) in enumerate(parallelizer_results):
            parallelizer_responses.append(parallelizer_response)
            all_parallelizer_cookies.update(parallelizer_cookies)
            self.logger.debug(f"[green]✅ Parallelizer {i + 1} completed[/green] (ID: {parallelizer_response.id})")

        # ------ STAGE 2: EXECUTOR ------- #
        self.logger.debug("[bold orange3]⚡ Starting executor stage[/bold orange3]")

        async def get_executor_response(parallelizer_response: NeMoGymResponse, parallelizer_cookies: dict):
            parallelizer_output = parallelizer_response.output[0].content[0].text

            if self.config.parallel_type == "planner":
                plan = ParallelReasoningUtils.parse_plan(parallelizer_output)[0]
                executor_prompt = ParallelReasoningUtils.construct_prompt_planner_execute(body.input[0].content, plan)
            elif self.config.parallel_type == "rewriter":
                rewrite = ParallelReasoningUtils.parse_rewrite(
                    body.input[0].content, parallelizer_output, use_identity=self.config.use_identity_rewrite
                )[0]
                executor_prompt = ParallelReasoningUtils.construct_prompt_rewriter_execute(
                    body.input[0].content, rewrite
                )

            executor_body = body.model_copy(
                update={
                    "input": [NeMoGymEasyInputMessage(role="user", content=executor_prompt)],
                    "max_output_tokens": 1024,
                }
            )
            executor_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=executor_body,
                cookies=parallelizer_cookies,
            )
            executor_cookies = executor_response.cookies
            try:
                executor_response_obj: NeMoGymResponse = NeMoGymResponse.model_validate(await executor_response.json())
                executor_response_obj.metadata = {
                    "parallelizer_resp_id": parallelizer_response.id,
                    "stage": Stage.EXECUTOR.value,
                }
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(await executor_response.json())}"
                ) from e

            return executor_response_obj, executor_cookies

        # Create all executor tasks
        executor_tasks = []
        for parallelizer_response in parallelizer_responses:
            for _ in range(num_executor):
                executor_tasks.append(get_executor_response(parallelizer_response, all_parallelizer_cookies))

        total_executors = len(executor_tasks)
        self.logger.debug(f"[orange3]🔄 Running {total_executors} executor requests concurrently[/orange3]")

        # Run all executor tasks concurrently
        executor_results = await asyncio.gather(*executor_tasks)

        # Extract executor responses and collect cookies
        executor_responses = []
        all_executor_cookies = {}
        for i, (executor_response, executor_cookies) in enumerate(executor_results):
            executor_responses.append(executor_response)
            all_executor_cookies.update(executor_cookies)
            parallelizer_id = executor_response.metadata.get("parallelizer_resp_id", "unknown")
            self.logger.debug(
                f"[green]✅ Executor {i + 1} completed[/green] (ID: {executor_response.id}, Parallelizer: {parallelizer_id})"
            )

        # ------ STAGE 3: REDUCER ------- #
        if self.config.use_reducer:
            self.logger.debug("[bold blue]⚡ Starting reducer stage[/bold blue]")
            if self.config.redcuer_type == "genselect":
                reducer_responses, all_reducer_cookies = self.get_reducer_response_genselect(
                    body, executor_responses, all_executor_cookies
                )
        else:
            reducer_responses = []
            all_reducer_cookies = {}
        # TODO(jk): appease linter and dispose
        # reducer_responses = []
        # all_reducer_cookies = {}
        # self.logger.debug(f"{reducer_responses} {reducer_body}")

        for k, v in (
            *resources_server_cookies.items(),
            *all_parallelizer_cookies.items(),
            *all_executor_cookies.items(),
            *all_reducer_cookies.items(),
        ):
            response.set_cookie(k, v)

        responses = parallelizer_responses + executor_responses + reducer_responses

        self.logger.debug("[bold green]🎉 Parallel reasoning completed successfully![/bold green]")
        self.logger.debug(
            f"[cyan]📊 Generated {len(parallelizer_responses)} parallelizer responses, {len(executor_responses)} executor responses and {len(reducer_responses)} reducer responses[/cyan]"
        )

        return responses

    async def run(self, request: Request, body: ParallelReasoningRunRequest) -> ParallelReasoningVerifyResponse:
        self.logger.debug("[bold purple]🏃 Starting parallel reasoning run workflow[/bold purple]")

        # Base input
        if isinstance(body.responses_create_params.input, str):
            body.responses_create_params.input = [
                NeMoGymEasyInputMessage(role="user", content=body.responses_create_params.input)
            ]

        cookies = request.cookies

        self.logger.debug("[blue]🌱 Seeding session with resources server[/blue]")
        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        cookies = seed_session_response.cookies
        self.logger.debug("[green]✅ Session seeded successfully[/green]")

        self.logger.debug("[cyan]🔄 Generating responses through parallel reasoning[/cyan]")
        responses = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        responses = await responses.json()
        self.logger.debug(f"[green]✅ Generated {len(responses)} total responses[/green]")

        parallelizer_responses = []
        executor_responses = []
        for response in responses:
            response = NeMoGymResponse.model_validate(response)
            if response.metadata["stage"] == Stage.PARALLELIZER.value:
                parallelizer_responses.append(response)
            elif response.metadata["stage"] == Stage.EXECUTOR.value:
                executor_responses.append(response)

        self.logger.debug(
            f"[magenta]🧠 Categorized responses:[/magenta] {len(parallelizer_responses)} parallelizers, {len(executor_responses)} executors"
        )

        self.logger.debug("[orange3]🔍 Starting verification of executor responses[/orange3]")
        executor_verify_responses = []
        for i, response in enumerate(executor_responses):
            verify_request = ParallelReasoningVerifyRequest.model_validate(
                body.model_dump() | {"response": response.model_dump()}
            )
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
                cookies=cookies,
            )
            try:
                executor_verify_response: BaseParallelReasoningVerifyResponse = (
                    BaseParallelReasoningVerifyResponse.model_validate(await verify_response.json())
                )
                executor_verify_responses.append(executor_verify_response)
                self.logger.debug(
                    f"[green]✅ Executor {i + 1} verified[/green] (Reward: {executor_verify_response.reward})"
                )
            except ValidationError as e:
                self.logger.error(f"[red]❌ Verification failed for executor {i + 1}[/red]")
                raise RuntimeError(
                    f"Received an invalid response from resources server: {json.dumps(await verify_response.json())}"
                ) from e

        # self.logger.debug("[orange3]🔍 Calculate reward of reducer responses[/orange3]")
        # reducer_verify_responses = []
        # for i, response in enumerate(reducer_responses):
        #     verify_request =

        # Aggregate executor rewards for each parallelizer response group.
        # Using the metadata information to aggregate rewards for each parallelizer response
        self.logger.debug("[yellow]📊 Aggregating rewards for parallelizer responses[/yellow]")
        parallelizer_verify_responses = []
        for i, parallelizer_response in enumerate(parallelizer_responses):
            parallelizer_response_group = [
                resp
                for resp in executor_verify_responses
                if resp.response.metadata["parallelizer_resp_id"] == parallelizer_response.id
            ]
            parallelizer_response_group_rewards = [resp.reward for resp in parallelizer_response_group]
            if parallelizer_response_group_rewards:
                parallelizer_reward = sum(parallelizer_response_group_rewards) / len(
                    parallelizer_response_group_rewards
                )
            else:
                parallelizer_reward = 0.0

            self.logger.debug(
                f"[cyan]Parallelizer {i + 1}:[/cyan] {len(parallelizer_response_group_rewards)} executors, avg reward: {parallelizer_reward:.3f}"
            )

            parallelizer_verify_response = BaseParallelReasoningVerifyResponse.model_validate(
                body.model_dump() | {"response": parallelizer_response.model_dump(), "reward": parallelizer_reward}
            )

            # Swap parallelizer input with parallelizer prompt
            if self.config.parallel_type == "planner":
                parallelizer_verify_response.responses_create_params.input[
                    0
                ].content = ParallelReasoningUtils.construct_prompt_planner_parallelize(
                    body.responses_create_params.input[0].content
                )
            elif self.config.parallel_type == "rewriter":
                parallelizer_verify_response.responses_create_params.input[
                    0
                ].content = ParallelReasoningUtils.construct_prompt_rewriter_parallelize(
                    body.responses_create_params.input[0].content
                )
            parallelizer_verify_responses.append(parallelizer_verify_response)

            # Optionally swap executor input with executor prompt
            if self.config.keep_executor_prompt:
                parallelizer_output = parallelizer_response.output[0].content[0].text
                for i in range(len(executor_verify_responses)):
                    resp = executor_verify_responses[i]
                    if not resp.response.metadata["parallelizer_resp_id"] == parallelizer_response.id:
                        continue

                    if self.config.parallel_type == "planner":
                        plan = ParallelReasoningUtils.parse_plan(parallelizer_output)[0]
                        executor_verify_responses[i].responses_create_params.input[
                            0
                        ].content = ParallelReasoningUtils.construct_prompt_planner_execute(
                            body.responses_create_params.input[0].content, plan
                        )
                    elif self.config.parallel_type == "rewriter":
                        rewrite = ParallelReasoningUtils.parse_rewrite(
                            body.responses_create_params.input[0].content,
                            parallelizer_output,
                            use_identity=self.config.use_identity_rewrite,
                        )[0]
                        executor_verify_responses[i].responses_create_params.input[
                            0
                        ].content = ParallelReasoningUtils.construct_prompt_rewriter_execute(
                            body.responses_create_params.input[0].content, rewrite
                        )
            else:
                # Swap the prompt_token_ids of the executor with the original problem.
                # Mostly to backprop on original prompt
                # TODO(jk): find better way to do this than actually using GPU even with max_output_tokens=1
                false_tokenize_body = body.responses_create_params
                false_tokenize_body = false_tokenize_body.model_copy(update={"max_output_tokens": 1})
                false_tokenize_response = await self.server_client.post(
                    server_name=self.config.model_server.name,
                    url_path="/v1/responses",
                    json=false_tokenize_body,
                    cookies=None,
                )
                try:
                    false_tokenize_response_obj: NeMoGymResponse = NeMoGymResponse.model_validate(
                        await false_tokenize_response.json()
                    )
                except ValidationError as e:
                    raise RuntimeError(
                        f"Received an invalid response from model server: {json.dumps(await false_tokenize_response.json())}"
                    ) from e
                original_problem_prompt_token_ids = false_tokenize_response_obj.output[0].prompt_token_ids
                for i in range(len(executor_verify_responses)):
                    resp = executor_verify_responses[i]
                    if not resp.response.metadata["parallelizer_resp_id"] == parallelizer_response.id:
                        continue
                    executor_verify_responses[i].response.output[
                        0
                    ].prompt_token_ids = original_problem_prompt_token_ids

        verify_responses = parallelizer_verify_responses + executor_verify_responses
        parallel_reasoning_verify_responses = ParallelReasoningVerifyResponse(responses=verify_responses)

        self.logger.debug("[bold green]🏆 Parallel reasoning run completed successfully![/bold green]")
        return parallel_reasoning_verify_responses


if __name__ == "__main__":
    ParallelReasoning.run_webserver()
