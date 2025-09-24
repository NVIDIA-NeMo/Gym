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
from typing import Any, List

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
    num_planner: int
    num_executor: int


class ParallelReasoningRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ParallelReasoningVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class BaseParallelReasoningVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class ParallelReasoningVerifyResponse(BaseModel):
    responses: List[BaseParallelReasoningVerifyResponse]


class Stage(StrEnum):
    PLANNER = "planner"
    EXECUTOR = "executor"


class ParallelReasoning(SimpleResponsesAPIAgent):
    config: ParallelReasoningConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._setup_logger()

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
            f"[blue]Configuration:[/blue] Planners={self.config.num_planner}, Executors={self.config.num_executor}"
        )

    @property
    def logger(self):
        """Get the configured rich logger for this agent."""
        return logging.getLogger("parallel_reasoning")

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> List[NeMoGymResponse]:
        self.logger.info("[bold cyan]🔄 Starting parallel reasoning process[/bold cyan]")

        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        model_server_cookies = None  # update the cookies on every model response
        resources_server_cookies = request.cookies  # update the cookies on every resources server response

        # CONFIG
        num_planner = self.config.num_planner
        num_executor = self.config.num_executor

        self.logger.info(
            f"[yellow]📝 Input query:[/yellow] {body.input[0].content[:100]}{'...' if len(body.input[0].content) > 100 else ''}"
        )
        self.logger.info(
            f"[blue]⚙️  Configuration:[/blue] {num_planner} planners, {num_executor} executors per planner"
        )

        # PLANNER STAGE
        self.logger.info("[bold magenta]🧠 Starting planner stage[/bold magenta]")

        async def get_planner_response(planner_prompt: str):
            new_body = body.model_copy(
                update={"input": [NeMoGymEasyInputMessage(role="user", content=planner_prompt)]}
            )
            planner_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=new_body,
                cookies=model_server_cookies,
            )
            model_response_json = planner_response.json()
            planner_cookies = planner_response.cookies
            try:
                planner_response_obj: NeMoGymResponse = NeMoGymResponse.model_validate(model_response_json)
                planner_response_obj.metadata = {"stage": Stage.PLANNER.value}
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e

            return planner_response_obj, planner_cookies

        planner_prompts = [
            ParallelReasoningUtils.construct_planner_prompt(body.input[0].content) for _ in range(num_planner)
        ]

        self.logger.info(f"[magenta]🔄 Running {len(planner_prompts)} planner requests concurrently[/magenta]")
        planner_results = await asyncio.gather(
            *(get_planner_response(planner_prompt) for planner_prompt in planner_prompts)
        )

        planner_responses = []
        all_planner_cookies = {}
        for i, (planner_response, planner_cookies) in enumerate(planner_results):
            planner_responses.append(planner_response)
            all_planner_cookies.update(planner_cookies)
            self.logger.info(f"[green]✅ Planner {i + 1} completed[/green] (ID: {planner_response.id})")

        # EXECUTOR STAGE
        self.logger.info("[bold orange3]⚡ Starting executor stage[/bold orange3]")

        async def get_executor_response(planner_response: NeMoGymResponse, planner_cookies: dict):
            planner_output = planner_response.output[0].content[0].text
            plan = ParallelReasoningUtils.parse_plan(planner_output)[0]
            executor_prompt = ParallelReasoningUtils.construct_executor_prompt(body.input[0].content, plan)
            executor_body = body.model_copy(
                update={"input": [NeMoGymEasyInputMessage(role="user", content=executor_prompt)]}
            )
            executor_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=executor_body,
                cookies=planner_cookies,
            )
            executor_cookies = executor_response.cookies
            try:
                executor_response_obj: NeMoGymResponse = NeMoGymResponse.model_validate(executor_response.json())
                executor_response_obj.metadata = {
                    "planner_resp_id": planner_response.id,
                    "stage": Stage.EXECUTOR.value,
                }
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(executor_response.json())}"
                ) from e

            return executor_response_obj, executor_cookies

        # Create all executor tasks
        executor_tasks = []
        for planner_response in planner_responses:
            for _ in range(num_executor):
                executor_tasks.append(get_executor_response(planner_response, all_planner_cookies))

        total_executors = len(executor_tasks)
        self.logger.info(f"[orange3]🔄 Running {total_executors} executor requests concurrently[/orange3]")

        # Run all executor tasks concurrently
        executor_results = await asyncio.gather(*executor_tasks)

        # Extract executor responses and collect cookies
        executor_responses = []
        all_executor_cookies = {}
        for i, (executor_response, executor_cookies) in enumerate(executor_results):
            executor_responses.append(executor_response)
            all_executor_cookies.update(executor_cookies)
            planner_id = executor_response.metadata.get("planner_resp_id", "unknown")
            self.logger.info(
                f"[green]✅ Executor {i + 1} completed[/green] (ID: {executor_response.id}, Planner: {planner_id})"
            )

        for k, v in (*resources_server_cookies.items(), *all_planner_cookies.items(), *all_executor_cookies.items()):
            response.set_cookie(k, v)

        responses = planner_responses + executor_responses

        self.logger.info("[bold green]🎉 Parallel reasoning completed successfully![/bold green]")
        self.logger.info(
            f"[cyan]📊 Generated {len(planner_responses)} planner responses and {len(executor_responses)} executor responses[/cyan]"
        )

        return responses

    async def run(self, request: Request, body: ParallelReasoningRunRequest) -> ParallelReasoningVerifyResponse:
        self.logger.info("[bold purple]🏃 Starting parallel reasoning run workflow[/bold purple]")

        if isinstance(body.responses_create_params.input, str):
            body.responses_create_params.input = [
                NeMoGymEasyInputMessage(role="user", content=body.responses_create_params.input)
            ]

        cookies = request.cookies

        self.logger.info("[blue]🌱 Seeding session with resources server[/blue]")
        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        cookies = seed_session_response.cookies
        self.logger.info("[green]✅ Session seeded successfully[/green]")

        self.logger.info("[cyan]🔄 Generating responses through parallel reasoning[/cyan]")
        responses = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        responses = responses.json()
        self.logger.info(f"[green]✅ Generated {len(responses)} total responses[/green]")

        planner_responses = []
        executor_responses = []
        for response in responses:
            if response["metadata"]["stage"] == Stage.PLANNER.value:
                planner_responses.append(response)
            elif response["metadata"]["stage"] == Stage.EXECUTOR.value:
                executor_responses.append(response)

        self.logger.info(
            f"[magenta]🧠 Categorized responses:[/magenta] {len(planner_responses)} planners, {len(executor_responses)} executors"
        )

        self.logger.info("[orange3]🔍 Starting verification of executor responses[/orange3]")
        executor_verify_responses = []
        for i, response in enumerate(executor_responses):
            verify_request = ParallelReasoningVerifyRequest.model_validate(body.model_dump() | {"response": response})
            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=verify_request.model_dump(),
                cookies=cookies,
            )
            try:
                executor_verify_response: BaseParallelReasoningVerifyResponse = (
                    BaseParallelReasoningVerifyResponse.model_validate(verify_response.json())
                )
                executor_verify_responses.append(executor_verify_response)
                self.logger.info(
                    f"[green]✅ Executor {i + 1} verified[/green] (Reward: {executor_verify_response.reward})"
                )
            except ValidationError as e:
                self.logger.error(f"[red]❌ Verification failed for executor {i + 1}[/red]")
                raise RuntimeError(
                    f"Received an invalid response from resources server: {json.dumps(verify_response.json())}"
                ) from e

        # Aggregate executor rewards for each planner response group.
        # Using the metadata information to aggregate rewards for each planner response
        self.logger.info("[yellow]📊 Aggregating rewards for planner responses[/yellow]")
        planner_verify_responses = []
        for i, planner_response in enumerate(planner_responses):
            planner_response_group = [
                resp
                for resp in executor_verify_responses
                if resp.response.metadata["planner_resp_id"] == planner_response["id"]
            ]
            planner_response_group_rewards = [resp.reward for resp in planner_response_group]
            if planner_response_group_rewards:
                planner_reward = sum(planner_response_group_rewards) / len(planner_response_group_rewards)
            else:
                planner_reward = 0.0

            self.logger.info(
                f"[cyan]Planner {i + 1}:[/cyan] {len(planner_response_group_rewards)} executors, avg reward: {planner_reward:.3f}"
            )

            planner_verify_response = BaseParallelReasoningVerifyResponse.model_validate(
                body.model_dump() | {"response": planner_response, "reward": planner_reward}
            )
            planner_verify_response.responses_create_params.input[
                0
            ].content = ParallelReasoningUtils.construct_planner_prompt(body.responses_create_params.input[0].content)
            planner_verify_responses.append(planner_verify_response)

        verify_responses = planner_verify_responses + executor_verify_responses
        parallel_reasoning_verify_responses = ParallelReasoningVerifyResponse(responses=verify_responses)

        self.logger.info("[bold green]🏆 Parallel reasoning run completed successfully![/bold green]")
        return parallel_reasoning_verify_responses


if __name__ == "__main__":
    ParallelReasoning.run_webserver()
