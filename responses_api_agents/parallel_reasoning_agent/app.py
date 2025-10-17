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
import copy
import json
import logging
from enum import StrEnum
from typing import Any, List, Literal, Optional, Tuple

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
    num_executor: int = 4
    num_reducer: int = 1
    max_output_tokens: Optional[int] = None
    use_reducer: bool = True
    return_reducer_only: bool = False
    reducer_type: Literal["genselect", "genselect_tournament"] = "genselect_tournament"
    reduce_across: Literal["all"] = "all"
    tournament_group_size: int = 4


class ParallelReasoningRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ParallelReasoningVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class BaseParallelReasoningVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class ParallelReasoningVerifyResponse(BaseModel):
    responses: List[BaseParallelReasoningVerifyResponse]


class Stage(StrEnum):
    EXECUTOR = "executor"
    REDUCER = "reducer"
    FINAL_WINNER = "final_winner"


class ParallelReasoning(SimpleResponsesAPIAgent):
    config: ParallelReasoningConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._setup_logger()

        if self.config.use_reducer:
            if self.config.reducer_type not in ["genselect", "genselect_tournament"]:
                raise NotImplementedError(
                    f"'reducer_type' must be one of ['genselect', 'genselect_tournament'], got {self.config.reducer_type}"
                )
            if self.config.reduce_across not in ["all"]:
                raise NotImplementedError(f"'reduce_across' must be one of ['all'], got {self.config.reduce_across}")
        else:
            if self.config.return_reducer_only:
                raise NotImplementedError("'return_reducer_only' must be used with 'use_reducer' !")

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
            f"[blue]Configuration:[/blue] Executors={self.config.num_executor}, Reducers={self.config.num_reducer}"
        )

    @property
    def logger(self):
        """Get the configured rich logger for this agent."""
        return logging.getLogger("parallel_reasoning")

    def get_contents(self, response: NeMoGymResponse):
        contents = []
        for output in response.output:
            if "content" in output.model_dump():
                contents.append(output.content[0].text)
        return "\n".join(contents)

    async def get_reducer_response_genselect(
        self, body, executor_responses: List[NeMoGymResponse], executor_cookies: dict
    ) -> Tuple[NeMoGymResponse, List[NeMoGymResponse], dict]:
        if self.config.reduce_across == "all":
            executor_outputs = [self.get_contents(executor_response) for executor_response in executor_responses]
            reducer_prompt = ParallelReasoningUtils.construct_prompt_genselect_reducer(
                self.config, body.input[0].content, executor_outputs
            )
            reducer_body = body.model_copy(
                update={"input": [NeMoGymEasyInputMessage(role="user", content=reducer_prompt)]}
            )
            if self.config.max_output_tokens is not None:
                reducer_body.max_output_tokens = self.config.max_output_tokens

            async def process_reducer_task():
                reducer_response = await self.server_client.post(
                    server_name=self.config.model_server.name,
                    url_path="/v1/responses",
                    json=reducer_body,
                    cookies=executor_cookies,
                )
                reducer_cookies = reducer_response.cookies
                try:
                    reducer_response_obj = await reducer_response.json()
                    reducer_response_obj: NeMoGymResponse = NeMoGymResponse.model_validate(reducer_response_obj)
                    reducer_response_obj.metadata = {
                        "executor_resp_ids": json.dumps(
                            [executor_response.id for executor_response in executor_responses]
                        ),
                        "stage": Stage.REDUCER.value,
                        "request": reducer_body.model_dump_json(),
                    }
                except ValidationError as e:
                    raise RuntimeError(
                        f"Received an invalid response from model server: {json.dumps(await reducer_response.json())}"
                    ) from e
                return reducer_response_obj, reducer_cookies

            reducer_tasks = [process_reducer_task() for _ in range(self.config.num_reducer)]
            reducer_results = await asyncio.gather(*reducer_tasks)

            reducer_responses = []
            all_reducer_cookies = {}
            for i, (reducer_response, reducer_cookies) in enumerate(reducer_results):
                reducer_responses.append(reducer_response)
                all_reducer_cookies.update(reducer_cookies)
                self.logger.debug(f"[green]✅ Reducer ({i + 1} / {self.config.num_reducer}) completed[/green]")

        return reducer_responses, reducer_cookies

    async def get_reducer_response_genselect_tournament(
        self,
        body,
        executor_responses: List[NeMoGymResponse],
        executor_cookies: dict,
        n: int,
    ) -> Tuple[NeMoGymResponse, List[NeMoGymResponse], dict]:
        """
        Tournament-style selection where executor_responses compete in groups of size n.
        Winners advance to the next round until a final winner is selected.

        Args:
            body: The original request body
            executor_responses: List of responses to compete
            executor_cookies: Cookies from executor
            n: Group size for each tournament round

        Returns:
            Final winner response and cookies
        """
        current_round_responses = executor_responses
        round_num = 0
        all_reducer_responses = []  # Track all reducer responses from all rounds

        while len(current_round_responses) > 1:
            round_num += 1
            self.logger.info(
                f"[blue]🏆 Tournament Round {round_num}: {len(current_round_responses)} contestants[/blue]"
            )

            # Group responses into batches of size n
            groups = []
            for i in range(0, len(current_round_responses), n):
                groups.append(current_round_responses[i : i + n])

            # Process each group in parallel
            async def process_group(group_responses: List[NeMoGymResponse], group_idx: int):
                # Extract outputs for this group
                # group_outputs = [response.output[0].content[0].text for response in group_responses]
                group_outputs = [self.get_contents(response) for response in group_responses]

                # Construct reducer prompt for this group
                reducer_prompt = ParallelReasoningUtils.construct_prompt_genselect_reducer(
                    self.config, body.input[0].content, group_outputs
                )
                reducer_body = body.model_copy(
                    update={"input": [NeMoGymEasyInputMessage(role="user", content=reducer_prompt)]}
                )
                if self.config.max_output_tokens is not None:
                    reducer_body.max_output_tokens = self.config.max_output_tokens

                # Get reducer response
                reducer_response = await self.server_client.post(
                    server_name=self.config.model_server.name,
                    url_path="/v1/responses",
                    json=reducer_body,
                    cookies=executor_cookies,
                )
                reducer_cookies = reducer_response.cookies

                try:
                    reducer_response_obj = await reducer_response.json()
                    reducer_response_obj: NeMoGymResponse = NeMoGymResponse.model_validate(reducer_response_obj)
                    reducer_response_obj.metadata = {
                        "executor_resp_ids": json.dumps([resp.id for resp in group_responses]),
                        "stage": f"{Stage.REDUCER.value}_round_{round_num}_group_{group_idx}",
                        "request": reducer_body.model_dump_json(),
                    }
                except ValidationError as e:
                    raise RuntimeError(
                        f"Received an invalid response from model server: {json.dumps(await reducer_response.json())}"
                    ) from e

                # Parse the winner IDX from reducer output
                # reducer_text = reducer_response_obj.output[0].content[0].text
                reducer_text = self.get_contents(reducer_response_obj)
                winner_idx = ParallelReasoningUtils.parse_genselect_reduction(reducer_text)

                if winner_idx is None or winner_idx < 0 or winner_idx >= len(group_responses):
                    self.logger.warning(
                        f"[yellow]⚠️  Group {group_idx}: Invalid winner IDX {winner_idx}, defaulting to 0[/yellow]"
                    )
                    winner_idx = 0

                winner_response = group_responses[winner_idx]
                self.logger.debug(
                    f"[green]✅ Group {group_idx} winner: IDX {winner_idx} (ID: {winner_response.id})[/green]"
                )
                reducer_response_obj.metadata.update(
                    {
                        "winner_idx": str(winner_idx),
                        "winner_resp_id": winner_response.id,
                    }
                )

                return winner_response, reducer_response_obj, reducer_cookies

            # Run all groups in parallel
            group_tasks = [process_group(group, group_idx) for group_idx, group in enumerate(groups)]
            group_results = await asyncio.gather(*group_tasks)

            # Extract winners and reducer responses for next round
            current_round_responses = []
            for winner_response, reducer_response_obj, reducer_cookies in group_results:
                current_round_responses.append(winner_response)
                all_reducer_responses.append(reducer_response_obj)

            self.logger.info(
                f"[green]✅ Round {round_num} complete: {len(current_round_responses)} winners advance[/green]"
            )

        # Final winner
        final_winner = copy.deepcopy(current_round_responses[0])
        final_winner.metadata["stage"] = Stage.FINAL_WINNER.value
        self.logger.info(f"[bold green]🏆 Tournament Champion: Response ID {final_winner.id}[/bold green]")

        # Return all reducer responses (one per group across all rounds) and final cookies
        return final_winner, all_reducer_responses, reducer_cookies

    async def get_reducer_verify_response_genselect(
        self,
        reducer_verify_request: ParallelReasoningVerifyRequest,
        executor_verify_responses: List[BaseParallelReasoningVerifyResponse],
    ) -> BaseParallelReasoningVerifyResponse:
        """
        Genselect Reducer Reward.
        We add in the reward calculation in the agent instead of adding a separate resource server as the reward calculation is tied to the executor.
        """
        # Check for 'correct' answers in executors
        executor_rewards = [executor_verify_response.reward for executor_verify_response in executor_verify_responses]
        executor_max_reward_idxs = [
            i for (i, reward) in enumerate(executor_rewards) if (reward == max(executor_rewards) and reward > 0)
        ]

        # Extract genselect answer
        # reducer_text = reducer_verify_request.response.output[0].content[0].text
        reducer_text = self.get_contents(reducer_verify_request.response)
        reducer_answer_idx = ParallelReasoningUtils.parse_genselect_reduction(reducer_text)

        if reducer_answer_idx in executor_max_reward_idxs:
            # Valid selection
            reducer_reward = 1.0
        else:
            # Even if impossible, score is 0.
            reducer_reward = 0.0

        # Construct verify response
        verify_request_fields = {
            "reward": reducer_reward,
            "allowed_answers": executor_max_reward_idxs,
            "predicted_answer": reducer_answer_idx,
        }

        reducer_verify_response = reducer_verify_request.model_dump()
        reducer_verify_response.update(verify_request_fields)
        reducer_verify_response = BaseParallelReasoningVerifyResponse.model_validate(reducer_verify_response)

        return reducer_verify_response

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

        resources_server_cookies = request.cookies  # update the cookies on every resources server response

        # CONFIG
        num_executor = self.config.num_executor

        self.logger.debug(
            f"[yellow]📝 Input query:[/yellow] {body.input[0].content[:100]}{'...' if len(body.input[0].content) > 100 else ''}"
        )
        self.logger.debug(f"[blue]⚙️  Configuration:[/blue] {num_executor} executors")

        # ------ STAGE 1: EXECUTOR ------- #
        self.logger.debug("[bold orange3]⚡ Starting executor stage[/bold orange3]")

        async def get_executor_response() -> Tuple[NeMoGymResponse, dict]:
            executor_prompt = body.input[0].content
            executor_body = body.model_copy(
                update={
                    "input": [NeMoGymEasyInputMessage(role="user", content=executor_prompt)],
                }
            )
            if self.config.max_output_tokens is not None:
                executor_body.max_output_tokens = self.config.max_output_tokens

            executor_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=executor_body,
                cookies=None,
            )
            executor_cookies = executor_response.cookies
            try:
                executor_response_obj: NeMoGymResponse = NeMoGymResponse.model_validate(await executor_response.json())
                executor_response_obj.metadata = {
                    "stage": Stage.EXECUTOR.value,
                }
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(await executor_response.json())}"
                ) from e

            return executor_response_obj, executor_cookies

        # Create all executor tasks
        executor_tasks = [get_executor_response() for _ in range(num_executor)]
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

        # ------ STAGE 2: REDUCER ------- #
        if self.config.use_reducer:
            self.logger.debug("[bold blue]⚡ Starting reducer stage[/bold blue]")
            if self.config.reducer_type == "genselect":
                reducer_responses, all_reducer_cookies = await self.get_reducer_response_genselect(
                    body, executor_responses, all_executor_cookies
                )
                final_winner_response = None
            elif self.config.reducer_type == "genselect_tournament":
                (
                    final_winner_response,
                    reducer_responses,
                    all_reducer_cookies,
                ) = await self.get_reducer_response_genselect_tournament(
                    body, executor_responses, all_executor_cookies, self.config.tournament_group_size
                )
        else:
            final_winner_response = None
            reducer_responses = []
            all_reducer_cookies = {}

        for k, v in (
            *resources_server_cookies.items(),
            *all_executor_cookies.items(),
            *all_reducer_cookies.items(),
        ):
            response.set_cookie(k, v)

        responses = executor_responses + reducer_responses
        if final_winner_response is not None:
            responses.append(final_winner_response)

        self.logger.debug("[bold green]🎉 Parallel reasoning completed successfully![/bold green]")
        self.logger.debug(
            f"[cyan]📊 Generated {len(executor_responses)} executor responses and {len(reducer_responses)} reducer responses[/cyan]"
        )

        return responses

    async def _run(self, request: Request, body: ParallelReasoningRunRequest) -> ParallelReasoningVerifyResponse:
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

        executor_responses = []
        reducer_responses = []
        final_winner_responses = []
        for response in responses:
            response = NeMoGymResponse.model_validate(response)
            if response.metadata["stage"] == Stage.EXECUTOR.value:
                executor_responses.append(response)
            elif response.metadata["stage"] == Stage.REDUCER.value:
                reducer_responses.append(response)
            elif response.metadata["stage"] == Stage.FINAL_WINNER.value:
                final_winner_responses.append(response)
        if len(final_winner_responses) > 1:
            self.logger.error(
                f"There should be at most one final winner response, but got {len(final_winner_responses)}"
            )
            final_winner_responses = []

        self.logger.debug(
            f"[magenta]🧠 Categorized responses:[/magenta] {len(executor_responses)} executors, {len(reducer_responses)} reducers, {len(final_winner_responses)} final winners"
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
                executor_verify_response = executor_verify_response.model_copy(
                    update={"question": body.responses_create_params.input[0].content}
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

        if self.config.use_reducer:
            self.logger.debug("[orange3]🔍 Calculate reward of reducer responses[/orange3]")
            reducer_verify_requests = []
            for reducer_response in reducer_responses:
                reducer_response_request = body.model_copy(
                    update={"responses_create_params": json.loads(reducer_response.metadata.pop("request"))}
                )
                verify_request = ParallelReasoningVerifyRequest.model_validate(
                    reducer_response_request.model_dump() | {"response": reducer_response.model_dump()}
                )
                reducer_verify_requests.append(verify_request)
            if self.config.reducer_type == "genselect":
                reducer_verify_requests = [
                    self.get_reducer_verify_response_genselect(request, executor_verify_responses)
                    for request in reducer_verify_requests
                ]
            reducer_verify_responses = await asyncio.gather(*reducer_verify_requests)
        else:
            reducer_verify_responses = []

        final_winner_verify_responses = []
        if len(final_winner_responses) > 0:
            for each_executor_verify_response in executor_verify_responses:
                if each_executor_verify_response.response.id == final_winner_responses[0].id:
                    final_winner_verify_response = BaseParallelReasoningVerifyResponse.model_validate(
                        body.model_dump()
                        | {
                            "response": final_winner_responses[0].model_dump(),
                            "reward": each_executor_verify_response.reward,
                        }
                    )
                    final_winner_verify_responses.append(final_winner_verify_response)

        if self.config.return_reducer_only:
            verify_responses = reducer_verify_responses
        else:
            verify_responses = executor_verify_responses + reducer_verify_responses + final_winner_verify_responses
        parallel_reasoning_verify_responses = ParallelReasoningVerifyResponse(responses=verify_responses)

        self.logger.debug("[bold green]🏆 Parallel reasoning run completed successfully![/bold green]")
        return parallel_reasoning_verify_responses

    async def run(self, request: Request, body: ParallelReasoningRunRequest) -> ParallelReasoningVerifyResponse:
        try:
            parallel_reasoning_verify_responses = await self._run(request, body)
        except Exception as e:
            self.logger.error(f"Caught Error: {e}")
            parallel_reasoning_verify_responses = ParallelReasoningVerifyResponse(responses=[])

        return parallel_reasoning_verify_responses


if __name__ == "__main__":
    ParallelReasoning.run_webserver()
