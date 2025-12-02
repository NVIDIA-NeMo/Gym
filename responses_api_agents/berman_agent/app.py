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
from typing import List, Tuple

from fastapi import Request, Response
from pydantic import ConfigDict, ValidationError

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


INITIAL_INSTRUCTION_GENERATION_PROMPT = """You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Here are training examples that demonstrate a transformation pattern:

{training_examples}

Your task is to describe the transformation rule in clear, step-by-step instructions. These instructions should explain the PATTERN that converts any input grid into its corresponding output grid.

Your instructions must:
- Describe the general pattern, not specific examples
- Be clear enough that someone could follow them to transform new inputs
- Work consistently across ALL training examples shown above

Write your instructions as a clear, step-by-step process in plain English."""

INDIVIDUAL_REVISION_PROMPT_TEMPLATE = """Your previous instructions were applied to solve this puzzle, but they did not produce the correct output.

Original Task:
{task_description}

Your Previous Instructions:
{instruction}

Output Generated from Your Instructions:
{output}

Expected Correct Output:
{ground_truth}

Differences (where your output was wrong):
{diff}

Based on this feedback, provide updated instructions that correctly describe the transformation pattern. Your revised instructions must:
- Fix the specific errors you observe
- Still work correctly for ALL training examples
- Remain clear, intuitive, and general

Analyze the differences between your incorrect output and the correct output to understand the true pattern, then write improved instructions."""

POOLED_REVISION_PROMPT_TEMPLATE = """Original Task:
{task_description}

Multiple expert puzzle solvers have attempted to describe the transformation pattern. Each attempt captured some aspects correctly but failed in other ways.

Below are the attempted instructions and their results:

{pooled_context}

Your task is to analyze why each approach partially failed and synthesize a complete, correct set of instructions.

By examining multiple flawed attempts, you can:
- Identify what each attempt got right
- Understand what each approach missed
- Recognize common misconceptions about the pattern
- Build comprehensive instructions that avoid all these pitfalls

Study the patterns of success and failure across all attempts, then write instructions that correctly describe the complete transformation rule that works for ALL training examples.

Your final instructions should be clear, intuitive, and capture the true underlying pattern."""


class BermanAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    num_initial_candidates: int = 30
    num_top_for_revision: int = 5
    num_pooled_candidates: int = 5
    initial_temperature: float = 0.9

    initial_prompt: str = INITIAL_INSTRUCTION_GENERATION_PROMPT
    individual_revision_prompt: str = INDIVIDUAL_REVISION_PROMPT_TEMPLATE
    pooled_revision_prompt: str = POOLED_REVISION_PROMPT_TEMPLATE


class BermanAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class BermanAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class BermanAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class BermanAgent(SimpleResponsesAPIAgent):
    config: BermanAgentConfig

    async def _generate_instruction_response(
        self,
        instruction_prompt: str,
        cookies: dict,
        temperature: float = None,
    ) -> Tuple[NeMoGymResponse, dict]:
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[NeMoGymEasyInputMessage(role="user", content=instruction_prompt)],
            temperature=temperature or self.config.initial_temperature,
        )

        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
            cookies=cookies,
        )

        model_response_json = await model_response.json()
        cookies = model_response.cookies

        try:
            response = NeMoGymResponse.model_validate(model_response_json)
        except ValidationError as e:
            raise RuntimeError(
                f"Received an invalid response from model server: {json.dumps(model_response_json)}"
            ) from e

        return response, cookies

    async def _verify_response(
        self,
        response: NeMoGymResponse,
        run_request: BermanAgentRunRequest,
        cookies: dict,
    ) -> Tuple[float, dict, dict]:
        """Return reward, verify_response_data, cookies"""
        verify_request = BermanAgentVerifyRequest.model_validate(
            run_request.model_dump() | {"response": response.model_dump()}
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )

        cookies = verify_response.cookies
        verify_data = await verify_response.json()
        reward = verify_data.get("reward", 0.0)

        return reward, verify_data, cookies

    def _format_grid(self, grid: List[List[int]]) -> str:
        if not grid:
            return "[]"
        return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

    def _generate_diff(self, predicted: List[List[int]], expected: List[List[int]]) -> str:
        """ASCII diff between predicted and expected grids"""
        if not predicted or not expected:
            return "Unable to generate diff"

        diff_lines = []
        max_rows = max(len(predicted), len(expected))

        for i in range(max_rows):
            pred_row = predicted[i] if i < len(predicted) else []
            exp_row = expected[i] if i < len(expected) else []

            if pred_row != exp_row:
                diff_lines.append(f"Row {i}:")
                diff_lines.append(f"  Predicted: {pred_row}")
                diff_lines.append(f"  Expected:  {exp_row}")

        return "\n".join(diff_lines) if diff_lines else "No differences"

    def _calculate_grid_accuracy(self, predicted: List[List[int]], expected: List[List[int]]) -> float:
        """Accuracy = fraction of correct cells"""  # should this be all or none?
        if not predicted or not expected:
            return 0.0

        if len(predicted) != len(expected):
            return 0.0

        total_cells = 0
        correct_cells = 0

        for pred_row, exp_row in zip(predicted, expected):
            if len(pred_row) != len(exp_row):
                return 0.0
            total_cells += len(pred_row)
            correct_cells += sum(1 for p, e in zip(pred_row, exp_row) if p == e)

        return correct_cells / total_cells if total_cells > 0 else 0.0

    def _extract_text_from_response(self, response: NeMoGymResponse) -> str:
        texts = []
        for output in response.output:
            if hasattr(output, "type") and output.type == "message" and hasattr(output, "content"):
                content = output.content
                if isinstance(content, list):
                    for part in content:
                        text = getattr(part, "text", None)
                        if isinstance(text, str):
                            texts.append(text)
                elif isinstance(content, str):
                    texts.append(content)
        return "\n".join(texts)

    async def _evaluate_on_training(
        self,
        generated_instructions: str,
        train_examples: List[dict],
        cookies: dict,
        temperature: float = 0.0,
    ) -> Tuple[float, List[dict]]:
        """
        Evaluate generated natural language grid pattern instructions on all training examples.
        Returns: average_accuracy, list of per-example results
        """
        results = []

        for idx, train_ex in enumerate(train_examples):
            train_input = train_ex["input"]
            train_output = train_ex["output"]

            train_input_str = self._format_grid(train_input)
            application_prompt = f"""Follow these instructions to transform the input grid:

Instructions:
{generated_instructions}

Input Grid:
{train_input_str}

Apply the instructions above to transform this input grid. Provide your solution as a 2D array inside \\boxed{{}} in this exact format: \\boxed{{[[row1],[row2],...]]}}"""

            response, _ = await self._generate_instruction_response(
                application_prompt, cookies, temperature=temperature
            )

            assistant_text = self._extract_text_from_response(response)

            from resources_servers.arc_agi.app import _parse_grid

            predicted_grid = _parse_grid(assistant_text)

            accuracy = self._calculate_grid_accuracy(predicted_grid, train_output)

            results.append(
                {
                    "train_example_idx": idx,
                    "input": train_input,
                    "expected_output": train_output,
                    "predicted_output": predicted_grid,
                    "accuracy": accuracy,
                }
            )

        avg_accuracy = sum(r["accuracy"] for r in results) / len(results) if results else 0.0

        return avg_accuracy, results

    async def _generate_and_evaluate_instruction(
        self,
        instruction_prompt: str,
        train_examples: List[dict],
        cookies: dict,
        temperature: float,
        candidate_id: int,
    ) -> Tuple[str, float, List[dict]]:
        """
        Generate instruction and evaluate on training
        Returns: generated_instructions, avg_train_accuracy, train_results
        """
        response, _ = await self._generate_instruction_response(instruction_prompt, cookies, temperature=temperature)

        generated_instructions = self._extract_text_from_response(response)

        avg_accuracy, train_results = await self._evaluate_on_training(
            generated_instructions, train_examples, cookies, temperature=0.0
        )

        print(f"  Candidate {candidate_id}: train_accuracy={avg_accuracy:.2%}")
        return (generated_instructions, avg_accuracy, train_results)

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """only here to satisfy the SimpleResponsesAPIAgent base class. unused"""
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
            cookies={},
        )

        model_response_json = await model_response.json()

        try:
            return NeMoGymResponse.model_validate(model_response_json)
        except ValidationError as e:
            raise RuntimeError(
                f"Received an invalid response from model server: {json.dumps(model_response_json)}"
            ) from e

    async def run(self, request: Request, body: BermanAgentRunRequest) -> BermanAgentVerifyResponse:
        """Main entry point to berman arc-agi agent"""
        cookies = request.cookies

        train_examples = body.train

        training_examples_str = ""
        for idx, ex in enumerate(train_examples):
            training_examples_str += f"\nExample {idx + 1}:\nInput:\n{self._format_grid(ex['input'])}\n\nOutput:\n{self._format_grid(ex['output'])}\n"

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        cookies = seed_session_response.cookies

        # STAGE 1: Generate N instruction candidates
        print(f"Stage 1: Generating {self.config.num_initial_candidates} instruction candidates in parallel...")

        instruction_gen_prompt = self.config.initial_prompt.format(training_examples=training_examples_str)

        tasks = [
            self._generate_and_evaluate_instruction(
                instruction_gen_prompt, train_examples, cookies, self.config.initial_temperature, i + 1
            )
            for i in range(self.config.num_initial_candidates)
        ]
        candidates = list(await asyncio.gather(*tasks))  # List of (instructions, accuracy, train_results)

        # Sort by training accuracy
        candidates.sort(key=lambda x: x[1], reverse=True)
        print(f"Stage 1 complete. Best training accuracy: {candidates[0][1]:.2%}")

        # Check for perfect solution on training - if found, submit
        best_instructions, best_accuracy, best_train_results = candidates[0]

        if best_accuracy < 1.0:
            # STAGE 2: Individual revision of top K instructions
            print(
                f"\nStage 2: Individual revision of top {self.config.num_top_for_revision} instructions in parallel..."
            )

            top_k = candidates[: self.config.num_top_for_revision]

            async def _revise_candidate(idx, instructions, accuracy, train_results):
                feedback_parts = []
                for result in train_results:
                    ex_idx = result["train_example_idx"]
                    predicted = result["predicted_output"]
                    expected = result["expected_output"]
                    acc = result["accuracy"]
                    diff = self._generate_diff(predicted, expected)

                    feedback_parts.append(f"""
Training Example {ex_idx + 1} (accuracy: {acc:.1%}):
Input:
{self._format_grid(result["input"])}

Your Output:
{self._format_grid(predicted)}

Correct Output:
{self._format_grid(expected)}

Differences:
{diff}
""")

                training_feedback = "\n".join(feedback_parts)

                revision_prompt = self.config.individual_revision_prompt.format(
                    task_description=training_examples_str,
                    instruction=instructions,
                    output=training_feedback,
                    ground_truth="",
                    diff="",
                )

                # Generate revised instructions
                revised_response, _ = await self._generate_instruction_response(
                    revision_prompt, cookies, temperature=0.7
                )
                revised_instructions = self._extract_text_from_response(revised_response)

                # Evaluate revised instructions on training
                revised_accuracy, revised_train_results = await self._evaluate_on_training(
                    revised_instructions, train_examples, cookies, temperature=0.0
                )

                print(f"  Revised {idx + 1}: train_accuracy={revised_accuracy:.2%} (was {accuracy:.2%})")
                return (revised_instructions, revised_accuracy, revised_train_results)

            # Revise top K candidates
            tasks = [_revise_candidate(idx, *item) for idx, item in enumerate(top_k)]
            revised_candidates = list(await asyncio.gather(*tasks))

            all_candidates = candidates + revised_candidates
            all_candidates.sort(key=lambda x: x[1], reverse=True)  # Sort by accuracy

            if all_candidates[0][1] >= 1.0:
                print("Stage 2 complete. Perfect solution on training found!")
                best_instructions, best_accuracy, best_train_results = all_candidates[0]
            else:
                print(f"Stage 2 complete. Best training accuracy: {all_candidates[0][1]:.2%}")

                # STAGE 3: Pooled revision
                print(
                    f"\nStage 3: Pooled revision - generating {self.config.num_pooled_candidates} from top {self.config.num_top_for_revision}..."
                )

                top_k_for_pool = all_candidates[: self.config.num_top_for_revision]
                pooled_context_parts = []

                for idx, (instructions, accuracy, train_results) in enumerate(top_k_for_pool):
                    summary = f"Instructions {idx + 1} (training accuracy: {accuracy:.1%}):\n{instructions}\n\nTraining Performance:"

                    for result in train_results:
                        summary += f"\n  Example {result['train_example_idx'] + 1}: {result['accuracy']:.1%} correct"

                    pooled_context_parts.append(f"<attempt_{idx + 1}>\n{summary}\n</attempt_{idx + 1}>")

                pooled_context = "\n\n".join(pooled_context_parts)
                pooled_prompt = self.config.pooled_revision_prompt.format(
                    task_description=training_examples_str, pooled_context=pooled_context
                )

                tasks = [
                    self._generate_and_evaluate_instruction(pooled_prompt, train_examples, cookies, 0.8, i + 1)
                    for i in range(self.config.num_pooled_candidates)
                ]
                pooled_candidates = list(await asyncio.gather(*tasks))

                # Pick best by training accuracy
                all_final_candidates = all_candidates + pooled_candidates
                all_final_candidates.sort(key=lambda x: x[1], reverse=True)

                best_instructions, best_accuracy, best_train_results = all_final_candidates[0]
                print(f"\nAll Stages complete. Best training accuracy: {best_accuracy:.2%}")
        else:
            print("  Perfect solution on training found in Stage 1! Skipping Stage 2 and 3.")

        print("\nApplying best instructions to test input...")

        test_input_str = self._format_grid(body.test_input)
        test_application_prompt = f"""Follow these instructions to transform the input grid:

Instructions:
{best_instructions}

Input Grid:
{test_input_str}

Apply the instructions above to transform this input grid. Provide your solution as a 2D array inside \\boxed{{}} in this exact format: \\boxed{{[[row1],[row2],...]]}}"""

        final_response, cookies = await self._generate_instruction_response(
            test_application_prompt, cookies, temperature=0.0
        )

        final_reward, final_verify_data, cookies = await self._verify_response(final_response, body, cookies)

        print(f"Test solution generated. Final verification reward: {final_reward}")

        return BermanAgentVerifyResponse.model_validate(final_verify_data)


if __name__ == "__main__":
    BermanAgent.run_webserver()
