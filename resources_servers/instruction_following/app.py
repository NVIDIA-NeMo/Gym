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
from typing import Any, Dict, List, Literal

from fastapi import FastAPI
from verifiable_instructions import instructions_registry

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class InstructionFollowingResourcesServerConfig(BaseResourcesServerConfig):
    pass


class InstructionFollowingRunRequest(BaseRunRequest):
    id: int
    instruction_id_list: List
    prompt: str
    kwargs: List
    grading_mode: Literal[
        "binary",
        "fraction",
    ] = "binary"


class InstructionFollowingVerifyRequest(InstructionFollowingRunRequest, BaseVerifyRequest):
    pass


def _get_loose_perturbations(text: str) -> list:
    """Return IFEval loose-mode perturbations following the NeMo Skills convention.

    Produces 4 line-removal variants of the text (original, without first line,
    without last line, without first and last line), each duplicated with asterisks
    removed. Empty variants are excluded.
    """

    def remove_stars(s: str) -> str:
        return s.replace("*", "")

    def without_first_line(s: str) -> str:
        idx = s.find("\n")
        return s[idx + 1 :] if idx >= 0 else ""

    def without_last_line(s: str) -> str:
        idx = s.rfind("\n")
        return s[:idx] if idx >= 0 else ""

    base = [
        text,
        without_first_line(text),
        without_last_line(text),
        without_last_line(without_first_line(text)),
    ]
    return [v for s in base for v in (s, remove_stars(s)) if v.strip()]


def _check_following_loose(instruction, text: str) -> bool:
    """Check instruction against native loose API or 8 perturbations."""
    if hasattr(instruction, "check_following_loose"):
        return instruction.check_following_loose(text)
    try:
        return instruction.check_following(text, mode="loose")
    except TypeError:
        return any(instruction.check_following(p) for p in _get_loose_perturbations(text))


class InstructionFollowingVerifyResponse(BaseVerifyResponse):
    follow_all_instructions: bool
    follow_instruction_list: List[bool]
    follow_all_instructions_loose: bool
    follow_instruction_list_loose: List[bool]
    kwargs: List
    instruction_id_list: List
    prompt: str
    grading_mode: Literal[
        "binary",
        "fraction",
    ] = "binary"


class InstructionFollowingResourcesServer(SimpleResourcesServer):
    config: InstructionFollowingResourcesServerConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        """Ensure required NLTK data is available at startup.

        nltk.download() always fetches the remote package index even when the
        data is already present. Guard with a local find() first to skip the
        download when the data already exists.
        """
        try:
            import nltk

            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
        except ImportError:
            # ifbench not available, skip
            pass
        except Exception as e:
            print(f"NLTK setup warning: {e}")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        return app

    async def verify(self, body: InstructionFollowingVerifyRequest) -> InstructionFollowingVerifyResponse:
        # Get the final text response from the last output item
        final_response_text = ""
        if body.response.output:
            last_output = body.response.output[-1]
            if hasattr(last_output, "content") and last_output.content:
                # Extract text from the nested content structure
                final_response_text = last_output.content[0].text

        # Verify each instruction using the verifiable instructions
        instruction_list = body.instruction_id_list
        kwargs_list = body.kwargs
        is_following_list = []
        is_following_list_loose = []

        for instruction_id, kwargs in zip(instruction_list, kwargs_list):
            try:
                # Create instruction instance
                instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
                instruction = instruction_cls(instruction_id)

                # Handle None kwargs
                if kwargs is None:
                    kwargs = {}

                # Filter out None values from kwargs
                filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

                # Build the instruction description with the provided kwargs
                instruction.build_description(**filtered_kwargs)

                # Check strict and loose from the same instruction instance
                is_following_list.append(instruction.check_following(final_response_text))
                is_following_list_loose.append(_check_following_loose(instruction, final_response_text))

            except Exception as e:
                # If there's an error processing the instruction, mark as failed
                print(f"Error processing instruction {instruction_id}: {e}")
                is_following_list.append(False)
                is_following_list_loose.append(False)

        # Calculate overall success
        reward_mode = getattr(body, "grading_mode", "binary")
        if reward_mode == "binary":
            reward = float(all(is_following_list))
        elif reward_mode == "fraction":
            reward = float((sum(is_following_list) / len(is_following_list)) if is_following_list else 0.0)
        else:
            raise ValueError(f"Invalid reward mode: {reward_mode}")

        return InstructionFollowingVerifyResponse(
            **body.model_dump(),
            reward=float(reward),
            follow_all_instructions=all(is_following_list),
            follow_instruction_list=is_following_list,
            follow_all_instructions_loose=all(is_following_list_loose),
            follow_instruction_list_loose=is_following_list_loose,
        )

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Compute the four IFEval accuracy metrics over all verify responses.

        tasks[i] is the list of rollout dicts for task i. Each dict contains
        follow_instruction_list and follow_instruction_list_loose.
        """
        prompt_strict: list = []
        instruction_strict: list = []
        prompt_loose: list = []
        instruction_loose: list = []

        for task_rollouts in tasks:
            for rd in task_rollouts:
                strict_list = rd.get("follow_instruction_list", [])
                loose_list = rd.get("follow_instruction_list_loose", [])
                prompt_strict.append(float(all(strict_list)) if strict_list else 0.0)
                prompt_loose.append(float(all(loose_list)) if loose_list else 0.0)
                instruction_strict.extend(float(v) for v in strict_list)
                instruction_loose.extend(float(v) for v in loose_list)

        def _mean(lst: list) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        return {
            "prompt_strict_accuracy": _mean(prompt_strict) * 100.0,
            "instruction_strict_accuracy": _mean(instruction_strict) * 100.0,
            "prompt_loose_accuracy": _mean(prompt_loose) * 100.0,
            "instruction_loose_accuracy": _mean(instruction_loose) * 100.0,
        }


if __name__ == "__main__":
    InstructionFollowingResourcesServer.run_webserver()
