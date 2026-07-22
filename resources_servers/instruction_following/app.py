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
from pydantic import model_validator
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
    verifier_metadata: Dict[str, Any]

    @model_validator(mode="after")
    def _validate_verifier_metadata(self) -> "InstructionFollowingRunRequest":
        missing = [f for f in ("instruction_id_list", "prompt", "kwargs") if f not in self.verifier_metadata]
        if missing:
            raise ValueError(f"verifier_metadata is missing required fields: {missing}")
        return self


class InstructionFollowingVerifyRequest(InstructionFollowingRunRequest, BaseVerifyRequest):
    pass


class InstructionFollowingVerifyResponse(BaseVerifyResponse):
    follow_all_instructions: bool
    follow_instruction_list: List[bool]
    verifier_metadata: Dict[str, Any]


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

        vm = body.verifier_metadata
        instruction_list = vm["instruction_id_list"]
        kwargs_list = vm["kwargs"]
        grading_mode = vm.get("grading_mode", "binary")
        is_following_list = []

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

                # Check if the response follows the instruction
                if instruction.check_following(final_response_text):
                    is_following_list.append(True)
                else:
                    is_following_list.append(False)

            except Exception as e:
                # If there's an error processing the instruction, mark as failed
                print(f"Error processing instruction {instruction_id}: {e}")
                is_following_list.append(False)

        # Calculate overall success
        if grading_mode == "binary":
            reward = float(all(is_following_list))
        elif grading_mode == "fraction":
            reward = float((sum(is_following_list) / len(is_following_list)) if is_following_list else 0.0)
        else:
            raise ValueError(f"Invalid grading_mode: {grading_mode}")

        return InstructionFollowingVerifyResponse(
            **body.model_dump(),
            reward=float(reward),
            follow_all_instructions=all(is_following_list),
            follow_instruction_list=is_following_list,
        )


if __name__ == "__main__":
    InstructionFollowingResourcesServer.run_webserver()
