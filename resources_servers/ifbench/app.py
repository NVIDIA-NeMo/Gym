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

"""IFBench resources server.

Evaluates instruction-following using the AllenAI IFBench library
(https://github.com/allenai/IFBench), which covers the full IFBench
instruction taxonomy (57 instruction types). The library is auto-installed
at server startup by setup_ifbench.py.

Reward is the fraction of instructions followed (grading_mode='fraction')
or 1.0 only if all instructions are followed (grading_mode='binary').
IFBench's primary metric is the average fraction, so 'fraction' is the default.
"""

import asyncio
from asyncio import get_running_loop
from typing import List, Literal

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.ifbench.setup_ifbench import ensure_ifbench


class IFBenchResourcesServerConfig(BaseResourcesServerConfig):
    num_processes: int = 32


class IFBenchRunRequest(BaseRunRequest):
    id: int
    instruction_id_list: List[str]
    prompt: str
    kwargs: List
    grading_mode: Literal["binary", "fraction"] = "fraction"


class IFBenchVerifyRequest(IFBenchRunRequest, BaseVerifyRequest):
    pass


class IFBenchVerifyResponse(BaseVerifyResponse):
    follow_all_instructions: bool
    follow_instruction_list: List[bool]
    instruction_id_list: List[str]
    prompt: str
    grading_mode: Literal["binary", "fraction"] = "fraction"


class IFBenchResourcesServer(SimpleResourcesServer):
    config: IFBenchResourcesServerConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ensure_ifbench()
        import instructions_registry  # available after ensure_ifbench adds to sys.path

        self._instructions_registry = instructions_registry
        self._semaphore = asyncio.Semaphore(value=self.config.num_processes)

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(self, body: IFBenchVerifyRequest) -> IFBenchVerifyResponse:
        # Extract model response text from the last output item
        final_response_text = ""
        if body.response.output:
            last_output = body.response.output[-1]
            if hasattr(last_output, "content") and last_output.content:
                final_response_text = last_output.content[0].text

        loop = get_running_loop()
        async with self._semaphore:
            is_following_list = await loop.run_in_executor(
                None,
                self._check_instructions,
                body.instruction_id_list,
                body.kwargs,
                body.prompt,
                final_response_text,
            )

        if body.grading_mode == "binary":
            reward = float(all(is_following_list))
        else:
            reward = float(sum(is_following_list) / len(is_following_list)) if is_following_list else 0.0

        return IFBenchVerifyResponse(
            **body.model_dump(),
            reward=reward,
            follow_all_instructions=all(is_following_list),
            follow_instruction_list=is_following_list,
        )

    def _check_instructions(
        self,
        instruction_id_list: List[str],
        kwargs_list: List,
        prompt: str,
        response: str,
    ) -> List[bool]:
        """Check each instruction synchronously (called from a thread executor)."""
        is_following_list = []
        response_has_content = bool(response.strip())

        for instruction_id, kwargs in zip(instruction_id_list, kwargs_list):
            if not response_has_content:
                is_following_list.append(False)
                continue
            try:
                instruction_cls = self._instructions_registry.INSTRUCTION_DICT[instruction_id]
                instruction = instruction_cls(instruction_id)

                filtered_kwargs = {k: v for k, v in (kwargs or {}).items() if v is not None}
                instruction.build_description(**filtered_kwargs)

                # Some instructions (e.g. repeat:*) need the original prompt text
                args = instruction.get_instruction_args()
                if args and "prompt" in args:
                    instruction.build_description(prompt=prompt)

                follows = bool(instruction.check_following(response))
            except Exception as e:
                print(f"Error checking instruction {instruction_id}: {e}")
                follows = False
            is_following_list.append(follows)

        return is_following_list


if __name__ == "__main__":
    IFBenchResourcesServer.run_webserver()
