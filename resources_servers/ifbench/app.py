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
import asyncio
import logging
from typing import List, Literal

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


try:
    from .setup_ifbench import ensure_ifbench
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from setup_ifbench import ensure_ifbench


logger = logging.getLogger(__name__)


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
    kwargs: List
    instruction_id_list: List[str]
    prompt: str
    grading_mode: Literal["binary", "fraction"] = "fraction"


class IFBenchResourcesServer(SimpleResourcesServer):
    config: IFBenchResourcesServerConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ensure_ifbench()
        import instructions_registry

        self._instructions_registry = instructions_registry
        self._semaphore = asyncio.Semaphore(value=self.config.num_processes)

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    def _check_instructions(
        self,
        instruction_id_list: List[str],
        kwargs_list: List,
        prompt: str,
        response: str,
    ) -> List[bool]:
        """Evaluate each instruction against the response.

        Individual instruction failures should never crash the server.
        """
        INSTRUCTION_DICT = self._instructions_registry.INSTRUCTION_DICT

        # Empty response: skip evaluation and fail all instructions
        if not response.strip():
            return [False] * len(instruction_id_list)

        is_following_list = []
        for instruction_id, kwargs in zip(instruction_id_list, kwargs_list):
            try:
                instruction_cls = INSTRUCTION_DICT[instruction_id]
                instruction = instruction_cls(instruction_id)

                # Filter None values from kwargs before calling build_description
                filtered_kwargs = {k: v for k, v in (kwargs or {}).items() if v is not None}
                instruction.build_description(**filtered_kwargs)

                # repeat:* instructions also need the original prompt text
                args = instruction.get_instruction_args()
                if args and "prompt" in args:
                    instruction.build_description(prompt=prompt)

                try:
                    follows = bool(instruction.check_following(response))
                except Exception:
                    logger.exception("check_following failed for instruction %s", instruction_id)
                    follows = False

                is_following_list.append(follows)

            except Exception:
                logger.exception("Error processing instruction %s", instruction_id)
                is_following_list.append(False)

        return is_following_list

    async def verify(self, body: IFBenchVerifyRequest) -> IFBenchVerifyResponse:
        # Extract final response text from the last output item
        final_response_text = ""
        if body.response.output:
            last_output = body.response.output[-1]
            if hasattr(last_output, "content") and last_output.content:
                final_response_text = last_output.content[0].text

        loop = asyncio.get_event_loop()
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


if __name__ == "__main__":
    IFBenchResourcesServer.run_webserver()
