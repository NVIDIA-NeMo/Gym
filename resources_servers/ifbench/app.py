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
    follow_all_instructions_loose: bool
    follow_instruction_list_loose: List[bool]
    reward_loose: float
    kwargs: List
    instruction_id_list: List[str]
    prompt: str
    grading_mode: Literal["binary", "fraction"] = "fraction"


def _loose_response_variants(response: str) -> List[str]:
    """Response variants for loose scoring (see AllenAI IFBench evaluation_lib)."""
    r = response.split("\n")
    remove_first = "\n".join(r[1:]).strip()
    remove_last = "\n".join(r[:-1]).strip()
    remove_both = "\n".join(r[1:-1]).strip()
    return [
        response,
        response.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]


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
    ) -> tuple[List[bool], List[bool]]:
        """Evaluate each instruction, returning (strict, loose) follow lists.

        Strict checks the raw response; loose passes if any response variant
        (the first of which is the raw response) satisfies the instruction.
        Individual instruction failures should never crash the server.
        """
        INSTRUCTION_DICT = self._instructions_registry.INSTRUCTION_DICT

        # Empty response: skip evaluation and fail all instructions
        if not response.strip():
            fail = [False] * len(instruction_id_list)
            return fail, list(fail)

        variants = _loose_response_variants(response)

        strict_list, loose_list = [], []
        for instruction_id, kwargs in zip(instruction_id_list, kwargs_list):
            strict = loose = False
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

                for index, variant in enumerate(variants):
                    if not variant.strip():
                        continue
                    try:
                        follows = bool(instruction.check_following(variant))
                    except Exception:
                        logger.exception("check_following failed for instruction %s", instruction_id)
                        follows = False
                    if index == 0:  # variant[0] is the raw response -> strict
                        strict = follows
                    if follows:
                        loose = True
                        break

            except Exception:
                logger.exception("Error processing instruction %s", instruction_id)

            strict_list.append(strict)
            loose_list.append(loose)

        return strict_list, loose_list

    def _reward(self, is_following_list: List[bool], grading_mode: str) -> float:
        if grading_mode == "binary":
            return float(all(is_following_list))
        return float(sum(is_following_list) / len(is_following_list)) if is_following_list else 0.0

    async def verify(self, body: IFBenchVerifyRequest) -> IFBenchVerifyResponse:
        # Extract final response text from the last output item
        final_response_text = ""
        if body.response.output:
            last_output = body.response.output[-1]
            if hasattr(last_output, "content") and last_output.content:
                final_response_text = last_output.content[0].text

        loop = asyncio.get_event_loop()
        async with self._semaphore:
            strict_list, loose_list = await loop.run_in_executor(
                None,
                self._check_instructions,
                body.instruction_id_list,
                body.kwargs,
                body.prompt,
                final_response_text,
            )

        return IFBenchVerifyResponse(
            **body.model_dump(),
            reward=self._reward(strict_list, body.grading_mode),
            follow_all_instructions=all(strict_list),
            follow_instruction_list=strict_list,
            reward_loose=self._reward(loose_list, body.grading_mode),
            follow_all_instructions_loose=all(loose_list),
            follow_instruction_list_loose=loose_list,
        )


if __name__ == "__main__":
    IFBenchResourcesServer.run_webserver()
