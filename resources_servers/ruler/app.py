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

from typing import List, Optional

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class RulerResourcesServerConfig(BaseResourcesServerConfig):
    pass


class RulerVerifyRequest(BaseVerifyRequest):
    outputs: List[str]
    length: int
    subset: str


class RulerVerifyResponse(RulerVerifyRequest, BaseVerifyResponse):
    string_match_all_single: float
    string_match_part_single: float
    cwe_string_match_all_single: Optional[float]
    cwe_string_match_part_single: Optional[float]
    fwe_string_match_all_single: Optional[float]
    fwe_string_match_part_single: Optional[float]
    niah_multikey_1_string_match_all_single: Optional[float]
    niah_multikey_1_string_match_part_single: Optional[float]
    niah_multikey_2_string_match_all_single: Optional[float]
    niah_multikey_2_string_match_part_single: Optional[float]
    niah_multikey_3_string_match_all_single: Optional[float]
    niah_multikey_3_string_match_part_single: Optional[float]
    niah_multiquery_string_match_all_single: Optional[float]
    niah_multiquery_string_match_part_single: Optional[float]
    niah_multivalue_string_match_all_single: Optional[float]
    niah_multivalue_string_match_part_single: Optional[float]
    niah_single_1_string_match_all_single: Optional[float]
    niah_single_1_string_match_part_single: Optional[float]
    niah_single_2_string_match_all_single: Optional[float]
    niah_single_2_string_match_part_single: Optional[float]
    niah_single_3_string_match_all_single: Optional[float]
    niah_single_3_string_match_part_single: Optional[float]
    qa_1_string_match_all_single: Optional[float]
    qa_1_string_match_part_single: Optional[float]
    qa_2_string_match_all_single: Optional[float]
    qa_2_string_match_part_single: Optional[float]
    vt_string_match_all_single: Optional[float]
    vt_string_match_part_single: Optional[float]


class RulerResourcesServer(SimpleResourcesServer):
    config: RulerResourcesServerConfig

    async def verify(self, body: RulerVerifyRequest) -> BaseVerifyResponse:
        prediction = body.response.output_text.strip()
        reward_string_match_all_single = self.string_match_all_single(prediction, body.outputs)
        reward_string_match_part_single = self.string_match_part_single(prediction, body.outputs)

        return RulerVerifyResponse(
            **body.model_dump(),
            reward=1.0,
            string_match_all_single=reward_string_match_all_single,
            string_match_part_single=reward_string_match_part_single,
            **{
                f"{body.subset}_string_match_all_single": reward_string_match_all_single,
                f"{body.subset}_string_match_part_single": reward_string_match_part_single,
            },
        )

    # These helper functions are taken from https://github.com/NVIDIA-NeMo/Skills/blob/54d2e113c2f64bf74bda72e15f23f01b524850da/nemo_skills/evaluation/evaluator/ruler.py#L36
    def string_match_all_single(self, pred: str, refs: List[str]) -> float:
        return sum([1.0 if r.lower() in pred.lower() else 0.0 for r in refs]) / len(refs)

    def string_match_part_single(self, pred: str, refs: List[str]) -> float:
        return max([1.0 if r.lower() in pred.lower() else 0.0 for r in refs])


if __name__ == "__main__":
    RulerResourcesServer.run_webserver()
