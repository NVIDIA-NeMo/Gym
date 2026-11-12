# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""LongBench v2 scorer: rule-based multiple-choice grading over long contexts.

Dataset: https://huggingface.co/datasets/THUDM/LongBench-v2

Each row is a four-way multiple-choice question whose context is already baked
into the rendered prompt. verify() extracts one letter from the model's answer
and scores an exact match against the gold letter; reward is 0.0 or 1.0.
"""

import re
from typing import Any, ClassVar, Optional

from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)


PARENS_ANSWER_PATTERN = re.compile(r"The correct answer is \(([A-D])\)")
BARE_ANSWER_PATTERN = re.compile(r"The correct answer is ([A-D])")


def extract_letter(text: str) -> Optional[str]:
    """Return the answer letter stated in ``text``, or None when absent.

    Asterisks are stripped first so bold markdown still matches. The whole
    string is searched for the parenthesised form before the bare form is tried
    at all, so a parenthesised statement anywhere wins over a bare one that
    comes earlier; within one form the leftmost match wins.
    """
    text = text.replace("*", "")
    match = PARENS_ANSWER_PATTERN.search(text)
    if match:
        return match.group(1)
    match = BARE_ANSWER_PATTERN.search(text)
    return match.group(1) if match else None


class LongbenchResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS


class LongbenchRunRequest(BaseRunRequest):
    # ``serialize_by_alias`` keeps the row id on the wire as ``_id``: Pydantic
    # reserves leading-underscore names for private attributes, so the field
    # itself must be called ``row_id``.
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    expected_answer: Optional[str] = None
    row_id: Optional[str] = Field(default=None, alias="_id")
    domain: Optional[str] = None
    sub_domain: Optional[str] = None
    difficulty: Optional[str] = None
    length: Optional[str] = None
    verifier_metadata: Optional[dict[str, Any]] = None


class LongbenchVerifyRequest(LongbenchRunRequest, BaseVerifyRequest):
    pass


class LongbenchVerifyResponse(BaseVerifyResponse, LongbenchRunRequest):
    expected_answer: str
    extracted_answer: Optional[str]


class LongbenchResourcesServer(SimpleResourcesServer):
    config: LongbenchResourcesServerConfig

    async def verify(self, body: LongbenchVerifyRequest) -> LongbenchVerifyResponse:
        # A caller may drop the top-level gold and forward only verifier_metadata.
        gold = body.expected_answer or (body.verifier_metadata or {}).get("expected_answer")
        gold = (gold or "").strip().upper()
        text = (body.response.output_text or "").strip()

        predicted = extract_letter(text) if text else None
        reward = 1.0 if (predicted is not None and gold and predicted == gold) else 0.0

        return LongbenchVerifyResponse(
            **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
            reward=reward,
            expected_answer=gold,
            extracted_answer=predicted,
        )


if __name__ == "__main__":
    LongbenchResourcesServer.run_webserver()
