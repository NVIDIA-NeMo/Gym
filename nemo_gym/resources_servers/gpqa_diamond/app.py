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

import re
from typing import Optional

from nemo_gym.resources_servers.mcqa.app import (
    MCQAResourcesServer,
    MCQAVerifyRequest,
    MCQAVerifyResponse,
    _extract_options_and_expected,
    _get_allowed_letters_from_options,
    _parse_answer_with_custom_regex,
)


def extract_letter(text: str) -> Optional[str]:
    """Extract the final GPQA answer letter from boxed or Answer: formats."""
    boxed_match = re.findall(r"\\boxed\{([^}]*)\}", text, re.DOTALL)
    if boxed_match:
        extracted = boxed_match[-1].strip()
        if len(extracted) == 1 and extracted.isupper():
            return extracted
        letter_match = re.findall(r"\b([A-Z])\b", extracted, re.DOTALL)
        if letter_match:
            return letter_match[-1].strip()

    answer_match = re.findall(r"(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])", text)
    if answer_match:
        return answer_match[-1].strip().upper()

    return None


class GPQADiamondResourcesServer(MCQAResourcesServer):
    """GPQA-Diamond verifier with GPQA-specific answer extraction."""

    async def verify(self, body: MCQAVerifyRequest) -> MCQAVerifyResponse:
        text = body.response.output_text.strip()
        options, expected_answer = _extract_options_and_expected(body)
        allowed_letters = _get_allowed_letters_from_options(options)

        pred: Optional[str] = None

        if body.template_metadata and "output_regex" in body.template_metadata:
            regex_pattern = body.template_metadata["output_regex"]
            pred = _parse_answer_with_custom_regex(text, regex_pattern, allowed_letters, options)

        if pred is None:
            pred = extract_letter(text)
            if pred is not None:
                pred = pred.upper()
                if allowed_letters and pred not in allowed_letters:
                    pred = None

        gold = (expected_answer or "").strip().upper()
        is_correct = (pred == gold) if (pred is not None and gold) else False
        reward = 1.0 if is_correct else 0.0

        return MCQAVerifyResponse(
            **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
            reward=reward,
            expected_answer=gold,
            extracted_answer=pred,
        )


if __name__ == "__main__":
    GPQADiamondResourcesServer.run_webserver()
