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
from typing import ClassVar, Collection, Optional

from nemo_gym.server_runtime import ReverifyMode
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.mcqa.app import (
    MCQAResourcesServer,
    MCQAResourcesServerConfig,
    MCQAVerifyRequest,
    MCQAVerifyResponse,
    _extract_options_and_expected,
    _get_allowed_letters_from_options,
    _parse_answer_with_custom_regex,
)


_BOXED = re.compile(r"\\boxed\{([^}]*)\}", re.DOTALL)
_ANSWER_LINE = re.compile(r"(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])")


def extract_letter(text: str, allowed_letters: Optional[Collection[str]] = None) -> Optional[str]:
    """Extract the final GPQA answer letter from boxed or Answer: formats.

    Every \\boxed{} and every "Answer: X" line offers a candidate; the one written last wins. A box that holds an
    intermediate result (\\boxed{q = 2 C}, \\boxed{T = 300 K}) is only a candidate if its letter is a valid choice,
    so it cannot hide the final "Answer:" line the prompt asks for.
    """
    candidates: list[tuple[int, str]] = []
    for match in _BOXED.finditer(text):
        extracted = match.group(1).strip()
        if len(extracted) == 1 and extracted.isupper():
            letter: Optional[str] = extracted
        else:
            letters = re.findall(r"\b([A-Z])\b", extracted, re.DOTALL)
            letter = letters[-1] if letters else None
        if letter is not None:
            candidates.append((match.start(), letter))
    for match in _ANSWER_LINE.finditer(text):
        candidates.append((match.start(), match.group(1).upper()))
    if allowed_letters:
        candidates = [(at, letter) for at, letter in candidates if letter in allowed_letters]
    return max(candidates)[1] if candidates else None


class GPQADiamondResourcesServerConfig(MCQAResourcesServerConfig):
    # Reuses mcqa's MCQAResourcesServerConfig (STATELESS); pin back to the safe default so this
    # benchmark does not inherit reverify support until it is separately reviewed.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNKNOWN


class GPQADiamondResourcesServer(MCQAResourcesServer):
    """GPQA-Diamond verifier with GPQA-specific answer extraction."""

    config: GPQADiamondResourcesServerConfig

    async def verify(self, body: MCQAVerifyRequest) -> MCQAVerifyResponse:
        text = body.response.output_text.strip()
        options, expected_answer = _extract_options_and_expected(body)
        allowed_letters = _get_allowed_letters_from_options(options)

        pred: Optional[str] = None

        if body.template_metadata and "output_regex" in body.template_metadata:
            regex_pattern = body.template_metadata["output_regex"]
            pred = _parse_answer_with_custom_regex(text, regex_pattern, allowed_letters, options)

        if pred is None:
            pred = extract_letter(text, allowed_letters)
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
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = GPQADiamondResourcesServer.run_webserver()  # noqa: F401
