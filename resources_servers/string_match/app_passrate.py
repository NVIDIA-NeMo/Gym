# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""String-match grading with trailing end-of-sequence markers removed."""

import re

from resources_servers.string_match.app import (
    StringMatchResourcesServer,
    StringMatchVerifyRequest,
    StringMatchVerifyResponse,
    _answers_match,
    _extract_answer,
    _extract_last_assistant_text,
)


_TRAILING_EOS = re.compile(r"(?:\s*(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>))+\s*$")


class PassrateStringMatchResourcesServer(StringMatchResourcesServer):
    """Normalize trailing EOS markers only for grading."""

    async def verify(self, body: StringMatchVerifyRequest) -> StringMatchVerifyResponse:
        text = _TRAILING_EOS.sub("", _extract_last_assistant_text(body)).strip()
        extracted = _extract_answer(text, body.extraction_mode)
        reward = 0.0
        if extracted is not None and body.expected_answer:
            reward = _answers_match(extracted, body.expected_answer, body.case_sensitive)
        return StringMatchVerifyResponse(
            **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
            reward=reward,
            expected_answer=body.expected_answer,
            extracted_answer=extracted,
        )


if __name__ == "__main__":
    PassrateStringMatchResourcesServer.run_webserver()
