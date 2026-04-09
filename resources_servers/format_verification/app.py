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
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class FormatVerificationResourcesServerConfig(BaseResourcesServerConfig):
    pass


class FormatVerificationVerifyRequest(BaseVerifyRequest):
    verifier: Dict[str, Any]


class FormatVerificationVerifyResponse(BaseVerifyResponse):
    verifier: Dict[str, Any]
    match_details: Dict[str, Any]


class FormatVerificationResourcesServer(SimpleResourcesServer):
    config: FormatVerificationResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(self, body: FormatVerificationVerifyRequest) -> FormatVerificationVerifyResponse:
        response_text = self._extract_assistant_text(body.response)
        verifier = body.verifier
        verifier_type = verifier.get("type", "")

        match verifier_type:
            case "regex":
                reward, details = self._verify_regex(response_text, verifier)
            case "string_match":
                reward, details = self._verify_string_match(response_text, verifier)
            case _:
                raise NotImplementedError(f"Verifier type must be 'regex' or 'string_match', got {verifier_type!r}")

        return FormatVerificationVerifyResponse(**body.model_dump(), reward=reward, match_details=details)

    # ----- Helpers ----- #
    @staticmethod
    def _extract_assistant_text(response) -> str:
        parts: List[str] = []
        for output_item in response.output:
            if output_item.type != "message":
                continue
            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue
                parts.append(content_item.text)
        return "".join(parts)

    @staticmethod
    def _verify_regex(text: str, verifier: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Count lines matching any pattern in verify_regex.

        Each line counts at most once (even if it matches multiple patterns).
        Reward is 1.0 if matches >= verify_min_matches, else 0.0.
        """
        patterns: List[str] = verifier.get("verify_regex", [])
        min_matches: int = verifier.get("verify_min_matches", 1)

        compiled = [re.compile(p) for p in patterns]
        matching_lines = 0
        for line in text.split("\n"):
            if any(rx.search(line) for rx in compiled):
                matching_lines += 1

        passed = matching_lines >= min_matches
        details = {
            "matching_lines": matching_lines,
            "min_matches": min_matches,
            "passed": passed,
        }
        return (1.0 if passed else 0.0), details

    @staticmethod
    def _verify_string_match(text: str, verifier: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Check that every expected_markers string appears in the text.

        Also detects spurious markers if patterns are provided.
        Reward is 1.0 if all expected markers are present (and no spurious), else 0.0.
        """
        expected_markers: List[str] = verifier.get("expected_markers", [])
        regex_patterns: List[str] = verifier.get("patterns", [])

        missing = [m for m in expected_markers if m not in text]

        spurious: List[str] = []
        if regex_patterns:
            expected_set = set(expected_markers)
            for pat in regex_patterns:
                for found in re.finditer(pat, text):
                    if found.group(0) not in expected_set:
                        spurious.append(found.group(0))

        passed = len(missing) == 0 and len(spurious) == 0
        details = {
            "expected": expected_markers,
            "missing": missing,
            "spurious": spurious,
            "passed": passed,
        }
        return (1.0 if passed else 0.0), details


if __name__ == "__main__":
    FormatVerificationResourcesServer.run_webserver()
