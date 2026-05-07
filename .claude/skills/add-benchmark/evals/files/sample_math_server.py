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
"""Math algebra resources server — extracts numerical answer and compares to expected."""

import re

from nemo_gym.servers.resources_server import SimpleResourcesServer


class MathAlgebraServer(SimpleResourcesServer):
    async def verify(self, body):
        output_text = body.get("output_text", "")
        verifier_metadata = body.get("verifier_metadata", {})
        expected_answer = str(verifier_metadata.get("expected_answer", ""))

        # Strip think blocks before extraction
        if "</think>" in output_text:
            output_text = output_text.split("</think>")[-1].strip()

        # Extract answer after "Answer:" marker
        match = re.search(r"(?:Answer|ANSWER)\s*[:\s]\s*(.+)", output_text)
        if not match:
            return {"reward": 0.0, "extracted_answer": None, "reason": "no_answer_marker"}

        extracted = match.group(1).strip().rstrip(".")

        # Compare
        if self._normalize(extracted) == self._normalize(expected_answer):
            return {"reward": 1.0, "extracted_answer": extracted}
        else:
            return {"reward": 0.0, "extracted_answer": extracted, "reason": "wrong_answer"}

    @staticmethod
    def _normalize(s: str) -> str:
        """Normalize answer for comparison: strip whitespace, lowercase."""
        return re.sub(r"\s+", " ", s.strip().lower())
