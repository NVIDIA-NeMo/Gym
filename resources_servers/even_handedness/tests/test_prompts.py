# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin normalized public rubric content to the upstream revision."""

import hashlib

from resources_servers.even_handedness.prompts import EVEN_HANDEDNESS_PROMPT, HEDGING_PROMPT, REFUSAL_PROMPT


def _digest(text: str) -> str:
    normalized = "\n".join(line.rstrip() for line in text.split("\n"))
    return hashlib.sha256(normalized.encode()).hexdigest()


def test_public_rubric_digests() -> None:
    assert _digest(EVEN_HANDEDNESS_PROMPT) == "f2aa097e78da365bb311f5a2a63b846b4c1981a13fff1c53e187a57fc5dbff7d"
    assert _digest(REFUSAL_PROMPT) == "4553839d237f25481d01f1e79964d86550c8989722fd3557b981023274fbb7dc"
    assert _digest(HEDGING_PROMPT) == "63fa4833c3d35120b8d95cd7f34a5b9329c720843f5a56fa2ab2e0aa1db69b0e"
