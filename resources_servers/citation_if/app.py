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
"""citation_if RL training resources server.

Grades model responses for citation instruction-following training. Each trajectory
row supplies a `verifier` dict that drives per-row reward logic.

Verifier dict schema (per training row):
    type            : "citation_if"
    mode            : "cite" | "no_cite"
    grammar         : grammar name (see GRAMMAR_TABLE)
    id_kind         : "full_source" | "snippet"
    id_regex        : per-row ID pattern, e.g. "citation_[a-z2-9]{4}:snippet_[0-9]+".
                      AUTHORITATIVE for grammar parsing — GRAMMAR_TABLE is
                      compiled per row against it, so random-ID and numeric-ID schemes
                      coexist. Distinct from the broad ATTEMPT_REGEX used by gate 1, which
                      must stay a SUPERSET of every row's id_regex.
    valid_id_set    : list[str]   — IDs visible in the trajectory evidence
    expected_ids    : list[str] | null  — set when correctness signal is available; null otherwise
    expected_slack  : int         — over-citation slack for the correctness gate; default 1
    min_valid_citations : int     — default 1

Gate sequence (cite mode) — authoritative implementation in scorer.py:
    0.  Structural        — non-empty text, no tool call in the response
    0b. Answer presence   — >=1 word character survives once citation markup is removed
                            A citation with NO answer scores 0.
                            Presence only, never quality: a one-word answer scores 1, and
                            claim text counts so claim_wrap_xml answers written entirely
                            inside the span are fine. Cite mode only.
    1.  Malformed-attempt — STRICT: strip well-formed citation spans, then
                            ANY residual ID-shaped token -> 0. Tag grammars additionally
                            require open == parsed == close.
    2.  Must-cite         — >= min_valid_citations parsed IDs in valid_id_set
    3.  No-hallucination  — every parsed citation ID must be in valid_id_set
    4.  Correctness       — (when expected_ids set) expected_ids ⊆ valid_cited AND
                            |valid_cited| ≤ |expected_ids| + expected_slack (every gold cited,
                            over-citation capped)

    Gate 1 was previously an EXISTENCE check ("id_regex matches but zero
    grammar-valid citations"), under which one well-formed citation whitelisted raw-ID
    leakage, "Sources:" trailers and comma+space lists for the rest of the answer. Do not
    reintroduce that form.

No-cite mode:
    Reward 1 iff zero ATTEMPT_REGEX matches AND zero citation markup in response.

Catch-all POST /{tool_name}:
    Returns a fixed terminal string; used with max_steps=1 so tool-calling
    rollouts terminate immediately and score 0 on Gate 0 (structural).

Grammar table (trained set -- 9 grammars):
    cite_xml, ascii_brackets, claim_wrap_xml, fullwidth_brackets,
    ref_colon, double_angle, web_brackets, paren_part, markdown_footnote

Holdout grammars (never assign to training rows):
    curly_double, angle_pipe
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from scorer import TERMINAL_TOOL_RESPONSE, score_citation_if

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


def extract_response_shape(response: Dict[str, Any]) -> Tuple[str, int]:
    """Return (output_text, function_call_count) from a Responses API output dict."""
    text_parts = []
    function_calls = 0
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            function_calls += 1
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        for block in item.get("content", []) or []:
            if (
                isinstance(block, dict)
                and block.get("type") in {"output_text", "text"}
                and isinstance(block.get("text"), str)
            ):
                text_parts.append(block["text"])
    return "\n".join(p for p in text_parts if p).strip(), function_calls


# Server


class CitationIfResourcesServerConfig(BaseResourcesServerConfig):
    pass


class CitationIfVerifyRequest(BaseVerifyRequest):
    verifier: Dict[str, Any]


class CitationIfVerifyResponse(BaseVerifyResponse):
    verifier: Dict[str, Any]
    match_details: Dict[str, Any]


class CitationIfResourcesServer(SimpleResourcesServer):
    config: CitationIfResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # Catch-all: any tool call during a rollout returns the terminal string.
        # Combined with max_steps=1 in the agent config, this ensures tool-calling
        # rollouts terminate and the final output has function_call_count > 0,
        # which fails Gate 0 (structural) and scores 0.
        app.post("/{tool_name}")(self._tool_catchall)
        return app

    async def _tool_catchall(self, tool_name: str, request: Request) -> PlainTextResponse:  # noqa: ARG002
        return PlainTextResponse(TERMINAL_TOOL_RESPONSE)

    async def verify(self, body: CitationIfVerifyRequest) -> CitationIfVerifyResponse:
        text, function_call_count = extract_response_shape(body.response.model_dump())
        reward, details = score_citation_if(text, function_call_count, body.verifier)
        return CitationIfVerifyResponse(
            **body.model_dump(),
            reward=reward,
            match_details=details,
        )


if __name__ == "__main__":
    CitationIfResourcesServer.run_webserver()
