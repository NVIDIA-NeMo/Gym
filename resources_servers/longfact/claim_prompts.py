# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned claim-extraction prompts shared by LongFact and replay tools."""

from __future__ import annotations

from typing import Literal


ClaimExtractionPrompt = Literal["legacy_v1", "claim_listing_v1"]


def build_claim_extraction_request(version: ClaimExtractionPrompt, prompt: str, response: str) -> str:
    if version == "claim_listing_v1":
        return f"""### Introduction
Your task is to list relevant facts in an assistant's response to a given prompt. Your output will be used as the first step in the following fact-checking pipeline used to evaluate an assistant's response for factual correctness.

Fact-Checking Pipeline:
1. Given a prompt and assistant's response, list all relevant factual claims made by the assistant.
2. Separate the list of N claims into M manageable groups.
3. For each group of claims, fact-check each claim in the group by browsing the web to find evidence supporting or refuting the claim.

### Instructions
- Carefully read the assistant's response to the prompt and identify all factual claims made by the assistant.
- You should isolate your focus to real-world facts (e.g., facts about news, people, places, events, etc.).
- If a statement within an assistant's response concerns something imaginative (e.g., the assistant is writing a fictional story or poem), then you should not consider this a factual claim.
- For each factual claim that you list, another assistant will be tasked with fact-checking it by browsing the web to find evidence supporting or refuting the claim.
- Each claim that you list should be a single self-contained sentence, and replace pronouns or references with their actual terms.
- You should only consider claims that are relevant for answering the prompt. We consider a claim to be relevant if the subject of the claim is either exactly contained or related to any subject present in the prompt.
- If the same claim is repeated multiple times, you should only list it once.
- Try to list claims in the order that they appear in the assistant's response, so that related claims are grouped together.

### Formatting
Return only a raw JSON array of strings, without Markdown fences or commentary:
[
  "fact_1",
  "fact_2"
]

### Example
Prompt:
Who is Barack Obama?

Response:
Barack Obama is an American politician and attorney who served as the 44th President of the United States from 2009 to 2017. A member of the Democratic Party, he was the first African American president in U.S. history.

Output:
[
  "Barack Obama is an American politician.",
  "Barack Obama is an attorney.",
  "Barack Obama served as the 44th President of the United States.",
  "Barack Obama served as president from 2009 to 2017.",
  "Barack Obama is a member of the Democratic Party.",
  "Barack Obama was the first African American president in United States history."
]

The assistant's response may be much longer than the example and may contain up to 100 separate claims.

### Task
Prompt:
{prompt}

Response:
{response}
"""

    if version != "legacy_v1":
        raise ValueError(f"Unknown claim extraction prompt: {version}")
    return f"""Break the RESPONSE into atomic, independently verifiable factual claims.
Exclude opinions, advice, formatting statements, and purely conversational text.
Return only a JSON array of strings. Each string must contain exactly one factual claim.

RESPONSE:
{response}
"""
