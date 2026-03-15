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
"""
Chemistry Direct — Nemo-Gym Resources Server

Verifiable chemistry question answering (direct generation variant).

The agent receives a natural-language chemistry question paired with a SMILES
string and must respond with a single number (integer or float) or a binary
0/1 flag.  No tool calls are permitted; the format instruction embedded in the
prompt specifies the exact output format.

Questions are drawn from a stratified sample of the ChEMBL database and cover
RDKit-computable molecular properties (logP, molecular weight, ring counts,
hydrogen bond donor/acceptor counts, fragment presence, etc.).

Reward signal
-------------
- Integer / count / bool / presence / fragment properties: exact match
  (reward = 1.0 iff round(predicted) == round(actual), else 0.0).
- Float properties: negative absolute error — reward = –|predicted – actual|.
  A perfect prediction scores 0.0; larger errors give more negative rewards.
  When no numeric value can be extracted from the response, reward = 0.0.

Dataset format (JSONL)
----------------------
Each row carries:
  responses_create_params.input  — user message (prompt + format instruction)
  expected_answer                — ground-truth numeric value
  property_type                  — "float" | "count" | "bool" | "presence" | "fragment"
  property                       — RDKit property name, e.g. "MolLogP"
  chembl_id                      — ChEMBL molecule identifier
  smiles                         — canonical SMILES string

See scripts/export_nemo_gym_data.py for data generation.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Property types that use binary 0/1 values and exact-match scoring
_BOOL_PROPERTY_TYPES = {"presence", "fragment", "bool"}

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_BOOL_TRUE_RE  = re.compile(r"\b(?:yes|true)\b",  re.IGNORECASE)
_BOOL_FALSE_RE = re.compile(r"\b(?:no|false)\b",  re.IGNORECASE)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ChemistryDirectConfig(BaseResourcesServerConfig):
    pass


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChemistryDirectRunRequest(BaseRunRequest):
    expected_answer: Union[float, int]
    property_type: str
    property: str
    chembl_id: Optional[str] = None
    smiles: Optional[str] = None


class ChemistryDirectVerifyRequest(ChemistryDirectRunRequest, BaseVerifyRequest):
    pass


class ChemistryDirectVerifyResponse(BaseVerifyResponse):
    predicted_value: Optional[float] = None
    correct: bool = False          # True only for exact-match property types
    absolute_error: Optional[float] = None  # populated for float properties
    property: str = ""
    property_type: str = ""
    chembl_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers: response text extraction
# ---------------------------------------------------------------------------

def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    """Extract the final assistant text from a Responses API output trajectory."""
    texts: list[str] = []
    for output_item in body.response.output:
        if (
            getattr(output_item, "type", None) == "message"
            and getattr(output_item, "role", None) == "assistant"
        ):
            content = getattr(output_item, "content", None)
            if isinstance(content, list):
                for part in content:
                    t = getattr(part, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


# ---------------------------------------------------------------------------
# Helpers: value extraction (mirrors compute_metrics.extract_predicted_value)
# ---------------------------------------------------------------------------

def extract_predicted_value(response: str, property_type: str) -> Optional[float]:
    """
    Extract a predicted numeric value from the model's response text.

    Replicates the three-step cascade from scripts/compute_metrics.py:
      1. Strict parse  — treat the entire stripped response as a number
      2. Permissive    — find the last number anywhere in the text
      3. Boolean text  — map yes/true → 1.0, no/false → 0.0 (presence/fragment)

    Returns None if no value can be extracted.
    """
    if not isinstance(response, str):
        return None

    text = response.strip()

    # 1. Strict
    try:
        return float(text.rstrip("."))
    except (ValueError, TypeError):
        pass

    # 2. Last number in text
    nums = _NUMBER_RE.findall(text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass

    # 3. Boolean fallback for binary-valued properties
    if property_type in _BOOL_PROPERTY_TYPES:
        if _BOOL_TRUE_RE.search(text):
            return 1.0
        if _BOOL_FALSE_RE.search(text):
            return 0.0

    return None


# ---------------------------------------------------------------------------
# Helpers: reward computation
# ---------------------------------------------------------------------------

def compute_reward(
    predicted: Optional[float],
    actual: float,
    property_type: str,
) -> float:
    """
    Compute a scalar reward given a prediction.

    Float properties: reward = –|predicted – actual|  (negative absolute error).
      A perfect prediction scores 0.0; larger errors give more negative rewards.
      No prediction (None / NaN) scores 0.0.

    Discrete properties (count / bool / presence / fragment):
      reward = 1.0 if round(predicted) == round(actual), else 0.0.
      No prediction scores 0.0.
    """
    if predicted is None or math.isnan(predicted):
        return 0.0

    if property_type == "float":
        return -abs(predicted - actual)

    return 1.0 if round(predicted) == round(actual) else 0.0


# ---------------------------------------------------------------------------
# Resources server
# ---------------------------------------------------------------------------

class ChemistryDirectResourcesServer(SimpleResourcesServer):
    config: ChemistryDirectConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(
        self, body: ChemistryDirectVerifyRequest
    ) -> ChemistryDirectVerifyResponse:
        text = _extract_last_assistant_text(body)
        predicted = extract_predicted_value(text, body.property_type)
        actual = float(body.expected_answer)

        reward = compute_reward(predicted, actual, body.property_type)

        absolute_error: Optional[float] = None
        if body.property_type == "float" and predicted is not None and not math.isnan(predicted):
            absolute_error = abs(predicted - actual)

        correct = reward == 1.0  # only meaningful for exact-match property types

        return ChemistryDirectVerifyResponse(
            **body.model_dump(),
            reward=reward,
            predicted_value=predicted,
            correct=correct,
            absolute_error=absolute_error,
            property=body.property,
            property_type=body.property_type,
            chembl_id=body.chembl_id,
        )

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        """Expose mean reward as the headline metric.

        For float-only rollouts mean/reward equals –MAE.
        For discrete-only rollouts mean/reward equals accuracy.
        """
        return {k: v for k, v in agent_metrics.items() if k in ("mean/reward", "mean/correct")}


if __name__ == "__main__":
    ChemistryDirectResourcesServer.run_webserver()
