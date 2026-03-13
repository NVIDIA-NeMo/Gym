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
  (reward = 1.0 iff predicted == actual, else 0.0).
- Float properties: percentile-based threshold.  A prediction earns 1.0 if
  its absolute error |pred - actual| is smaller than
  FLOAT_ACCURACY_THRESHOLD (default 0.95) of all |v - actual| across the
  full ChEMBL distribution for that property.  Equivalently: the model must
  be closer to the true value than 95% of a random draw from the prior.
  This threshold is pre-computed from features.parquet and stored as quantile
  arrays in reward_stats.json alongside this environment.

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

import json
import math
import re
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


# ---------------------------------------------------------------------------
# Constants — must match scripts/precompute_reward_stats.py
# ---------------------------------------------------------------------------

FLOAT_ACCURACY_THRESHOLD = 0.95

# Property types that use binary 0/1 values and exact-match scoring
_BOOL_PROPERTY_TYPES = {"presence", "fragment", "bool"}

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_BOOL_TRUE_RE  = re.compile(r"\b(?:yes|true)\b",  re.IGNORECASE)
_BOOL_FALSE_RE = re.compile(r"\b(?:no|false)\b",  re.IGNORECASE)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ChemistryDirectConfig(BaseResourcesServerConfig):
    reward_stats_path: str = Field(
        default="resources_servers/chemistry_direct/data/reward_stats.json",
        description=(
            "Path to reward_stats.json produced by precompute_reward_stats.py. "
            "Contains per-property quantile arrays used for percentile-based "
            "float scoring.  Relative paths are resolved from the working "
            "directory where the server is launched."
        ),
    )


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
    correct: bool = False
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

def _float_reward(
    predicted: float,
    actual: float,
    quantiles: list[float],
    threshold: float,
) -> float:
    """
    Percentile-based reward for continuous (float) properties.

    Approximates the fraction of the ChEMBL distribution whose absolute
    error w.r.t. `actual` is >= the prediction's absolute error, using the
    stored quantile array.

    Scoring rule (matches compute_metrics._float_accuracy_for_group):
      reward = 1.0  if  P(|v - actual| >= |pred - actual|) > threshold
               0.0  otherwise

    The quantile array stores N evenly-spaced quantile values of the
    property distribution.  We reconstruct |v - actual| values from those
    quantiles, then compute the fraction >= pred_error via bisect.
    """
    if math.isnan(predicted):
        return 0.0

    pred_error = abs(predicted - actual)
    q = np.asarray(quantiles, dtype=np.float64)
    feat_errors = np.abs(q - actual)
    fraction_ge = float(np.mean(feat_errors >= pred_error))
    return 1.0 if fraction_ge > threshold else 0.0


def compute_reward(
    predicted: Optional[float],
    actual: float,
    property_type: str,
    property_name: str,
    reward_stats: dict[str, Any],
    threshold: float,
) -> float:
    """
    Compute a binary reward (0.0 or 1.0) given a prediction.

    - None / NaN prediction → 0.0 (no answer or unparseable)
    - Integer / count / bool / presence / fragment → exact match
    - Float → percentile-based threshold (requires reward_stats entry)
    """
    if predicted is None or math.isnan(predicted):
        return 0.0

    if property_type != "float":
        return 1.0 if round(predicted) == round(actual) else 0.0

    prop_stats = reward_stats.get("properties", {}).get(property_name)
    if prop_stats is None:
        # Property not found in reward_stats — fall back to a generous relative-error check
        if actual == 0:
            return 1.0 if abs(predicted) < 1e-6 else 0.0
        return 1.0 if abs(predicted - actual) / abs(actual) < 0.05 else 0.0

    return _float_reward(predicted, actual, prop_stats["quantiles"], threshold)


# ---------------------------------------------------------------------------
# Resources server
# ---------------------------------------------------------------------------

class ChemistryDirectResourcesServer(SimpleResourcesServer):
    config: ChemistryDirectConfig

    def __init__(self, config: ChemistryDirectConfig) -> None:
        super().__init__(config)
        self._reward_stats: dict[str, Any] = self._load_reward_stats(config.reward_stats_path)
        self._threshold: float = self._reward_stats.get(
            "float_accuracy_threshold", FLOAT_ACCURACY_THRESHOLD
        )

    @staticmethod
    def _load_reward_stats(path: str) -> dict[str, Any]:
        p = Path(path)
        if not p.exists():
            # Graceful degradation: float properties will use the fallback rule
            import warnings
            warnings.warn(
                f"reward_stats.json not found at {p}. Float properties will use "
                "a relative-error fallback (|pred-actual|/|actual| < 5%). "
                "Run 'make precompute-reward-stats' and copy the output to "
                f"{p} for accurate percentile-based scoring.",
                stacklevel=2,
            )
            return {}
        with p.open() as fh:
            stats = json.load(fh)
        n_props = len(stats.get("properties", {}))
        print(f"Loaded reward_stats from {p} ({n_props} float properties)")
        return stats

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(
        self, body: ChemistryDirectVerifyRequest
    ) -> ChemistryDirectVerifyResponse:
        text = _extract_last_assistant_text(body)
        predicted = extract_predicted_value(text, body.property_type)
        actual = float(body.expected_answer)

        reward = compute_reward(
            predicted,
            actual,
            body.property_type,
            body.property,
            self._reward_stats,
            self._threshold,
        )

        return ChemistryDirectVerifyResponse(
            **body.model_dump(),
            reward=reward,
            predicted_value=predicted,
            correct=(reward == 1.0),
            property=body.property,
            property_type=body.property_type,
            chembl_id=body.chembl_id,
        )

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        """Expose mean reward (= accuracy) as the headline metric."""
        return {k: v for k, v in agent_metrics.items() if k in ("mean/reward", "mean/correct")}


if __name__ == "__main__":
    ChemistryDirectResourcesServer.run_webserver()
