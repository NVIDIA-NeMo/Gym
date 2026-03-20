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
RDKit Chemistry — Nemo-Gym Resources Server

Verifiable chemistry question answering with optional Python tool-use.

The agent receives a natural-language chemistry question paired with a SMILES
string and must respond with a single number (integer or float) or a binary
0/1 flag.

Questions are drawn from a stratified sample of the ChEMBL database and cover
RDKit-computable molecular properties (logP, molecular weight, ring counts,
hydrogen bond donor/acceptor counts, fragment presence, etc.).

Two question methods are supported (selected per-row via the ``method`` field):

* **direct** — the model answers from parametric knowledge alone.
* **mcp-python** — the model may call a Python tool (via ``ns_tools`` wrapper)
  to compute the answer using RDKit.

This server is a pure verifier: it only implements ``verify()``.  When tool-use
is needed, pair this server with ``ns_tools`` via
``rdkit_chemistry_with_tools.yaml`` — ``ns_tools`` handles tool execution and
delegates verification here.

Reward signal
-------------
- Integer / count / bool / presence / fragment properties: exact match
  (reward = 1.0 iff round(predicted) == round(actual), else 0.0).
- Float properties: reward = 1 / (1 + |predicted - actual|) for continuous
  properties in _INVERSE_ERROR_PROPERTIES list; reward ranges from (0, 1]
  with 1.0 for a perfect prediction. Other float properties: reward =
  -|predicted - actual| (negative absolute error). A perfect prediction
  scores 0.0; larger errors give more negative rewards. When no numeric
  value can be extracted from the response, reward = 0.0.

Dataset format (JSONL)
----------------------
Each row carries:
  responses_create_params.input  — user message (prompt + format instruction)
  responses_create_params.tools  — [] for direct, [stateful_python_code_exec] for mcp-python
  expected_answer                — ground-truth numeric value
  property_type                  — "float" | "count" | "bool" | "presence" | "fragment"
  property                       — RDKit property name, e.g. "MolLogP"
  chembl_id                      — ChEMBL molecule identifier
  smiles                         — canonical SMILES string
  method                         — "direct" | "mcp-python"
"""

from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI

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

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_DOUBLE_PAREN_RE = re.compile(r"\(\(([^)]+)\)\)")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class RDKitChemistryConfig(BaseResourcesServerConfig):
    pass


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChemistryRunRequest(BaseRunRequest):
    expected_answer: Union[str, float, int]
    property_type: str
    property: str
    chembl_id: Optional[str] = None
    smiles: Optional[str] = None
    method: Optional[str] = None
    use_box_format: bool = False


class ChemistryVerifyRequest(ChemistryRunRequest, BaseVerifyRequest):
    pass


class ChemistryVerifyResponse(BaseVerifyResponse):
    predicted_value: Optional[float] = None
    correct: bool = False
    absolute_error: Optional[float] = None
    property: str = ""
    property_type: str = ""
    chembl_id: Optional[str] = None
    method: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers: response text extraction
# ---------------------------------------------------------------------------


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    """Extract the final assistant text from a Responses API output trajectory."""
    texts: list[str] = []
    for output_item in body.response.output:
        if getattr(output_item, "type", None) == "message" and getattr(output_item, "role", None) == "assistant":
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
# Helpers: value extraction
# ---------------------------------------------------------------------------


def _extract_from_boxed(text: str) -> Optional[float]:
    """Extract a numeric value from the last ``\\boxed{...}`` in *text*.

    Returns None if no boxed expression is found or the content is not numeric.
    """
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    inner = matches[-1].strip()
    try:
        return float(inner)
    except (ValueError, TypeError):
        pass
    nums = _NUMBER_RE.findall(inner)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass
    return None


def _extract_from_double_parens(text: str) -> Optional[float]:
    """Extract a numeric value from the last ``((...))`` in *text*.

    Returns None if no double-parenthesised expression is found or the
    content is not numeric.
    """
    matches = _DOUBLE_PAREN_RE.findall(text)
    if not matches:
        return None
    inner = matches[-1].strip()
    try:
        return float(inner)
    except (ValueError, TypeError):
        pass
    nums = _NUMBER_RE.findall(inner)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass
    return None


def extract_predicted_value(
    response: str,
    property_type: str,
    *,
    use_box_format: bool = False,
) -> Optional[float]:
    """
    Extract a predicted numeric value from the model's response text.

    When *use_box_format* is True the answer **must** appear inside a
    ``\\boxed{...}`` expression (as requested in the prompt).  Only the
    content of the last ``\\boxed`` is considered; if none is found the
    function returns None (→ reward 0).

    When *use_box_format* is False the answer **must** appear inside
    double parentheses ``((...))``.  Only the content of the last ``((...))``
    is considered; if none is found the function returns None (→ reward 0).

    Returns None if no value can be extracted.
    """
    if not isinstance(response, str):
        return None

    text = response.strip()

    if use_box_format:
        return _extract_from_boxed(text)

    return _extract_from_double_parens(text)


# ---------------------------------------------------------------------------
# Helpers: reward computation
# ---------------------------------------------------------------------------


# List of properties for which the reward is computed as 1 / (1 + |predicted - actual|)
_INVERSE_ERROR_PROPERTIES = frozenset(
    {"TPSA", "ExactMolWt", "FractionCSP3", "HeavyAtomMolWt", "MolLogP", "MolWt", "qed"}

def compute_reward(
    predicted: Optional[float],
    actual: float,
    property_type: str,
    property_name: str = "",
) -> float:
    """
    Compute a scalar reward given a prediction.

    Float properties in _INVERSE_ERROR_PROPERTIES:
      reward = 1 / (1 + |predicted - actual|)  (ranges (0, 1], perfect = 1.0).
    Other float properties: reward = -|predicted - actual|  (negative absolute error).
    Discrete properties (count / bool / presence / fragment):
      reward = 1.0 if round(predicted) == round(actual), else 0.0.
    No prediction (None / NaN) scores 0.0.
    """
    if predicted is None or math.isnan(predicted):
        return 0.0

    if property_type == "float":
        # Alternatively can remove the list and remove property_name from function call
        # to run on all float properties
        error = abs(predicted - actual)
        if property_name in _INVERSE_ERROR_PROPERTIES:
            return 1.0 / (1.0 + error)
        return -error

    return 1.0 if round(predicted) == round(actual) else 0.0


# ---------------------------------------------------------------------------
# Resources server
# ---------------------------------------------------------------------------


class RDKitChemistryResourcesServer(SimpleResourcesServer):
    config: RDKitChemistryConfig

    def setup_webserver(self) -> FastAPI:
        return super().setup_webserver()

    async def verify(
        self,
        body: ChemistryVerifyRequest,
    ) -> ChemistryVerifyResponse:
        text = _extract_last_assistant_text(body)
        predicted = extract_predicted_value(text, body.property_type, use_box_format=body.use_box_format)
        actual = float(body.expected_answer)

        reward = compute_reward(predicted, actual, body.property_type, property_name=body.property)

        absolute_error: Optional[float] = None
        if body.property_type == "float" and predicted is not None and not math.isnan(predicted):
            absolute_error = abs(predicted - actual)

        correct = reward == 1.0

        return ChemistryVerifyResponse(
            **body.model_dump(),
            reward=reward,
            predicted_value=predicted,
            correct=correct,
            absolute_error=absolute_error,
        )

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        rollouts = [r for task in tasks for r in task]

        grouped: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for r in rollouts:
            method = r.get("method", "unknown") or "unknown"
            ptype = r.get("property_type", "unknown") or "unknown"
            grouped[method][ptype].append(r)

        def _ptype_stats(group: list) -> Dict[str, Any]:
            rewards = [r["reward"] for r in group]
            corrects = [int(r.get("correct", False)) for r in group]
            stats: Dict[str, Any] = {
                "count": len(group),
                "accuracy": statistics.mean(corrects),
                "mean_reward": statistics.mean(rewards),
            }
            errors = [r["absolute_error"] for r in group if r.get("absolute_error") is not None]
            if errors:
                stats["mean_abs_error"] = statistics.mean(errors)
                stats["median_abs_error"] = statistics.median(errors)
            return stats

        result: Dict[str, Any] = {}
        for method in sorted(grouped):
            method_rollouts = [r for ptype_group in grouped[method].values() for r in ptype_group]
            method_rewards = [r["reward"] for r in method_rollouts]
            method_corrects = [int(r.get("correct", False)) for r in method_rollouts]
            by_ptype = {ptype: _ptype_stats(g) for ptype, g in sorted(grouped[method].items())}
            result[method] = {
                "count": len(method_rollouts),
                "accuracy": statistics.mean(method_corrects),
                "mean_reward": statistics.mean(method_rewards),
                "by_property_type": by_ptype,
            }
        return result

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        keys = {"mean/reward", "mean/correct"}
        return {k: v for k, v in agent_metrics.items() if k in keys or k in ("direct", "mcp-python")}


if __name__ == "__main__":
    RDKitChemistryResourcesServer.run_webserver()
