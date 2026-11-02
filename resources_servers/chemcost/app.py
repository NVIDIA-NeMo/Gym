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
"""ChemCost resources server: chemical reaction procurement-cost estimation with supplier-quote tools.

ChemCost (Wu et al., NeurIPS 2026, https://github.com/HeinzVonHank/ChemCost) gives an agent a reaction as it
was run in the laboratory and asks for the US-dollar cost of buying its materials, per gram of product, at
1 mmol of a named scale reference. The agent has the benchmark's two tools: ``get_supplier_quotes`` over the
frozen supplier-quote snapshot and ``calculate``. Both routes call the authors' ``chemcost`` package (MIT,
pinned in requirements.txt), and ``verify()`` uses its CTA@k rule and a linear-time port of its answer parser
(answer_parsing.py), so a score here is the
upstream computation: an answer within k% of the label counts, an abstention or an unparseable answer counts
as a failure. ``reward`` is CTA@25 for one reaction (0 or 1); CTA@10 is reported beside it.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.failure_kinds import AGENT_INVALID_OUTPUT, AGENT_LIMIT, VERIFIER_ERROR
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.chemcost.answer_parsing import parse_final_answer
from resources_servers.chemcost.pricing_data import ensure_pricing_db


REPO_ROOT = Path(__file__).absolute().parents[2]
CTA_TOLERANCES = (10, 25)
REWARD_TOLERANCE = 25


class ChemCostResourcesServerConfig(BaseResourcesServerConfig):
    # Frozen pricing SQLite; null downloads the pinned release into resources_servers/chemcost/.data/.
    pricing_db_path: Optional[str] = None


class ChemCostVerifierMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    reaction_id: str
    type: Literal["single_step", "multi_step"]
    procurement_cost_usd_per_g_product: Optional[float] = None


class ChemCostVerifyRequest(BaseVerifyRequest):
    verifier_metadata: ChemCostVerifierMetadata


class ChemCostVerifyResponse(ChemCostVerifyRequest, BaseVerifyResponse):
    # False only when the row has no label to score against (gym-p0/v1 TE-6 requires the field).
    evaluation_completed: bool = True
    predicted_cost_per_gram: Optional[float] = None
    relative_error: Optional[float] = None
    abstained: bool = True
    cta_at_10: float = 0.0
    cta_at_25: float = 0.0


class SupplierQuotesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    smiles_or_name: Any = None


class CalculateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    expression: Any = None


def final_message_text(response: Any) -> Optional[str]:
    """Text of the terminal assistant message: what the upstream Codex harness reads as the final answer.

    None when the episode ended on a tool call or its output (simple_agent stopped at ``max_steps``):
    a message before that is interim commentary, not an answer.
    """
    for item in reversed(response.output):
        item_type = getattr(item, "type", None)
        if item_type == "reasoning":
            continue
        if item_type == "message":
            return "".join(getattr(part, "text", "") or "" for part in (item.content or []))
        return None
    return None


def truncation_reason(response: Any) -> Optional[str]:
    """Why the model's output was cut (max_output_tokens, content filter), or None when it finished."""
    details = getattr(response, "incomplete_details", None)
    return details.get("reason") if isinstance(details, dict) else getattr(details, "reason", None)


def abstention_cause(response: Any, text: Optional[str]) -> tuple[str, str]:
    """failure_kind and failure_reason of a rollout without a numeric prediction."""
    reason = truncation_reason(response)
    if reason is not None:
        return AGENT_LIMIT, f"response ended early ({reason})"
    termination = (getattr(response, "metadata", None) or {}).get("ng_termination_reason")
    if termination:
        # simple_agent marks reasoning-only or empty model output incomplete; it is not a limit.
        return AGENT_INVALID_OUTPUT, f"model stopped without an answer ({termination})"
    tool_items = ("function_call", "function_call_output")
    if text is None and any(getattr(item, "type", None) in tool_items for item in response.output):
        return AGENT_LIMIT, "step limit reached with tool calls pending"
    if not text:
        return AGENT_INVALID_OUTPUT, "no final assistant message"
    return AGENT_INVALID_OUTPUT, "no numeric predicted_cost_per_gram"


class ChemCostVerifier:
    async def verify(self, body: ChemCostVerifyRequest) -> ChemCostVerifyResponse:
        from chemcost.evaluation.evaluator import _coerce_cost
        from chemcost.evaluation.metrics import CostPrediction, cta_at_k, tcre

        payload = body.model_dump()
        meta = body.verifier_metadata
        true_cost = meta.procurement_cost_usd_per_g_product
        if true_cost is None or true_cost <= 0:
            return ChemCostVerifyResponse(
                **payload,
                reward=0.0,
                mask_sample=True,
                evaluation_completed=False,
                failure_kind=VERIFIER_ERROR,
                failure_reason=f"{meta.reaction_id} has no positive ground-truth label",
            )

        # A cut-off message is an unfinished estimate, not an answer, even when it already holds a figure.
        text = None if truncation_reason(body.response) else final_message_text(body.response)
        parsed = (await asyncio.to_thread(parse_final_answer, text) if text else None) or {}
        try:
            predicted = _coerce_cost(parsed.get("predicted_cost_per_gram"))
        except OverflowError:
            # An integer too large for a float: upstream's _coerce_cost catches only TypeError and ValueError.
            predicted = None
        prediction = CostPrediction(
            reaction_id=meta.reaction_id,
            predicted_cost=predicted,
            true_cost=true_cost,
            predicted_components=[],
            true_component_names=[],
        )
        cta = {k: cta_at_k([prediction], k) for k in CTA_TOLERANCES}
        failure_kind = failure_reason = None
        if predicted is None:
            # An abstention is a scored failure (upstream keeps it in the CTA denominator), not a masked sample.
            failure_kind, failure_reason = abstention_cause(body.response, text)
        return ChemCostVerifyResponse(
            **payload,
            reward=cta[REWARD_TOLERANCE],
            predicted_cost_per_gram=predicted,
            relative_error=tcre(predicted, true_cost) if predicted is not None else None,
            abstained=predicted is None,
            cta_at_10=cta[10],
            cta_at_25=cta[25],
            failure_kind=failure_kind,
            failure_reason=failure_reason,
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """CTA@k by reaction type, as the upstream README reports CTA@10 for single- and multi-step routes."""
        by_type: dict[str, list[dict[str, Any]]] = {}
        for task in tasks:
            for rollout in task:
                if rollout.get("mask_sample"):
                    continue
                by_type.setdefault(rollout["verifier_metadata"]["type"], []).append(rollout)
        metrics: dict[str, Any] = {}
        for reaction_type, rollouts in sorted(by_type.items()):
            for k in CTA_TOLERANCES:
                metrics[f"{reaction_type}/cta@{k}"] = sum(r[f"cta_at_{k}"] for r in rollouts) / len(rollouts)
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        keys = ("mean/cta_at_10", "mean/cta_at_25", "mean/abstained")
        return {k: agent_metrics[k] for k in keys if k in agent_metrics}


class ChemCostResourcesServer(ChemCostVerifier, SimpleResourcesServer):
    # Tool results depend only on the request and the frozen database.
    checkpoint_mode = "stateless"
    config: ChemCostResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        configured = self.config.pricing_db_path
        db_path = ensure_pricing_db(REPO_ROOT / configured if configured else None)
        # chemcost reads the database path from this variable when it is first imported.
        os.environ["CHEMCOST_PRICING_DB"] = str(db_path)
        from chemcost.pricing.pricing_db import DEFAULT_DB_PATH

        if Path(DEFAULT_DB_PATH) != db_path:
            raise RuntimeError(f"chemcost was imported before its database was set ({DEFAULT_DB_PATH})")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/get_supplier_quotes")(self.get_supplier_quotes)
        app.post("/calculate")(self.calculate)
        return app

    async def get_supplier_quotes(self, body: SupplierQuotesRequest) -> dict:
        from chemcost.tools.agent_tools import get_supplier_quotes

        if body.smiles_or_name is None:
            return {"error": "Missing argument(s) for get_supplier_quotes: smiles_or_name"}
        # SQLite and RDKit lookups block; the package keeps one connection per thread.
        return await asyncio.to_thread(_call_tool, get_supplier_quotes, str(body.smiles_or_name))

    async def calculate(self, body: CalculateRequest) -> dict:
        from chemcost.tools.agent_tools import calculate

        if body.expression is None:
            return {"error": "Missing argument(s) for calculate: expression"}
        return _call_tool(calculate, str(body.expression))


def _call_tool(fn: Any, arg: str) -> dict:
    """Run one upstream tool; a failure goes back to the model as an error result, as upstream's run_tool does."""
    try:
        return fn(arg)
    except Exception as e:  # noqa: BLE001 - tool boundary: the model sees the error, the episode continues
        return {"error": f"{type(e).__name__}: {e}"}


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=ChemCostVerifier,
    request_model=ChemCostVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    ChemCostResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ChemCostResourcesServer.run_webserver()  # noqa: F401
