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
#
# Adapted from ChemCost, https://github.com/HeinzVonHank/ChemCost at d7de381d782835e51840d7d371586d3648836cfd
# (MIT License): src/chemcost/evaluation/evaluator.py (prepare_agent_input
# and its helpers), src/chemcost/baselines/react_agent.py (TASK_DEFINITION, SYSTEM_PROMPT, shown,
# build_components_text_for_prompt, build_task_prompt, the Responses API tool definitions) and
# src/chemcost/tools/agent_tools.py (TOOL_REGISTRY descriptions). Only the clean (no noise) view is kept.
# tests/test_prompts.py checks every prepared row against the pinned package.
# The original code carries this notice:
#
# Copyright (c) 2026 The ChemCost Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""The agent-facing view, prompts and tool definitions of ChemCost's ReAct and Codex harnesses.

``benchmarks/chemcost/prepare.py`` runs in Gym's own environment, which does not install the ``chemcost``
package, so the prompt code is transcribed here rather than imported.
"""

from typing import Any, Optional


SYSTEM_PROMPT = """\
## Task
You are given a reaction as it was run in the laboratory: each component's name and the amount used
(not recorded for some components), the product and the mass of it that was isolated, and the scale
reference compound. Estimate what it costs to buy the materials for this reaction, per gram of product.

## Definition of the answer
- Scale: the reaction is run with exactly 1 mmol of the scale reference compound; every other
  component is scaled in proportion to the laboratory amounts.
- Purchases: for each component that has to be bought, buy whole packs (any combination of pack sizes
  and suppliers), the cheapest way that gives at least the amount needed. Solvents are not bought.
- Which offers count: only offers whose purity is reported and is at least 95% (a purity written as
  a fraction, such as 0.98, means 98%). Reagents whose ordinary commercial grade is below that count
  at their grade: potassium hydroxide from 85%, sodium hydride from 55%, mCPBA from 70%.
- Product: the product obtained scales with the run, in proportion to the laboratory run.
- Multi-step routes: every step is run with 1 mmol of that step's scale reference. A component named
  "product of step k" is made in step k, not bought: it costs what step k costs per gram of its product.
  Report the cost per gram of the final product.
- The answer is the total purchase cost divided by the grams of product obtained.

## Tools
- get_supplier_quotes(smiles_or_name): every supplier offer for a chemical as listed (supplier, pack
  size with unit, price_usd, reported purity). These are the only offers you may buy from; not all of
  them count (see the definition above).
- calculate(expression): evaluate a mathematical expression.

This is a non-interactive benchmark: never ask questions and always end with a final answer. If you
cannot complete the estimate, answer {"predicted_cost_per_gram": null, "predicted_components": []}.

Final answer, as JSON only:
{"predicted_cost_per_gram": <number>, "predicted_components": [{"name": <str>, "price_per_gram": <number>}]}
"""

_TOOL_REGISTRY = {
    "get_supplier_quotes": (
        "Get every supplier quote for a chemical from the frozen pricing database, as the suppliers "
        "listed them: supplier, pack size with its unit, price in USD, and reported purity (null when "
        "not reported). Quotes are not filtered: decide yourself which ones qualify. "
        "If no pack quotes are returned, the chemical is unpriced.",
        {"smiles_or_name": "SMILES string (preferred) or chemical name."},
    ),
    "calculate": (
        "Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, log.",
        {"expression": "A mathematical expression string."},
    ),
}

# The function tools as upstream's Responses API loop declares them.
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {p: {"type": "string", "description": d} for p, d in params.items()},
            "required": list(params),
            "additionalProperties": False,
        },
        "strict": True,
    }
    for name, (description, params) in _TOOL_REGISTRY.items()
]


def _visible_name(component: dict) -> str:
    if component.get("from_step") is not None:
        return f"product of step {component['from_step']}"
    return (component.get("name") or "").strip() or "UNKNOWN_COMPONENT"


def _format_amount(component: dict) -> Optional[str]:
    grams = component.get("amount_g")
    if grams is None or component.get("role") == "solvent":
        return None
    return f"{grams:.5g} g" if grams >= 1 else f"{grams * 1000:.5g} mg"


def _product_label(product: Optional[dict]) -> Optional[str]:
    product = product or {}
    if product.get("display_name"):
        return product["display_name"]
    return f"compound of formula {product['formula']}" if product.get("formula") else None


def _product_mass(product: Optional[dict], yield_percent: Any) -> Optional[str]:
    grams = (product or {}).get("mass_g")
    if grams is None or yield_percent is None:
        return None
    return f"{grams:.5g} g" if grams >= 1 else f"{grams * 1000:.5g} mg"


def agent_view(record: dict) -> dict:
    """``prepare_agent_input``: names and laboratory amounts, the product and its isolated mass, the scale reference.

    No roles, structures, equivalents, molecular weights, yields, prices or ids.
    """
    components = []
    for c in record.get("components", []):
        if c.get("role") == "product" or c.get("original_role") == "product":
            continue
        row = {"name": _visible_name(c), "amount": _format_amount(c)}
        if c.get("original_step") is not None:
            row["step"] = c["original_step"]
        components.append(row)
    product = record.get("product") if isinstance(record.get("product"), dict) else {}
    view = {
        "product": _product_label(product),
        "product_mass": _product_mass(product, record.get("yield_percent")),
        "scale_reference": record.get("scale_reference"),
        "components": components,
    }
    if record.get("steps"):
        view["steps"] = [
            {
                "step_number": st.get("step_number"),
                "product": _product_label(st.get("product")),
                "product_mass": _product_mass(st.get("product"), st.get("yield_percent")),
                "scale_reference": st.get("scale_reference"),
            }
            for st in record["steps"]
        ]
    return view


def _shown(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def task_prompt(view: dict) -> str:
    """``build_task_prompt`` with tools: the user message of every upstream tool-using agent."""
    lines = []
    for c in view.get("components", []):
        step = f"[step {c['step']}] " if c.get("step") is not None else ""
        lines.append(f"  - {step}{c['name'] or '(unnamed)'} | amount used: {c.get('amount') or 'not recorded'}")
    steps = view.get("steps")
    if steps:
        lines.append("")
        lines.append("Steps:")
        for st in steps:
            lines.append(
                f"  - step {st.get('step_number')}: product {_shown(st.get('product'))};"
                f" isolated {_shown(st.get('product_mass'))}; scale reference {_shown(st.get('scale_reference'))}"
            )
    body = "\n".join(lines)
    head = (
        f"Product: {_shown(view.get('product'))}\n"
        f"Product isolated: {_shown(view.get('product_mass'))}\n"
        f"Scale reference: {_shown(view.get('scale_reference'))}\n"
    )
    if steps:
        head = f"Multi-step route ({len(steps)} steps)\n"
    how = "Work out each chemical's structure from its name, then use the tools to get supplier quotes."
    return (
        "Estimate the procurement cost, in US dollars per gram of product, for this reaction.\n\n"
        f"{head}\nComponents as used in the laboratory:\n{body}\n\n{how}\n"
    )


def build_row(record: dict) -> dict:
    """One Gym row: the system and user messages, the two tools, and what ``verify()`` needs."""
    return {
        "task_id": record["reaction_id"],
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task_prompt(agent_view(record))},
            ],
            "tools": TOOLS,
        },
        "verifier_metadata": {
            "reaction_id": record["reaction_id"],
            "type": record["type"],
            "procurement_cost_usd_per_g_product": record.get("procurement_cost_usd_per_g_product"),
        },
    }
