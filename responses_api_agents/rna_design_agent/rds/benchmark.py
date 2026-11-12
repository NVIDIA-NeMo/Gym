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
"""Running campaigns as a benchmark. One implementation of the loop, so every arm sees the same one."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .budget import Budget, Limits
from .campaign import (
    Campaign,
    applicable_tools,
    default_cycle,
    objective_view,
    score_campaign,
    write_result,
)
from .controllers import FixedPipeline, LLMController
from .runner import run_campaign
from .state import CampaignState
from .targets import from_dict, register


ARMS = ("generator", "fixed", "fixed_deep", "agent")


def anonymised(campaign):
    """The same campaign under a name that names nothing."""
    import dataclasses
    import hashlib

    if not campaign.inline_target:
        return campaign
    real = campaign.inline_target["name"]
    alias = "target_" + hashlib.sha256(real.encode()).hexdigest()[:8]
    target = dict(campaign.inline_target, name=alias)
    register(from_dict(target))
    # The campaign's own name carries the suite and the puzzle too.
    return dataclasses.replace(campaign, name=f"design/{alias}", target=alias, inline_target=target)


def _code_version() -> str:
    """Which revision produced this row: a hash of the package source, plus the git revision when there is one."""
    import hashlib
    import subprocess

    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.rglob("*.py")):
        digest.update(path.read_bytes())
    stamp = digest.hexdigest()[:12]
    try:
        named = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        named = ""
    return f"{stamp} {named}".strip()


def run_one(
    campaign: Campaign,
    out_dir: str | Path,
    arm: str = "agent",
    generator: str = "generate.inverse_fold",
    provider: Any = None,
    rounds: int = 0,
    max_tokens: int = 8192,
    seed: int = 101,
) -> dict:
    """Run one campaign under one arm and return its scored result.

    `generator` runs the derived cycle once, `fixed` repeats it, `agent` lets a model choose. All share one budget.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {', '.join(ARMS)}")
    if campaign.inline_target:
        register(from_dict(campaign.inline_target))
    if campaign.objective_visibility != "metric":
        # A published puzzle name would let a model recall a known solution.
        campaign = anonymised(campaign)

    if campaign.control == "handwritten":
        if not campaign.control_cycle:
            raise ValueError(
                f"{campaign.name} sets control: handwritten but declares no "
                f"control_cycle; a hand-written control has to be written"
            )
        # Fill in the campaign's target for steps that omit it.
        from .campaign import _field_names

        cycle = []
        for step in campaign.control_cycle:
            arguments = dict(step[1])
            if "target" in _field_names(step[0]) and "target" not in arguments:
                arguments["target"] = campaign.target_name
            cycle.append((step[0], arguments))
    elif campaign.control == "derived":
        cycle = default_cycle(campaign, generator=generator, depth="full" if arm == "fixed_deep" else "minimal")
    else:
        raise ValueError(f"unknown control {campaign.control!r}; expected derived or handwritten")
    if arm == "agent":
        if provider is None:
            raise ValueError("the agent arm needs a provider")
        from .controllers.llm import BENCHMARK_PROMPT, SYSTEM_PROMPT

        controller = LLMController(
            provider,
            max_tokens=max_tokens,
            system_prompt=(SYSTEM_PROMPT if campaign.objective_visibility == "metric" else BENCHMARK_PROMPT),
            allowed=[spec.name for spec in applicable_tools(campaign)],
        )
    else:
        controller = FixedPipeline(cycle, seed=seed)  # fixed and fixed_deep differ only in depth

    limits = dict(campaign.budget)
    if rounds:
        limits["rounds"] = rounds
    budget = Budget(Limits(**limits))
    max_rounds = len(cycle) if arm == "generator" else limits.get("rounds", 10)

    os.environ["RDS_RUN_DIR"] = str(out_dir)
    # The sandbox sees no campaign state, so it is told the target.
    os.environ["RDS_TARGET"] = campaign.target_name
    # Model-written tools are shared across targets but keyed by model, protocol and seed, so
    # replicates and protocols never inherit each other's work.
    if campaign.allow_code:
        model = (getattr(provider, "model", "") or "control").replace("/", "_")
        protocol = f"{campaign.control}-{campaign.disclosure}-{campaign.objective_visibility}-{campaign.exposure}"
        toolbox = Path(out_dir).parent / "toolbox" / model / protocol / f"s{seed}"
        os.environ["RDS_TOOLBOX"] = str(toolbox)
    else:
        # Clear a value left by an earlier campaign in this process.
        os.environ.pop("RDS_TOOLBOX", None)
    state = CampaignState(out_dir, campaign.scoring.direction)
    summary = run_campaign(
        controller,
        state,
        budget,
        rank_by=campaign.scoring.primary,
        max_rounds=max_rounds,
        objective=objective_view(campaign),
        reflect_every=campaign.reflect_every,
    )
    summary["arm"] = arm
    summary["seed"] = seed
    summary["code"] = _code_version()
    summary["protocol"] = {
        "control": campaign.control,
        "disclosure": campaign.disclosure,
        "exposure": campaign.exposure,
        "visibility": campaign.objective_visibility,
        "allow_code": campaign.allow_code,
        "toolbox": os.environ.get("RDS_TOOLBOX", ""),
    }
    summary["model"] = getattr(provider, "model", None) if arm == "agent" else None
    result = score_campaign(state, campaign)
    if campaign.objective_visibility != "metric":
        # The agent was not shown the protocol, so it is applied here.
        from .evaluate import official_result

        pool = [row["sequence"] for row in state.candidates()]
        result["official"] = official_result(campaign, Path(out_dir), pool)
    result["run"] = summary
    result["arm"] = arm
    result["campaign"] = campaign.name
    usage = getattr(controller, "usage_total", None)
    if usage:
        result["tokens"] = {"in": usage.input_tokens, "out": usage.output_tokens}
    write_result(out_dir, result)
    state.close()
    return result
