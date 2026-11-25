# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select a frozen RoboDojo input and bind it to an EmberBridge configuration."""

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


RoboDojoSimulator = Literal["isaaclab3_newton", "isaacsim_physx"]
ROBODOJO_SIMULATORS: dict[str, RoboDojoSimulator] = {
    "isaaclab3": "isaaclab3_newton",
    "isaacsim": "isaacsim_physx",
}


class SourceFile(BaseModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RoboDojoCase(BaseModel):
    case_id: str
    runtime_task: str
    layout_id: int = Field(ge=0)
    eval_seed: int = Field(ge=0)
    native_step_limit: int = Field(gt=0)
    layout: SourceFile
    native_support_trajectories: list[SourceFile] = Field(default_factory=list)


class Panel(BaseModel):
    panel_id: str | None = None
    panel_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    cases: list[RoboDojoCase] = Field(min_length=1)
    substitutes: list[RoboDojoCase] = Field(default_factory=list)
    native_source_files: list[SourceFile]


def load_panel(path: Path) -> Panel:
    panel = Panel.model_validate_json(path.read_bytes())
    cases = [case.case_id for case in panel.cases + panel.substitutes]
    if len(set(cases)) != len(cases):
        raise ValueError("RoboDojo case IDs must be unique")
    return panel


def parse_pool_cases(value: str) -> tuple[str, ...]:
    cases = tuple(case.strip() for case in value.split(","))
    if not 2 <= len(cases) <= 5 or not all(cases) or len(set(cases)) != len(cases):
        raise ValueError("Select two to five distinct RoboDojo pool cases")
    return cases


def bind_case(
    *,
    template_path: Path,
    panel_path: Path,
    validity_path: Path | None,
    case_id: str,
    horizon: Literal["native_x1", "native_x2"],
    model_url: str,
    model_name: str,
) -> tuple[dict, RoboDojoCase]:
    """Resolve a requested slot and reject changed native inputs before launching."""
    panel = load_panel(panel_path)
    if case_id not in {case.case_id for case in panel.cases}:
        raise ValueError(f"Unknown RoboDojo case {case_id!r}")

    config = json.loads(template_path.read_text())
    if config.get("integration") != "robodojo" or config.get("simulator_backend") not in ROBODOJO_SIMULATORS:
        raise ValueError(
            "The EmberBridge configuration must select RoboDojo with simulator_backend isaaclab3 or isaacsim"
        )
    sdk = config.get("sdk", {})
    if sdk.get("execution") != "ttc" or sdk.get("gates") is not False:
        raise ValueError("The evaluation requires an ungated TTC SDK session")
    if config.get("state_access") != "none" or config.get("policies"):
        raise ValueError("The evaluation requires private state and auxiliary policies to be disabled")

    assigned_id = case_id
    config.pop("eval_case", None)
    panel_file_sha256 = _file_sha256(panel_path)
    if validity_path is not None:
        validity_bytes = validity_path.read_bytes()
        validity_file_sha256 = hashlib.sha256(validity_bytes).hexdigest()
        validity = json.loads(validity_bytes)
        if (
            panel.panel_id is None
            or panel.panel_sha256 is None
            or validity["panel_id"] != panel.panel_id
            or validity["panel_sha256"] != panel.panel_sha256
        ):
            raise ValueError("RoboDojo validity data belongs to a different panel")
        matches = [entry["case_id"] for entry in validity["assignment"] if entry["slot"] == case_id]
        if len(matches) != 1:
            raise ValueError(f"Case {case_id!r} must have one frozen assignment")
        assigned_id = matches[0]
        config["eval_case"] = {
            "manifest": str(panel_path.resolve()),
            "panel_sha256": panel.panel_sha256,
            "panel_file_sha256": panel_file_sha256,
            "case_id": assigned_id,
            "validity": str(validity_path.resolve()),
            "validity_sha256": validity["validity_sha256"],
            "validity_file_sha256": validity_file_sha256,
        }

    cases = {case.case_id: case for case in panel.cases + panel.substitutes}
    if assigned_id not in cases:
        raise ValueError(f"Frozen assignment {assigned_id!r} is absent from the panel")
    case = cases[assigned_id]
    asset_root = Path(config["robodojo"]).resolve(strict=True)
    for record in [*panel.native_source_files, *case.native_support_trajectories, case.layout]:
        path = (asset_root / record.path).resolve(strict=True)
        if not path.is_relative_to(asset_root):
            raise ValueError(f"Source record escapes the RoboDojo checkout: {record.path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record.sha256:
            raise ValueError(f"RoboDojo input changed: {record.path}")

    config.update(
        task=case.runtime_task,
        eval_seed=case.eval_seed,
        layout_id=case.layout_id,
        horizon_profile=horizon,
        horizon_record=True,
        step_limit=case.native_step_limit * (2 if horizon == "native_x2" else 1),
        provider_auth="api_key",
    )
    config["model"].update(id=model_name, base_url=model_url)
    if table := os.environ.get("ROBODOJO_SHARED_PLANNER_TABLE"):
        routes = json.loads(Path(table).read_text())
        socket = routes.get(case_id)
        if not isinstance(socket, str) or not Path(socket).is_absolute() or len(os.fsencode(socket)) > 107:
            raise ValueError(f"Case {case_id!r} must have a valid shared planner socket")
        config["shared_planner_socket"] = socket
    return config, case
