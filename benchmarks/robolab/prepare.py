# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the Gym dataset from a staged RoboLab case manifest."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


SDK_VISION_BY_VERSION = {
    "1.5": "rgb",
    "open_model": "rgbd",
}
OPEN_MODEL_BUNDLES = frozenset({"core", "assisted"})


class Case(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    task: str
    run: Annotated[int, Field(ge=10, le=14)]
    native_horizon: Annotated[int, Field(gt=0)]
    official_bundle: Annotated[str, Field(pattern=r"^[0-9a-f]{12}$")]


class RunPlan(BaseModel):
    model_config = ConfigDict(strict=True)

    schema_id: Literal["robolab.native-sequence.v2"] = Field(alias="schema")
    arm: Literal["sdk_v15", "open_model"]
    task: str
    run_indices: list[int]
    reset_only_prefix: int


class Manifest(BaseModel):
    model_config = ConfigDict(strict=True)

    schema_id: Literal["robolab.efb-candidate.v1"] = Field(alias="schema")
    official_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    case: Case
    native_run_plan: RunPlan
    bridge_config: dict[str, object]
    harness: Literal["opencode"]
    tools: list[Literal["observe"]]
    score_source: Literal["native_summary"]


def load_case_manifest(path: Path) -> tuple[Manifest, str]:
    """Read and validate a single RoboLab case manifest and its staged digest."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if path.parent.parent.name == "staged-files" and path.parent.name != digest:
        raise ValueError("RoboLab manifest does not match its staged SHA-256")
    manifest = Manifest.model_validate_json(raw)
    case, plan = manifest.case, manifest.native_run_plan
    if not case.task or case.id != f"{case.task}@run-{case.run:03d}" or plan.task != case.task:
        raise ValueError("RoboLab case identity and run plan task must agree")
    if plan.run_indices != [case.run] or plan.reset_only_prefix != case.run:
        raise ValueError("RoboLab run plan must select the single assigned run after its reset-only prefix")

    bridge = manifest.bridge_config
    native, sdk, model = (bridge.get(key) for key in ("native", "sdk", "model"))
    if not isinstance(native, dict) or not isinstance(sdk, dict) or not isinstance(model, dict):
        raise ValueError("RoboLab manifest requires native, SDK and model bindings")
    if (
        bridge.get("integration") != "robolab"
        or native.get("task") != case.task
        or type(native.get("seed")) is not int
    ):
        raise ValueError("RoboLab runtime task or seed does not match the case contract")
    if native["seed"] != 0:
        raise ValueError("RoboLab candidate requires seed 0")

    sdk_version = sdk.get("version")
    expected_vision = SDK_VISION_BY_VERSION.get(str(sdk_version)) if isinstance(sdk_version, str) else None
    profile = sdk.get("profile")
    if (
        expected_vision is None
        or sdk.get("execution") != "ttc"
        or profile
        != {
            "vision": expected_vision,
            "oracle": "none",
            "contact": False,
            "perception": False,
        }
    ):
        raise ValueError("RoboLab candidate requires SDK 1.5 RGB or open_model RGB-D with TTC")
    if sdk_version == "open_model":
        bundle = sdk.get("bundle")
        if bundle is not None and bundle not in OPEN_MODEL_BUNDLES:
            raise ValueError("RoboLab open_model bundle must be 'core' or 'assisted'")

    command = bridge.get("harness_command")
    if (
        not isinstance(command, list)
        or len(command) != 4
        or not all(isinstance(item, str) for item in command)
        or not command[1].endswith("/harnesses/opencode/main.mts")
        or command[2] != "--opencode"
    ):
        raise ValueError("RoboLab candidate requires the OpenCode harness command")
    model_id = model.get("id")
    reasoning = model.get("reasoning")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("RoboLab candidate requires a non-empty model id")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("RoboLab candidate requires a non-empty reasoning setting")
    if manifest.tools != ["observe"] or bridge.get("policies") != []:
        raise ValueError("RoboLab candidate requires the observe-only tool surface")
    return manifest, digest


def prepare(
    manifest_path: str | Path | None = None,
    output_fpath: str | Path | None = None,
) -> Path:
    """Write the single frozen episode request consumed by Gym."""
    source = manifest_path or os.environ.get("ROBOLAB_CASE_MANIFEST")
    if not source:
        raise ValueError("Set ROBOLAB_CASE_MANIFEST to the staged RoboLab case manifest")
    manifest, digest = load_case_manifest(Path(source))
    row = {
        "case_id": manifest.case.id,
        "responses_create_params": {"input": []},
        "verifier_metadata": {
            "manifest_sha256": digest,
            "official_manifest_sha256": manifest.official_manifest_sha256,
            "case": manifest.case.model_dump(),
            "seed": 0,
            "native_run_plan": manifest.native_run_plan.model_dump(by_alias=True),
            "score_source": manifest.score_source,
        },
    }
    if output_fpath is None:
        output_dir = os.environ.get("NEMO_GYM_BENCH_DIR", "benchmarks/robolab/data")
        output_fpath = Path(output_dir) / "case.jsonl"
    output = Path(output_fpath)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(json.dumps(row) + "\n")
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output
