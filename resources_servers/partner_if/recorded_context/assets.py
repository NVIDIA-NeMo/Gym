# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned data loading and receipt construction; policy execution belongs to Gym."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from resources_servers.partner_if.native_evidence import LiveAgentCapture
from resources_servers.partner_if.recorded_context.fixtures import canonical_path, probe_options, shell_tokens
from resources_servers.partner_if.recorded_context.runtime_contract import RuntimeContract, read_runtime_contract


GYM_ROOT = Path(__file__).resolve().parents[3]


class ReplayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime_contract_path: Path | None = None
    runtime_contract_sha256: str | None = None
    package_dir: Path
    coverage_profile: Literal["recorded-v1", "expanded-v2", "expanded-v3", "expanded-v4"] = "recorded-v1"
    max_policy_calls: int = Field(default=8, ge=1, le=64)
    missing_runtime_reads: bool = False
    shell_control_flow: bool = False
    capture_stdout: bool = False
    temporal_history: bool = False
    max_delegate_turns: int = Field(default=6, ge=1, le=12)
    max_tokens: int = Field(default=8192, ge=1, le=8192)
    temperature: float = 0.7
    top_p: float = 0.95
    timeout_s: int = Field(default=180, ge=1, le=180)

    @field_validator("runtime_contract_path", "package_dir")
    @classmethod
    def checkout_paths(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        path = value.expanduser()
        return (path if path.is_absolute() else GYM_ROOT / path).resolve()


def actor_identity(params: dict) -> dict:
    """Allow Gym to omit completed message status; retain all actor content."""
    identity = copy.deepcopy(params)
    for item in identity.get("input", []):
        if item.get("type", "message") == "message" and item.get("status") == "completed":
            item.pop("status")
    return identity


def load_runtime(config: ReplayConfig) -> RuntimeContract:
    if config.runtime_contract_path is not None and config.runtime_contract_sha256 is None:
        raise ValueError("An external replay contract requires runtime_contract_sha256")
    path = config.runtime_contract_path or config.package_dir / "ngc-recorded-context/runtime.json"
    return read_runtime_contract(path, expected_sha256=config.runtime_contract_sha256)


def load_actor_params(config: ReplayConfig) -> dict:
    """Return the original pinned actor row before Gym adds typed API defaults."""
    runtime = load_runtime(config)
    data = (config.package_dir / "ngc-recorded-context/actor-params.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != runtime.input_hashes["actor-params.json"]:
        raise ValueError("Frozen replay input changed: actor-params.json")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Frozen actor parameters must be an object")
    return value


def load_inputs(config: ReplayConfig, params: dict) -> tuple[dict, dict]:
    runtime = load_runtime(config)
    root = config.package_dir / "ngc-recorded-context"

    def checked(name):
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != runtime.input_hashes[name]:
            raise ValueError(f"Frozen replay input changed: {name}")
        return json.loads(gzip.decompress(data) if name.endswith(".gz") else data)

    case, bundle = checked("case.json"), checked("bundle.json")
    if actor_identity(checked("actor-params.json")) != actor_identity(params):
        raise ValueError("Gym row differs from the pinned actor history/tools")
    if config.coverage_profile != "recorded-v1":
        coverage = checked("coverage-v2.json.gz")
        if coverage["schema_version"] != runtime.source_schemas["coverage"]:
            raise ValueError("Unexpected coverage schema")
        for path, text in coverage["files"].items():
            relative = path.removeprefix(runtime.source_root.rstrip("/") + "/")
            if relative == path or ".." in Path(relative).parts:
                raise ValueError("Unsafe coverage source path")
            if hashlib.sha256(text.encode()).hexdigest() != coverage["source_file_sha256"][relative]:
                raise ValueError("Coverage source checksum mismatch")
        bundle["expanded_coverage"] = coverage
    if config.coverage_profile in {"expanded-v3", "expanded-v4"}:
        inventory = checked("source-inventory.json")
        if (
            inventory["schema_version"] != runtime.source_schemas["inventory"]
            or inventory["root"] != runtime.source_root
            or inventory["source_commit"] != coverage["provenance"]["source_commit"]
        ):
            raise ValueError("Source inventory identity mismatch")
        if any(Path(path).is_absolute() or ".." in Path(path).parts for path in inventory["paths"]):
            raise ValueError("Unsafe source inventory path")
        bundle["source_inventory"] = inventory
    if config.coverage_profile == "expanded-v4":
        extra = checked("source-extra-v4.json.gz")
        if (
            extra["schema_version"] != runtime.source_schemas["extra"]
            or extra["source_commit"] != inventory["source_commit"]
        ):
            raise ValueError("Extra source identity mismatch")
        for path, text in extra["files"].items():
            relative = path.removeprefix(inventory["root"] + "/")
            if (
                relative not in inventory["paths"]
                or hashlib.sha256(text.encode()).hexdigest() != extra["file_sha256"][path]
            ):
                raise ValueError("Extra source checksum/path mismatch")
            if path in coverage["files"] or path in bundle["files"]:
                raise ValueError("Extra source cannot replace recorded content")
        coverage["files"].update(extra["files"])
    if config.temporal_history:
        bundle["temporal_world"] = checked("temporal-world.json")
    bundle.update(
        runtime=runtime.model_dump(),
        missing_runtime_reads=config.missing_runtime_reads,
        shell_control_flow=config.shell_control_flow,
        capture_stdout=config.capture_stdout,
    )
    return case, bundle


def probe_identity(event: dict, bundle: dict) -> dict:
    """Validate a fixture-backed exact probe, independently of prose/label claims.

    The interpreter is deliberately small. An unrecognized equivalent check is
    a coverage exclusion, never silently treated as no probe or a successful one.
    """
    if event["kind"] not in {"primary_probe", "sibling_probe"}:
        return {}
    fn = event["function"]
    command = json.loads(fn["arguments"])["command"]
    tokens = shell_tokens(canonical_path(command, bundle["runtime"].get("path_aliases", {})))
    if not tokens or tokens.pop(0) not in {"python", "python3", "/usr/bin/python3"}:
        raise ValueError("Unverified probe interpreter")
    if tokens and tokens[0] == "-I":
        tokens.pop(0)
    if not tokens or tokens.pop(0) != bundle["runtime"]["probe_script"]:
        raise ValueError("Probe script path is not the pinned script")
    options = probe_options(tokens)
    label = "sibling" if event["kind"] == "sibling_probe" else "primary"
    expected = bundle[label]
    if fn["name"] != "Bash" or options.get("--uri") != expected["input_uri"]:
        raise ValueError("Probe identity mismatch")
    if "--output" in options:
        output = options["--output"]
        if ".." in Path(output).parts or not Path(output).is_relative_to(bundle["runtime"]["workspace_prefix"]):
            raise ValueError("Probe output outside modeled workspace")
    if json.loads(event["result"]) != expected:
        raise ValueError("Probe result differs from returned fixture")
    return {"namespace": label, "uri": expected["input_uri"], "returned_result": expected}


def capture(
    params: dict,
    output: list[dict],
    events: list[dict],
    drafts: dict,
    final: str,
    *,
    prefix: bool = False,
    answer_paths: list[str] | None = None,
) -> LiveAgentCapture:
    # Receipts include response-turn numbers: all calls in one response were
    # chosen before any of their results could be seen by the model.
    text = next((drafts[path] for path in (answer_paths or list(drafts)) if path in drafts), final)
    receipts = {
        "supplied_history_is_context": True,
        "same_response_results_are_not_prior_evidence": True,
        "new_events": events,
    }
    if prefix:
        receipts["evaluation_boundary"] = "ordering_prefix_v1"
        receipts["scope"] = "new captured prefix; supplied history is context"
    return LiveAgentCapture(
        input=params["input"],
        artifacts={
            "final_reply": {
                "exists": bool(text) if prefix else True,
                **({"text": text} if text or not prefix else {}),
            },
            "replay_receipts": {"exists": True, "text": json.dumps(receipts, ensure_ascii=False, indent=2)},
        },
    )
