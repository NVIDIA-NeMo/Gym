# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a versioned model preset into a native Gym submit-config fragment.

This module intentionally uses only the standard library. A consumer can run the
file from an exact Gym checkout without upgrading its benchmark runtime. The
output is ordinary JSON (also valid YAML) accepted by Hydra's defaults imports.
"""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate preset key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict:
    """Read a JSON object, refusing duplicate keys and non-finite numbers."""

    def invalid(value: str) -> None:
        raise ValueError(f"Non-finite preset number: {value}")

    result = json.loads(path.read_text(), object_pairs_hook=_unique_object, parse_constant=invalid)
    _require(isinstance(result, dict), f"Expected an object in {path}")
    return result


def _keys(value: dict, required: set[str], optional: set[str], label: str) -> None:
    _require(isinstance(value, dict), f"{label} must be an object")
    _require(required <= value.keys(), f"{label}: missing {sorted(required - value.keys())}")
    _require(value.keys() <= required | optional, f"{label}: unknown {sorted(value.keys() - required - optional)}")


def _sha(value: object, length: int, label: str) -> None:
    _require(
        isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None,
        f"{label} must be an immutable {length}-character hash",
    )


def _text(value: object, label: str) -> None:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be a nonempty string")


def render_preset(
    preset_path: Path,
    *,
    bindings: dict,
    benchmarks: list[str],
    source: dict,
    overrides: dict | None = None,
) -> dict:
    """Bind verified asset identities and locations to one shared model preset.

    Asset hashes here are declarations; consumers must separately verify staged
    bytes. Overrides name an existing service or run field and require a reason.
    The result contains no cluster, benchmark definition, credentials or scores.
    """
    preset = read_json(preset_path)
    _keys(preset, {"schema_version", "name", "checkpoint", "runtime", "service", "run", "sources"}, set(), "preset")
    _require(type(preset["schema_version"]) is int and preset["schema_version"] == 1, "Unsupported preset schema")
    _text(preset["name"], "preset.name")
    _keys(preset["checkpoint"], {"repository", "revision"}, set(), "checkpoint")
    _keys(preset["runtime"], {"source", "sha256"}, set(), "runtime")
    revision = preset["checkpoint"]["revision"]
    if isinstance(revision, str) and revision.startswith("sha256:"):
        _sha(revision.removeprefix("sha256:"), 64, "checkpoint manifest revision")
    else:
        _sha(revision, 40, "checkpoint.revision")
    _sha(preset["runtime"]["sha256"], 64, "runtime.sha256")
    _require(isinstance(preset["sources"], list) and bool(preset["sources"]), "Preset needs source evidence")
    for evidence in preset["sources"]:
        _keys(evidence, {"url", "sha256", "description"}, set(), "source evidence")
        _sha(evidence["sha256"], 64, "source evidence.sha256")
        _text(evidence["url"], "source evidence.url")
        _text(evidence["description"], "source evidence.description")
    _keys(
        bindings,
        {
            "checkpoint",
            "checkpoint_repository",
            "checkpoint_revision",
            "container",
            "runtime_sha256",
            "cache",
            "served_model_name",
        },
        {"pre_command", "mounts"},
        "bindings",
    )
    for field in ("checkpoint", "container", "cache", "served_model_name"):
        _text(bindings[field], f"bindings.{field}")
    _require(bindings["checkpoint_repository"] == preset["checkpoint"]["repository"], "Checkpoint repository mismatch")
    _require(bindings["checkpoint_revision"] == preset["checkpoint"]["revision"], "Checkpoint revision mismatch")
    _require(bindings["runtime_sha256"] == preset["runtime"]["sha256"], "Serving runtime digest mismatch")
    _keys(source, {"repository", "revision", "path"}, set(), "preset reference")
    _sha(source["revision"], 40, "preset reference.revision")
    _text(source["repository"], "preset reference.repository")
    _text(source["path"], "preset reference.path")
    _require(bool(benchmarks) and len(set(benchmarks)) == len(benchmarks), "Benchmarks must be nonempty and unique")
    for benchmark in benchmarks:
        _require(
            isinstance(benchmark, str) and re.fullmatch(r"[A-Za-z0-9_-]+", benchmark) is not None,
            "Invalid benchmark key",
        )

    service = copy.deepcopy(preset["service"])
    _keys(
        service,
        {
            "type",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "number_of_instances",
            "extra_args",
            "pre_command",
            "env",
            "health_check",
        },
        set(),
        "service",
    )
    _require(service["type"] == "vllm", "Only native vLLM services are supported")
    for key in ("tensor_parallel_size", "pipeline_parallel_size", "number_of_instances"):
        _require(type(service[key]) is int and service[key] > 0, f"Invalid service.{key}")
    _require(isinstance(service["env"], dict), "service.env must be an object")
    run = copy.deepcopy(preset["run"])
    _require(isinstance(run, dict) and bool(run), "Preset needs model request settings")
    for key in run:
        _require(
            key.startswith(("responses_create_params.", "policy_model.responses_api_models.vllm_model.")),
            f"Not a model request setting: {key}",
        )
        _require(
            key not in {"responses_create_params.input", "responses_create_params.model"},
            f"Model presets cannot supply benchmark input or endpoint identity: {key}",
        )

    changes = overrides or {}
    _require(isinstance(changes, dict), "Overrides must be an object")
    for key, change in changes.items():
        _keys(change, {"value", "reason"}, set(), f"override {key}")
        _text(change["reason"], f"override {key}.reason")
        group, separator, field = key.partition(".")
        _require(separator == "." and group in {"service", "run"}, f"Unsupported override: {key}")
        target = service if group == "service" else run
        _require(field in target, f"Override is not a preset field: {key}")
        _require(type(change["value"]) is type(target[field]), f"Override changes field type: {key}")
        target[field] = change["value"]

    additional_mounts = bindings.get("mounts", [])
    _require(isinstance(additional_mounts, list), "bindings.mounts must be a list")
    targets = {"/checkpoint", "/cache"}
    for mount in additional_mounts:
        _require(isinstance(mount, str), "Each mount must be source:target[:ro]")
        parts = mount.split(":")
        _require(len(parts) in (2, 3), "Each mount must be source:target[:ro]")
        source_path, target_path = parts[:2]
        source_reference = re.fullmatch(r"\$\{[A-Za-z0-9_.-]+\}", source_path)
        _require(
            source_reference or (Path(source_path).is_absolute() and ".." not in Path(source_path).parts),
            "Mount source must be absolute or a configuration reference",
        )
        _require(
            Path(target_path).is_absolute() and ".." not in Path(target_path).parts, "Mount target must be absolute"
        )
        _require(len(parts) == 2 or parts[2] in {"ro", "rw"}, "Unsupported mount mode")
        target = str(Path(parts[1]))
        _require(target not in targets, "Duplicate or reserved mount target")
        _require(target != "/" and not target.startswith(("/checkpoint/", "/cache/")), "Reserved mount target")
        targets.add(target)

    service.update(
        {
            "model": "/checkpoint",
            "container": bindings["container"],
            "served_model_name": bindings["served_model_name"],
            "mounts": [f"{bindings['checkpoint']}:/checkpoint:ro", f"{bindings['cache']}:/cache", *additional_mounts],
        }
    )
    if bindings.get("pre_command"):
        service["pre_command"] = service["pre_command"].rstrip() + "\n" + bindings["pre_command"]
    provenance = {
        **source,
        "name": preset["name"],
        "sha256": hashlib.sha256(preset_path.read_bytes()).hexdigest(),
        "checkpoint": preset["checkpoint"],
        "runtime": preset["runtime"],
        "overrides": changes,
    }
    return {
        "services": {"policy": service},
        "driver": {
            "policy_model": "policy",
            "policy_model_type": "vllm_model",
            "benchmarks": {name: {"run": copy.deepcopy(run)} for name in benchmarks},
            # The native submit schema saves driver.env in resolved-config.yaml.
            # This retains the exact source reference even on older Gym runtimes.
            "env": {
                "GYM_MODEL_PRESET_PROVENANCE": "lit:" + json.dumps(provenance, sort_keys=True, separators=(",", ":"))
            },
        },
    }


def main() -> None:
    """Write a fragment for a direct Gym run or an external recipe composer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--benchmark", action="append", required=True)
    parser.add_argument("--overrides", type=Path)
    args = parser.parse_args()
    fragment = render_preset(
        args.preset,
        bindings=read_json(args.bindings),
        benchmarks=args.benchmark,
        source=read_json(args.source),
        overrides=read_json(args.overrides) if args.overrides else None,
    )
    print(json.dumps(fragment, indent=2))


if __name__ == "__main__":
    main()
