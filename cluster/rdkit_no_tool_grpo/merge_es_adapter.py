#!/usr/bin/env python3
"""Merge the validated Inferno Step 140 LoRA adapter into HF safetensor shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

EXPECTED_BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
EXPECTED_ADAPTER_SHA256 = "e281afb74b78d5ef337c233c1b1faeb42ac72208ca76ed4b5563e1ca4adb3de4"
EXPECTED_ADAPTER_TENSOR_COUNT = 11916
EXPECTED_UPDATE_COUNT = 140
LORA_A_PATTERN = re.compile(r"^(?P<prefix>.+)\.lora_A(?:\.default)?\.weight$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_adapter(adapter_dir: Path, expected_sha256: str) -> dict:
    weights = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    provenance_path = adapter_dir / "provenance.json"
    for path in (weights, config_path, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    actual_sha256 = sha256(weights)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Adapter SHA-256 is {actual_sha256}, expected {expected_sha256}")
    config = json.loads(config_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    if config.get("base_model_name_or_path") != EXPECTED_BASE_MODEL:
        raise ValueError("Unexpected PEFT base model")
    if config.get("r") != 8 or config.get("lora_alpha") != 8.0:
        raise ValueError("Expected rank-8, alpha-8 ES adapter")
    if config.get("fan_in_fan_out") is not False or config.get("bias") != "none":
        raise ValueError("Direct merge requires fan_in_fan_out=false and bias=none")
    if config.get("use_rslora", False) or config.get("use_dora", False):
        raise ValueError("Direct merge does not support RSLoRA or DoRA scaling")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError("Direct merge requires uniform rank and alpha")
    if config.get("modules_to_save"):
        raise ValueError("Direct merge does not support additional modules_to_save")
    if provenance.get("selection", {}).get("inferno_update_count") != EXPECTED_UPDATE_COUNT:
        raise ValueError("Adapter provenance is not post-Step 140")
    if provenance.get("lora", {}).get("tensor_count") != EXPECTED_ADAPTER_TENSOR_COUNT:
        raise ValueError("Unexpected adapter tensor count in provenance")
    return {"config": config, "provenance": provenance, "sha256": actual_sha256}


def base_weight_map(base_model: Path) -> tuple[dict[str, str], Path | None]:
    index_path = base_model / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid safetensors index: {index_path}")
        return weight_map, index_path

    model_path = base_model / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Expected model.safetensors or model.safetensors.index.json in {base_model}"
        )
    from safetensors import safe_open

    with safe_open(model_path, framework="pt", device="cpu") as handle:
        return {key: model_path.name for key in handle.keys()}, None


def resolve_base_key(adapter_prefix: str, weight_map: dict[str, str]) -> str:
    candidates = [adapter_prefix]
    for prefix in ("base_model.model.", "base_model."):
        if adapter_prefix.startswith(prefix):
            candidates.insert(0, adapter_prefix.removeprefix(prefix))
    matches = [f"{candidate}.weight" for candidate in candidates if f"{candidate}.weight" in weight_map]
    if len(matches) != 1:
        raise ValueError(
            f"Adapter module {adapter_prefix!r} maps to {matches}, expected exactly one base weight"
        )
    return matches[0]


def build_lora_pairs(
    adapter_tensors: dict[str, object],
    weight_map: dict[str, str],
    target_modules: set[str],
) -> dict[str, tuple[str, str]]:
    pairs: dict[str, tuple[str, str]] = {}
    used_adapter_keys: set[str] = set()
    for a_key in sorted(adapter_tensors):
        match = LORA_A_PATTERN.fullmatch(a_key)
        if match is None:
            continue
        b_key = a_key.replace(".lora_A.", ".lora_B.", 1)
        if b_key not in adapter_tensors:
            raise ValueError(f"Missing LoRA B tensor for {a_key}")
        base_key = resolve_base_key(match.group("prefix"), weight_map)
        module_name = base_key.rsplit(".", 2)[-2]
        if module_name not in target_modules:
            raise ValueError(f"Unexpected LoRA target {module_name!r} for {base_key}")
        if base_key in pairs:
            raise ValueError(f"Multiple LoRA pairs map to {base_key}")
        pairs[base_key] = (a_key, b_key)
        used_adapter_keys.update((a_key, b_key))

    unused = sorted(set(adapter_tensors) - used_adapter_keys)
    if unused:
        raise ValueError(f"Unmapped adapter tensors ({len(unused)}): {unused[:8]}")
    if len(used_adapter_keys) != EXPECTED_ADAPTER_TENSOR_COUNT:
        raise ValueError(
            f"Mapped {len(used_adapter_keys)} adapter tensors, expected "
            f"{EXPECTED_ADAPTER_TENSOR_COUNT}"
        )
    return pairs


def merge(args: argparse.Namespace) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    base_model = args.base_model.resolve()
    adapter_dir = args.adapter_dir.resolve()
    output_dir = args.output_dir.resolve()
    validation_output = args.validation_output.resolve()
    adapter = validate_adapter(adapter_dir, args.expected_adapter_sha256)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    adapter_tensors = load_file(adapter_dir / "adapter_model.safetensors", device="cpu")
    if len(adapter_tensors) != EXPECTED_ADAPTER_TENSOR_COUNT:
        raise ValueError(
            f"Adapter contains {len(adapter_tensors)} tensors, expected "
            f"{EXPECTED_ADAPTER_TENSOR_COUNT}"
        )
    weight_map, _ = base_weight_map(base_model)
    target_modules = set(adapter["config"].get("target_modules", []))
    pairs = build_lora_pairs(adapter_tensors, weight_map, target_modules)
    scaling = float(adapter["config"]["lora_alpha"]) / float(adapter["config"]["r"])
    if not math.isfinite(scaling):
        raise ValueError("LoRA scaling is not finite")

    shard_names = sorted(set(weight_map.values()))
    modules_by_shard: dict[str, list[str]] = {name: [] for name in shard_names}
    for base_key in pairs:
        modules_by_shard[weight_map[base_key]].append(base_key)

    temporary_dir = output_dir.with_name(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    temporary_dir.mkdir(parents=True)
    output_manifest: list[dict] = []
    merged_modules = 0
    max_abs_delta = 0.0
    try:
        for source in sorted(base_model.iterdir()):
            destination = temporary_dir / source.name
            if source.name in shard_names:
                continue
            if source.is_dir():
                shutil.copytree(source, destination, symlinks=True)
            elif source.is_file():
                shutil.copy2(source, destination)

        for shard_index, shard_name in enumerate(shard_names, start=1):
            source = base_model / shard_name
            destination = temporary_dir / shard_name
            tensors = load_file(source, device="cpu")
            with safe_open(source, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
            shard_modules = sorted(modules_by_shard[shard_name])
            print(
                f"Merging shard {shard_index}/{len(shard_names)}: {shard_name} "
                f"({len(shard_modules)} LoRA modules)",
                flush=True,
            )
            for base_key in shard_modules:
                a_key, b_key = pairs[base_key]
                base = tensors[base_key]
                lora_a = adapter_tensors[a_key]
                lora_b = adapter_tensors[b_key]
                delta = torch.matmul(lora_b.float(), lora_a.float()).mul_(scaling)
                if delta.shape != base.shape:
                    raise ValueError(
                        f"Delta shape {tuple(delta.shape)} does not match "
                        f"{base_key} shape {tuple(base.shape)}"
                    )
                if not torch.isfinite(delta).all():
                    raise ValueError(f"Non-finite LoRA delta for {base_key}")
                max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))
                merged = base.float().add_(delta).to(dtype=base.dtype)
                if not torch.isfinite(merged).all():
                    raise ValueError(f"Non-finite merged weight for {base_key}")
                tensors[base_key] = merged
                merged_modules += 1
            save_file(tensors, destination, metadata=metadata)
            del tensors

        if merged_modules != len(pairs):
            raise ValueError(f"Merged {merged_modules} modules, expected {len(pairs)}")

        for path in sorted(temporary_dir.iterdir()):
            if path.is_file() and path.name != "es140_merge_manifest.json":
                output_manifest.append(
                    {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                )

        validation = {
            "schema_version": 2,
            "validation_type": "direct_lora_weight_merge",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "passed": True,
            "adapter_tensor_count": len(adapter_tensors),
            "lora_module_count": len(pairs),
            "merged_module_count": merged_modules,
            "unmapped_adapter_tensor_count": 0,
            "nonfinite_delta_count": 0,
            "scaling": scaling,
            "maximum_absolute_delta": max_abs_delta,
        }
        manifest = {
            "schema_version": 2,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "base_model": EXPECTED_BASE_MODEL,
            "base_model_path": str(base_model),
            "dtype": "preserved_per_tensor",
            "merge_method": "direct_safetensors_lora_b_matmul_a",
            "source_adapter": {
                "path": str(adapter_dir),
                "sha256": adapter["sha256"],
                "inferno_update_count": adapter["provenance"]["selection"][
                    "inferno_update_count"
                ],
                "inferno_commit": adapter["provenance"]["source"]["inferno_commit"],
                "rank": adapter["config"]["r"],
                "alpha": adapter["config"]["lora_alpha"],
            },
            "gym_commit": args.gym_commit,
            "validation": validation,
            "output_files": output_manifest,
        }
        (temporary_dir / "es140_merge_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        validation_output.parent.mkdir(parents=True, exist_ok=True)
        validation_output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_dir, output_dir)
        print(json.dumps(validation, indent=2, sort_keys=True), flush=True)
        print(f"Published merged model atomically: {output_dir}", flush=True)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--gym-commit", required=True)
    parser.add_argument("--expected-adapter-sha256", default=EXPECTED_ADAPTER_SHA256)
    return parser.parse_args()


if __name__ == "__main__":
    merge(parse_args())
