#!/usr/bin/env python3
"""Merge the validated Inferno Step 140 PEFT adapter into a frozen HF base."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

EXPECTED_BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
EXPECTED_ADAPTER_SHA256 = "e281afb74b78d5ef337c233c1b1faeb42ac72208ca76ed4b5563e1ca4adb3de4"
EXPECTED_UPDATE_COUNT = 140


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
    if provenance.get("selection", {}).get("inferno_update_count") != EXPECTED_UPDATE_COUNT:
        raise ValueError("Adapter provenance is not post-Step 140")
    return {"config": config, "provenance": provenance, "sha256": actual_sha256}


def git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def output_files(root: Path) -> list[dict]:
    files = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name == "es140_merge_manifest.json":
            continue
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return files


def existing_output_matches(output_dir: Path, expected_sha256: str) -> bool:
    manifest_path = output_dir / "es140_merge_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source_adapter", {}).get("sha256") != expected_sha256:
        return False
    for item in manifest.get("output_files", []):
        path = output_dir / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            return False
        if sha256(path) != item["sha256"]:
            return False
    return True


def merge(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = args.adapter_dir.resolve()
    output_dir = args.output_dir.resolve()
    adapter = validate_adapter(adapter_dir, args.expected_adapter_sha256)
    if output_dir.exists():
        if existing_output_matches(output_dir, args.expected_adapter_sha256):
            print(f"Validated existing merged model: {output_dir}")
            return
        raise FileExistsError(f"Refusing to overwrite non-matching output: {output_dir}")

    temporary_dir = output_dir.with_name(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    temporary_dir.mkdir(parents=True)
    try:
        print(f"Loading frozen base model: {args.base_model}", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        print(f"Loading ES adapter: {adapter_dir}", flush=True)
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
        print("Merging ES adapter into frozen base", flush=True)
        merged = peft_model.merge_and_unload(safe_merge=True)
        merged.eval()
        merged.save_pretrained(
            temporary_dir,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        tokenizer.save_pretrained(temporary_dir)

        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "base_model": args.base_model,
            "dtype": "bfloat16",
            "merge_method": "peft.PeftModel.merge_and_unload(safe_merge=True)",
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
            "gym_commit": git_commit(args.gym_dir.resolve()),
            "output_files": output_files(temporary_dir),
        }
        (temporary_dir / "es140_merge_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary_dir, output_dir)
        print(f"Published merged model atomically: {output_dir}", flush=True)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=EXPECTED_BASE_MODEL)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gym-dir", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256", default=EXPECTED_ADAPTER_SHA256)
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


if __name__ == "__main__":
    merge(parse_args())
