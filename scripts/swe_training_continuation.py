# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare another physical attempt of an existing logical SWE training run."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


_OUTPUT_FIELDS = frozenset({"cache_dir", "results_dir", "artifacts_dir", "verification_log_dir", "sandbox_logs_dir"})


def _move_outputs(value: object, *, original: Path, output: Path) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _OUTPUT_FIELDS and isinstance(child, str):
                try:
                    relative = Path(child).relative_to(original)
                except ValueError:
                    continue
                value[key] = str(output / relative)
            else:
                _move_outputs(child, original=original, output=output)
    elif isinstance(value, list):
        for child in value:
            _move_outputs(child, original=original, output=output)


def prepare_continuation(prepared: Path, output: Path, *, wandb_id: str, prior_job_id: str) -> dict[str, object]:
    """Retain dataset/checkpoint identity and scientific settings, changing only physical outputs.

    ``prepared`` must be the original checkpoint-producing preparation. The caller
    reconciles its terminal job and published checkpoint; this function neither
    inspects checkpoints nor submits jobs. Original inputs remain referenced in
    place, including Harbor task/grader assets and harness settings.
    """
    prepared, output = prepared.resolve(), output.resolve()
    if not wandb_id.strip() or not prior_job_id.isdigit():
        raise ValueError("Supply the original W&B run ID and numeric prior scheduler job ID")
    config = OmegaConf.to_container(OmegaConf.load(prepared / "config.yaml"), resolve=False)
    provenance = json.loads((prepared / "provenance.json").read_text())
    if "continuation" in provenance:
        raise ValueError("Use the original checkpoint-producing preparation, not a continuation attempt")
    if not Path(config["data"]["train"]["data_path"]).is_absolute():
        raise ValueError("Original training data_path must be absolute to preserve dataset identity")
    checkpoint_dir = config["checkpointing"]["checkpoint_dir"]
    if not Path(checkpoint_dir).is_absolute():
        raise ValueError("Original checkpoint_dir must be absolute to preserve the logical run")
    # Dataset paths determine durable task names. Never relocate the data block
    # or its task assets, even when their contents would otherwise be identical.
    _move_outputs(config["env"], original=prepared, output=output)
    config["logger"]["log_dir"] = str(output / "logs")
    config["logger"]["wandb"].update(id=wandb_id, resume="must", mode="online")
    config["logger"]["wandb_enabled"] = True
    receipt = {
        "kind": "same_logical_run_new_physical_attempt",
        "original_preparation": str(prepared),
        "continuation_preparation": str(output),
        "prior_job_id": prior_job_id,
        "wandb_id": wandb_id,
        "wandb_resume": "must",
        "checkpoint_dir": checkpoint_dir,
        "original_train_data_path": config["data"]["train"]["data_path"],
        "total_target_updates": config["grpo"]["max_num_steps"],
        "checkpoint_selection": "trainer chooses latest published checkpoint at startup",
        "checkpoint_inspected": False,
        "terminal_state_verified": False,
        "training_started": False,
        "rollback_accounting": "Keep original physical attempts; distinguish retained checkpoint prefix from discarded updates and repeated logical steps.",
    }
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(config), output / "config.yaml")
    result = copy.deepcopy(provenance)
    result["inherited_preparation_files_sha256"] = result.pop("files_sha256", {})
    result["files_sha256"] = {"config.yaml": hashlib.sha256((output / "config.yaml").read_bytes()).hexdigest()}
    source_mount = prepared / "source-mount.json"
    if source_mount.is_file():
        payload = source_mount.read_bytes()
        (output / source_mount.name).write_bytes(payload)
        result["files_sha256"][source_mount.name] = hashlib.sha256(payload).hexdigest()
    result["continuation"] = receipt
    (output / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "continuation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv: list[str]) -> None:
    """Handle the maintained recipe's ``continue`` command without launching training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True, help="Original checkpoint-producing preparation")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh physical attempt directory")
    parser.add_argument("--wandb-id", required=True, help="Original actual W&B run ID")
    parser.add_argument("--prior-job-id", required=True, help="Actual preceding physical scheduler job ID")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare_continuation(
                args.prepared, args.output_dir, wandb_id=args.wandb_id, prior_job_id=args.prior_job_id
            )
        )
    )
