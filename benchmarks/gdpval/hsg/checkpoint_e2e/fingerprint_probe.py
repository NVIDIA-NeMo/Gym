#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute the exact GDPVal multistage fingerprint without starting servers.

This mirrors the configuration and materialization steps used by ``gym eval
run`` up to (but excluding) resume-state loading and rollout dispatch.  It is a
provider-free migration guard: a package/config cutover must not proceed unless
the recomputed fingerprint matches the frozen campaign journal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from resources_servers.gdpval.multistage_elo import ensure_distribution
from resources_servers.gdpval.multistage_orchestrator import (
    compute_fingerprint,
    find_gdpval_reference_elos,
    parse_multistage_config,
)


FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")
STAGES = [
    {
        "num_tasks": 45,
        "partial_completion": {
            "min_success_fraction": 0.9,
            "min_per_reference_success_fraction": 0.5,
            "min_successful_rows_per_reference": 1,
            "waivable_failure_classes": ["timeout_exceeded", "transient"],
        },
    },
    {"num_tasks": 220, "num_models": 4},
]


def _regular(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved


def _directory(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"{label} must be a directory: {resolved}")
    return resolved


def _initial_config(args: argparse.Namespace) -> DictConfig:
    return OmegaConf.create(
        {
            "config_paths": [str(path) for path in args.config],
            "split": "benchmark",
            "output_jsonl_fpath": str(args.output),
            "policy_base_url": args.policy_url,
            "policy_api_key": "dummy",  # pragma: allowlist secret
            "policy_model_name": args.model_name,
            "num_samples_in_parallel": args.concurrency,
            "num_repeats": 1,
            "responses_create_params": {"max_output_tokens": 262144},
            "resume_from_cache": True,
            # The sibling preprocessed JSONL is materialized and frozen by the
            # provider-free bootstrap. Production must reuse that exact path:
            # ``input_jsonl_fpath`` is part of the multistage fingerprint even
            # when another file has byte-identical rows.
            "reuse_existing_data_preparation": True,
            "overwrite_metrics_conflicts": True,
            "skip_venv_if_present": True,
            "uv_venv_dir": str(args.venv_dir),
            "policy_model": {
                "responses_api_models": {
                    "vllm_model": {
                        "base_url": [args.policy_url],
                        "model": args.model_name,
                    }
                }
            },
            "gdpval_stirrup_agent": {
                "responses_api_agents": {
                    "stirrup_agent": {
                        "judge_only": True,
                        "rerun_incomplete": True,
                        "persist_deliverables_dir": str(args.candidate_view),
                    }
                }
            },
            "gdpval_resources_server": {
                "resources_servers": {
                    "gdpval": {
                        "reward_mode": "comparison",
                        "num_comparison_trials": 4,
                        "preconvert_office_to_pdf": False,
                    }
                }
            },
            "multistage": {
                "enabled": True,
                "stages": STAGES,
                "dataset_path": str(args.dataset),
                # Never let provider-free probing fall back to the module-owned
                # distribution cache.  The runtime overlay is immutable and
                # must remain byte-for-byte equal to its manifest.
                "distribution_path": str(args.distribution_path),
                "seed": 42,
            },
        }
    )


def compute(args: argparse.Namespace) -> dict[str, Any]:
    # The production command resolves relative configs from the pinned Gym root
    # while patched Python modules/configs are found through this exact overlay.
    os.environ["NEMO_GYM_EXTRA_ROOTS"] = str(args.runtime_root)
    os.chdir(args.gym_root)

    parser = GlobalConfigDictParser()
    global_config = parser.parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=_initial_config(args),
            skip_load_from_cli=True,
        )
    )

    # ``gym eval run`` first prepares benchmark data, then swaps this exact
    # prepared JSONL into the collection config before calling the multistage
    # driver.  Reusing the frozen file keeps every materialized row identical.
    collection_config_dict = deepcopy(global_config)
    with open_dict(collection_config_dict):
        collection_config_dict["input_jsonl_fpath"] = str(args.preprocessed_input)
    collection_config = RolloutCollectionConfig.model_validate(
        OmegaConf.to_container(collection_config_dict, resolve=True)
    )
    materialized_rows = RolloutCollectionHelper()._preprocess_rows_from_config(collection_config)

    resolved_global_config = OmegaConf.to_container(global_config, resolve=True)
    if not isinstance(resolved_global_config, dict):
        raise TypeError("resolved global config is not a mapping")
    multistage_config = parse_multistage_config(resolved_global_config.get("multistage") or {})
    reference_elos = find_gdpval_reference_elos(resolved_global_config)
    if not reference_elos:
        raise ValueError("resolved config contains no GDPVal reference ELOs")
    distribution, distribution_path = ensure_distribution(
        multistage_config.distribution_path,
        dataset_path=multistage_config.dataset_path or collection_config.input_jsonl_fpath,
        columns=multistage_config.column,
    )
    distribution_path = _regular(distribution_path, "occupation distribution")
    if distribution_path != args.distribution_path:
        raise ValueError(
            f"resolved occupation distribution path drift: {distribution_path} != {args.distribution_path}"
        )
    fingerprint = compute_fingerprint(
        multistage_config,
        reference_elos,
        distribution,
        materialized_rows=materialized_rows,
        rollout_collection_config=collection_config,
        resolved_global_config=resolved_global_config,
    )
    preprocessed_sha256 = hashlib.sha256(args.preprocessed_input.read_bytes()).hexdigest()
    distribution_sha256 = hashlib.sha256(args.distribution_path.read_bytes()).hexdigest()
    return {
        "schema": "gdpval.multistage-fingerprint-probe.v1",
        "status": "PASS" if fingerprint == args.expected else "MISMATCH",
        "fingerprint": fingerprint,
        "expected_fingerprint": args.expected,
        "materialized_row_count": len(materialized_rows),
        "reference_count": len(reference_elos),
        "config_count": len(args.config),
        "runtime_root": str(args.runtime_root),
        "preprocessed_input": str(args.preprocessed_input),
        "preprocessed_input_sha256": preprocessed_sha256,
        "distribution_path": str(args.distribution_path),
        "distribution_sha256": distribution_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--preprocessed-input", type=Path, required=True)
    parser.add_argument("--distribution-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--venv-dir", type=Path, required=True)
    parser.add_argument("--candidate-view", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--policy-url", default="http://127.0.0.1:9/v1")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    args = parser.parse_args()

    args.gym_root = _directory(args.gym_root, "Gym root")
    args.runtime_root = _directory(args.runtime_root, "runtime root")
    args.dataset = _regular(args.dataset, "dataset")
    args.preprocessed_input = _regular(args.preprocessed_input, "preprocessed input")
    if args.distribution_path.is_symlink():
        raise ValueError("occupation distribution must not be a symlink")
    distribution_parent = _directory(args.distribution_path.parent, "occupation distribution parent")
    args.distribution_path = distribution_parent / args.distribution_path.name
    expected_distribution = args.preprocessed_input.parent.parent / "occupation_distribution.json"
    if args.distribution_path != expected_distribution:
        raise ValueError(
            "occupation distribution must be isolated in the judge namespace: "
            f"{args.distribution_path} != {expected_distribution}"
        )
    if args.distribution_path.exists() and not args.distribution_path.is_file():
        raise ValueError(f"occupation distribution must be a regular file: {args.distribution_path}")
    args.config = [_regular(path, "config") for path in args.config]
    if len(args.config) != 6:
        raise ValueError(f"expected the production six-config stack, found {len(args.config)}")
    if FINGERPRINT_RE.fullmatch(args.expected) is None:
        raise ValueError("expected fingerprint must be 64 lowercase hex characters")
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")

    # Configuration/materialization helpers emit human diagnostics. Keep stdout
    # as one machine-readable receipt for guarded shell migrations.
    with redirect_stdout(sys.stderr):
        result = compute(args)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
