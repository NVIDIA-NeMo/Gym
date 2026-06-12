# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare Tau2 and Tau3 banking_knowledge benchmark data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from benchmarks.tau2.prepare_utils import (
    BANKING_RETRIEVAL_CONFIGS,
    prepare_all_banking_knowledge,
    prepare_banking_knowledge,
    prepare_tau2,
)


def prepare(
    dataset: str = "tau2",
    retrieval_config: str | None = None,
    all_retrieval_configs: bool = False,
) -> Path:
    """Prepare a Tau dataset and return the generated JSONL path.

    Gym calls this with keyword arguments from the benchmark YAML
    ``prepare_args`` field. Direct CLI calls use the same argument names.
    """

    if dataset == "tau2":
        if all_retrieval_configs or retrieval_config is not None:
            raise ValueError("Tau2 base prepare does not accept banking arguments")
        return prepare_tau2()

    if dataset in ("banking", "banking_knowledge"):
        if all_retrieval_configs:
            prepared = prepare_all_banking_knowledge()
            return prepared["terminal_use"]
        if retrieval_config is None:
            raise ValueError("banking_knowledge prepare requires retrieval_config")
        if retrieval_config not in BANKING_RETRIEVAL_CONFIGS:
            supported = ", ".join(BANKING_RETRIEVAL_CONFIGS)
            raise ValueError(
                f"Unsupported banking_knowledge retrieval_config {retrieval_config!r}. Supported: {supported}"
            )
        return prepare_banking_knowledge(retrieval_config)

    raise ValueError(f"Unsupported Tau2 prepare dataset: {dataset!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        default="tau2",
        choices=("tau2", "banking", "banking_knowledge"),
    )
    parser.add_argument(
        "--retrieval-config",
        choices=BANKING_RETRIEVAL_CONFIGS,
        help="banking_knowledge retrieval config to prepare",
    )
    parser.add_argument(
        "--all",
        dest="all_retrieval_configs",
        action="store_true",
        help="prepare JSONLs for every pinned banking_knowledge retrieval config",
    )
    args = parser.parse_args()

    prepare(
        dataset=args.dataset,
        retrieval_config=args.retrieval_config,
        all_retrieval_configs=args.all_retrieval_configs,
    )


if __name__ == "__main__":
    main()
