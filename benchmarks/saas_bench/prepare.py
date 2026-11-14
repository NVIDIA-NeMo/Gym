# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the bundled native tasks; no separate SaaS checkout is required."""

import json
import os
import sys
import tempfile
from pathlib import Path

from responses_api_agents.saas_bench_agent.runtime import HARNESS_VERSIONS, materialize_bundle


def prepare(output_fpath=None, task_selector="all", harness="opencode", data_dir=None, bundle_root=None, **kwargs):
    if harness not in HARNESS_VERSIONS:
        raise ValueError("Unknown SaaS harness")
    root = materialize_bundle(data_dir or os.environ["SAAS_BENCH_DATA_DIR"], bundle_root)
    sys.path.insert(0, str(root))
    from saas_bench.opencode.core import dataset_rows, inventory

    selected = inventory(root, "tasks", task_selector)
    if task_selector == "all" and len(selected) != 106:
        raise ValueError("Expected the pinned 106-task benchmark")
    destination = Path(output_fpath or Path(__file__).parent / "data/saas_bench_benchmark.jsonl")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for row in dataset_rows(selected):
                row["verifier_metadata"].update(harness=harness, harness_version=HARNESS_VERSIONS[harness])
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
