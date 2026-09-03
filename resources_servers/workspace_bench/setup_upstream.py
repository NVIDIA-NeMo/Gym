# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import tempfile
from pathlib import Path


UPSTREAM_COMMIT = "3fbd0f1a136720fece86786545983e26642c3db2"
UPSTREAM_URL = "https://github.com/OpenDataBox/Workspace-Bench.git"


def ensure_upstream() -> Path:
    configured = os.environ.get("WORKSPACE_BENCH_UPSTREAM_DIR")
    root = Path(configured) if configured else Path(__file__).parent / ".upstream"
    evaluator = root / "evaluation" / "src" / "agent_as_a_judge.py"
    if evaluator.is_file():
        return root
    if root.exists():
        raise RuntimeError(f"Invalid Workspace-Bench checkout: {root}")
    with tempfile.TemporaryDirectory(dir=root.parent) as temporary_dir:
        checkout = Path(temporary_dir) / "Workspace-Bench"
        subprocess.run(["git", "clone", UPSTREAM_URL, str(checkout)], check=True)
        subprocess.run(["git", "checkout", UPSTREAM_COMMIT], cwd=checkout, check=True)
        subprocess.run(["npm", "install", "--prefix", str(checkout / "evaluation")], check=True)
        checkout.rename(root)
    return root
