# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from responses_api_agents.kernel_gym.prepare import build_row


def test_build_row_creates_gpu_kernel_task(tmp_path: Path) -> None:
    repo = tmp_path / "KernelBench"
    problem_dir = repo / "KernelBench" / "level1"
    problem_dir.mkdir(parents=True)
    (problem_dir / "19_ReLU.py").write_text("import torch\nclass Model(torch.nn.Module):\n    pass\n")

    row = build_row(repo, tmp_path / "tasks", "registry/kernelbench:test", 1, 19)

    metadata = row["responses_create_params"]["metadata"]
    task_dir = Path(metadata["task_dir"])
    assert metadata["gpus"] == "1"
    assert metadata["workdir"] == "/workspace"
    assert metadata["instruction"].startswith("Optimize KernelBench Level 1")
    assert "class ModelNew(Model):" in (task_dir / "solution.py").read_text()
    assert "eval_kernel_against_ref" in (task_dir / "tests" / "verify.py").read_text()


def test_build_row_rejects_missing_problem(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected one"):
        build_row(tmp_path, tmp_path / "tasks", "image", 1, 19)
