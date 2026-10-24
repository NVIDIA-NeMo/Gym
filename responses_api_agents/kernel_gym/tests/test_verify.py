# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest


def _run_verifier(
    tmp_path: Path,
    *,
    result=None,
    cuda_available: bool = True,
    cuda_error: Exception | None = None,
    evaluator_error: Exception | None = None,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "reference.py").write_text("trusted reference")
    (workspace / "solution.py").write_text("submitted solution")
    logs = tmp_path / "logs" / "verifier"
    logs.mkdir(parents=True)

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=Mock(return_value=cuda_available), synchronize=Mock(side_effect=cuda_error)
    )
    torch.device = Mock(return_value="cuda:0")
    torch.float32 = "float32"
    evaluator = ModuleType("kernelbench.eval")
    evaluator.eval_kernel_against_ref = Mock(return_value=result, side_effect=evaluator_error)
    pathlib = ModuleType("pathlib")
    pathlib.Path = lambda path: tmp_path / str(path).lstrip("/")
    script = Path(__file__).parents[1] / "verify.py"
    with patch.dict(
        sys.modules,
        {"torch": torch, "kernelbench": ModuleType("kernelbench"), "kernelbench.eval": evaluator, "pathlib": pathlib},
    ):
        runpy.run_path(str(script))
    return json.loads((logs / "result.json").read_text()), float((logs / "reward.txt").read_text()), evaluator


@pytest.mark.parametrize("cuda_available,cuda_error", [(False, None), (True, RuntimeError("CUDA driver unavailable"))])
def test_verifier_masks_cuda_setup_failures(
    tmp_path: Path, cuda_available: bool, cuda_error: Exception | None
) -> None:
    result, reward, evaluator = _run_verifier(tmp_path, cuda_available=cuda_available, cuda_error=cuda_error)

    assert result["evaluation_completed"] is False
    assert "CUDA" in result["error"]
    assert reward == 0
    evaluator.eval_kernel_against_ref.assert_not_called()


def test_verifier_preserves_retryable_compilation_failure(tmp_path: Path) -> None:
    result, reward, evaluator = _run_verifier(tmp_path, result=None)

    evaluator.eval_kernel_against_ref.assert_called_once()
    assert result["evaluation_completed"] is False
    assert "retryable compilation" in result["error"]
    assert reward == 0


@pytest.mark.parametrize(
    "compiled,correctness,runtime,ref_runtime,reward",
    [(False, False, -1, -1, 0), (True, False, -1, -1, 0), (True, True, 2, 4, 1), (True, True, 4, 2, 0)],
)
def test_verifier_preserves_official_scoring(
    tmp_path: Path, compiled, correctness, runtime, ref_runtime, reward
) -> None:
    result, actual_reward, evaluator = _run_verifier(
        tmp_path,
        result=SimpleNamespace(compiled=compiled, correctness=correctness, runtime=runtime, ref_runtime=ref_runtime),
    )

    assert result["evaluation_completed"] is True
    assert result["compiled"] is compiled
    assert result["correctness"] is correctness
    assert result["runtime"] == runtime
    assert actual_reward == reward
    assert evaluator.eval_kernel_against_ref.call_args.kwargs["original_model_src"] == "trusted reference"
    assert evaluator.eval_kernel_against_ref.call_args.kwargs["custom_model_src"] == "submitted solution"


def test_verifier_does_not_mask_exceptions_from_submitted_code(tmp_path: Path) -> None:
    result, reward, _ = _run_verifier(tmp_path, evaluator_error=ValueError("invalid ModelNew constructor"))

    assert result["evaluation_completed"] is True
    assert result["correctness"] is False
    assert "invalid ModelNew constructor" in result["error"]
    assert reward == 0
