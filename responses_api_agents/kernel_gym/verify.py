# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import torch
from kernelbench.eval import eval_kernel_against_ref


reference = Path("/workspace/reference.py").read_text()
solution = Path("/workspace/solution.py").read_text()
evaluation_completed = False
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the verifier sandbox")
    device = torch.device("cuda:0")
    torch.cuda.synchronize(device=device)
    # After the device preflight, exceptions can originate in the submitted code.
    evaluation_completed = True
    result = eval_kernel_against_ref(
        original_model_src=reference,
        custom_model_src=solution,
        measure_performance=True,
        timing_method="cuda_event",
        num_correct_trials=5,
        num_perf_trials=100,
        device=device,
        backend="triton",
        precision=torch.float32,
    )
    if result is None:
        evaluation_completed = False
        raise RuntimeError("KernelBench reported a retryable compilation failure")
    speedup = result.ref_runtime / result.runtime if result.correctness and result.runtime > 0 else 0.0
    output = {
        "compiled": result.compiled,
        "correctness": result.correctness,
        "runtime": result.runtime,
        "ref_runtime": result.ref_runtime,
        "speedup": speedup,
    }
except Exception as error:
    output = {
        "compiled": False,
        "correctness": False,
        "runtime": -1,
        "ref_runtime": -1,
        "speedup": 0,
        "error": repr(error),
    }
output["evaluation_completed"] = evaluation_completed
Path("/logs/verifier/result.json").write_text(json.dumps(output, default=str))
Path("/logs/verifier/reward.txt").write_text("1\n" if output["correctness"] and output["speedup"] > 1 else "0\n")
print(json.dumps(output, default=str))
