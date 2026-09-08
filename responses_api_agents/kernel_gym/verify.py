# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import sys
import types
from pathlib import Path

import torch


dotenv = types.ModuleType("dotenv")
dotenv.load_dotenv = lambda *args, **kwargs: False
openai = types.ModuleType("openai")
openai.OpenAI = object
litellm = types.ModuleType("litellm")
litellm.completion = None
sys.modules.update(dotenv=dotenv, openai=openai, litellm=litellm)

from kernelbench.eval import eval_kernel_against_ref  # noqa: E402


reference = Path("/workspace/reference.py").read_text()
solution = Path("/workspace/solution.py").read_text()
result = eval_kernel_against_ref(
    original_model_src=reference,
    custom_model_src=solution,
    measure_performance=True,
    timing_method="cuda_event",
    num_correct_trials=5,
    num_perf_trials=100,
    device=torch.device("cuda:0"),
    backend="cuda",
    precision=torch.float32,
)
speedup = result.ref_runtime / result.runtime if result.correctness and result.runtime > 0 else 0.0
output = {
    "compiled": result.compiled,
    "correctness": result.correctness,
    "runtime": result.runtime,
    "ref_runtime": result.ref_runtime,
    "speedup": speedup,
}
Path("/logs/verifier/result.json").write_text(json.dumps(output, default=str))
Path("/logs/verifier/reward.txt").write_text("1\n" if output["correctness"] and output["speedup"] > 1 else "0\n")
print(json.dumps(output, default=str))
