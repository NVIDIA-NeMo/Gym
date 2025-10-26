# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

atropos_path = Path(__file__).parent.parent.parent / "atropos"
if str(atropos_path) not in sys.path:
    sys.path.insert(0, str(atropos_path))

from datasets import load_dataset

try:
    from src.eval import eval_kernel_against_ref
    from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
    KERNELBENCH_AVAILABLE = True
except ImportError as e:
    KERNELBENCH_AVAILABLE = False
    print(f"Warning: KernelBench import failed with error: {e}")
    print("Full traceback:")
    import traceback
    traceback.print_exc()

from resources_servers.atropos.app import AtroposResourcesServer

KERNELBENCH_DEVICE = os.getenv("KERNELBENCH_DEVICE", "cuda:0")
BUILD_DIR = os.getenv("KERNELBENCH_BUILD_DIR", "build/kernelbench")

SYSTEM_PROMPT = """You are an expert CUDA kernel developer. Your task is to optimize CUDA kernels for performance.

When responding:
1. Analyze the reference implementation
2. Identify optimization opportunities
3. Provide an optimized CUDA kernel implementation
4. Wrap your code in ```python ... ``` markdown blocks"""


def get_level_and_idx(task_idx: int) -> tuple[int, int]:
    if task_idx < 100:
        return 1, task_idx + 1
    elif task_idx < 200:
        return 2, task_idx - 100 + 1
    elif task_idx < 250:
        return 3, task_idx - 200 + 1
    elif task_idx < 270:
        return 4, task_idx - 250 + 1
    else:
        raise ValueError(f"task_idx {task_idx} out of range [0, 269]")


def load_hf_dataset(level: int, problem_id: int) -> str:
    split = f"level_{level}"
    ds = load_dataset("ScalingIntelligence/KernelBench", split=split)
    row = ds.filter(lambda x: x["problem_id"] == problem_id)
    if len(row) == 0:
        raise ValueError(f"Problem {problem_id} not found in {split}")
    return row[0]["code"]


class KernelBenchAtroposServer(AtroposResourcesServer[None]):
    system_prompt: Optional[str] = SYSTEM_PROMPT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def env_factory(self, task_idx: int) -> dict:
        level, problem_id = get_level_and_idx(task_idx)
        print(f"Loading KernelBench problem: task_idx={task_idx}, level={level}, problem={problem_id}")
        ref_code = load_hf_dataset(level, problem_id)

        prompt = prompt_generate_custom_cuda_from_prompt_template(ref_code)

        env = {
            "level": level,
            "problem_id": problem_id,
            "ref_code": ref_code,
            "prompt": prompt,
        }
        print(f"Loaded level {level} problem {problem_id} ({len(ref_code)} chars)")
        return env

    async def get_initial_item(self, env: dict, task_idx: int) -> dict:
        return {
            "level": env["level"],
            "problem_id": env["problem_id"],
            "prompt": env["prompt"],
        }

    async def format_item_as_message(self, item: dict) -> str:
        return item["prompt"]

    async def score_response(
        self,
        env: dict,
        item: dict,
        response: str,
        messages: list,
    ) -> tuple[float, bool, Optional[dict]]:
        if not KERNELBENCH_AVAILABLE:
            print("ERROR: KernelBench not available, cannot score")
            return 0.0, True, {"error": "KernelBench not installed"}

        # TODO: probably should do simple extration and let model learn, but for now
        code = response
        if "```python" in response:
            parts = response.split("```python")
            candidates = []
            for i in range(1, len(parts)):
                block = parts[i].split("```")[0].strip()
                if block:
                    candidates.append(block)

            # take block with "class ModelNew", otherwise the longest
            code_with_modelnew = [c for c in candidates if "class ModelNew" in c]
            if code_with_modelnew:
                code = code_with_modelnew[-1]
            elif candidates:
                code = max(candidates, key=len)
        elif "```" in response:
            parts = response.split("```")
            candidates = [parts[i].strip() for i in range(1, len(parts), 2) if parts[i].strip()]
            code_with_modelnew = [c for c in candidates if "class ModelNew" in c]
            if code_with_modelnew:
                code = code_with_modelnew[-1]
            elif candidates:
                code = max(candidates, key=len)

        code = code.strip()

        print("\n" + "="*80)
        print(f"Level {env['level']}, Problem {env['problem_id']}")
        print("="*80)
        print(f"Extracted code ({len(code)} chars):\n{code}")
        print("="*80)

        build_dir = os.path.join(
            BUILD_DIR,
            f"level_{env['level']}",
            f"problem_{env['problem_id']}"
        )
        os.makedirs(build_dir, exist_ok=True)

        # eval kernel
        try:
            loop = asyncio.get_event_loop()
            eval_result = await loop.run_in_executor(
                self._executor,
                lambda: eval_kernel_against_ref(
                    original_model_src=env["ref_code"],
                    custom_model_src=code,
                    measure_performance=True,
                    verbose=False,
                    num_correct_trials=1,
                    num_perf_trials=1,
                    build_dir=build_dir,
                    device=KERNELBENCH_DEVICE,
                )
            )

            compiled = bool(getattr(eval_result, "compiled", False))
            runtime = float(getattr(eval_result, "runtime", -1.0))

            reward = 0.3 * (1 if compiled else 0) + max(runtime, 0.0)

            print("\n" + "="*80)
            print(f"Result: Level {env['level']}, Problem {env['problem_id']}")
            print("="*80)
            print(f"Compiled: {compiled}")
            print(f"Runtime speedup: {runtime:.4f}")
            print(f"Reward: {reward:.4f} (0.3 * compiled + speedup)")
            print("="*80 + "\n")

            return reward, True, {
                "compiled": compiled,
                "runtime": runtime,
                "code_length": len(code),
            }

        except Exception as e:
            print(f"\nError during kernel evaluation: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, True, {"error": str(e), "compiled": False}

    def __del__(self):
        # cleanup thread pool
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)


if __name__ == "__main__":
    KernelBenchAtroposServer.run_webserver()
