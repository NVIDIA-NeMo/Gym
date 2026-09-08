# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).parent


def build_row(repo: Path, tasks_dir: Path, image: str, level: int, problem_id: int) -> dict:
    matches = list((repo / "KernelBench" / f"level{level}").glob(f"{problem_id}_*.py"))
    if len(matches) != 1:
        raise ValueError(f"expected one Level {level} problem {problem_id}, found {len(matches)}")

    reference = matches[0].read_text()
    task_dir = tasks_dir / f"level{level}_{problem_id}"
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "reference.py").write_text(reference)
    (task_dir / "solution.py").write_text(reference + "\n\nclass ModelNew(Model):\n    pass\n")
    shutil.copy2(ROOT / "test.sh", tests_dir / "test.sh")
    shutil.copy2(ROOT / "verify.py", tests_dir / "verify.py")

    name = matches[0].stem.split("_", 1)[1].replace("_", " ")
    prompt = (
        f"Optimize KernelBench Level {level} problem {problem_id}: {name}.\n"
        "Work in /workspace. reference.py defines the required Model behavior and inputs. "
        "Edit solution.py so it defines ModelNew with exactly equivalent outputs but lower CUDA runtime. "
        "You may use CUDA extensions or Triton. Do not modify reference.py."
    )
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": prompt}],
            "metadata": {
                "instance_id": f"kernelbench::level{level}::{problem_id}",
                "task_name": f"level{level}_{problem_id}",
                "instruction": prompt,
                "task_dir": str(task_dir.resolve()),
                "docker_image": image,
                "workdir": "/workspace",
                "agent_timeout_sec": "1200",
                "verifier_timeout_sec": "600",
                "cpus": "4",
                "memory_mb": "16384",
                "storage_mb": "30720",
                "gpus": "1",
            },
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernelbench", type=Path, required=True)
    parser.add_argument("--image", required=True, help="Prebuilt image containing KernelBench and CUDA PyTorch")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--problem-id", type=int, action="append", default=[])
    parser.add_argument("--all", action="store_true", help="Prepare all 250 public Level 1-3 tasks")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "kernelbench.jsonl")
    args = parser.parse_args()

    tasks_dir = ROOT / "data" / "tasks"
    problems = (
        [(level, problem_id) for level, count in ((1, 100), (2, 100), (3, 50)) for problem_id in range(1, count + 1)]
        if args.all
        else [(args.level, problem_id) for problem_id in (args.problem_id or [19])]
    )
    rows = [build_row(args.kernelbench, tasks_dir, args.image, level, problem_id) for level, problem_id in problems]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    print(f"Wrote {len(rows)} task(s) to {args.output}")


if __name__ == "__main__":
    main()
