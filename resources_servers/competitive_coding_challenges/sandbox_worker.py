# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import os
import signal
import subprocess
from pathlib import Path

from .ccc_eval import CCCEvaluator, LocalSandbox


class ProcessSandbox(LocalSandbox):
    """Run the native grader shell commands inside an already isolated sandbox."""

    def __init__(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def execute_code(
        self,
        generated_code: str,
        std_input: str = "",
        language: str = "shell",
        timeout: float = 10.0,
        max_output_characters: int = 1000,
        session_id: object | None = None,
        traceback_verbosity: str = "plain",
    ) -> tuple[dict[str, str], None]:
        if language != "shell" or session_id is not None:
            raise ValueError("The CCC worker supports stateless shell execution only")
        process = subprocess.Popen(
            ["/bin/bash", "-c", generated_code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        status = "completed"
        try:
            stdout, stderr = process.communicate(std_input, timeout=timeout)
            if process.returncode != 0:
                status = "error"
        except subprocess.TimeoutExpired:
            status = "timeout"
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            stderr += "\nExecution timed out"
        return {
            "process_status": status,
            "stdout": stdout[:max_output_characters],
            "stderr": stderr[:max_output_characters],
        }, None


async def evaluate(request: dict, directory: Path) -> dict:
    """Evaluate one candidate using the original CCC scripts and aggregation rules."""
    metadata = directory / "metadata.jsonl"
    metadata.write_text(json.dumps(request["metadata"]) + "\n")
    evaluator = CCCEvaluator(
        {
            **request["config"],
            "test_file": str(metadata),
            "shared_dir": str(directory),
            "local_compile_dir": str(directory / "compile"),
        },
        sandbox_factory=ProcessSandbox,
    )
    try:
        result = await evaluator.eval_single(request["entry"])
        errors = [
            output["error"]
            for subtask in result["test_case_results"].values()
            for output in subtask["outputs"]
            if output.get("error")
        ]
        if errors:
            raise RuntimeError(f"CCC worker infrastructure errors: {errors}")
        return result
    finally:
        if evaluator.pool is not None:
            evaluator.pool.shutdown(wait=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    result = asyncio.run(evaluate(json.loads(args.request.read_text()), args.request.parent))
    temporary = args.result.with_suffix(".pending")
    temporary.write_text(json.dumps(result))
    temporary.replace(args.result)
