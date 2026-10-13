# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entry point that runs inside the verification sandbox.

Usage: ``python sandbox_runner.py REQUEST_JSON RESULT_JSON``

Grades one submission with the unchanged ``CCCEvaluator``; only the executor differs.
``ProcessSandbox`` runs each shell command the way the NeMo Skills sandbox's
``execute_shell`` does, which is what the competitive-coding baselines were scored with.
The per-test time and memory limits come from each problem's ``run.sh``.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from ccc_eval import CCCEvaluator


# NeMo Skills sandbox default RLIMIT_AS/RLIMIT_DATA (NEMO_SKILLS_SANDBOX_MEM_LIMIT, 50 GiB).
# run.sh lowers the address-space limit to the problem's memory limit.
_MEM_LIMIT_KB = 50 * 1024 * 1024


class ProcessSandbox:
    """Local-process stand-in for the NeMo Skills sandbox client used by ``ccc_eval``.

    Like Skills' ``execute_shell``: runs ``/bin/bash`` on a script file in a new session
    with the memory rlimits above, reports ``completed`` only for exit code 0, and cuts
    stdout/stderr at ``max_output_characters``. Unlike it, a timeout kills the whole
    session instead of only bash; the result is the same.
    """

    async def execute_code(
        self,
        generated_code: str,
        std_input: str = "",
        language: str = "shell",
        timeout: float = 10.0,
        max_output_characters: int = 1000,
        session_id: object = None,
        traceback_verbosity: str = "plain",
    ) -> tuple[dict[str, str], None]:
        if language != "shell" or session_id is not None:
            raise ValueError("ProcessSandbox only runs stateless shell commands")
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as script:
            script.write(f"ulimit -v {_MEM_LIMIT_KB}\nulimit -d {_MEM_LIMIT_KB}\n{generated_code}")
        try:
            process = subprocess.Popen(
                ["/bin/bash", script.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                return {
                    "process_status": "timeout",
                    "stdout": "",
                    "stderr": f"Execution timed out after {timeout} seconds\n",
                }, None
        finally:
            os.unlink(script.name)

        result = {
            "process_status": "completed" if process.returncode == 0 else "error",
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }
        for key in ("stdout", "stderr"):
            if len(result[key]) > max_output_characters:
                result[key] = result[key][:max_output_characters] + "<output cut>"
        return result, None

    async def close(self) -> None:
        pass


def main(request_path: Path, result_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    # Absolute: ccc_eval cds into per-test directories and exports TMPDIR relative to this path.
    work_dir = request_path.resolve().parent / "work"
    work_dir.mkdir(exist_ok=True)
    evaluator = CCCEvaluator(
        {
            **request["config"],
            "test_file": request["problem_file"],
            "shared_dir": str(work_dir),
            # The sandbox disk is already local; no staging directory is needed.
            "local_compile_dir": None,
        },
        sandbox_factory=ProcessSandbox,
    )
    try:
        result = asyncio.run(evaluator.eval_single(request["entry"]))
    finally:
        if evaluator.pool is not None:
            evaluator.pool.shutdown(wait=True)
    # Publish atomically so the server never reads a partially written result.
    pending = result_path.with_name(result_path.name + ".pending")
    pending.write_text(json.dumps(result), encoding="utf-8")
    os.replace(pending, result_path)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
