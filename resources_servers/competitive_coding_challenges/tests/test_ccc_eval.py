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

import asyncio
import shutil
import subprocess
from pathlib import Path

from resources_servers.competitive_coding_challenges import ccc_eval


def _execute_shell(_tls, _sandbox, command, *, language, timeout):
    assert language == "shell"
    completed = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "process_status": "completed" if completed.returncode == 0 else "error",
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
    }


def _write_compile_script(work_dir, artifact_text):
    script = work_dir / "compile.sh"
    script.write_text(
        "#!/bin/bash\n"
        "pwd > compile_pwd.txt\n"
        "printf 'compile\\n' >> compile_invocations.txt\n"
        "cat > compiled_artifact <<'EOF'\n"
        "#!/bin/bash\n"
        f"printf '%s\\n' {artifact_text!r}\n"
        "EOF\n"
        "chmod +x compiled_artifact\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


def test_compile_stages_on_local_storage_and_publishes_artifacts(monkeypatch, tmp_path):
    shared_work_dir = tmp_path / "shared" / "ccc_run_1"
    shared_work_dir.mkdir(parents=True)
    _write_compile_script(shared_work_dir, "compiled locally")
    local_compile_dir = tmp_path / "local"
    monkeypatch.setattr(ccc_eval, "_exec_sync", _execute_shell)

    result = ccc_eval._compile_in_sandbox(
        object(),
        object(),
        str(shared_work_dir),
        str(local_compile_dir),
    )

    assert result["returncode"] == 0
    published_artifact = subprocess.run(
        [shared_work_dir / "compiled_artifact"], capture_output=True, text=True, check=True
    )
    assert published_artifact.stdout == "compiled locally\n"
    compile_pwd = (shared_work_dir / "compile_pwd.txt").read_text().strip()
    assert compile_pwd == str(local_compile_dir / shared_work_dir.name)
    assert not (local_compile_dir / shared_work_dir.name).exists()


def test_compile_without_local_storage_preserves_shared_behavior(monkeypatch, tmp_path):
    shared_work_dir = tmp_path / "shared" / "ccc_run_2"
    shared_work_dir.mkdir(parents=True)
    _write_compile_script(shared_work_dir, "compiled shared")
    monkeypatch.setattr(ccc_eval, "_exec_sync", _execute_shell)

    result = ccc_eval._compile_in_sandbox(object(), object(), str(shared_work_dir), None)

    assert result["returncode"] == 0
    published_artifact = subprocess.run(
        [shared_work_dir / "compiled_artifact"], capture_output=True, text=True, check=True
    )
    assert published_artifact.stdout == "compiled shared\n"
    assert (shared_work_dir / "compile_pwd.txt").read_text().strip() == str(shared_work_dir)


def test_compile_success_uses_process_status_when_stderr_has_warnings():
    result = ccc_eval._test_result_from_compile(
        {
            "process_status": "completed",
            "stdout": "",
            "stderr": "compiler warning\n",
        }
    )

    assert result["compile_success"] is True
    assert result["compile_stderr"] == "compiler warning\n"


def test_compile_failure_uses_process_status_when_stderr_is_empty():
    result = ccc_eval._test_result_from_compile(
        {
            "process_status": "error",
            "stdout": "",
            "stderr": "",
        }
    )

    assert result["compile_success"] is False


def test_compiled_solution_is_reused_without_recompiling_per_test(monkeypatch, tmp_path):
    shared_dir = tmp_path / "shared"
    precompiled_dir = shared_dir / "ccc_pre_toy"
    (precompiled_dir / "graders").mkdir(parents=True)
    _write_compile_script(precompiled_dir, "reusable artifact")
    run_script = precompiled_dir / "run.sh"
    run_script.write_text(
        "#!/bin/bash\ntest -x ./compiled_artifact\n./compiled_artifact >/dev/null\nprintf '1\\n'\n",
        encoding="utf-8",
    )
    run_script.chmod(0o755)
    monkeypatch.setattr(ccc_eval, "_exec_sync", _execute_shell)

    compiled_solution_dir, compile_result = ccc_eval._compile_solution_once(
        "toy",
        "Batch",
        "int solve() { return 0; }\n",
        str(precompiled_dir),
        str(shared_dir),
        str(tmp_path / "local"),
    )
    compiled_solution_path = Path(compiled_solution_dir)

    task = {
        "compiled_solution_dir": compiled_solution_dir,
        "compile_result": compile_result,
        "test_input": "input one\n",
        "test_output": "output one\n",
        "time_scale": 1.0,
        "shared_dir": str(shared_dir),
    }
    first_result = ccc_eval.run_test_case(task, worker_id=0)
    task["test_input"] = "input two\n"
    task["test_output"] = "output two\n"
    second_result = ccc_eval.run_test_case(task, worker_id=1)

    assert first_result["compile_success"] is True
    assert first_result["score"] == 1.0
    assert second_result["compile_success"] is True
    assert second_result["score"] == 1.0
    assert (tmp_path / "local").exists()
    assert (compiled_solution_path / "compile_invocations.txt").read_text(encoding="utf-8") == "compile\n"
    assert list(shared_dir.glob("ccc_run_*")) == []
    shutil.rmtree(compiled_solution_path)


def test_compile_failure_skips_test_staging_and_sandbox_execution(monkeypatch, tmp_path):
    def fail_if_called():
        raise AssertionError("sandbox must not be called after a compile failure")

    monkeypatch.setattr(ccc_eval, "_get_thread_test_sandbox", fail_if_called)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    result = ccc_eval.run_test_case(
        {
            "compiled_solution_dir": str(shared_dir / "missing"),
            "compile_result": {"process_status": "error", "stdout": "", "stderr": "compile failed"},
            "test_input": "",
            "test_output": "",
            "time_scale": 1.0,
            "shared_dir": str(shared_dir),
        },
        worker_id=0,
    )

    assert result["compile_success"] is False
    assert result["compile_stderr"] == "compile failed"
    assert list(shared_dir.glob("ccc_run_*")) == []


def _score_from_input(task_args, worker_id):
    """Stand-in for run_test_case: each toy test's input is the score the submission earns on it."""
    return {"compile_success": True, "score": float(task_args["test_input"]), "run_time_s": 0.0}


def _evaluate_toy(monkeypatch, *, grade_all_subtasks: bool, partial_credit: bool) -> dict:
    # Subtask a: one full and one partial test; b: all full; c: a zero, then a test that would pass.
    problem = {
        "subtasks": {
            "a": {"aggregation": "min", "score": 10.0, "score_precision": 2, "test_names": ["a1", "a2"]},
            "b": {"aggregation": "min", "score": 20.0, "score_precision": 2, "test_names": ["b1", "b2"]},
            "c": {"aggregation": "min", "score": 30.0, "score_precision": 2, "test_names": ["c1", "c2"]},
        },
        "all_tests": {
            name: {"input": score, "output": ""}
            for name, score in [("a1", "1"), ("a2", "0.5"), ("b1", "1"), ("b2", "1"), ("c1", "0"), ("c2", "1")]
        },
    }
    monkeypatch.setattr(ccc_eval, "run_test_case", _score_from_input)
    evaluator = ccc_eval.CCCEvaluator(
        {"test_batch_size": 1, "grade_all_subtasks": grade_all_subtasks, "partial_credit": partial_credit}
    )
    evaluator.pool = ccc_eval.ThreadPoolExecutor(max_workers=1)
    try:
        result = asyncio.run(
            evaluator._evaluate_compiled_solution(
                {"name": "toy", "subtask": "a"}, problem, "compiled", {"process_status": "completed"}, 1.0
            )
        )
    finally:
        evaluator.pool.shutdown()
    return {
        name: (subtask["score"], [output["test_name"] for output in subtask["outputs"]])
        for name, subtask in result["test_case_results"].items()
    }


def test_default_scoring_grades_only_the_rows_subtask_and_stops_below_full_marks(monkeypatch):
    assert _evaluate_toy(monkeypatch, grade_all_subtasks=False, partial_credit=False) == {
        "a": (0.0, ["a1", "a2"]),
        "b": (0.0, []),
        "c": (0.0, []),
    }


def test_ioi_benchmark_scoring_grades_every_subtask_with_partial_credit(monkeypatch):
    # As in NeMo Skills: a partial test keeps the subtask alive (min 0.5 x 10 points); only a zero
    # stops it, so c2 never runs.
    assert _evaluate_toy(monkeypatch, grade_all_subtasks=True, partial_credit=True) == {
        "a": (5.0, ["a1", "a2"]),
        "b": (20.0, ["b1", "b2"]),
        "c": (0.0, ["c1"]),
    }


def test_grade_all_subtasks_without_partial_credit_scores_all_or_nothing(monkeypatch):
    assert _evaluate_toy(monkeypatch, grade_all_subtasks=True, partial_credit=False) == {
        "a": (0.0, ["a1", "a2"]),
        "b": (20.0, ["b1", "b2"]),
        "c": (0.0, ["c1"]),
    }


def test_partial_credit_alone_grades_only_the_rows_subtask(monkeypatch):
    assert _evaluate_toy(monkeypatch, grade_all_subtasks=False, partial_credit=True) == {
        "a": (5.0, ["a1", "a2"]),
        "b": (0.0, []),
        "c": (0.0, []),
    }
