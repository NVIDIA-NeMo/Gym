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

import logging
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

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


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_execute_code_names_a_connection_failure_instead_of_a_timeout():
    sandbox = ccc_eval.LocalSandbox(host="127.0.0.1", port=str(_closed_port()))

    output = ccc_eval._exec_sync(ccc_eval._precompile_loop_tls, sandbox, "echo hello world", language="shell")

    assert output["process_status"] == "timeout"
    assert "ConnectError" in output["stderr"]
    assert "timed out" not in output["stderr"]


def test_wait_for_sandbox_error_names_url_cause_and_how_to_start_it():
    port = _closed_port()
    sandbox = ccc_eval.LocalSandbox(host="127.0.0.1", port=str(port))

    with pytest.raises(RuntimeError) as exc_info:
        ccc_eval.wait_for_sandbox(sandbox, timeout=0.3, poll=0.05)

    message = str(exc_info.value)
    assert f"http://127.0.0.1:{port}/execute" in message
    assert "ConnectError" in message
    assert "README" in message
    assert "NEMO_SKILLS_SANDBOX_HOST" in message and "NEMO_SKILLS_SANDBOX_PORT" in message


def test_wait_for_sandbox_reports_an_unexpected_probe_answer(monkeypatch):
    def wrong_answer(_tls, _sandbox, _command, *, language, timeout):
        return {"process_status": "error", "stdout": "", "stderr": "unsupported language: shell"}

    monkeypatch.setattr(ccc_eval, "_exec_sync", wrong_answer)

    with pytest.raises(RuntimeError, match="unsupported language: shell"):
        ccc_eval.wait_for_sandbox(ccc_eval.LocalSandbox(), timeout=0.2, poll=0.05)


def test_wait_for_sandbox_waits_for_a_sandbox_that_is_still_booting(monkeypatch, caplog):
    # NeMo Skills `with_sandbox=True` starts the sandbox next to this server, so early probes are refused.
    answers = iter(
        [
            {"process_status": "timeout", "stdout": "", "stderr": "ConnectError: connection refused\n"},
            {"process_status": "timeout", "stdout": "", "stderr": "ConnectError: connection refused\n"},
            {"process_status": "completed", "stdout": "hello world\n", "stderr": ""},
        ]
    )
    monkeypatch.setattr(ccc_eval, "_exec_sync", lambda *_args, **_kwargs: next(answers))
    sandbox = ccc_eval.LocalSandbox(host="sandbox-node", port="6123")

    with caplog.at_level(logging.WARNING, logger=ccc_eval.__name__):
        ccc_eval.wait_for_sandbox(sandbox, timeout=5, poll=0.01)

    assert next(answers, None) is None
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "http://sandbox-node:6123/execute" in warnings[0]


def _sandbox_that_cannot_see(hidden_dir):
    """A sandbox that runs the same shell but has no mount for ``hidden_dir``."""

    def execute(_tls, _sandbox, command, *, language, timeout):
        command = command.replace(str(hidden_dir), "/nonexistent-ccc-shared-dir")
        return _execute_shell(_tls, _sandbox, command, language=language, timeout=timeout)

    return execute


def _run_task(compiled_solution_dir, shared_dir):
    return {
        "compiled_solution_dir": str(compiled_solution_dir),
        "compile_result": {"process_status": "completed", "stdout": "", "stderr": ""},
        "test_input": "",
        "test_output": "",
        "time_scale": 1.0,
        "shared_dir": str(shared_dir),
    }


def _write_run_script(work_dir):
    script = work_dir / "run.sh"
    script.write_text("#!/bin/bash\nprintf '1\\n'\n", encoding="utf-8")
    script.chmod(0o755)


@pytest.mark.parametrize("local_compile", [True, False])
def test_compile_in_an_unshared_work_dir_is_a_sandbox_infra_error(monkeypatch, tmp_path, local_compile):
    shared_dir = tmp_path / "shared"
    work_dir = shared_dir / "ccc_solution"
    work_dir.mkdir(parents=True)
    _write_compile_script(work_dir, "never built")
    monkeypatch.setattr(ccc_eval, "_exec_sync", _sandbox_that_cannot_see(shared_dir))

    result = ccc_eval._compile_in_sandbox(
        ccc_eval._test_loop_tls, None, str(work_dir), str(tmp_path / "local") if local_compile else None
    )

    assert ccc_eval._test_result_from_compile(result)["compile_success"] is False
    assert "cannot see" in ccc_eval.compile_infra_error(result)


def test_a_real_compile_failure_is_not_an_infra_error_even_when_it_echoes_the_marker(monkeypatch, tmp_path):
    # The solution's text can reach compiler output; it must not be able to turn a failure into a masked sample.
    work_dir = tmp_path / "ccc_solution"
    (work_dir / "graders").mkdir(parents=True)
    (work_dir / "graders" / "toy.cpp").write_text(f"// {ccc_eval.WORK_DIR_NOT_VISIBLE}\nnot c++\n", encoding="utf-8")
    script = work_dir / "compile.sh"
    script.write_text("#!/bin/bash\necho Compiling toy\ncat graders/toy.cpp >&2\nexit 1\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setattr(ccc_eval, "_exec_sync", _execute_shell)

    result = ccc_eval._compile_in_sandbox(ccc_eval._test_loop_tls, None, str(work_dir), None)

    assert ccc_eval._test_result_from_compile(result)["compile_success"] is False
    assert ccc_eval.compile_infra_error(result) is None


def test_a_run_sh_that_prints_the_marker_stays_a_scored_failure(monkeypatch, tmp_path):
    # run.sh executes the solution, so nothing it prints or returns can mark the sample as infra (review on #4770).
    compiled_dir = tmp_path / "ccc_solution"
    compiled_dir.mkdir()
    script = compiled_dir / "run.sh"
    script.write_text(f"#!/bin/bash\necho {ccc_eval.WORK_DIR_NOT_VISIBLE}\nexit 3\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setattr(ccc_eval, "_exec_sync", _execute_shell)

    result = ccc_eval.run_test_case(_run_task(compiled_dir, tmp_path), worker_id=0)

    assert result["score"] == 0.0
    assert "infra_error" not in result
    assert result["run_time_s"] >= 0.0


def test_generated_code_that_cannot_be_written_as_utf8_fails_before_the_sandbox(monkeypatch, tmp_path):
    monkeypatch.setattr(ccc_eval, "_exec_sync", lambda *_args, **_kwargs: pytest.fail("sandbox must not be called"))

    with pytest.raises(UnicodeEncodeError):
        ccc_eval._compile_solution_once("toy", "Batch", "int main() { return 0; } // \ud800", "", str(tmp_path), None)

    assert list(tmp_path.glob("ccc_solution_*")) == []


def test_run_against_an_unreachable_sandbox_is_an_infra_error(monkeypatch, tmp_path):
    compiled_dir = tmp_path / "ccc_solution"
    compiled_dir.mkdir()
    _write_run_script(compiled_dir)
    sandbox = ccc_eval.LocalSandbox(host="127.0.0.1", port=str(_closed_port()))
    monkeypatch.setattr(ccc_eval, "_get_thread_test_sandbox", lambda: sandbox)

    result = ccc_eval.run_test_case(_run_task(compiled_dir, tmp_path), worker_id=0)

    assert result["score"] == 0.0
    assert "ConnectError" in result["infra_error"]


def test_a_connection_lost_while_the_solution_runs_stays_a_scored_zero(monkeypatch, tmp_path):
    # The solution itself can crash or overload the sandbox; masking that would remove its penalty in training.
    compiled_dir = tmp_path / "ccc_solution"
    compiled_dir.mkdir()
    _write_run_script(compiled_dir)
    monkeypatch.setattr(
        ccc_eval,
        "_exec_sync",
        lambda *_args, **_kwargs: {"process_status": "timeout", "stdout": "", "stderr": "ReadError: \n"},
    )

    result = ccc_eval.run_test_case(_run_task(compiled_dir, tmp_path), worker_id=0)

    assert result["score"] == 0.0
    assert "infra_error" not in result


async def test_evaluation_details_carry_the_sandbox_infra_error():
    evaluator = ccc_eval.CCCEvaluator(config={"test_batch_size": 1})
    problem_metadata = {
        "subtasks": {"s1": {"aggregation": "min", "score": 1, "test_names": ["t1"]}},
        "all_tests": {"t1": {"input": "", "output": ""}},
    }
    compile_result = {"process_status": "error", "stdout": f"{ccc_eval.WORK_DIR_NOT_VISIBLE}\n", "stderr": ""}

    details = await evaluator._evaluate_compiled_solution(
        {"name": "toy", "subtask": "s1"}, problem_metadata, "/unused", compile_result, 1.0
    )

    assert details["test_case_results"]["s1"]["score"] == 0.0
    assert "cannot see" in details["infra_error"]


def _two_test_problem():
    return {
        "subtasks": {"s1": {"aggregation": "sum_tests", "score": 2, "test_names": ["t1", "t2"]}},
        "all_tests": {"t1": {"input": "", "output": ""}, "t2": {"input": "", "output": ""}},
    }


async def test_a_run_stage_infra_error_masks_only_when_no_test_was_measured(monkeypatch):
    unreachable = {"score": 0.0, "infra_error": "NeMo Skills sandbox unreachable: ConnectError"}
    evaluator = ccc_eval.CCCEvaluator(config={"test_batch_size": 2})
    compiled = {"process_status": "completed", "stdout": "", "stderr": ""}

    monkeypatch.setattr(ccc_eval, "run_test_case", lambda _task, _idx: dict(unreachable))
    all_unreachable = await evaluator._evaluate_compiled_solution(
        {"name": "toy", "subtask": "s1"}, _two_test_problem(), "/unused", compiled, 1.0
    )
    # One test ran the solution and failed; the other could not reach a sandbox the solution may have crashed.
    monkeypatch.setattr(
        ccc_eval, "run_test_case", lambda _task, idx: dict(unreachable) if idx else {"score": 0.0, "run_time_s": 0.1}
    )
    one_measured = await evaluator._evaluate_compiled_solution(
        {"name": "toy", "subtask": "s1"}, _two_test_problem(), "/unused", compiled, 1.0
    )

    assert all_unreachable["infra_error"] == unreachable["infra_error"]
    assert "infra_error" not in one_measured


async def test_a_measured_test_is_not_hidden_by_min_aggregation(monkeypatch):
    # With `min`, results after the first failure in a batch are dropped from `outputs`, but those tests still ran.
    # Here the unreachable result sorts first and the measured failure second (review on #4770).
    problem = _two_test_problem()
    problem["subtasks"]["s1"]["aggregation"] = "min"
    evaluator = ccc_eval.CCCEvaluator(config={"test_batch_size": 2})
    unreachable = {"score": 0.0, "infra_error": "NeMo Skills sandbox unreachable: ConnectError"}
    monkeypatch.setattr(
        ccc_eval, "run_test_case", lambda _task, idx: {"score": 0.0, "run_time_s": 0.1} if idx else dict(unreachable)
    )

    details = await evaluator._evaluate_compiled_solution(
        {"name": "toy", "subtask": "s1"}, problem, "/unused", {"process_status": "completed"}, 1.0
    )

    assert details["num_tests_run"] == 1
    assert "infra_error" not in details
