# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from resources_servers.matharena_aime import setup_parser, worker
from resources_servers.matharena_aime.parser import UnsafeMathExpression


@pytest.mark.parametrize("uv", ["/usr/bin/uv", None])
def test_runtime_setup_isolated_and_fingerprinted(monkeypatch, tmp_path, uv):
    runtime = tmp_path / "parser-runtime"
    monkeypatch.setattr(setup_parser.shutil, "which", lambda name: uv)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "venv" in command:
            (runtime / "bin").mkdir(parents=True)
            (runtime / "bin" / "python").touch()
        return subprocess.CompletedProcess(command, 0, stdout='{"valid":true,"reward":1.0}')

    monkeypatch.setattr(setup_parser.subprocess, "run", run)
    python = setup_parser.ensure_parser_runtime(runtime)
    assert python == runtime / "bin" / "python"
    initial_calls = 3 if uv else 4
    assert len(calls) == initial_calls
    install = calls[initial_calls - 2][0]
    assert "--no-deps" in install
    assert install[-1].endswith("parser-requirements.txt")
    assert str(python) in install and sys.executable != str(python)
    assert (runtime / ".installed").read_text()
    setup_parser.ensure_parser_runtime(runtime)
    assert len(calls) == initial_calls
    (runtime / ".installed").write_text("stale-source-fingerprint")
    setup_parser.ensure_parser_runtime(runtime)
    assert len(calls) == initial_calls + (2 if uv else 3)  # Never recreate an existing venv.


@pytest.mark.parametrize("result", [{"valid": False}, {"valid": True, "reward": 0.0}])
def test_runtime_failed_self_test_never_marks_installed(monkeypatch, tmp_path, result):
    runtime = tmp_path / "parser-runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin" / "python").touch()
    monkeypatch.setattr(setup_parser.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        setup_parser.subprocess,
        "run",
        MagicMock(return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps(result))),
    )
    with pytest.raises(RuntimeError, match="self-test failed"):
        setup_parser.ensure_parser_runtime(runtime)
    assert not (runtime / ".installed").exists()


def test_runtime_install_failure_is_not_silently_accepted(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_parser.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(
        setup_parser.subprocess, "run", MagicMock(side_effect=subprocess.CalledProcessError(1, ["uv"]))
    )
    with pytest.raises(subprocess.CalledProcessError):
        setup_parser.ensure_parser_runtime(tmp_path / "runtime")
    assert not (tmp_path / "runtime" / ".installed").exists()


@pytest.fixture
def worker_environment(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    limits = MagicMock()
    monkeypatch.setattr(worker.resource, "setrlimit", limits)
    versions = {
        "sympy": "1.14.0",
        "antlr4-python3-runtime": "4.11.1",
        "regex": "2026.2.28",
        "loguru": "0.7.3",
        "mpmath": "1.3.0",
    }
    monkeypatch.setattr(worker.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(
        sys, "stdin", io.StringIO(json.dumps({"text": r"\boxed{42}", "strict": False, "expected_answer": 42}))
    )
    return limits


def test_worker_json_protocol_limits_and_success(monkeypatch, capsys, worker_environment):
    parser = MagicMock(return_value={"reward": 1.0, "extracted_answer": "42", "parser_warning": 0})
    monkeypatch.setattr("resources_servers.matharena_aime.parser.parse_result", parser)
    worker.main()
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] and result["reward"] == 1.0
    parser.assert_called_once_with(r"\boxed{42}", strict=False, expected_answer=42, output_tokens=0)
    worker_environment.assert_any_call(worker.resource.RLIMIT_AS, (2048 * 1024**2, 2048 * 1024**2))
    worker_environment.assert_any_call(worker.resource.RLIMIT_CPU, (5, 6))


@pytest.mark.parametrize(
    "exception,prefix", [(UnsafeMathExpression("blocked"), "unsafe_expression"), (ValueError("bad"), "parser_error")]
)
def test_worker_parsing_failure_is_invalid(monkeypatch, capsys, worker_environment, exception, prefix):
    monkeypatch.setattr("resources_servers.matharena_aime.parser.parse_result", MagicMock(side_effect=exception))
    worker.main()
    result = json.loads(capsys.readouterr().out)
    assert not result["valid"] and result["verifier_error"].startswith(prefix)


def test_worker_rejects_wrong_dependency_versions(monkeypatch, capsys, worker_environment):
    monkeypatch.setattr(worker.importlib.metadata, "version", lambda package: "wrong")
    worker.main()
    result = json.loads(capsys.readouterr().out)
    assert not result["valid"] and "dependency must be sympy==1.14.0" in result["verifier_error"]


def test_worker_rejects_malformed_request(monkeypatch, capsys, worker_environment):
    monkeypatch.setattr(sys, "stdin", io.StringIO("not JSON"))
    worker.main()
    result = json.loads(capsys.readouterr().out)
    assert not result["valid"] and "JSONDecodeError" in result["verifier_error"]
