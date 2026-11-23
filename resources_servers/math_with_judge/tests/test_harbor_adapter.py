# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the math_with_judge Harbor adapter: deterministic output, refusals, the provenance README, the id map,
and the vendored grader scoring exactly like the Resources Server's own ``verify``."""

import ast
import asyncio
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from resources_servers.math_with_judge.app import (
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
    LibraryJudgeMathVerifyRequest,
)
from resources_servers.math_with_judge.harbor import adapter


QUESTIONS = [
    ("What is 2 + 3?", "5"),
    ('Compute "half" of 1, as a fraction.', "\\frac{1}{2}"),
    ("How many primes are below 10?", "4"),
]


def _row(question: str, answer: str) -> dict:
    return {
        "responses_create_params": {"input": [{"role": "user", "content": question}]},
        "question": question,
        "expected_answer": answer,
    }


@pytest.fixture
def rows_file(tmp_path: Path) -> Path:
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(_row(*pair)) + "\n" for pair in QUESTIONS), encoding="utf-8")
    return path


@pytest.fixture
def native_config(tmp_path: Path) -> Path:
    """A copy of the pairing YAML the adapter reads by default."""
    path = tmp_path / "native.yaml"
    path.write_text(adapter.DEFAULT_NATIVE_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def _edit_config(path: Path, edit) -> Path:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(config)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-B", str(Path(adapter.__file__)), *args], capture_output=True, text=True, check=False
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_two_conversions_are_byte_identical(tmp_path: Path, rows_file: Path) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text("# header\n2\ndapo17k-00000\n")
    for name in ("a", "b"):
        result = _run("--rows", str(rows_file), "--output-dir", str(tmp_path / name), "--ids-file", str(ids))
        assert result.returncode == 0, result.stderr
    assert _tree(tmp_path / "a") == _tree(tmp_path / "b")
    records = [json.loads(line) for line in (tmp_path / "a" / "tasks.jsonl").read_text().splitlines()]
    assert [record["task_id"] for record in records] == ["dapo17k-00002", "dapo17k-00000"]
    assert records[0]["task_data"] == {"question": QUESTIONS[2][0], "expected_answer": QUESTIONS[2][1]}


def test_instruction_folds_the_native_system_prompt(tmp_path: Path, rows_file: Path) -> None:
    assert _run("--rows", str(rows_file), "--output-dir", str(tmp_path / "out")).returncode == 0
    system_prompt = yaml.safe_load(adapter.DEFAULT_NATIVE_CONFIG.read_text())["opencode_math_agent"][
        "responses_api_agents"
    ]["opencode_agent"]["system_prompt"]
    template = (tmp_path / "out" / "instruction.template.md").read_text()
    assert template == system_prompt + "\n\n{{ task_data.question }}"
    # What the agent sees never names the expected answer; only solution/ and tests/ read it.
    for agent_visible in ("instruction.template.md", "environment/Dockerfile", "task.toml"):
        assert "expected_answer" not in (tmp_path / "out" / agent_visible).read_text(), agent_visible
    assert "task_data.expected_answer" in (tmp_path / "out" / "solution" / "answer.template.md").read_text()


def test_native_rows_out_and_id_map_pair_the_arms(tmp_path: Path, rows_file: Path) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text("1\n0\n")
    native, id_map = tmp_path / "native.jsonl", tmp_path / "id_map.json"
    result = _run(
        "--rows", str(rows_file), "--output-dir", str(tmp_path / "out"), "--ids-file", str(ids),
        "--native-rows-out", str(native), "--id-map-out", str(id_map),
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    source_lines = rows_file.read_text().splitlines(keepends=True)
    assert native.read_text() == source_lines[1] + source_lines[0]
    assert json.loads(id_map.read_text()) == {"0": "dapo17k-00001", "1": "dapo17k-00000"}
    task_ids = [json.loads(line)["task_id"] for line in (tmp_path / "out" / "tasks.jsonl").read_text().splitlines()]
    assert task_ids == list(json.loads(id_map.read_text()).values())


def test_readme_records_provenance(tmp_path: Path, rows_file: Path) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text("0\n1\n")
    result = _run(
        "--rows", str(rows_file), "--output-dir", str(tmp_path / "out"), "--ids-file", str(ids),
        "--native-rows-out", str(tmp_path / "native.jsonl"), "--id-map-out", str(tmp_path / "id_map.json"),
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    readme = (tmp_path / "out" / "README.md").read_text()
    grader = tmp_path / "out" / "tests" / adapter.GRADER_MODULE
    assert "`rows.jsonl`" in readme and f"{len(QUESTIONS)} rows" in readme
    assert hashlib.sha256(rows_file.read_bytes()).hexdigest() in readme
    assert "Tasks: 2;" in readme
    assert hashlib.sha256(ids.read_bytes()).hexdigest() in readme
    assert f"Gym commit: `{adapter.gym_commit()}`" in readme
    assert hashlib.sha256(grader.read_bytes()).hexdigest() in readme
    assert hashlib.sha256(adapter.SERVER_APP.read_bytes()).hexdigest() in readme
    regenerate = (
        "python resources_servers/math_with_judge/harbor/adapter.py --rows rows.jsonl --output-dir <dir>"
        " --ids-file ids.txt --native-rows-out native.jsonl --id-map-out id_map.json"
    )
    assert f"Regenerate: `{regenerate}`" in readme
    assert str(tmp_path) not in readme


def test_grader_header_names_its_source(tmp_path: Path, rows_file: Path) -> None:
    assert _run("--rows", str(rows_file), "--output-dir", str(tmp_path / "out")).returncode == 0
    header = (tmp_path / "out" / "tests" / adapter.GRADER_MODULE).read_text().splitlines()[2]
    digest = hashlib.sha256(adapter.SERVER_APP.read_bytes()).hexdigest()
    assert header == (
        f"# Vendored unchanged from resources_servers/math_with_judge/app.py at {adapter.gym_commit()};"
        f" source sha256 {digest}; license Apache-2.0"
    )


@pytest.mark.parametrize(
    ("head", "status", "expected"),
    [("abc", "", "abc"), ("abc", "?? resources_servers/math_with_judge/harbor/", "abc-dirty"), (None, "", "unknown")],
)
def test_gym_commit_marks_uncommitted_changes(monkeypatch, head: str | None, status: str, expected: str) -> None:
    calls = []

    def fake_git(*args: str) -> str | None:
        calls.append(args)
        return head if args[0] == "rev-parse" else status

    monkeypatch.setattr(adapter, "_git", fake_git)
    assert adapter.gym_commit() == expected
    if head:
        # Only the vendored source and the adapter's own folder decide "dirty", not the rest of the checkout.
        assert calls[1] == ("status", "--porcelain", "--", str(adapter.SERVER_APP), str(adapter.HERE))


def test_non_empty_output_is_refused(tmp_path: Path, rows_file: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    assert _run("--rows", str(rows_file), "--output-dir", str(out)).returncode == 0  # an empty folder is accepted
    result = _run("--rows", str(rows_file), "--output-dir", str(out))
    assert result.returncode == 2
    assert "not an empty folder" in result.stderr


@pytest.mark.parametrize(
    ("ids_text", "message"),
    [
        ("0\n0\n", "duplicate ids"),
        ("7\n", "name no row"),
        ("task-1\n", "not a row index"),
        # str.isdigit() accepts a superscript digit that int() rejects.
        ("\u00b2\n", "not a row index"),
        ("# only a comment\n\n", "no ids"),
    ],
)
def test_bad_ids_are_refused(tmp_path: Path, rows_file: Path, ids_text: str, message: str) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text(ids_text)
    result = _run("--rows", str(rows_file), "--output-dir", str(tmp_path / "out"), "--ids-file", str(ids))
    assert result.returncode == 2
    assert message in result.stderr
    assert not (tmp_path / "out").exists()


def _agent(config: dict) -> dict:
    return config["opencode_math_agent"]["responses_api_agents"]["opencode_agent"]


def _server(config: dict) -> dict:
    return config["math_with_judge"]["resources_servers"]["math_with_judge"]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda config: _server(config).update(should_use_judge=True), "should_use_judge"),
        # The server's default turns the judge on.
        (lambda config: _server(config).pop("should_use_judge"), "should_use_judge"),
        (lambda config: _server(config).update(format_tolerant_answer_extraction=True), "format_tolerant"),
        (lambda config: _agent(config).update(entrypoint="app.py"), "legacy_app.py"),
        (lambda config: _agent(config).update(system_prompt_mode="replace"), "system_prompt_mode"),
        (lambda config: _server(config).update(library_verifier_timeout_seconds=0), "library_verifier_timeout"),
        (lambda config: config.pop("opencode_math_agent"), "opencode_math_agent"),
        (lambda config: _agent(config).update(system_prompt="Answer {{ x }}"), "'{{'"),
        (lambda config: _agent(config)["resources_server"].update(name="other"), "not a math_with_judge instance"),
        # The Harbor arm's agent settings come from task.toml and harbor_arm_harness.yaml, not from this config.
        (lambda config: _agent(config).update(timeout=1800), "Harbor arm runs the agent"),
        (lambda config: _agent(config).pop("timeout"), "Harbor arm runs the agent"),
        (lambda config: _agent(config).update(thinking=True), "Harbor arm runs the agent"),
        (lambda config: _agent(config).update(opencode_version="1.17.10"), "Harbor arm runs the agent"),
    ],
)
def test_unreproduced_native_settings_are_refused(
    tmp_path: Path, rows_file: Path, native_config: Path, edit, message: str
) -> None:
    _edit_config(native_config, edit)
    result = _run(
        "--rows", str(rows_file), "--output-dir", str(tmp_path / "out"), "--native-config", str(native_config)
    )
    assert result.returncode == 2
    assert message in result.stderr
    assert not (tmp_path / "out").exists()


def test_malformed_native_yaml_is_refused(tmp_path: Path, rows_file: Path) -> None:
    config = tmp_path / "native.yaml"
    config.write_text("opencode_math_agent: [unclosed\n")
    result = _run("--rows", str(rows_file), "--output-dir", str(tmp_path / "out"), "--native-config", str(config))
    assert result.returncode == 2
    assert "not valid YAML" in result.stderr
    assert not (tmp_path / "out").exists()


def test_native_verifier_timeout_reaches_the_grader(tmp_path: Path, rows_file: Path, native_config: Path) -> None:
    _edit_config(native_config, lambda config: _server(config).update(library_verifier_timeout_seconds=30))
    result = _run(
        "--rows", str(rows_file), "--output-dir", str(tmp_path / "out"), "--native-config", str(native_config)
    )
    assert result.returncode == 0, result.stderr
    grader = (tmp_path / "out" / "tests" / adapter.GRADER_MODULE).read_text()
    assert "\nLIBRARY_VERIFIER_TIMEOUT_SECONDS = 30.0\n" in grader


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({**_row("q", "1"), "responses_create_params": {"input": []}}, "exactly one user message"),
        ({**_row("q", "1"), "question": "other"}, "differs from its user message"),
        ({**_row("q", "1"), "expected_answer": None}, "expected_answer"),
        ({**_row("q", "1"), "responses_create_params": {"input": ["q"]}}, "exactly one user message"),
        (["not", "an", "object"], "not a JSON object"),
        # The native agent honours a per-row timeout the Harbor task cannot carry.
        (
            {
                **_row("q", "1"),
                "responses_create_params": {
                    **_row("q", "1")["responses_create_params"],
                    "metadata": {"agent_timeout_sec": "60"},
                },
            },
            "cannot carry",
        ),
    ],
)
def test_rows_the_task_cannot_carry_are_refused(tmp_path: Path, row: object, message: str) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(row) + "\n")
    result = _run("--rows", str(rows), "--output-dir", str(tmp_path / "out"))
    assert result.returncode == 2
    assert message in result.stderr
    assert not (tmp_path / "out").exists()


def test_non_utf8_rows_are_refused(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b'{"x": "\xff"}\n')
    result = _run("--rows", str(rows), "--output-dir", str(tmp_path / "out"))
    assert result.returncode == 2
    assert "adapter:" in result.stderr


def test_grader_segments_are_byte_copies_of_app_py(tmp_path: Path) -> None:
    grader = adapter.grader_source(library_verifier_timeout_seconds=10.0)
    app_source = adapter.SERVER_APP.read_text()
    app_lines = app_source.splitlines(keepends=True)
    names = set(adapter.GRADER_FUNCTIONS) | set(adapter.GRADER_METHODS)
    found = set()
    for node in ast.walk(ast.parse(app_source)):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            start = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
            assert "".join(app_lines[start - 1 : node.end_lineno]) in grader, node.name
            found.add(node.name)
    assert found == names


def _response(*texts: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="m",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id=f"msg-{index}",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
            for index, text in enumerate(texts)
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


# (expected answer, assistant messages): right, equivalent, wrong, unboxed, an empty or blank last box after a right
# answer (math-verify alone awards both; only the gate scores them 0), the box split over messages, only the last box
# counts, and a reference with math delimiters.
GRADING_CASES = [
    ("5", ["The sum is \\boxed{5}."]),
    ("\\frac{1}{2}", ["So it is \\boxed{0.5}"]),
    ("4", ["\\boxed{5}"]),
    ("4", ["The answer is 4."]),
    ("4", ["The answer is 4 \\boxed{}"]),
    ("4", ["The answer is 4. \\boxed{ }"]),
    ("4", ["Thinking... \\box", "ed{4}"]),
    ("4", ["\\boxed{4} no wait \\boxed{3}"]),
    ("$4$", ["\\boxed{4}"]),
]


def test_vendored_verifier_scores_like_the_resources_server(tmp_path: Path, rows_file: Path, monkeypatch) -> None:
    assert _run("--rows", str(rows_file), "--output-dir", str(tmp_path / "out")).returncode == 0
    monkeypatch.syspath_prepend(str(tmp_path / "out" / "tests"))
    for name in ("verifier", "math_with_judge_grader"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    verifier = importlib.import_module("verifier")

    server = LibraryJudgeMathResourcesServer(
        config=LibraryJudgeMathResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            should_use_judge=False,
        ),
        server_client=MagicMock(spec=ServerClient),
    )

    rewards = []
    for expected, texts in GRADING_CASES:
        response = _response(*texts)
        request = LibraryJudgeMathVerifyRequest(
            question="q",
            expected_answer=expected,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=response,
        )
        native = asyncio.run(server.verify(request)).reward
        harbor = verifier.verify({"question": "q", "expected_answer": expected}, response.model_dump())["reward"]
        assert harbor == native, (expected, texts)
        rewards.append(native)
    # The cases cover both verdicts, so equality is not two graders that always say 0.
    assert set(rewards) == {0.0, 1.0}


@pytest.fixture
def verifier(tmp_path: Path, rows_file: Path, monkeypatch):
    """The generated dataset's ``tests/verifier.py``, imported with its generated grader."""
    assert _run("--rows", str(rows_file), "--output-dir", str(tmp_path / "out")).returncode == 0
    monkeypatch.syspath_prepend(str(tmp_path / "out" / "tests"))
    for name in ("verifier", "math_with_judge_grader"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("verifier")


def test_library_reward_times_out(verifier, monkeypatch) -> None:
    # The native cap (library_verifier_timeout_seconds) scores a wedged math-verify 0 instead of waiting for it.
    monkeypatch.setattr(verifier, "LIBRARY_VERIFIER_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(verifier, "_run_math_verify_in_subprocess", lambda *args: time.sleep(60))
    start = time.monotonic()
    assert verifier.library_reward("4", "\\boxed{4}") == 0.0
    assert time.monotonic() - start < 10


@pytest.mark.parametrize(
    "target",
    [lambda expected, generated, conn: os._exit(1), lambda expected, generated, conn: conn.close()],
    ids=["child-exits-non-zero", "child-sends-no-result"],
)
def test_library_reward_scores_a_failed_child_zero(verifier, monkeypatch, target) -> None:
    # The fork context does not pickle the target, so a lambda runs in the child.
    monkeypatch.setattr(verifier, "_run_math_verify_in_subprocess", target)
    assert verifier.library_reward("4", "\\boxed{4}") == 0.0
