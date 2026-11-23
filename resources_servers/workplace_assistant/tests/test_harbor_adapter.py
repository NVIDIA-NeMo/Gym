# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the workplace_assistant Harbor adapter: deterministic output, each refusal, the dataset README and id map,
and the vendored code grading exactly as the Resources Server's own ``is_correct``."""

import asyncio
import copy
import filecmp
import hashlib
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pytest
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse

from nemo_gym.data.tasks.harbor import load_task_data_schema
from resources_servers.workplace_assistant.app import WorkbenchResponse, WorkbenchVerifyRequest
from resources_servers.workplace_assistant.harbor import adapter
from resources_servers.workplace_assistant.utils import get_tools, is_correct


ADAPTER = Path(adapter.__file__)
SERVER_DIR = ADAPTER.parents[1]
SEND = {
    "name": "email_send_email",
    "arguments": json.dumps({"recipient": "dana@example.com", "subject": "Q4 plan", "body": "Draft attached."}),
}
# A read-only call: it changes no table, so a journal that adds it to SEND ends in the same state as SEND alone.
SEARCH = {"name": "email_search_emails", "arguments": json.dumps({"query": "Q4"})}


def _row(row_id: int, ground_truth: list[dict]) -> dict:
    return {
        "id": row_id,
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "Today's date is Thursday, 2023-11-30."},
                {"role": "user", "content": f"Task {row_id}: email dana about the Q4 plan."},
            ],
            "tools": [{"type": "function", "name": "email_send_email", "parameters": {"type": "object"}}],
            "parallel_tool_calls": False,
            "temperature": 1.0,
        },
        "ground_truth": ground_truth,
        "category": "workplace_assistant_email",
        "environment_name": "workplace_assistant",
        "agent_ref": {"type": "responses_api_agents", "name": "workplace_assistant_simple_agent"},
    }


ROWS = [_row(7, [SEND]), _row(3, []), _row(12, [SEND, SEND])]


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.fixture
def rows_file(tmp_path: Path) -> Path:
    return _write_rows(tmp_path / "rows.jsonl", ROWS)


@pytest.fixture
def ids_file(tmp_path: Path) -> Path:
    path = tmp_path / "ids.txt"
    path.write_text("# smoke subset\n12\n7  # the email row\n")
    return path


def _run(*args: str | Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ADAPTER), *map(str, args)], capture_output=True, text=True, check=False, timeout=120
    )


def _convert(tmp_path: Path, name: str, rows_file: Path, ids_file: Path) -> Path:
    out = tmp_path / name
    result = _run(
        "--rows", rows_file, "--output-dir", out, "--ids-file", ids_file,
        "--native-rows-out", f"{out}.native.jsonl", "--opencode-rows-out", f"{out}.opencode.jsonl",
        "--id-map-out", f"{out}.id_map.json",
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    return out


def _tree_equal(left: Path, right: Path) -> bool:
    comparison = filecmp.dircmp(left, right)
    if comparison.left_only or comparison.right_only or comparison.funny_files:
        return False
    _, mismatch, errors = filecmp.cmpfiles(left, right, comparison.common_files, shallow=False)
    return (
        not mismatch and not errors and all(_tree_equal(left / name, right / name) for name in comparison.common_dirs)
    )


def test_two_conversions_are_byte_identical(tmp_path: Path, rows_file: Path, ids_file: Path) -> None:
    first = _convert(tmp_path, "a", rows_file, ids_file)
    second = _convert(tmp_path, "b", rows_file, ids_file)
    assert _tree_equal(first, second)
    for suffix in (".native.jsonl", ".opencode.jsonl", ".id_map.json"):
        assert Path(f"{first}{suffix}").read_bytes() == Path(f"{second}{suffix}").read_bytes()


def test_tasks_native_rows_and_id_map_follow_the_ids_file(tmp_path: Path, rows_file: Path, ids_file: Path) -> None:
    out = _convert(tmp_path, "a", rows_file, ids_file)
    tasks = [json.loads(line) for line in (out / "tasks.jsonl").read_text().splitlines()]
    assert [task["task_id"] for task in tasks] == ["wa-validation-0012", "wa-validation-0007"]
    assert tasks[1]["task_data"] == {
        "system_prompt": "Today's date is Thursday, 2023-11-30.",
        "prompt": "Task 7: email dana about the Q4 plan.",
        "ground_truth": [SEND],
        "id": 7,
        "category": "workplace_assistant_email",
        "environment_name": "workplace_assistant",
    }
    raw_lines = rows_file.read_bytes().splitlines(keepends=True)
    assert Path(f"{out}.native.jsonl").read_bytes() == raw_lines[2] + raw_lines[0]
    opencode = [json.loads(line) for line in Path(f"{out}.opencode.jsonl").read_text().splitlines()]
    assert [row["id"] for row in opencode] == [12, 7]
    assert "agent_ref" not in opencode[0]
    assert opencode[0]["responses_create_params"] == {
        "input": ROWS[2]["responses_create_params"]["input"],
        "temperature": 1.0,
    }
    id_map = json.loads(Path(f"{out}.id_map.json").read_text())
    assert id_map == {"12": "wa-validation-0012", "7": "wa-validation-0007"}


def test_every_converted_row_satisfies_the_datasets_own_task_data(tmp_path: Path, rows_file: Path) -> None:
    out = tmp_path / "out"
    assert _run("--rows", rows_file, "--output-dir", out).returncode == 0
    schema = load_task_data_schema(out)
    tasks = [json.loads(line) for line in (out / "tasks.jsonl").read_text().splitlines()]
    assert len(tasks) == len(ROWS)
    for task in tasks:
        # Gym's own check (strict types, unknown keys refused); every declared field is written, defaulted or not.
        schema.validate(task["task_data"], task_id=task["task_id"])
        assert set(task["task_data"]) == set(schema.model.model_fields)


# The contract of each TaskData field: the native row value it carries, and the template files that read it. id,
# category and environment_name are provenance: no task file reads them.
FIELD_OWNERS = {
    "system_prompt": lambda row: row["responses_create_params"]["input"][0]["content"],
    "prompt": lambda row: row["responses_create_params"]["input"][1]["content"],
    "ground_truth": lambda row: row["ground_truth"],
    "id": lambda row: row["id"],
    "category": lambda row: row["category"],
    "environment_name": lambda row: row["environment_name"],
}
FIELD_READERS = {
    "system_prompt": ["instruction.template.md"],
    "prompt": ["instruction.template.md"],
    "ground_truth": ["solution/calls.template.json", "tests/ground_truth.template.json"],
    "id": [],
    "category": [],
    "environment_name": [],
}
# Gym's two placeholder spellings, {{ task_data.key }} and {{ key }}, in the files it renders (*.template*).
PLACEHOLDER = re.compile(r"\{\{\s*(?:task_data\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def test_each_task_field_carries_its_native_owner_to_the_files_that_read_it() -> None:
    template = adapter.TEMPLATE_DIR
    schema = load_task_data_schema(template)
    # Types: TaskData's against the native verify request's, which owns the four carried fields.
    fields = schema.model.model_fields
    template_types = {name: field.annotation for name, field in fields.items()}
    native_types = {
        name: WorkbenchVerifyRequest.model_fields[name].annotation
        for name in ("ground_truth", "id", "category", "environment_name")
    }
    assert all(field.is_required() for field in fields.values())
    assert template_types == {
        "system_prompt": str,
        "prompt": str,
        **native_types,
        "ground_truth": list[dict[str, str]],
    }
    # The native request also accepts ground_truth as JSON text; the tasks carry only the list form (the adapter
    # refuses text).
    assert native_types["ground_truth"] == list[Dict[str, str]] | str
    assert set(FIELD_OWNERS) == set(FIELD_READERS) == set(schema.model.model_fields)
    assert adapter.task_data(ROWS[2]) == {name: owner(ROWS[2]) for name, owner in FIELD_OWNERS.items()}
    readers: dict[str, list[str]] = {name: [] for name in FIELD_READERS}
    for path in sorted(template.rglob("*.template*")):
        for name in PLACEHOLDER.findall(path.read_text(encoding="utf-8")):
            readers.setdefault(name, []).append(path.relative_to(template).as_posix())
    assert readers == FIELD_READERS
    # The oracle and the verifier json.load the rendered files, so each holds the ground truth alone: the native list.
    for reader in FIELD_READERS["ground_truth"]:
        assert (template / reader).read_text(encoding="utf-8").strip() == "{{ task_data.ground_truth }}"
    assert 'replay.py" http://tools:8000/mcp "$here/calls.json"' in (template / "solution" / "solve.sh").read_text()
    assert 'grade.py" "$journal" "$tests/ground_truth.json"' in (template / "tests" / "test.sh").read_text()


def test_non_empty_output_is_refused(tmp_path: Path, rows_file: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    result = _run("--rows", rows_file, "--output-dir", out)
    assert result.returncode == 2
    assert "not empty" in result.stderr
    assert sorted(path.name for path in out.iterdir()) == ["keep.txt"]


def _system_user_roles(row: dict) -> None:
    row["responses_create_params"]["input"].insert(0, {"role": "developer", "content": "Be brief."})


def _list_content(row: dict) -> None:
    row["responses_create_params"]["input"][1]["content"] = [{"type": "input_text", "text": "Task"}]


def _top_p(row: dict) -> None:
    row["responses_create_params"]["top_p"] = 0.9


def _temperature(row: dict) -> None:
    row["responses_create_params"]["temperature"] = 0.2


def _ground_truth_dict_arguments(row: dict) -> None:
    row["ground_truth"] = [{"name": "email_send_email", "arguments": {"recipient": "dana@example.com"}}]


def _no_category(row: dict) -> None:
    del row["category"]


def _null_category(row: dict) -> None:
    row["category"] = None


def _environment_name_not_text(row: dict) -> None:
    row["environment_name"] = ["workplace_assistant"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_system_user_roles, "expected a system and a user message"),
        (_list_content, "plain-text messages"),
        (_top_p, "['top_p']"),
        (_temperature, "temperature 0.2"),
        (_ground_truth_dict_arguments, "ground_truth is not a list"),
        (_no_category, "no ['category']"),
        (_null_category, "task-template/task_data.py: category: Input should be a valid string"),
        (_environment_name_not_text, "environment_name: Input should be a valid string"),
    ],
)
def test_rows_the_tasks_do_not_reproduce_are_refused(tmp_path: Path, change, message: str) -> None:
    row = copy.deepcopy(ROWS[0])
    change(row)
    rows_file = _write_rows(tmp_path / "rows.jsonl", [ROWS[1], row])
    out = tmp_path / "out"
    result = _run("--rows", rows_file, "--output-dir", out)
    assert result.returncode == 2
    assert message in result.stderr
    assert not out.exists()


@pytest.mark.parametrize(
    ("ids", "message"),
    [("7\n99\n", "ids not in rows.jsonl: [99]"), ("7\n7\n", "duplicate ids [7]"), ("seven\n", "is not a row id")],
)
def test_bad_ids_are_refused(tmp_path: Path, rows_file: Path, ids: str, message: str) -> None:
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text(ids)
    result = _run("--rows", rows_file, "--output-dir", tmp_path / "out", "--ids-file", ids_file)
    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize("with_ids_file", [True, False])
def test_duplicate_row_ids_are_refused(tmp_path: Path, ids_file: Path, with_ids_file: bool) -> None:
    rows_file = _write_rows(tmp_path / "rows.jsonl", [*ROWS, ROWS[0]])
    out = tmp_path / "out"
    result = _run("--rows", rows_file, "--output-dir", out, *(["--ids-file", ids_file] if with_ids_file else []))
    assert result.returncode == 2
    assert "more than one row has id 7" in result.stderr
    assert not out.exists()


def test_a_non_integer_row_id_is_refused(tmp_path: Path) -> None:
    row = copy.deepcopy(ROWS[0])
    row["id"] = "7"
    out = tmp_path / "out"
    result = _run("--rows", _write_rows(tmp_path / "rows.jsonl", [row]), "--output-dir", out)
    assert result.returncode == 2
    assert "row id '7' is not an integer" in result.stderr
    assert not out.exists()


@pytest.mark.parametrize("empty", ["rows file", "ids file"])
def test_an_empty_selection_is_refused(tmp_path: Path, empty: str) -> None:
    out = tmp_path / "out"
    if empty == "rows file":
        selection = ["--rows", _write_rows(tmp_path / "rows.jsonl", [])]
    else:
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("# only a comment\n\n")
        selection = ["--rows", _write_rows(tmp_path / "rows.jsonl", ROWS), "--ids-file", ids_file]
    result = _run(*selection, "--output-dir", out)
    assert result.returncode == 2
    assert "no rows selected from rows.jsonl" in result.stderr
    assert not out.exists()


def test_a_rows_line_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    rows_file = tmp_path / "rows.jsonl"
    rows_file.write_bytes(json.dumps(ROWS[0]).encode() + b"\n\xff\n")
    out = tmp_path / "out"
    result = _run("--rows", rows_file, "--output-dir", out)
    assert result.returncode == 2
    assert "rows.jsonl line 2: not UTF-8 JSON" in result.stderr
    assert not out.exists()


def test_an_ids_file_that_is_not_utf8_is_refused(tmp_path: Path, rows_file: Path) -> None:
    ids_file = tmp_path / "ids.txt"
    ids_file.write_bytes(b"12\n\xff\n")
    out = tmp_path / "out"
    result = _run("--rows", rows_file, "--output-dir", out, "--ids-file", ids_file)
    assert result.returncode == 2
    assert result.stderr == "adapter: ids.txt: not UTF-8 text\n"
    assert not out.exists()


def test_an_unwritable_output_dir_stops_before_anything_is_written(tmp_path: Path, rows_file: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder")
    native = tmp_path / "native.jsonl"
    result = _run("--rows", rows_file, "--output-dir", blocker / "out", "--native-rows-out", native)
    assert result.returncode == 2
    assert result.stderr.startswith("adapter: ")
    assert not native.exists()


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_a_limit_below_one_is_refused(tmp_path: Path, rows_file: Path, limit: str) -> None:
    out = tmp_path / "out"
    result = _run("--rows", rows_file, "--output-dir", out, "--limit", limit)
    assert result.returncode == 2
    assert not out.exists()


def test_an_unwritable_side_file_stops_before_the_dataset(tmp_path: Path, rows_file: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder")
    out = tmp_path / "out"
    result = _run("--rows", rows_file, "--output-dir", out, "--native-rows-out", blocker / "native.jsonl")
    assert result.returncode == 2
    assert not out.exists()


@pytest.mark.parametrize(("solution", "calls"), [("none", []), ("wrong", adapter.WRONG_CALLS)])
def test_negative_controls_replace_the_reference_calls(
    tmp_path: Path, rows_file: Path, solution: str, calls: list[dict]
) -> None:
    out = tmp_path / "out"
    assert _run("--rows", rows_file, "--output-dir", out, "--solution", solution).returncode == 0
    assert json.loads((out / "solution" / "calls.json").read_text()) == calls
    assert not (out / "solution" / "calls.template.json").exists()


def test_readme_records_the_provenance(tmp_path: Path, rows_file: Path, ids_file: Path) -> None:
    out = _convert(tmp_path, "a", rows_file, ids_file)
    readme = (out / "README.md").read_text()
    assert f"`rows.jsonl`: 3 lines, sha256 `{hashlib.sha256(rows_file.read_bytes()).hexdigest()}`" in readme
    assert f"Tasks: 2; ids file `ids.txt`, sha256 `{hashlib.sha256(ids_file.read_bytes()).hexdigest()}`" in readme
    assert f"Gym commit: `{adapter.gym_commit()}`" in readme
    for relative in adapter.vendored_files():
        digest = hashlib.sha256((SERVER_DIR / relative).read_bytes()).hexdigest()
        assert f"`{relative.as_posix()}` sha256 `{digest}`" in readme
    assert (
        "Regenerate: `python resources_servers/workplace_assistant/harbor/adapter.py --rows rows.jsonl "
        "--output-dir <dir> --ids-file ids.txt --native-rows-out <native rows> --opencode-rows-out <opencode rows> "
        "--id-map-out <id map>`"
    ) in readme
    assert str(tmp_path) not in readme


def test_vendored_files_are_the_servers_own_below_their_header(tmp_path: Path, rows_file: Path) -> None:
    out = tmp_path / "out"
    assert _run("--rows", rows_file, "--output-dir", out).returncode == 0
    relatives = adapter.vendored_files()
    assert Path("utils.py") in relatives
    assert any(relative.suffix == ".csv" for relative in relatives)
    for relative in relatives:
        source = (SERVER_DIR / relative).read_bytes()
        for target in ("environment/wa", "tests/wa"):
            copied = (out / target / "resources_servers" / "workplace_assistant" / relative).read_bytes()
            if relative.suffix != ".py":
                assert copied == source
                continue
            header, rest = copied.split(b"\n", 1)
            assert rest == source
            assert header.decode() == (
                f"# Vendored unchanged from resources_servers/workplace_assistant/{relative.as_posix()} at "
                f"{adapter.gym_commit()}; source sha256 {hashlib.sha256(source).hexdigest()}; license Apache-2.0"
            )


GRADE = """
import runpy, sys
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
print(sys.modules["resources_servers.workplace_assistant.utils"].__file__)
"""


@pytest.mark.parametrize(
    ("predicted", "ground_truth", "expected"),
    [
        ([SEND], [SEND], 1.0),
        ([SEARCH, SEND], [SEND], 1.0),
        (adapter.WRONG_CALLS, [SEND], 0.0),
        (adapter.WRONG_CALLS, [], 0.0),
        ([], [], 1.0),
        ([], [SEND], 0.0),
    ],
)
def test_vendored_grader_scores_as_the_servers_is_correct(
    tmp_path: Path, rows_file: Path, predicted: list[dict], ground_truth: list[dict], expected: float
) -> None:
    out = tmp_path / "out"
    assert _run("--rows", rows_file, "--output-dir", out).returncode == 0
    journal = tmp_path / "calls.jsonl"
    journal.write_text("".join(json.dumps(call) + "\n" for call in predicted))
    truth = tmp_path / "ground_truth.json"
    truth.write_text(json.dumps(ground_truth))
    reward_file = tmp_path / "reward.json"
    tests = out / "tests"
    result = subprocess.run(
        [sys.executable, "-c", GRADE, str(tests / "grade.py"), str(journal), str(truth), str(reward_file)],
        capture_output=True, text=True, check=False, timeout=120, cwd=tmp_path,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.splitlines()[-1]).is_relative_to(tests / "wa")
    reward = json.loads(reward_file.read_text())
    assert reward == {"reward": expected, "journal_calls": len(predicted)}
    assert is_correct(predicted, ground_truth, None) * 1.0 == expected


def test_the_sidecar_journals_every_call_and_replies_as_the_servers_mcp_exposure(
    tmp_path: Path, rows_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    types = pytest.importorskip("mcp.types")
    out = tmp_path / "out"
    assert _run("--rows", rows_file, "--output-dir", out).returncode == 0
    journal_file = tmp_path / "calls.jsonl"
    monkeypatch.setenv("WA_CODE_DIR", str(out / "environment" / "wa"))
    monkeypatch.setenv("WA_JOURNAL", str(journal_file))
    monkeypatch.setattr(sys, "path", list(sys.path))
    sidecar = runpy.run_path(str(out / "environment" / "tool_server.py"), run_name="wa_tool_server")
    calls = [
        ("email_send_email", json.loads(SEND["arguments"])),
        ("email_send_email", {"recipient": None, "subject": "Q4 plan", "body": "Draft attached."}),
        ("no_such_tool", {"query": "Q4"}),
        # page None reaches the tool as its default 1; passed through, it would make the search fail.
        ("email_search_emails", {"query": "Q4 plan", "page": None}),
    ]
    results = [
        asyncio.run(sidecar["on_call_tool"](None, types.CallToolRequestParams(name=name, arguments=arguments)))
        for name, arguments in calls
    ]
    # Every call, the unknown and the failing one included, with its None arguments kept, in call order.
    journal = [json.loads(line) for line in journal_file.read_text().splitlines()]
    assert journal == [{"name": name, "arguments": json.dumps(arguments)} for name, arguments in calls]
    # By alias: the MCP SDK names the fields isError and structuredContent on the wire, whatever its attributes are.
    replies = [result.model_dump(by_alias=True) for result in results]
    assert replies[2]["isError"] is True
    assert replies[2]["content"][0]["text"].startswith("Unknown tool: 'no_such_tool'")
    # The known calls answer as the server's POST /{path} route does over Gym's MCP exposure, replayed here on a
    # fresh set of the server's own tables: None arguments dropped, the reply the JSON of {"output": <result>}.
    functions = get_tools(sidecar["TOOLKITS"])["functions"]
    for (name, arguments), reply in zip(calls, replies):
        if name == "no_such_tool":
            continue
        output = functions[name](**{key: value for key, value in arguments.items() if value is not None})
        payload = jsonable_encoder(WorkbenchResponse(output=output))
        expected = types.CallToolResult(
            content=[types.TextContent(type="text", text=JSONResponse(payload).body.decode("utf-8"))],
            structuredContent=payload,
        )
        assert reply == expected.model_dump(by_alias=True)
        assert reply["isError"] is False
    assert replies[0]["structuredContent"] == {"output": "Email sent successfully."}
    assert replies[1]["structuredContent"] == {"output": "Recipient, subject, or body not provided."}
    assert "Draft attached." in replies[3]["content"][0]["text"]
    assert {route.path for route in sidecar["build_app"]().routes} == {"/mcp"}
