# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the reasoning_gym Harbor adapter: deterministic output, refusals, the id map, what the agent sees, the
generated Dockerfile, and the copied grader scoring like the Resources Server itself."""

import ast
import base64
import copy
import gzip
import importlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import reasoning_gym
import yaml

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.data.tasks.harbor import load_task_data_schema
from resources_servers.reasoning_gym.app import (
    ReasoningGymResourcesServer,
    ReasoningGymResourcesServerConfig,
    ReasoningGymVerifyRequest,
)
from resources_servers.reasoning_gym.harbor import adapter
from resources_servers.reasoning_gym.task_data import TaskData as ServerTaskData


SYSTEM_MESSAGE = "Put only your final answer inside <answer></answer> tags or \\boxed{...}."


def _row(entry: dict) -> dict:
    entry = copy.deepcopy(entry)
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": f"{entry['question']}\n\n{SYSTEM_MESSAGE}"}]
        },
        **entry,
    }


@pytest.fixture(scope="module")
def entries() -> list[dict]:
    dataset = reasoning_gym.create_dataset("knights_knaves", size=4, seed=7)
    return [dataset[index] for index in range(4)]


@pytest.fixture
def rows_file(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(_row(entry)) + "\n" for entry in entries))
    return path


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(Path(adapter.__file__)), *args], capture_output=True, text=True, check=False
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_two_conversions_are_byte_identical(tmp_path: Path, rows_file: Path) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text("# header\n2\n0\n")
    for name in ("a", "b"):
        result = _run("--rows", str(rows_file), "--output-dir", str(tmp_path / name), "--ids-file", str(ids))
        assert result.returncode == 0, result.stderr
    assert _tree(tmp_path / "a") == _tree(tmp_path / "b")
    task_ids = [json.loads(line)["task_id"] for line in (tmp_path / "a" / "tasks.jsonl").read_text().splitlines()]
    assert task_ids == ["rg-00002", "rg-00000"]


def test_template_without_an_environment_folder_converts(tmp_path, rows_file, monkeypatch) -> None:
    # git does not track the template's empty environment/ folder, so a clone has none.
    template = tmp_path / "template"
    shutil.copytree(adapter.TEMPLATE_DIR, template, ignore=shutil.ignore_patterns("environment"))
    assert not (template / "environment").exists()
    monkeypatch.setattr(adapter, "TEMPLATE_DIR", template)
    adapter.convert(rows_file, tmp_path / "out")
    assert (tmp_path / "out" / "environment" / "Dockerfile").read_text() == adapter.dockerfile()


def test_non_empty_output_is_refused(tmp_path: Path, rows_file: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "keep.txt").write_text("x")
    result = _run("--rows", str(rows_file), "--output-dir", str(tmp_path / "out"))
    assert result.returncode == 2
    assert "not empty" in result.stderr


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda agent: agent.update(entrypoint="app.py"), "local CLI path"),
        (lambda agent: agent.update(system_prompt_mode="replace"), "system_prompt_mode"),
        (lambda agent: agent.update(system_prompt="{{ x }}"), "placeholder"),
        (lambda agent: agent.update(timeout=30), "the Harbor arm runs the agent with"),
        (lambda agent: agent.pop("timeout"), "the Harbor arm runs the agent with"),
        (lambda agent: agent.update(thinking=True), "the Harbor arm runs the agent with"),
        (lambda agent: agent.update(opencode_version="1.17.10"), "the Harbor arm runs the agent with"),
    ],
)
def test_native_settings_the_task_cannot_reproduce_are_refused(tmp_path, rows_file, change, message) -> None:
    config = yaml.safe_load(adapter.DEFAULT_NATIVE_CONFIG.read_text())
    change(config[adapter.NATIVE_AGENT_INSTANCE]["responses_api_agents"]["opencode_agent"])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = _run("--rows", str(rows_file), "--output-dir", str(tmp_path / "out"), "--native-config", str(config_path))
    assert result.returncode == 2
    assert message in result.stderr
    assert not (tmp_path / "out").exists()


def test_harbor_arm_runs_the_agent_as_the_default_pairing() -> None:
    config = yaml.safe_load(adapter.DEFAULT_NATIVE_CONFIG.read_text())
    native = config[adapter.NATIVE_AGENT_INSTANCE]["responses_api_agents"]["opencode_agent"]
    harbor = adapter.harbor_agent_settings()
    assert harbor == {key: native[key] for key in ("timeout", "thinking", "opencode_version")}
    assert harbor == {"timeout": 900, "thinking": False, "opencode_version": "1.17.11"}


@pytest.mark.parametrize("answer", [7, ["a"]], ids=["int", "list"])
def test_rows_the_template_schema_refuses_exit_2_before_output(tmp_path, entries, answer) -> None:
    row = _row(entries[0])
    row["answer"] = answer
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(row) + "\n")
    result = _run("--rows", str(rows), "--output-dir", str(tmp_path / "out"))
    assert result.returncode == 2, result.stderr
    assert "row 0: task 'rg-00000': task_data does not satisfy TaskData" in result.stderr
    assert "answer" in result.stderr
    assert not (tmp_path / "out").exists()


def test_generated_rows_load_with_the_shipped_schema(tmp_path: Path, rows_file: Path) -> None:
    out = tmp_path / "out"
    adapter.convert(rows_file, out)
    schema = load_task_data_schema(out)
    rows = [json.loads(line) for line in (out / "tasks.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    for row in rows:
        schema.validate(row["task_data"], task_id=row["task_id"])
        # Reasoning Gym's metadata stays open: each task's own keys pass through.
        assert {"source_dataset", "solution"} <= set(row["task_data"]["metadata"])


def test_task_data_contract_ties_adapter_template_server_and_verifier(tmp_path: Path, rows_file: Path) -> None:
    out = tmp_path / "out"
    adapter.convert(rows_file, out)
    fields = {name: field.annotation for name, field in load_task_data_schema(out).model.model_fields.items()}
    # Adapter -> template: every row carries exactly the declared fields.
    for line in (out / "tasks.jsonl").read_text().splitlines():
        assert set(json.loads(line)["task_data"]) == set(fields)
    # Template -> owners: the grading fields are the native verify request's own fields, with the same types, and the
    # fields the server's task_data.py declares for its rows.
    base = ReasoningGymVerifyRequest.__mro__[1].model_fields
    owned = {
        name: field.annotation for name, field in ReasoningGymVerifyRequest.model_fields.items() if name not in base
    }
    assert {name: kind for name, kind in fields.items() if name != "user_message"} == owned
    assert set(ServerTaskData.model_fields) == set(owned)
    # Template -> readers: the verifier reads the grading fields, the instruction the user message.
    verifier = ast.parse((out / "tests" / "verifier.py").read_text())
    reads = {
        node.slice.value
        for node in ast.walk(verifier)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "task_data"
        and isinstance(node.slice, ast.Constant)
    }
    reads |= {
        node.args[0].value
        for node in ast.walk(verifier)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "task_data"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert reads == set(owned)
    assert re.findall(r"task_data\.(\w+)", (out / "instruction.template.md").read_text()) == ["user_message"]


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda row: row["responses_create_params"]["input"].insert(0, {"role": "system", "content": "s"}),
            "one user",
        ),
        (lambda row: row["responses_create_params"].update(temperature=0.6), "temperature"),
        (lambda row: row["metadata"].pop("source_dataset"), "source_dataset"),
        (lambda row: row.update(answer=None), "no reference answer"),
        (lambda row: row.pop("question"), "'question' string"),
    ],
)
def test_rows_the_task_cannot_reproduce_are_refused(tmp_path, entries, change, message) -> None:
    row = _row(entries[0])
    change(row)
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(row) + "\n")
    result = _run("--rows", str(rows), "--output-dir", str(tmp_path / "out"))
    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize(
    "rows_tail, ids_text, message",
    [
        ("\n", None, "rows.jsonl:5: not a JSON row"),
        ("", "1\n²\n", "is not a 0-based row index"),
    ],
    ids=["blank-rows-line", "non-ascii-digit-id"],
)
def test_malformed_inputs_exit_2(tmp_path, rows_file, rows_tail, ids_text, message) -> None:
    rows_file.write_text(rows_file.read_text() + rows_tail)
    args = ["--rows", str(rows_file), "--output-dir", str(tmp_path / "out")]
    if ids_text is not None:
        ids = tmp_path / "ids.txt"
        ids.write_text(ids_text, encoding="utf-8")
        args += ["--ids-file", str(ids)]
    result = _run(*args)
    assert result.returncode == 2, result.stderr
    assert message in result.stderr
    assert "Traceback" not in result.stderr


def test_id_map_names_the_lines_of_native_rows_out(tmp_path: Path, rows_file: Path) -> None:
    ids = tmp_path / "ids.txt"
    ids.write_text("3\n1\n")
    native, id_map = tmp_path / "native.jsonl", tmp_path / "id_map.json"
    result = _run(
        "--rows", str(rows_file), "--output-dir", str(tmp_path / "out"), "--ids-file", str(ids),
        "--native-rows-out", str(native), "--id-map-out", str(id_map),
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    source = rows_file.read_text().splitlines()
    assert native.read_text().splitlines() == [source[3], source[1]]
    assert json.loads(id_map.read_text()) == {"0": "rg-00003", "1": "rg-00001"}


def test_instruction_is_the_native_prompt_and_hides_the_answer(tmp_path: Path, rows_file: Path) -> None:
    out = tmp_path / "out"
    adapter.convert(rows_file, out)
    config = yaml.safe_load(adapter.DEFAULT_NATIVE_CONFIG.read_text())
    system_prompt = config[adapter.NATIVE_AGENT_INSTANCE]["responses_api_agents"]["opencode_agent"]["system_prompt"]
    instruction = (out / "instruction.template.md").read_text()
    assert system_prompt
    assert instruction == system_prompt + "\n\n{{ task_data.user_message }}"
    assert adapter.native_system_prompt(adapter.DEFAULT_NATIVE_CONFIG) == system_prompt
    visible = [instruction] + [path.read_text() for path in sorted((out / "environment").rglob("*")) if path.is_file()]
    rows = [json.loads(line) for line in (out / "tasks.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    for row in rows:
        task_data = row["task_data"]
        for value in (str(task_data["answer"]), json.dumps(task_data["metadata"]["solution"])):
            assert all(value not in text for text in visible), value
        assert str(task_data["answer"]) not in task_data["user_message"]


def test_dockerfile_installs_the_hashed_requirements_on_a_pinned_base(tmp_path: Path, rows_file: Path) -> None:
    out = tmp_path / "out"
    adapter.convert(rows_file, out)
    text = (out / "environment" / "Dockerfile").read_text()
    assert "@sha256:" in adapter.BASE_IMAGE
    assert f"FROM {adapter.BASE_IMAGE}\n" in text
    assert "--require-hashes" in text
    token = re.search(r"^RUN .*\becho (\S+) \| base64 -d", text, re.MULTILINE).group(1)
    assert gzip.decompress(base64.b64decode(token)) == adapter.REQUIREMENTS.read_bytes()


def _response(*texts: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="m",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id=f"m{index}",
                role="assistant",
                type="message",
                status="completed",
                content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
            )
            for index, text in enumerate(texts)
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _flip_first_role(entry: dict) -> str:
    """The reference answer with the first person's role swapped: knights_knaves gives it partial credit."""
    metadata = entry["metadata"]
    terms = metadata["knight_knave_terms"]
    name = metadata["names"][0]
    current, other = (terms["a_knight"], terms["a_knave"])[:: 1 if metadata["solution"][0] else -1]
    flipped = entry["answer"].replace(f"{name} is {current}", f"{name} is {other}", 1)
    assert flipped != entry["answer"]
    return flipped


# Each case: the assistant messages, and whether the native reward is 0, 1 or partial credit.
GRADER_CASES = [
    (("<answer>{answer}</answer>",), "full"),
    (("thinking", "\\boxed{{\\text{{{answer}}}}}"), "full"),
    (("<answer>nobody</answer>",), "zero"),
    (("{answer}",), "full"),
    (("",), "zero"),
    (("<answer>{flipped}</answer>",), "partial"),
]


def _case_response(entry: dict, texts: tuple[str, ...]) -> NeMoGymResponse:
    return _response(*(text.format(answer=entry["answer"], flipped=_flip_first_role(entry)) for text in texts))


async def _native_reward(entry: dict, response: NeMoGymResponse) -> float:
    server = ReasoningGymResourcesServer(
        config=ReasoningGymResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient),
    )
    native = await server.verify(
        ReasoningGymVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "q"}]),
            response=response,
            question=entry["question"],
            answer=entry["answer"],
            metadata=entry["metadata"],
        )
    )
    return native.reward


@pytest.mark.asyncio
@pytest.mark.parametrize("texts, kind", GRADER_CASES)
async def test_copied_grader_scores_like_the_server(tmp_path, entries, monkeypatch, texts, kind) -> None:
    dataset = tmp_path / "dataset"
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps(_row(entries[0])) + "\n")
    adapter.convert(rows, dataset)
    monkeypatch.syspath_prepend(str(dataset / "tests"))
    for name in ("verifier", "reasoning_gym_grader"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    verifier = importlib.import_module("verifier")
    assert Path(verifier.__file__).parent == dataset / "tests"
    entry = entries[0]
    response = _case_response(entry, texts)
    native = await _native_reward(entry, response)
    task_data = json.loads((dataset / "tasks.jsonl").read_text())["task_data"]
    assert verifier.verify(task_data, response.model_dump(mode="json"))["reward"] == native
    if kind == "partial":
        assert 0 < native < 1


@pytest.mark.asyncio
async def test_grader_cases_cover_zero_full_and_partial_credit(entries) -> None:
    # The equalities above prove nothing if both graders return the same constant for every case.
    rewards = {kind: [] for _, kind in GRADER_CASES}
    for texts, kind in GRADER_CASES:
        rewards[kind].append(await _native_reward(entries[0], _case_response(entries[0], texts)))
    assert set(rewards["zero"]) == {0.0}
    assert set(rewards["full"]) == {1.0}
    assert all(0 < reward < 1 for reward in rewards["partial"])
