# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from environments.ui_generation import prepare as prepare_module

from ..app import UiGenerationVerifyRequest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ANSWER = "```html\n<!doctype html>\n<p>hello</p>\n```"


def write_tasks(path: Path, tasks: list[dict]) -> Path:
    path.write_text("".join(json.dumps(task) + "\n" for task in tasks) + "\n")
    return path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_prompt_tasks_become_rows_with_the_answer_kept_from_the_model(tmp_path: Path) -> None:
    source = write_tasks(
        tmp_path / "tasks.jsonl",
        [{"id": "a", "prompt": "Make a page that says hello.", "reference_answer": ANSWER, "category": "html"}],
    )

    (row,) = read_rows(prepare_module.prepare(source, tmp_path / "out.jsonl"))

    assert row == {
        "task_id": "a",
        "responses_create_params": {"input": [{"role": "user", "content": "Make a page that says hello."}]},
        "verifier_metadata": {"reference_answer": ANSWER, "category": "html"},
    }


def test_message_tasks_take_the_final_assistant_message_as_the_reference(tmp_path: Path) -> None:
    messages = [
        {"role": "system", "content": "You write web pages."},
        {"role": "user", "content": "Make a page that says hello."},
        {"role": "assistant", "content": ANSWER},
    ]
    source = write_tasks(tmp_path / "tasks.jsonl", [{"messages": messages, "meta": {"stack": "html"}}])

    (row,) = read_rows(prepare_module.prepare(source, tmp_path / "out.jsonl", category_field="meta.stack"))

    assert row["responses_create_params"]["input"] == messages[:2]
    assert row["verifier_metadata"] == {"reference_answer": ANSWER, "category": "html"}


def test_an_explicit_reference_keeps_every_message_in_the_prompt(tmp_path: Path) -> None:
    messages = [{"role": "user", "content": "Start a page."}, {"role": "assistant", "content": "Which kind?"}]
    messages.append({"role": "user", "content": "A hello page."})
    source = write_tasks(tmp_path / "tasks.jsonl", [{"messages": messages, "reference_answer": ANSWER}])

    (row,) = read_rows(prepare_module.prepare(source, tmp_path / "out.jsonl"))

    assert row["responses_create_params"]["input"] == messages


def test_a_final_assistant_message_and_a_reference_answer_are_refused_together(tmp_path: Path) -> None:
    # The final assistant message would otherwise stay in the policy's prompt next to the reference.
    messages = [{"role": "user", "content": "Make a page."}, {"role": "assistant", "content": ANSWER}]
    source = write_tasks(tmp_path / "tasks.jsonl", [{"messages": messages, "reference_answer": ANSWER}])
    with pytest.raises(prepare_module.TaskFormatError, match="conflicts with 'reference_answer'"):
        prepare_module.prepare(source, tmp_path / "out.jsonl")


def test_limit_keeps_the_first_tasks(tmp_path: Path) -> None:
    tasks = [{"id": str(i), "prompt": f"Page {i}", "reference_answer": ANSWER} for i in range(4)]
    output = prepare_module.prepare(write_tasks(tmp_path / "tasks.jsonl", tasks), tmp_path / "out.jsonl", limit=2)
    assert [row["task_id"] for row in read_rows(output)] == ["0", "1"]


@pytest.mark.parametrize(
    "task,error",
    [
        ({"reference_answer": ANSWER}, "exactly one of"),
        ({"prompt": "p", "messages": [], "reference_answer": ANSWER}, "exactly one of"),
        ({"prompt": "p"}, "missing reference answer"),
        ({"prompt": "p", "reference_answer": "  "}, "missing reference answer"),
        ({"prompt": "", "reference_answer": ANSWER}, "nonempty string content"),
        ({"messages": "hi", "reference_answer": ANSWER}, "list of objects"),
        ({"messages": [{"role": "tool", "content": "x"}], "reference_answer": ANSWER}, "roles"),
        ({"messages": [{"role": "system", "content": "x"}], "reference_answer": ANSWER}, "user message"),
        ({"messages": [{"role": "user", "content": "only a prompt"}]}, "missing reference answer"),
        ({"prompt": "p", "reference_answer": ANSWER, "category": 3}, "nonempty string"),
    ],
)
def test_unusable_tasks_are_refused(tmp_path: Path, task: dict, error: str) -> None:
    source = write_tasks(tmp_path / "tasks.jsonl", [task])
    with pytest.raises(prepare_module.TaskFormatError, match=error):
        prepare_module.prepare(source, tmp_path / "out.jsonl")


@pytest.mark.parametrize("content,error", [("[1, 2]\n", "JSON object"), ("{not json\n", "invalid JSON")])
def test_lines_that_are_not_task_objects_are_refused(tmp_path: Path, content: str, error: str) -> None:
    source = tmp_path / "tasks.jsonl"
    source.write_text(content)
    with pytest.raises(prepare_module.TaskFormatError, match=error):
        prepare_module.prepare(source, tmp_path / "out.jsonl")


def test_duplicate_ids_are_refused(tmp_path: Path) -> None:
    tasks = [{"id": "same", "prompt": "p", "reference_answer": ANSWER}] * 2
    with pytest.raises(prepare_module.TaskFormatError, match="duplicate id"):
        prepare_module.prepare(write_tasks(tmp_path / "tasks.jsonl", tasks), tmp_path / "out.jsonl")


def test_the_committed_examples_are_rows_prepare_writes(tmp_path: Path) -> None:
    committed = REPOSITORY_ROOT / "resources_servers/ui_generation/data/example.jsonl"
    rows = read_rows(committed)
    tasks = [
        {"id": row["task_id"], "prompt": row["responses_create_params"]["input"][0]["content"]}
        | row["verifier_metadata"]
        for row in rows
    ]
    converted = prepare_module.prepare(write_tasks(tmp_path / "tasks.jsonl", tasks), tmp_path / "example.jsonl")
    assert converted.read_text() == committed.read_text()

    assert len(rows) == 5
    for row in rows:
        request = UiGenerationVerifyRequest.model_validate(row | {"response": _empty_response()})
        assert request.verifier_metadata.reference_answer.startswith("```")
    assert {row["verifier_metadata"]["category"] for row in rows} == {"nextjs", "react", "html"}


def _empty_response() -> dict:
    return {
        "id": "r",
        "created_at": 0,
        "model": "m",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }
