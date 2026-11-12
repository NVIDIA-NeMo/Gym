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
"""The judge's message and verdict parsing, task preparation, and the task fields shared by data and server."""

import json
import sys
from pathlib import Path

import pytest
import yaml

from environments.ui_generation_agentic.prepare import (
    DEFAULT_TEMPLATE,
    STACKS,
    RowError,
    _render,
    convert,
)
from environments.ui_generation_agentic.prepare import (
    main as prepare_main,
)
from nemo_gym.core import NeMoGymChatCompletion
from nemo_gym.verification import JudgeError
from resources_servers.ui_generation_agentic.app import UiGenTask, UiGenVerifyRequest
from resources_servers.ui_generation_agentic.judge import SCHEMA_VERSION, judge_case, parse_judge_completion
from resources_servers.ui_generation_agentic.task_data import TaskData, UiGenTaskFields


REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_CONFIG = REPO_ROOT / "resources_servers/ui_generation_agentic/configs/ui_generation_agentic.yaml"
VERDICT = {
    "schema_version": SCHEMA_VERSION,
    "prompt_adherence": {"score": 2, "rationale": "r"},
    "quality": {"score": 3, "rationale": "r"},
    "critical_issues": [],
    "uncertainty": [],
    "execution_verified": False,
}


def _completion(content: str, finish_reason: str = "stop") -> NeMoGymChatCompletion:
    return NeMoGymChatCompletion.model_validate(
        {
            "id": "judge-reply",
            "object": "chat.completion",
            "created": 0,
            "model": "judge",
            "choices": [
                {"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}
            ],
        }
    )


# ---------------------------------------------------------------------------------------------- judge


@pytest.mark.parametrize(
    "content",
    [
        json.dumps(VERDICT),
        f"<think>weighing it</think>\n{json.dumps(VERDICT)}",
        f"```json\n{json.dumps(VERDICT)}\n```",
        f"```JSON\n{json.dumps(VERDICT)}\n```\nThat is my verdict.",
        f"Here it is:\n```\n{json.dumps(VERDICT, indent=2)}\n```",
    ],
)
def test_a_valid_verdict_is_parsed_in_any_wrapping(content: str) -> None:
    verdict, _ = parse_judge_completion(_completion(content))

    assert verdict == VERDICT


def test_the_leading_think_block_is_the_judge_reasoning() -> None:
    _, reasoning = parse_judge_completion(_completion(f"<think>weighing it</think>{json.dumps(VERDICT)}"))

    assert reasoning == "weighing it"


@pytest.mark.parametrize("score", [True, 5, -1, 2.0, "3"])
def test_a_score_outside_the_integer_scale_is_refused(score: object) -> None:
    verdict = {**VERDICT, "quality": {"score": score, "rationale": "r"}}

    with pytest.raises(JudgeError, match="invalid_schema"):
        parse_judge_completion(_completion(json.dumps(verdict)))


@pytest.mark.parametrize(
    ("content", "finish_reason", "match"),
    [
        ("no verdict here", "stop", "invalid_json"),
        ("```json\n[1, 2]\n```", "stop", "invalid"),
        (json.dumps(VERDICT), "length", "finish_length"),
        (json.dumps({**VERDICT, "extra": 1}), "stop", "invalid_schema"),
    ],
)
def test_a_reply_without_a_valid_verdict_raises(content: str, finish_reason: str, match: str) -> None:
    with pytest.raises(JudgeError, match=match):
        parse_judge_completion(_completion(content, finish_reason))


def test_the_code_fence_is_longer_than_any_backtick_run_in_the_code() -> None:
    code = "const md = `a`;\nconst block = '```js\\nx\\n```';\nconst five = '`````';\n"

    message = json.loads(judge_case("prompt", "vite", "src/App.tsx", code, "tsx"))

    candidate = message["candidate_answer"]
    assert candidate.startswith("### src/App.tsx\n``````tsx\n")
    assert candidate.endswith("\n``````")
    assert candidate[len("### src/App.tsx\n``````tsx\n") : -len("\n``````")] == code


# ---------------------------------------------------------------------------------------------- prepare


def test_the_default_task_prompt_asks_every_harness_to_build_the_file_in_steps() -> None:
    row = {"id": "bakery", "stack": "ui-static", "prompt": "Build a landing page for a small bakery."}

    task = convert(row, 1, DEFAULT_TEMPLATE.read_text())

    (message,) = task["responses_create_params"]["input"]
    assert message["role"] == "user"
    assert message["content"].startswith("Build a landing page for a small bakery.")
    assert "Build that file in steps" in message["content"]
    assert "/usr/share/nginx/html/index.html" in message["content"]
    assert task["user_prompt"] == "Build a landing page for a small bakery."


def test_template_syntax_inside_a_prompt_stays_literal() -> None:
    prompt = "Show {{ count }} items and print {{ workdir }} literally."
    row = {"id": "vue-like", "stack": "ui-vite", "prompt": prompt}

    task = convert(row, 1, DEFAULT_TEMPLATE.read_text())

    content = task["responses_create_params"]["input"][0]["content"]
    assert content.startswith(prompt)
    assert task["user_prompt"] == prompt
    assert "`/app`" in content  # the template's own {{ workdir }} is still filled


@pytest.mark.parametrize("template", ["{{ user_prompt }} {{ customer }}", "{{ user_prompt }} {{ workdir"])
def test_an_unknown_or_unclosed_template_placeholder_raises(template: str) -> None:
    with pytest.raises(RowError, match="unfilled placeholder"):
        _render(template, {"user_prompt": "p", "workdir": "/app"})


def test_prepare_skips_invalid_rows_and_keeps_one_id_per_stack(tmp_path: Path, monkeypatch, capsys) -> None:
    rows = [
        {"id": "a", "stack": "ui-vite", "prompt": "Build {{ count }} cards."},
        {"id": "a", "stack": "ui-next", "prompt": "Same id, other stack."},
        {"id": "a", "stack": "ui-vite", "prompt": "Duplicate."},
        {"id": "b", "stack": "ui-unknown", "prompt": "Unknown stack."},
        {"id": "c", "stack": "ui-static"},
    ]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows) + "not json\n")
    output = tmp_path / "out" / "tasks.jsonl"
    monkeypatch.setattr(sys, "argv", ["prepare.py", "--input", str(source), "--output", str(output)])

    assert prepare_main() == 0

    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["task_id"] for row in written] == ["ui-vite-a", "ui-next-a"]
    manifest = json.loads(output.with_name("tasks.jsonl.manifest.json").read_text())
    assert manifest["rows"] == 2 and manifest["skipped_rows"] == 4
    assert "duplicate id" in capsys.readouterr().err


def test_prepare_stacks_match_the_server_stacks() -> None:
    config = yaml.safe_load(SERVER_CONFIG.read_text())
    stacks = config["ui_generation_agentic_resources_server"]["resources_servers"]["ui_generation_agentic"]["stacks"]

    assert set(STACKS) == set(stacks)
    for name, spec in STACKS.items():
        assert (spec["workdir"], spec["entry_path"]) == (stacks[name]["workdir"], stacks[name]["entry_path"]), name


# ---------------------------------------------------------------------------------------------- task fields


def test_the_row_schema_and_the_server_models_share_one_task_field_declaration() -> None:
    shared = UiGenTaskFields.model_fields
    for model in (TaskData, UiGenTask, UiGenVerifyRequest):
        assert issubclass(model, UiGenTaskFields)
        for name, declared in shared.items():
            assert model.model_fields[name].annotation == declared.annotation, (model.__name__, name)
            assert model.model_fields[name].is_required(), (model.__name__, name)


def test_a_prepared_row_is_valid_task_data_flat_and_materialized() -> None:
    from environments.ui_generation_agentic.prepare import materialize

    row = convert({"id": "x", "stack": "ui-next", "prompt": "Build a clock."}, 1, DEFAULT_TEMPLATE.read_text())
    task_data = materialize(row, "ui_generation_agentic:train")["task_input"]["task_data"]

    flat = TaskData.model_validate({k: v for k, v in row.items() if k != "responses_create_params"})
    assert TaskData.model_validate(task_data) == flat
    assert UiGenTask.model_validate(task_data).stack == "ui-next"
