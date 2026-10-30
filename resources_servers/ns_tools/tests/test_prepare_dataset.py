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
import json
import subprocess
import sys
from pathlib import Path


SERVER_DIR = Path(__file__).parents[1]
SHIPPED_COMPMATH = SERVER_DIR / "data" / "compmath_test.jsonl"


def run_readme_command(source: Path, output: Path, prompt_config: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SERVER_DIR / "prepare_dataset.py"),
            "--input",
            str(source),
            "--output",
            str(output),
            "--prompt_config",
            prompt_config,
            "--tools",
            "nemo_skills.mcp.servers.python_tool::DirectPythonTool",
            "--verifier_type",
            "math_with_judge",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )


def write_source_rows(path: Path, shipped: list[dict]) -> None:
    """Write the comp-math-24-25 source rows that the shipped rows were prepared from."""
    fields = ("id", "expected_answer", "subset_for_metrics", "reference_solution")
    with open(path, "w") as f:
        for row in shipped:
            f.write(json.dumps({"problem": row["question"], **{k: row[k] for k in fields}}) + "\n")


def test_readme_command_reproduces_shipped_compmath_rows(tmp_path: Path) -> None:
    with open(SHIPPED_COMPMATH) as f:
        shipped = [json.loads(next(f)) for _ in range(3)]
    source = tmp_path / "test.txt"
    write_source_rows(source, shipped)
    output = tmp_path / "compmath_test.jsonl"

    result = run_readme_command(source, output, "generic/math")

    assert result.returncode == 0, result.stderr
    with open(output) as f:
        prepared = [json.loads(line) for line in f]
    # The shipped rows describe a 120 s tool timeout, while the pinned DirectPythonTool and configs/ns_tools.yaml use
    # 10 s, so tool descriptions are left out of the comparison.
    for row in prepared + shipped:
        for tool in row["responses_create_params"]["tools"]:
            tool.pop("description")
    assert prepared == shipped


def test_missing_prompt_config_fails_before_writing_output(tmp_path: Path) -> None:
    source = tmp_path / "test.txt"
    source.write_text(json.dumps({"id": 0, "problem": "What is 1+1?", "expected_answer": "2"}) + "\n")
    output = tmp_path / "out.jsonl"

    result = run_readme_command(source, output, "generic/no_such_prompt")

    assert result.returncode != 0
    assert "Prompt config" in result.stderr and "was not found" in result.stderr
    assert not output.exists()


def test_prompt_config_yaml_path_is_used_as_given(tmp_path: Path) -> None:
    prompt = tmp_path / "custom.yaml"
    prompt.write_text('system: "Answer briefly."\nuser: "Q: {problem}"\n')
    source = tmp_path / "test.txt"
    source.write_text(json.dumps({"id": 0, "problem": "What is 1+1?", "expected_answer": "2"}) + "\n")
    output = tmp_path / "out.jsonl"

    result = run_readme_command(source, output, str(prompt))

    assert result.returncode == 0, result.stderr
    with open(output) as f:
        assert json.loads(f.readline())["responses_create_params"]["input"] == [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "Q: What is 1+1?"},
        ]


def test_system_template_is_filled_like_the_user_template(tmp_path: Path) -> None:
    prompt = tmp_path / "custom.yaml"
    prompt.write_text("system: 'Put the answer in \\boxed{{}}.'\nuser: 'Q: {problem}'\n")
    source = tmp_path / "test.txt"
    source.write_text(json.dumps({"id": 0, "problem": "What is 1+1?", "expected_answer": "2"}) + "\n")
    output = tmp_path / "out.jsonl"

    result = run_readme_command(source, output, str(prompt))

    assert result.returncode == 0, result.stderr
    with open(output) as f:
        assert json.loads(f.readline())["responses_create_params"]["input"][0] == {
            "role": "system",
            "content": "Put the answer in \\boxed{}.",
        }


def test_template_error_fails_before_the_existing_output_is_truncated(tmp_path: Path) -> None:
    prompt = tmp_path / "custom.yaml"
    prompt.write_text("user: 'Q: {question}'\n")
    source = tmp_path / "test.txt"
    source.write_text(json.dumps({"id": 0, "problem": "What is 1+1?", "expected_answer": "2"}) + "\n")
    output = tmp_path / "out.jsonl"
    output.write_text('{"id": "kept"}\n')

    result = run_readme_command(source, output, str(prompt))

    assert result.returncode != 0
    assert "KeyError: 'question'" in result.stderr
    assert output.read_text() == '{"id": "kept"}\n'
