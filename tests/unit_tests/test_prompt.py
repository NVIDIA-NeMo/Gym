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

import pytest
import yaml

from nemo_gym import PARENT_DIR
from nemo_gym.prompt import (
    PromptConfig,
    _resolve_path,
    apply_prompt_to_row,
    fill_prompt,
    load_prompt_config,
    materialize_prompts,
    validate_prompt_compatibility,
)


@pytest.fixture(autouse=True)
def _clear_prompt_cache():
    """Clear the lru_cache on load_prompt_config between tests."""
    load_prompt_config.cache_clear()
    yield
    load_prompt_config.cache_clear()


class TestLoadPromptConfig:
    def test_load_valid_config(self, tmp_path):
        config_path = tmp_path / "prompt.yaml"
        config_path.write_text(yaml.dump({"system": "You are a math tutor.", "user": "{question}"}))
        result = load_prompt_config(str(config_path))
        assert result.system == "You are a math tutor."
        assert result.user == "{question}"

    def test_load_user_only(self, tmp_path):
        config_path = tmp_path / "prompt.yaml"
        config_path.write_text(yaml.dump({"user": "Solve: {question}"}))
        result = load_prompt_config(str(config_path))
        assert result.system is None
        assert result.user == "Solve: {question}"

    def test_missing_user_key_raises(self, tmp_path):
        config_path = tmp_path / "prompt.yaml"
        config_path.write_text(yaml.dump({"system": "Hello"}))
        with pytest.raises(Exception):
            load_prompt_config(str(config_path))

    def test_non_mapping_yaml_raises(self, tmp_path):
        config_path = tmp_path / "prompt.yaml"
        config_path.write_text("- item1\n- item2\n")
        with pytest.raises(Exception):
            load_prompt_config(str(config_path))

    def test_caching(self, tmp_path):
        config_path = tmp_path / "prompt.yaml"
        config_path.write_text(yaml.dump({"user": "{question}"}))
        result1 = load_prompt_config(str(config_path))
        result2 = load_prompt_config(str(config_path))
        assert result1 is result2


class TestResolvePath:
    def test_absolute_path(self, tmp_path):
        p = tmp_path / "prompt.yaml"
        p.write_text("")
        assert _resolve_path(str(p)) == p

    def test_relative_path_resolves_to_parent_dir(self, tmp_path):
        # Create a file relative to PARENT_DIR to test resolution
        test_file = PARENT_DIR / "test_resolve_path_temp.yaml"
        test_file.write_text("user: '{q}'")
        try:
            resolved = _resolve_path("test_resolve_path_temp.yaml")
            assert resolved.exists()
        finally:
            test_file.unlink()


class TestFillPrompt:
    def test_system_and_user(self):
        config = PromptConfig(system="You solve math.", user="{question}")
        row = {"question": "What is 2+2?"}
        messages = fill_prompt(config, row)
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You solve math."}
        assert messages[1] == {"role": "user", "content": "What is 2+2?"}

    def test_user_only(self):
        config = PromptConfig(user="Solve: {question}")
        row = {"question": "What is 2+2?"}
        messages = fill_prompt(config, row)
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Solve: What is 2+2?"}

    def test_literal_braces(self):
        config = PromptConfig(user="Put answer in \\boxed{{{question}}}")
        row = {"question": "42"}
        messages = fill_prompt(config, row)
        assert messages[0]["content"] == "Put answer in \\boxed{42}"

    def test_multiple_placeholders(self):
        config = PromptConfig(user="Question: {question}\nHint: {hint}")
        row = {"question": "What is pi?", "hint": "It's irrational"}
        messages = fill_prompt(config, row)
        assert "What is pi?" in messages[0]["content"]
        assert "It's irrational" in messages[0]["content"]

    def test_missing_field_raises_with_context(self):
        config = PromptConfig(user="{nonexistent_field}")
        row = {"question": "test", "answer": "42"}
        with pytest.raises(KeyError, match="nonexistent_field.*question.*answer"):
            fill_prompt(config, row)


class TestValidatePromptCompatibility:
    def _config(self):
        return PromptConfig(user="{question}")

    def test_raw_data_passes(self):
        rows = [{"question": "test", "responses_create_params": {"tools": []}}]
        validate_prompt_compatibility(rows, self._config())

    def test_conflicting_raises(self):
        rows = [{"responses_create_params": {"input": [{"role": "user", "content": "test"}]}}]
        with pytest.raises(ValueError, match="mutually exclusive"):
            validate_prompt_compatibility(rows, self._config())

    def test_reports_all_conflicting_indices(self):
        rows = [
            {"responses_create_params": {"input": [{"role": "user", "content": "baked"}]}},
            {"question": "raw"},
            {"responses_create_params": {"input": [{"role": "user", "content": "also baked"}]}},
        ]
        with pytest.raises(ValueError, match=r"mutually exclusive.*Violating rows: \[0, 2\]"):
            validate_prompt_compatibility(rows, self._config())


class TestApplyPromptToRow:
    def _config(self):
        return PromptConfig(user="{question}")

    def test_preserves_other_fields(self):
        row = {
            "question": "What is 2+2?",
            "expected_answer": "4",
            "responses_create_params": {"tools": [{"type": "function", "name": "python"}], "temperature": 0.7},
        }
        result = apply_prompt_to_row(row, self._config())
        assert result["expected_answer"] == "4"
        assert result["responses_create_params"]["tools"] == [{"type": "function", "name": "python"}]
        assert result["responses_create_params"]["temperature"] == 0.7
        assert result["responses_create_params"]["input"] == [{"role": "user", "content": "What is 2+2?"}]

    def test_does_not_mutate_original(self):
        row = {"question": "test", "responses_create_params": {"temperature": 0.5}}
        result = apply_prompt_to_row(row, self._config())
        assert "input" not in row["responses_create_params"]
        assert "input" in result["responses_create_params"]

    def test_creates_responses_create_params_if_missing(self):
        row = {"question": "test"}
        result = apply_prompt_to_row(row, self._config())
        assert result["responses_create_params"]["input"] == [{"role": "user", "content": "test"}]

    def test_non_dict_responses_create_params_replaced(self):
        row = {"question": "test", "responses_create_params": "not_a_dict"}
        result = apply_prompt_to_row(row, self._config())
        assert result["responses_create_params"] == {"input": [{"role": "user", "content": "test"}]}


class TestMaterializePrompts:
    def test_end_to_end(self, tmp_path):
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"system": "You are a math tutor.", "user": "Solve: {problem}"}))

        input_path = tmp_path / "input.jsonl"
        rows = [
            {"problem": "2+2", "verifier_metadata": {"answer": "4"}},
            {"problem": "3*5", "verifier_metadata": {"answer": "15"}},
        ]
        with open(input_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        output_path = tmp_path / "output.jsonl"

        materialize_prompts(str(input_path), str(prompt_path), str(output_path))

        with open(output_path) as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        assert results[0]["responses_create_params"]["input"] == [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Solve: 2+2"},
        ]
        assert results[0]["verifier_metadata"] == {"answer": "4"}
        assert results[0]["prompt_config_used"] == str(prompt_path)
        assert results[1]["responses_create_params"]["input"][1]["content"] == "Solve: 3*5"

    def test_creates_output_dirs(self, tmp_path):
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"user": "{q}"}))

        input_path = tmp_path / "input.jsonl"
        with open(input_path, "w") as f:
            f.write(json.dumps({"q": "hello"}) + "\n")

        output_path = tmp_path / "nested" / "dir" / "output.jsonl"

        materialize_prompts(str(input_path), str(prompt_path), str(output_path))

        assert output_path.exists()
        with open(output_path) as f:
            results = [json.loads(line) for line in f]
        assert len(results) == 1

    def test_rejects_prebaked_input(self, tmp_path):
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"user": "{question}"}))

        input_path = tmp_path / "input.jsonl"
        with open(input_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "question": "test",
                        "responses_create_params": {"input": [{"role": "user", "content": "already baked"}]},
                    }
                )
                + "\n"
            )

        output_path = tmp_path / "output.jsonl"

        with pytest.raises(ValueError, match=r"mutually exclusive.*Violating rows: \[0\]"):
            materialize_prompts(str(input_path), str(prompt_path), str(output_path))

    def test_rejects_mixed_rows_with_indices(self, tmp_path):
        """A dataset where row 0 is fine but row 2 has pre-baked input should fail upfront listing violating rows."""
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"user": "{question}"}))

        input_path = tmp_path / "input.jsonl"
        with open(input_path, "w") as f:
            f.write(json.dumps({"question": "good row 0"}) + "\n")
            f.write(json.dumps({"question": "good row 1"}) + "\n")
            f.write(
                json.dumps(
                    {
                        "question": "bad row 2",
                        "responses_create_params": {"input": [{"role": "user", "content": "pre-baked"}]},
                    }
                )
                + "\n"
            )

        output_path = tmp_path / "output.jsonl"

        with pytest.raises(ValueError, match=r"mutually exclusive.*Violating rows: \[2\]"):
            materialize_prompts(str(input_path), str(prompt_path), str(output_path))
        # No partial output file should be written
        assert not output_path.exists()
