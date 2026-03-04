# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

from nemo_gym.prompt import FewShotExamplesConfig, Prompt, PromptConfig, load_prompt
from nemo_gym.rollout_collection import RolloutCollectionConfig


# TODO: Eventually we want to add more tests to ensure that the rollout collection flow does not break
class TestRolloutCollection:
    def test_sanity(self) -> None:
        RolloutCollectionConfig(
            agent_name="",
            input_jsonl_fpath="",
            output_jsonl_fpath="",
        )

    def test_prompt_config_field(self) -> None:
        config = RolloutCollectionConfig(
            agent_name="agent",
            input_jsonl_fpath="input.jsonl",
            output_jsonl_fpath="output.jsonl",
            prompt_config="prompt_configs/math.yaml",
        )
        assert config.prompt_config == "prompt_configs/math.yaml"

    def test_prompt_config_default_none(self) -> None:
        config = RolloutCollectionConfig(
            agent_name="agent",
            input_jsonl_fpath="input.jsonl",
            output_jsonl_fpath="output.jsonl",
        )
        assert config.prompt_config is None


class TestPrompt:
    def test_fill_system_and_user(self) -> None:
        config = PromptConfig(system="You are a math tutor.", user="Solve: {problem}")
        prompt = Prompt(config)
        messages = prompt.fill({"problem": "2+2"})
        assert messages == [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Solve: 2+2"},
        ]

    def test_fill_user_only(self) -> None:
        config = PromptConfig(user="{problem}")
        prompt = Prompt(config)
        messages = prompt.fill({"problem": "What is 5*3?"})
        assert messages == [{"role": "user", "content": "What is 5*3?"}]

    def test_fill_with_few_shot_examples(self) -> None:
        config = PromptConfig(
            user="Solve: {problem}",
            few_shot_examples=FewShotExamplesConfig(
                prefix="Here are some examples:\n",
                template="Q: {question}\nA: {answer}\n",
                suffix="\nNow solve the following:\n",
                examples=[
                    {"question": "1+1", "answer": "2"},
                    {"question": "2+3", "answer": "5"},
                ],
            ),
        )
        prompt = Prompt(config)
        messages = prompt.fill({"problem": "3+4"})
        assert len(messages) == 1
        content = messages[0]["content"]
        assert content.startswith("Here are some examples:\n")
        assert "Q: 1+1\nA: 2\n" in content
        assert "Q: 2+3\nA: 5\n" in content
        assert content.endswith("Solve: 3+4")

    def test_fill_missing_field_raises_key_error(self) -> None:
        config = PromptConfig(user="{problem}")
        prompt = Prompt(config)
        with pytest.raises(KeyError):
            prompt.fill({"other_field": "value"})

    def test_load_prompt_from_yaml(self, tmp_path) -> None:
        yaml_content = 'system: "Be helpful."\nuser: "{question}"'
        yaml_file = tmp_path / "test_prompt.yaml"
        yaml_file.write_text(yaml_content)

        prompt = load_prompt(str(yaml_file))
        messages = prompt.fill({"question": "Hello?"})
        assert messages == [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello?"},
        ]
