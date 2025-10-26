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
import random
import sys
from pathlib import Path
from typing import Optional

atropos_path = Path(__file__).parent.parent.parent / "atropos"
if str(atropos_path) not in sys.path:
    sys.path.insert(0, str(atropos_path))

from environments.letter_counting_environment import LetterCountingEnv, system_prompt

from resources_servers.atropos.app import AtroposResourcesServer

# TODO: fix hf datasets package error to use text passages too

class LetterCountingAtroposServer(AtroposResourcesServer[LetterCountingEnv]):
    system_prompt: Optional[str] = system_prompt
    _shared_env: Optional[LetterCountingEnv] = None

    async def env_factory(self, task_idx: int) -> LetterCountingEnv:
        if self._shared_env is None:
            env_config, server_configs = LetterCountingEnv.config_init()
            env_config.use_text_passages = False
            self._shared_env = LetterCountingEnv(
                config=env_config,
                server_configs=server_configs,
                slurm=False,
                testing=True,
            )
            await self._shared_env.setup()
        return self._shared_env

    async def get_initial_item(self, env: LetterCountingEnv, task_idx: int) -> str:
        return env.train_words[task_idx % len(env.train_words)]

    async def format_item_as_message(self, item: str) -> str:
        env = self._shared_env

        use_multiple = (
            env.config.max_letters_to_count > 1
            and random.random() < env.config.multi_letter_probability
        )
        
        if use_multiple:
            num_letters = random.randint(2, env.config.max_letters_to_count)
            target_letters = env._select_target_letters(item, num_letters)
        else:
            target_letters = env._select_target_letters(item, 1)

        is_text_passage = len(item) > 50 or " " in item or any(c in item for c in ".,!?;:")

        if len(target_letters) == 1:
            target_letter = target_letters[0]
            if is_text_passage:
                question_text = f"How many of the letter {target_letter} are in the following text: {item}?"
            else:
                question_text = f"How many of the letter {target_letter} are in the string {item}?"
            return f"{question_text}\n\nProvide your answer in the format: <answer>{{number}}</answer>"
        else:
            letters_str = ", ".join(f"'{letter}'" for letter in target_letters[:-1]) + f", and '{target_letters[-1]}'"
            if is_text_passage:
                question_text = f"Count the occurrences of the letters {letters_str} in the following text: {item}"
            else:
                question_text = f"Count the occurrences of the letters {letters_str} in the string {item}"
            example_json = "{" + ", ".join(f'"{letter}": 0' for letter in target_letters) + "}"
            return f"{question_text}\n\nProvide your answer as JSON in the format: <answer>{example_json}</answer>"

    async def score_response(
        self,
        env: LetterCountingEnv,
        item: str,
        response: str,
        messages: list,
    ) -> tuple[float, bool, Optional[dict]]:
        user_message = [msg for msg in messages if msg.get("role") == "user"][-1]["content"]

        if "Count the occurrences" in user_message:
            import re
            letter_pattern = r"'([a-zA-Z])'"
            target_letters = re.findall(letter_pattern, user_message)
            expected_format = "multi"
        else:
            import re
            match = re.search(r"How many ([a-zA-Z])s", user_message)
            if match:
                target_letters = [match.group(1)]
                expected_format = "single"
            else:
                return 0.0, True, {"correct": False, "error": "Could not parse question"}

        text_for_counting = env._prepare_text_for_counting(item)

        expected_counts = {}
        for letter in target_letters:
            expected_counts[letter] = text_for_counting.lower().count(letter.lower())

        if "</think>" in response.lower() and "<think>" not in response.lower():
            response = "<think>\n" + response

        model_answer = env._extract_answer(response, expected_format)

        if model_answer is None:
            correct = False
            reward = 0.0
        elif expected_format == "single":
            expected_count = expected_counts[target_letters[0]]
            correct = model_answer == expected_count
            reward = 1.0 if correct else 0.0
        else:
            correct = (
                set(model_answer.keys()) == set(target_letters)
                and all(model_answer.get(letter, -1) == expected_counts[letter] for letter in target_letters)
            )
            reward = 1.0 if correct else 0.0

        print("\n" + "="*80)
        print("Scoring results:")
        print("="*80)
        print(f"Word/Text: {item}")
        print(f"Target letters: {target_letters}")
        print(f"Question: {user_message[:200]}...")
        print(f"\nModel response:\n{response}")
        print(f"\nExpected counts: {expected_counts}")
        print(f"Model answer: {model_answer}")
        print(f"Correct: {correct}")
        print(f"Reward: {reward}")
        print("="*80 + "\n")

        return reward, True, {
            "correct": correct,
            "expected_counts": expected_counts,
            "model_answer": model_answer,
            "target_letters": target_letters,
        }


if __name__ == "__main__":
    LetterCountingAtroposServer.run_webserver()
