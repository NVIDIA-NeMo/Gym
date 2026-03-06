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
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel


class FewShotExamplesConfig(BaseModel):
    prefix: str = ""
    template: str = ""
    suffix: str = ""
    examples: List[Dict[str, str]] = []


class PromptConfig(BaseModel):
    user: str
    system: Optional[str] = None
    few_shot_examples: FewShotExamplesConfig = FewShotExamplesConfig()


class Prompt:
    def __init__(self, config: PromptConfig):
        self.config = config

    def fill(self, input_dict: dict) -> List[dict]:
        messages = []

        if self.config.system is not None:
            messages.append({"role": "system", "content": self.config.system.format(**input_dict)})

        # Build few-shot examples string, available as {examples} in the user template.
        # Matches NeMo Skills: <prefix><filled_template1><filled_template2>...<suffix>
        fs = self.config.few_shot_examples
        if fs.examples:
            filled = "".join(fs.template.format(**ex) for ex in fs.examples)
            examples = f"{fs.prefix}{filled}{fs.suffix}"
        else:
            examples = ""

        user_content = self.config.user.format(examples=examples, **input_dict)
        messages.append({"role": "user", "content": user_content})

        return messages


def load_prompt(config_path: str) -> Prompt:
    with open(config_path) as f:
        data = yaml.safe_load(f)
    config = PromptConfig.model_validate(data)
    return Prompt(config)
