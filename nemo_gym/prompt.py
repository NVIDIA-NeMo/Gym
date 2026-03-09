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
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    user: str
    system: Optional[str] = None


class Prompt:
    def __init__(self, config: PromptConfig):
        self.config = config

    def fill(self, input_dict: dict) -> List[dict]:
        messages = []

        if self.config.system is not None:
            messages.append({"role": "system", "content": self.config.system.format(**input_dict)})

        user_content = self.config.user.format(**input_dict)
        messages.append({"role": "user", "content": user_content})

        return messages


@lru_cache(maxsize=64)
def load_prompt(config_path: str) -> Prompt:
    with open(config_path) as f:
        data = yaml.safe_load(f)
    config = PromptConfig.model_validate(data)
    return Prompt(config)


def materialize_prompts(input_jsonl: str, prompt_config: str, output_jsonl: str) -> None:
    """Apply a prompt template to raw JSONL data, producing materialized JSONL for RL training.

    Reads each row from input_jsonl, applies the prompt template to build
    responses_create_params.input, and writes the result to output_jsonl.

    Args:
        input_jsonl: Path to raw JSONL (no responses_create_params.input).
        prompt_config: Path to prompt YAML file.
        output_jsonl: Path to write materialized JSONL (with responses_create_params.input).
    """
    prompt = load_prompt(prompt_config)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_jsonl) as fin, open(output_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            if row.get("responses_create_params", {}).get("input"):
                raise ValueError(
                    "Row already has responses_create_params.input. "
                    "materialize_prompts expects raw data without pre-baked prompts."
                )
            row.setdefault("responses_create_params", {})
            row["responses_create_params"]["input"] = prompt.fill(row)
            row["prompt_config_used"] = prompt_config
            fout.write(json.dumps(row) + "\n")
            count += 1

    print(f"Materialized {count} rows with prompt '{prompt_config}' -> {output_path}")


def materialize_prompts_cli() -> None:  # pragma: no cover
    """CLI entry point for ng_materialize_prompts."""
    from nemo_gym.config_types import BaseNeMoGymCLIConfig
    from nemo_gym.global_config import get_global_config_dict

    class MaterializePromptsConfig(BaseNeMoGymCLIConfig):
        input_jsonl_fpath: str = Field(description="Path to raw JSONL data (no responses_create_params.input).")
        prompt_config: str = Field(description="Path to prompt YAML file.")
        output_jsonl_fpath: str = Field(description="Path to write materialized JSONL.")

    config = MaterializePromptsConfig.model_validate(get_global_config_dict())
    materialize_prompts(config.input_jsonl_fpath, config.prompt_config, config.output_jsonl_fpath)
