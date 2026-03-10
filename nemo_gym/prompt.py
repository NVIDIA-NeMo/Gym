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
"""Prompt configuration: YAML-based prompt templates applied at rollout time.

Prompt templates are mutually exclusive with pre-baked
``responses_create_params.input`` values. This separation enables prompt
sweeps without re-preparing data.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from nemo_gym import PARENT_DIR
from nemo_gym.config_types import BaseNeMoGymCLIConfig
from nemo_gym.global_config import GlobalConfigDictParserConfig, get_global_config_dict


class PromptConfig(BaseModel):
    """Schema for a prompt YAML file. ``user`` is required, ``system`` is optional."""

    user: str
    system: Optional[str] = None


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to the Gym root (PARENT_DIR), consistent with config_paths resolution."""
    p = Path(path)
    if not p.is_absolute():
        p = PARENT_DIR / p
    return p


@lru_cache(maxsize=64)
def load_prompt_config(path: str) -> PromptConfig:
    """Load and validate a YAML prompt config file.

    Relative paths are resolved against the Gym root directory (``PARENT_DIR``),
    consistent with how ``config_paths`` and other Gym paths are resolved.

    Returns a ``PromptConfig`` with required ``user`` and optional ``system`` fields.
    Each value is a string template with ``{placeholder}`` syntax.
    Results are cached so the same file is only parsed once.
    """
    resolved = _resolve_path(path)
    with open(resolved) as f:
        data = yaml.safe_load(f)
    return PromptConfig.model_validate(data)


def fill_prompt(prompt_config: PromptConfig, row: dict) -> List[Dict[str, str]]:
    """Apply a prompt template to a data row, producing message dicts.

    Placeholders (``{field_name}``) are filled from the row's top-level
    fields. Literal braces must be doubled (``{{`` / ``}}``).
    """
    try:
        messages = []
        if prompt_config.system is not None:
            messages.append({"role": "system", "content": prompt_config.system.format_map(row)})
        messages.append({"role": "user", "content": prompt_config.user.format_map(row)})
        return messages
    except KeyError as e:
        raise KeyError(
            f"Prompt template references field {e} but the data row only has fields: {list(row.keys())}"
        ) from None


def validate_prompt_compatibility(rows: List[dict], prompt_config: PromptConfig) -> None:
    """Validate that no rows have pre-baked responses_create_params.input when a prompt_config is provided.

    Collects all violating row indices and reports them in a single error.
    """
    conflicting_indices = [i for i, row in enumerate(rows) if row.get("responses_create_params", {}).get("input")]
    if conflicting_indices:
        raise ValueError(
            "Some rows have responses_create_params.input but prompt_config is also specified. "
            f"These are mutually exclusive — use one or the other. Violating rows: {conflicting_indices}"
        )


def apply_prompt_to_row(row: dict, prompt_config: PromptConfig) -> dict:
    """Apply prompt_config to a row, building responses_create_params.input.

    Other fields in responses_create_params (tools, metadata, temperature,
    max_output_tokens) are preserved. Returns a new dict (does not mutate the original).
    """
    messages = fill_prompt(prompt_config, row)
    row = row.copy()
    rcp = row.get("responses_create_params", {})
    if isinstance(rcp, dict):
        rcp = rcp.copy()
    else:
        rcp = {}
    rcp["input"] = messages
    row["responses_create_params"] = rcp
    return row


def materialize_prompts(input_jsonl: str, prompt_config: str, output_jsonl: str) -> None:
    """Apply a prompt template to raw JSONL data, producing materialized JSONL.

    Reads each row from ``input_jsonl``, validates that no row has pre-baked
    ``responses_create_params.input``, applies the prompt template, and writes
    the result to ``output_jsonl``.

    Args:
        input_jsonl: Path to raw JSONL (no responses_create_params.input).
        prompt_config: Path to prompt YAML file.
        output_jsonl: Path to write materialized JSONL (with responses_create_params.input).
    """
    prompt_cfg = load_prompt_config(prompt_config)
    resolved_prompt_path = str(_resolve_path(prompt_config))
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_jsonl) as f_in:
        rows = [json.loads(line) for line in f_in]

    validate_prompt_compatibility(rows, prompt_cfg)

    with open(output_path, "w") as f_out:
        for row in rows:
            materialized = apply_prompt_to_row(row, prompt_cfg)
            materialized["prompt_config_used"] = resolved_prompt_path
            f_out.write(json.dumps(materialized) + "\n")

    print(f"Materialized {len(rows)} rows to {output_path}")


class MaterializePromptsConfig(BaseNeMoGymCLIConfig):
    """
    Apply a prompt template to raw JSONL data, producing materialized JSONL
    with baked ``responses_create_params.input`` for RL training.

    Examples:

    See example prompt configs in ``nemo_gym/prompt_configs/examples/``.

    ```bash
    ng_materialize_prompts \\
        +input_jsonl_fpath=data/my_dataset.jsonl \\
        +prompt_config=nemo_gym/prompt_configs/examples/math_cot.yaml \\
        +output_jsonl_fpath=my_dataset_materialized.jsonl
    ```
    """

    input_jsonl_fpath: str = Field(description="Raw JSONL data (no responses_create_params.input).")
    prompt_config: str = Field(description="Path to prompt YAML file to apply.")
    output_jsonl_fpath: str = Field(description="Output path for materialized JSONL with baked prompts.")


def materialize_prompts_cli() -> None:  # pragma: no cover
    """CLI entry point for ng_materialize_prompts."""
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    config = MaterializePromptsConfig.model_validate(global_config_dict)
    materialize_prompts(config.input_jsonl_fpath, config.prompt_config, config.output_jsonl_fpath)
