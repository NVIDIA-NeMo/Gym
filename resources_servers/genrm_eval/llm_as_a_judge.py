# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrap GenRM comparisons in the full judging instructions for a regular chat model."""

import json
from pathlib import Path

import yaml

from resources_servers.genrm_eval.prepare import atomic_output


def _conversation_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(part, dict)
        and part.get("type") in {"input_text", "output_text"}
        and isinstance(part.get("text"), str)
        for part in content
    ):
        return "".join(part["text"] for part in content)
    raise ValueError("LLM preparation requires text conversation messages")


def prepare_llm_input(*, source: Path, output: Path) -> Path:
    if output.resolve() == source.resolve():
        raise ValueError("LLM preparation must not overwrite the GenRM input")
    template = yaml.safe_load((Path(__file__).parent / "configs/llm_as_a_judge_prompt.yaml").read_text())[
        "judge_prompt"
    ]
    with source.open() as stream, atomic_output(output) as destination:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            params = row["responses_create_params"]
            turns = params["input"]
            if not turns:
                raise ValueError("LLM preparation requires text conversation messages")
            candidates = params.pop("metadata")
            # Keep labels and order unchanged; only the model-facing request is reformatted.
            params["input"] = [
                {
                    "role": "user",
                    "content": template.format(
                        conversation="\n".join(
                            f"{turn['role'].capitalize()}: {_conversation_text(turn.get('content'))}" for turn in turns
                        ),
                        response_1=candidates["response_1"],
                        response_2=candidates["response_2"],
                    ),
                }
            ]
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output
