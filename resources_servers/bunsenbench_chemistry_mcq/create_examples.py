# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate synthetic smoke-test example.jsonl for BunsenBench Chemistry MCQ.

These rows are not redistributed benchmark-source questions. They are built from
the same materialize + prompt pipeline used for the full benchmark.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

from benchmarks.bunsenbench_chemistry_mcq.materialize import materialize_row
from benchmarks.bunsenbench_chemistry_mcq.upstream import UPSTREAM_CONFIG_METADATA
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config
from nemo_gym.server_utils import ServerClient
from resources_servers.bunsenbench_chemistry_mcq.app import (
    BunsenChemResourcesServer,
    BunsenChemResourcesServerConfig,
    BunsenChemVerifyRequest,
)


SERVER_DIR = Path(__file__).resolve().parent
DATA_DIR = SERVER_DIR / "data"
EXAMPLE_FPATH = DATA_DIR / "example.jsonl"
ROLLOUTS_FPATH = DATA_DIR / "example_rollouts.jsonl"
PROMPT_CONFIG_FPATH = (
    SERVER_DIR.parent.parent / "benchmarks" / "bunsenbench_chemistry_mcq" / "prompts" / "default.yaml"
)
AGENT_REF = {"type": "responses_api_agents", "name": "bunsenbench_chemistry_mcq_simple_agent"}

COMMITTED_FIELDS = ("responses_create_params", "options", "expected_answer", "uuid", "metadata", "agent_ref")

SYNTHETIC_CONFIG_METADATA = {
    **UPSTREAM_CONFIG_METADATA,
}

SYNTHETIC_RECONSTITUTED_ROWS: list[dict] = [
    {
        **SYNTHETIC_CONFIG_METADATA,
        "bunsen_id": "bunsen:example:1",
        "source": "example",
        "source_dataset": "synthetic",
        "source_config": "example",
        "source_split": "example",
        "source_revision": "example",
        "source_record_id": "example-1",
        "source_row_index": 0,
        "source_record_sha256": "example-hash-1",
        "canonical_problem_sha256": "example-problem-1",
        "bct_field": "general",
        "bct_subfield": "acids_bases",
        "question": "Which species is the conjugate base of hydronium?",
        "choices": ["H2O", "OH-", "H3O+", "O2"],
        "answer": "H2O",
        "answer_index": 0,
    },
    {
        **SYNTHETIC_CONFIG_METADATA,
        "bunsen_id": "bunsen:example:2",
        "source": "example",
        "source_dataset": "synthetic",
        "source_config": "example",
        "source_split": "example",
        "source_revision": "example",
        "source_record_id": "example-2",
        "source_row_index": 1,
        "source_record_sha256": "example-hash-2",
        "canonical_problem_sha256": "example-problem-2",
        "bct_field": "organic",
        "bct_subfield": "structure",
        "question": "Which option names a common organic functional group?",
        "choices": ["Ketone", "Kelvin", "Nucleus", "Photon"],
        "answer": "Ketone",
        "answer_index": 0,
    },
    {
        **SYNTHETIC_CONFIG_METADATA,
        "bunsen_id": "bunsen:example:3",
        "source": "example",
        "source_dataset": "synthetic",
        "source_config": "example",
        "source_split": "example",
        "source_revision": "example",
        "source_record_id": "example-3",
        "source_row_index": 2,
        "source_record_sha256": "example-hash-3",
        "canonical_problem_sha256": "example-problem-3",
        "bct_field": "general",
        "bct_subfield": "nomenclature",
        "question": "Which formula represents carbon dioxide?",
        "choices": ["CO", "CO2", "CH4", "C2H6"],
        "answer": "CO2",
        "answer_index": 1,
    },
    {
        **SYNTHETIC_CONFIG_METADATA,
        "bunsen_id": "bunsen:example:4",
        "source": "example",
        "source_dataset": "synthetic",
        "source_config": "example",
        "source_split": "example",
        "source_revision": "example",
        "source_record_id": "example-4",
        "source_row_index": 3,
        "source_record_sha256": "example-hash-4",
        "canonical_problem_sha256": "example-problem-4",
        "bct_field": "analytical",
        "bct_subfield": "chromatography",
        "question": "Which technique separates compounds by interaction with a stationary phase?",
        "choices": ["Chromatography", "Calorimetry", "Titration", "Crystallography"],
        "answer": "Chromatography",
        "answer_index": 0,
    },
    {
        **SYNTHETIC_CONFIG_METADATA,
        "bunsen_id": "bunsen:example:5",
        "source": "example",
        "source_dataset": "synthetic",
        "source_config": "example",
        "source_split": "example",
        "source_revision": "example",
        "source_record_id": "example-5",
        "source_row_index": 4,
        "source_record_sha256": "example-hash-5",
        "canonical_problem_sha256": "example-problem-5",
        "bct_field": "physical",
        "bct_subfield": "thermodynamics",
        "question": "Which quantity is measured in kelvin?",
        "choices": ["Temperature", "Pressure", "Mass", "Charge"],
        "answer": "Temperature",
        "answer_index": 0,
    },
]


def build_example_rows() -> list[dict]:
    prompt_cfg = load_prompt_config(str(PROMPT_CONFIG_FPATH))
    rows: list[dict] = []
    for reconstituted in SYNTHETIC_RECONSTITUTED_ROWS:
        materialized = materialize_row(reconstituted)
        prompted = apply_prompt_to_row(materialized, prompt_cfg)
        row = {key: prompted[key] for key in COMMITTED_FIELDS if key in prompted}
        row["agent_ref"] = AGENT_REF
        rows.append(row)
    return rows


def write_example_jsonl(output_path: Path = EXAMPLE_FPATH) -> Path:
    rows = build_example_rows()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {output_path}")
    return output_path


def _response(text: str, response_id: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id=response_id,
        created_at=0.0,
        model="example-model",
        object="response",
        output=[
            {
                "id": f"msg_{response_id}",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _choice_text_for_letter(row: dict, letter: str) -> str:
    for option in row["options"]:
        if letter in option:
            return str(option[letter])
    raise KeyError(f"Letter {letter} not found in options for {row.get('uuid')}")


async def build_example_rollouts(example_rows: list[dict]) -> list[dict]:
    server = BunsenChemResourcesServer(
        config=BunsenChemResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient),
    )
    rollouts: list[dict] = []
    for task_idx, row in enumerate(example_rows):
        letter = row["expected_answer"]
        choice_text = _choice_text_for_letter(row, letter)
        request = BunsenChemVerifyRequest(
            responses_create_params=row["responses_create_params"],
            response=_response(f"<choice>{choice_text}</choice>", f"resp_bunsen_example_{task_idx}"),
            options=row["options"],
            expected_answer=letter,
            uuid=row["uuid"],
            metadata=row.get("metadata"),
        )
        verified = await server.verify(request)
        rollout = row.copy()
        rollout.update(
            {
                "_ng_task_index": task_idx,
                "_ng_rollout_index": 0,
                "response": request.response.model_dump(),
                "reward": verified.reward,
                "extracted_answer": verified.extracted_answer,
                "no_answer": verified.no_answer,
                "error_mode": verified.error_mode,
                "source": verified.source,
                "bct_field": verified.bct_field,
                "bct_subfield": verified.bct_subfield,
            }
        )
        rollouts.append(rollout)
    return rollouts


def write_example_rollouts(
    example_path: Path = EXAMPLE_FPATH,
    output_path: Path = ROLLOUTS_FPATH,
) -> Path:
    example_rows = [json.loads(line) for line in example_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rollouts = asyncio.run(build_example_rollouts(example_rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rollout in rollouts:
            f.write(json.dumps(rollout, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rollouts)} rollouts to {output_path}")
    return output_path


def main() -> None:
    write_example_jsonl()
    write_example_rollouts()


if __name__ == "__main__":
    main()
