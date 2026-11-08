# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Prepare the AA-aligned SciCode evaluation data for NeMo Gym.

Downloads the SciCode problems from HuggingFace and converts to Gym JSONL, one row per problem.
Each row carries the full sub_steps list so the agent can build the per-sub-step prompts and the
resources server can run each sub-step's test cases.

The benchmark uses only HuggingFace's test split, matching the current AA Intelligence Index setup.

This does NOT download test_data.h5 (the ~1 GB numeric test targets used for scoring). That file
must be staged manually at the path the resources server is configured to read.
"""

from pathlib import Path

from pydantic import ValidationError

from nemo_gym.config import HF_TOKEN_KEY_NAME, get_global_config_dict
from nemo_gym.server_runtime import AgentContext


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "scicode_benchmark.jsonl"

# Keep this aligned with the current AA SciCode problem set.
SPLITS = ["test"]
# The prompt material of a sub-step. Its test cases and reference code stay with the resources server.
AGENT_CONTEXT_STEP_KEYS = (
    "step_number",
    "step_description_prompt",
    "step_background",
    "function_header",
    "return_line",
)


def scicode_agent_context(row: dict) -> dict:
    """The row's ``agent_context``: what scicode_agent needs to build every sub-step's prompt in an agent session.

    An agent's ``/run`` reads the same fields from the row itself.
    """
    values = {
        "problem_id": row["problem_id"],
        "sub_steps": [{key: step[key] for key in AGENT_CONTEXT_STEP_KEYS} for step in row["sub_steps"]],
        "required_dependencies": row["required_dependencies"],
    }
    if row.get("prefilled_steps_code"):
        values["prefilled_steps_code"] = row["prefilled_steps_code"]
    return {"values": values}


def prepare() -> Path:
    """Download SciCode problems and convert to Gym JSONL format."""
    import json

    from datasets import load_dataset

    print("Downloading SciCode from HuggingFace...")
    hf_token = get_global_config_dict().get(HF_TOKEN_KEY_NAME)
    ds = load_dataset("SciCode1/SciCode", token=hf_token)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in SPLITS:
        for entry in ds[split]:
            row = {
                # The agent builds each sub-step's messages from sub_steps, so the row carries an
                # empty input; rollout collection still requires the key to be present.
                "responses_create_params": {"input": []},
                "problem_id": entry["problem_id"],
                "problem_name": entry["problem_name"],
                "required_dependencies": entry["required_dependencies"],
                "sub_steps": entry["sub_steps"],
                # problem_id is unique across both splits, so it doubles as the rollout uuid.
                "uuid": entry["problem_id"],
            }
            agent_context = scicode_agent_context(row)
            try:
                AgentContext.model_validate(agent_context)
            except ValidationError as error:
                # Its /run still works; an agent session refuses the row with 422 for lack of context.
                print(f"Problem {row['problem_id']} gets no agent_context: {error}")
            else:
                row["agent_context"] = agent_context
            rows.append(json.dumps(row) + "\n")

    with open(OUTPUT_FPATH, "w") as f:
        f.writelines(rows)

    print(f"Wrote {len(rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
