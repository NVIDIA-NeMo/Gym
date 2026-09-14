# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Generate a Gym JSONL for the prime general-agent verifiers environment.

The local solver does ``task_dir = Path(state["info"]["task_dir"])`` so the
``info`` dict must live at the top level of each row (not inside
``verifier_metadata``). This script writes rows in that lifted shape.

The ``info.task_dir`` paths point into the installed ``general-agent`` package's
bundled tasks/ tree, so the package version must match the corpus version that
produced the dataset.

Requires the agent venv (or a venv with verifiers + general-agent installed):
    pip install 'verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git@main' \\
        --extra-index-url https://hub.primeintellect.ai/primeintellect/simple/
    pip install general-agent==0.1.4

Usage:
    python environments/prime_general_agent/prepare.py --output environments/prime_general_agent/data/train.jsonl
"""

import argparse
import json
from pathlib import Path


VF_ENV_ID = "general-agent-solver-local"
AGENT_NAME = "prime_general_agent"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", help="verifiers env split (typically 'train' or 'test')")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        import verifiers as vf
    except ImportError:
        raise ImportError("Install verifiers + general-agent first; see module docstring.")

    env = vf.load_environment(VF_ENV_ID)
    dataset = env.get_dataset(args.split)
    rows = list(dataset)
    if args.limit is not None:
        rows = rows[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for idx, row in enumerate(rows):
            # The local solver reads info.task_dir at the row top level.
            info = row.get("info")
            if isinstance(info, str):
                info = json.loads(info)
            info = info or {}

            prompt_input = []
            if "prompt" in row and row["prompt"]:
                prompt_input.append({"role": "user", "content": row["prompt"]})
            elif "question" in row and row["question"]:
                prompt_input.append({"role": "user", "content": row["question"]})

            out = {
                "task_idx": idx,
                "vf_env_id": VF_ENV_ID,
                "agent_ref": {"type": "responses_api_agents", "name": AGENT_NAME},
                "responses_create_params": {"input": prompt_input},
                "info": info,
                **{k: v for k, v in row.items() if k not in {"responses_create_params", "info"}},
            }
            f.write(json.dumps(out) + "\n")

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
