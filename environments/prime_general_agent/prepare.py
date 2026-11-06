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

``info.task_dir`` is written relative to the ``general-agent`` package's bundled
tasks/ tree and the agent resolves it against its own install, so rows work in
any venv. The package version must still match the corpus that produced the rows.

Run with the agent venv, which ``gym env start`` builds from the agent's requirements.txt:
    responses_api_agents/prime_general_agent/.venv/bin/python environments/prime_general_agent/prepare.py \\
        --output environments/prime_general_agent/data/train.jsonl
"""

import argparse
import json
from pathlib import Path


VF_ENV_ID = "general-agent-solver-local"
AGENT_NAME = "prime_general_agent"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=-1, help="number of tasks to write (default: all)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        import verifiers as vf
    except ImportError:
        raise ImportError("Install verifiers + general-agent first; see module docstring.")

    env = vf.load_environment(VF_ENV_ID)
    rows = list(env.get_dataset(n=args.limit))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for idx, row in enumerate(rows):
            # The local solver reads info.task_dir at the row top level.
            info = row["info"]
            info["task_dir"] = Path(info["task_dir"]).relative_to(env.taskset.tasks_dir).as_posix()
            out = {
                "task_idx": idx,
                "vf_env_id": VF_ENV_ID,
                "agent_ref": {"type": "responses_api_agents", "name": AGENT_NAME},
                # verifiers already formats the instruction as a message list.
                "responses_create_params": {"input": row["prompt"]},
                "info": info,
                "answer": row["answer"],
                "example_id": row["example_id"],
            }
            f.write(json.dumps(out) + "\n")

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
