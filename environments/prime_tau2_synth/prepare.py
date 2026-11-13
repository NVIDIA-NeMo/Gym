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
"""Generate a Gym JSONL for the tau2-synth verifiers environment.

Loads the tau2-synth env for a given domain and writes one row per task.
The agent builds its env from the config's vf_env_args.domain, so --domain must match it.

Requires the agent venv (or a venv with verifiers + tau2-synth installed):
    pip install 'verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git@v0.1.14' \\
        --extra-index-url https://hub.primeintellect.ai/prime/simple/
    pip install 'tau2 @ git+https://github.com/eligotts/tau2-bench.git@4839dd6' tau2-synth==0.2.0

Usage:
    python environments/prime_tau2_synth/prepare.py --domain library --output environments/prime_tau2_synth/data/library_train.jsonl [--limit N]
"""

import argparse
import json
from pathlib import Path


VF_ENV_ID = "tau2-synth"
DOMAINS = [
    "library",
    "fitness_gym",
    "tech_support",
    "telecom",
    "cloud_incident_response",
    "daily_planner",
    "ev_charging_support",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="library", choices=DOMAINS)
    parser.add_argument("--limit", type=int, default=-1, help="number of tasks to write (-1 for all)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        import verifiers as vf
    except ImportError:
        raise ImportError("Install verifiers + tau2-synth first; see module docstring.")

    env = vf.load_environment(VF_ENV_ID, domain=args.domain)
    rows = list(env.get_dataset(n=args.limit))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for idx, row in enumerate(rows):
            # prompt is the system message with the domain policy. info is the tau2 Task as a JSON string.
            out = {
                "task_idx": idx,
                "vf_env_id": VF_ENV_ID,
                "responses_create_params": {"input": row["prompt"]},
                "info": json.loads(row["info"]),
                "example_id": row["example_id"],
            }
            f.write(json.dumps(out) + "\n")

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
