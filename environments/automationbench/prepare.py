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
from __future__ import annotations

import argparse
import json
from pathlib import Path


VF_ENV_ID = "automationbench_env"
TOOLSET = "api"


def prepare(
    domains: list[str] | None = None,
    size: int = -1,
    max_turns: int = 50,
    out: Path | None = None,
) -> Path:
    """Write the taskset to JSONL and return the path written.

    `gym eval prepare` imports this module and calls `prepare(**prepare_script_args)`,
    requiring the returned path to equal the dataset's `jsonl_fpath`, so this is
    the entry point the benchmark config points at. `main` is the CLI wrapper.
    """
    try:
        import verifiers as vf  # noqa: F401
        from automationbench_env import load_environment
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "automationbench_env is not installed. Install this environment first:\n"
            "    uv pip install -e environments/automationbench"
        ) from exc

    env = load_environment(domains=domains, max_turns=max_turns, toolset=TOOLSET)
    dataset = env.dataset
    n = len(dataset) if size < 0 else min(size, len(dataset))

    out = out or Path(__file__).parent / "data" / f"automationbench-{n}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for i in range(n):
            row = dataset[i]
            prompt = row["prompt"]
            info = row.get("info", {})
            if isinstance(info, str):
                info = json.loads(info) if info else {}
            output_row = {
                "task_idx": i,
                "vf_env_id": VF_ENV_ID,
                "responses_create_params": {"input": prompt},
                "agent_ref": {
                    "type": "responses_api_agents",
                    "name": "verifiers_agent",
                },
                "question": prompt[-1]["content"] if prompt else "",
                "answer": row.get("answer", ""),
                "example_id": row["example_id"],
                "info": info,
            }
            f.write(json.dumps(output_row) + "\n")

    print(f"wrote {n} rows -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="*", default=None, help="subset of domains (default: all public domains)")
    parser.add_argument("--size", type=int, default=-1, help="number of tasks (-1 for the full taskset)")
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    prepare(domains=args.domains, size=args.size, max_turns=args.max_turns, out=args.out)


if __name__ == "__main__":
    main()
