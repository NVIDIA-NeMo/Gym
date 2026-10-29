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
"""Build the 392 BaxBench tasks (28 scenarios x 14 frameworks) from upstream at the pinned commit.

Prompts are upstream's own (``Scenario.build_prompt``) with the leaderboard's main setting: the OpenAPI
spec and no security reminder. The HF dataset LogicStar/BaxBench is older than the pinned commit (it
still has Alpine-era package lines and older specs), so prompts are built from the repository instead.
"""

import argparse
from pathlib import Path

import orjson

from resources_servers.baxbench.upstream import UPSTREAM_SHA, fetch_upstream, image_slug, import_upstream


DATA_DIR = Path(__file__).parent / "data"


def build_rows(src_dir: Path) -> list[dict]:
    bx_env, bx_scenarios = import_upstream(src_dir)
    rows = []
    for environment in bx_env.all_envs:
        for scenario in bx_scenarios.all_scenarios:
            rows.append(
                {
                    "question": scenario.build_prompt(environment, "openapi", "none", agent=False),
                    "verifier_metadata": {
                        "task_id": f"{environment.id}/{scenario.id}",
                        "env_id": environment.id,
                        "image_slug": image_slug(environment.id),
                        "scenario_id": scenario.id,
                        "language": environment.language,
                        "framework": environment.framework,
                        "is_multi_file": environment.is_multi_file,
                        "code_filename": environment.code_filename,
                        "functional_tests": [test.__name__ for test in scenario.functional_tests],
                        "security_tests": [test.__name__ for test in scenario.security_tests],
                        "upstream_sha": UPSTREAM_SHA,
                    },
                }
            )
    return rows


def prepare(output_path: Path = DATA_DIR / "baxbench_benchmark.jsonl") -> Path:
    rows = build_rows(fetch_upstream(DATA_DIR / "upstream"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for row in rows:
            f.write(orjson.dumps(row) + b"\n")
    print(f"Wrote {len(rows)} BaxBench tasks to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-path", type=Path, default=DATA_DIR / "baxbench_benchmark.jsonl")
    prepare(parser.parse_args().output_path)
