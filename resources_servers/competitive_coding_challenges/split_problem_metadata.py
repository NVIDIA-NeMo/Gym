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

"""Write one metadata file per problem for sandboxed verification.

Usage, from the repository root:
``python -m resources_servers.competitive_coding_challenges.split_problem_metadata TEST_FILE OUTPUT_DIR``

Reads any metadata file the evaluator accepts (``test_file``) and writes
``OUTPUT_DIR/<competition>/<problem>.jsonl``, the layout ``sandbox_config.problem_data_dir``
expects. Bake OUTPUT_DIR into the sandbox image or mount it read-only at that path.
"""

import argparse
import json
from pathlib import Path

from resources_servers.competitive_coding_challenges.ccc_eval import _load_metadata_file
from resources_servers.competitive_coding_challenges.sandbox_evaluator import problem_file


def split_problem_metadata(test_file: str, output_dir: Path) -> int:
    """Write each problem's normalized metadata to its own file; return the number written."""
    metadata_by_competition, _ = _load_metadata_file(test_file)
    written = 0
    for competition_id, problems in metadata_by_competition.items():
        for problem_id, problem_metadata in problems.items():
            path = Path(problem_file(str(output_dir), competition_id or None, problem_id))
            path.parent.mkdir(parents=True, exist_ok=True)
            entry = {"competition_id": competition_id, "metadata": {problem_id: problem_metadata}}
            path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
            written += 1
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test_file")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    count = split_problem_metadata(args.test_file, args.output_dir)
    print(f"Wrote {count} problem files under {args.output_dir}")
