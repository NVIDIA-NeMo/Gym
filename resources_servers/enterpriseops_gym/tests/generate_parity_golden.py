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
"""Generate verifier parity goldens by running the fixture cases through the ORIGINAL
EnterpriseOps-Gym engine. Run this inside the EOG checkout with its venv, e.g.:

    cd /path/to/enterpriseops-gym
    .venv/bin/python /path/to/gym/resources_servers/enterpriseops_gym/tests/generate_parity_golden.py

It imports ``benchmark.verifier.VerifierEngine`` from the EOG repo (must be on sys.path /
be the CWD) and writes fixtures/verifier_parity_golden.json next to this script.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def main() -> None:
    from benchmark.verifier import VerifierEngine  # the ORIGINAL EOG engine

    engine = VerifierEngine(MagicMock(), MagicMock())

    cases = json.loads((FIXTURES_DIR / "verifier_parity_cases.json").read_text())

    golden = {
        "extraction": [engine._extract_value_from_sql_result(case) for case in cases["extraction"]],
        "comparison": [
            engine._compare_values(case["actual"], case["expected"], case["comparison_type"])
            for case in cases["comparison"]
        ],
    }

    output_path = FIXTURES_DIR / "verifier_parity_golden.json"
    output_path.write_text(json.dumps(golden, indent=2))
    print(f"Wrote goldens for {len(golden['extraction'])} extraction and {len(golden['comparison'])} comparison cases")
    print(f"-> {output_path}")


if __name__ == "__main__":
    main()
