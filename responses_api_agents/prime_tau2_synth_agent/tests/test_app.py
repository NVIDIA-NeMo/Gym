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
import json
import sys
from pathlib import Path

from environments.prime_tau2_synth.prepare import main as prepare_main
from responses_api_agents.verifiers_agent.app import VerifiersAgentRunRequest


EXAMPLE_JSONL = Path(__file__).resolve().parents[3] / "environments" / "prime_tau2_synth" / "data" / "example.jsonl"


def test_prepare_writes_runnable_rows_that_match_the_example(tmp_path, monkeypatch) -> None:
    output = tmp_path / "rows.jsonl"
    monkeypatch.setattr(sys, "argv", ["prepare.py", "--domain", "library", "--limit", "2", "--output", str(output)])
    prepare_main()

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [json.loads(line) for line in EXAMPLE_JSONL.read_text().splitlines()[:2]]
    for row in rows:
        request = VerifiersAgentRunRequest.model_validate(row)
        assert request.responses_create_params.input[0].role == "system"
        assert {"id", "user_scenario"} <= request.info.keys()
