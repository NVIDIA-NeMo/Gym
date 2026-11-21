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
"""The official harness's run: the run clock and the pin check."""

import hashlib
import json
import time
from pathlib import Path

import pytest

from responses_api_agents.arc_agi_3_agent.harness_runtime import GameStartError
from responses_api_agents.arc_agi_3_official_agent.official_game import OfficialRun


def write_game(root: Path, game: str = "ls20", version: str = "v1") -> dict:
    game_dir = root / game / version
    game_dir.mkdir(parents=True)
    (game_dir / f"{game}.py").write_text("# game\n")
    (game_dir / "metadata.json").write_text(json.dumps({"game_id": f"{game}-{version}", "baseline_actions": [3]}))
    sha = {f"{game}.py": hashlib.sha256(b"# game\n").hexdigest()}
    return {"game_id": game, "full_game_id": f"{game}-{version}", "version": version, "files_sha256": sha}


def test_the_run_clock_starts_once_in_utc(tmp_path: Path) -> None:
    run = OfficialRun(tmp_path, tmp_path / "run", play_seconds=0.2)
    assert run.start_time_utc is None and run.seconds_left() == 0.2
    run.start_clock()
    first = run.start_time_utc
    assert first.endswith("+00:00")
    time.sleep(0.25)
    run.start_clock()  # later games do not move the origin
    assert run.start_time_utc == first and run.seconds_left() < 0


def test_pins(tmp_path: Path) -> None:
    meta = write_game(tmp_path)
    run = OfficialRun(tmp_path, tmp_path / "run", play_seconds=60)
    assert run.check_game_pins(meta) == "ls20-v1"
    for change, match in (
        ({"files_sha256": {"ls20.py": "0" * 64}}, "sha256"),
        ({"version": "v9"}, "not under"),
        ({"full_game_id": "ls20-other"}, "does not match"),
        ({"full_game_id": None}, "lacks"),
    ):
        with pytest.raises(GameStartError, match=match):
            run.check_game_pins({**meta, **change})
