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
"""The official harness's run: the one run clock and the pin check. The games themselves are on the official game
server (the resources server's, the authority); the worker plays them through it.

One run clock, as in the Kaggle kit agent (its ``run_clock: shared``): the origin is taken just before the first
game's scorecard is opened, and every game's play window ends at origin + ``play_seconds``. With ``run_clock: per_game``
the agent gives each game its own window and does not use this clock.
"""

import datetime as dt
import time
from pathlib import Path
from typing import Any

from responses_api_agents.arc_agi_3_agent.harness_runtime import GameStartError, game_files_sha256


class OfficialRun:
    """The run's directory, clock and pin check."""

    def __init__(self, environments_dir: Path, run_dir: Path, play_seconds: float):
        self.environments_dir = environments_dir
        self.run_dir = run_dir
        self.play_seconds = float(play_seconds)
        self.run_config_sha256: str | None = None
        self.closed = False
        self._start_time: dt.datetime | None = None
        self._start_monotonic: float | None = None

    @property
    def start_time_utc(self) -> str | None:
        """The run clock's origin as a UTC ISO time (what the verifier measures the server's action times from)."""
        return self._start_time.isoformat() if self._start_time else None

    def start_clock(self) -> None:
        """Start the run clock once, before the first game's scorecard is opened."""
        if self._start_time is None:
            self._start_time = dt.datetime.now(dt.timezone.utc)
            self._start_monotonic = time.monotonic()

    def seconds_left(self) -> float:
        if self._start_monotonic is None:
            return self.play_seconds
        return self._start_monotonic + self.play_seconds - time.monotonic()

    def check_game_pins(self, verifier_metadata: dict[str, Any]) -> str:
        """The row's full game id, after checking its version and file sha256 against the files to be played."""
        game = verifier_metadata.get("game_id")
        version = verifier_metadata.get("version")
        pinned = verifier_metadata.get("files_sha256")
        full_game_id = verifier_metadata.get("full_game_id")
        if not (game and version and pinned and full_game_id):
            raise GameStartError(f"{game}: the dataset row lacks full_game_id/version/files_sha256 pins")
        game_dir = self.environments_dir / str(game) / str(version)
        if not game_dir.is_dir():
            raise GameStartError(f"{game}: pinned version {version} is not under {self.environments_dir}")
        if game_files_sha256(game_dir) != pinned:
            raise GameStartError(f"{game}: files under {game_dir} differ from the dataset's sha256 pins")
        if full_game_id != f"{game}-{version}":
            raise GameStartError(f"{game}: full_game_id {full_game_id!r} does not match version {version!r}")
        return str(full_game_id)
