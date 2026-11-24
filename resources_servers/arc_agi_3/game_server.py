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
"""The official ``arc_agi`` game server: the authority for every played game.

The resources server starts one per evaluation, as a child process:

    python -m resources_servers.arc_agi_3.game_server --environments-dir D --out O --port P --scorecard-timeout-min M

It is ``arc_agi``'s own Flask server (``Arcade(OFFLINE).listen_and_serve``) on the pinned game files, with RESET keeping
the level (``ONLY_RESET_LEVELS=true``, the kit's semantics). Harnesses play through it over HTTP, one scorecard per
rollout. It writes:

- ``O/cards/<card_id>.json``: every scorecard when it is closed (by its harness, by the verifier, or idle). A closed
  card is gone from the server, so this file is the authority's record of the game.
- ``O/game_server.json``: its URL and process id (also logged at start in ``O/game_server.log``).
- ``O/recordings/<card_id>/<game>-<guid>.jsonl``: one line per response with the server's UTC ``timestamp``, the action,
  ``levels_completed`` and ``state`` (no frames). The verifier computes RHAE@T from it.

One change to arc_agi's server: its game-info routes (``/api/games``, ``/api/games/<id>``) also publish each game's
human ``baseline_actions``, which arc_agi leaves out (the competition setting). The kit's taaf records them (the kit's
``rhae.py`` scores ``benchmark.json`` with them) and the official harness sets its per-level action budgets from them
(without them it falls back to 10 actions per game). Scoring and game logic stay arc_agi's.
"""

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

from nemo_gym import PARENT_DIR


LOG = logging.getLogger(__name__)
# Keys of a closed card's JSON that are not written to disk.
PRIVATE_CARD_KEYS = ("api_key",)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def card_json(card: Any) -> dict[str, Any]:
    """A closed ``EnvironmentScorecard`` as the JSON the close endpoint returns, without its API key."""
    data = json.loads(card.model_dump_json())
    for key in PRIVATE_CARD_KEYS:
        data.pop(key, None)
    return data


def game_info(env: Any) -> dict[str, Any]:
    """A game's public info as arc_agi's server gives it, plus its human ``baseline_actions``."""
    return env.model_dump(mode="json", exclude={"private_tags", "level_tags"})


def publish_baselines(arcade: Any, app: Any) -> None:
    """Replace the two game-info views of arc_agi's Flask app (endpoints ``games`` and ``game_by_id``)."""
    from flask import jsonify

    def games() -> Any:
        return jsonify([game_info(env) for env in arcade.available_environments]), 200

    def game_by_id(game_id: str) -> Any:
        for env in arcade.available_environments:
            if env.game_id == game_id or env.game_id.startswith(f"{game_id}-"):
                return jsonify(game_info(env)), 200
        return jsonify({"error": "SERVER_ERROR", "message": f"game {game_id} not found"}), 404

    app.view_functions["games"] = games
    app.view_functions["game_by_id"] = game_by_id


def serve(environments_dir: str, out: Path, host: str, port: int, scorecard_timeout_min: int) -> None:
    import arc_agi

    os.environ["ONLY_RESET_LEVELS"] = "true"
    (out / "cards").mkdir(parents=True, exist_ok=True)

    def on_close(card: Any) -> None:
        write_json_atomic(out / "cards" / f"{card.card_id}.json", card_json(card))

    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=environments_dir,
        recordings_dir=str(out / "recordings"),
        logger=logging.getLogger("arc_agi_3.game_server"),
    )
    print(f"ARC-AGI-3 game server: http://{host}:{port} ({environments_dir}; files in {out})", flush=True)
    arcade.listen_and_serve(
        host=host,
        port=port,
        save_all_recordings=True,
        include_frame_data=False,
        scorecard_timeout=scorecard_timeout_min,
        on_scorecard_close=on_close,
        extra_api_routes=publish_baselines,
    )


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class GameServer:
    """The game server child process and the files it writes."""

    def __init__(
        self,
        environments_dir: str,
        out_dir: Path,
        port: int | None = None,
        scorecard_timeout_min: int = 240,
        python_path: list[str] | None = None,
    ):
        self.environments_dir = environments_dir
        # Put first on the server's PYTHONPATH: runtimes that some games import (e.g. param25's bundled arcgames).
        self.python_path = list(python_path or [])
        self.out_dir = out_dir
        self.port = port
        self.scorecard_timeout_min = scorecard_timeout_min
        self.host = "127.0.0.1"
        self._process: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, expected_games: int, timeout_s: float = 120.0) -> None:
        """Start the server and wait until it serves all ``expected_games`` games."""
        if self.running:
            return
        self.port = self.port or free_port()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.out_dir / "game_server.log", "ab")
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "resources_servers.arc_agi_3.game_server",
                "--environments-dir",
                self.environments_dir,
                "--out",
                str(self.out_dir),
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--scorecard-timeout-min",
                str(self.scorecard_timeout_min),
            ],
            cwd=str(PARENT_DIR),
            env=dict(
                os.environ,
                ONLY_RESET_LEVELS="true",
                PYTHONPATH=os.pathsep.join(self.python_path + [p for p in [os.environ.get("PYTHONPATH")] if p]),
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.close()
        write_json_atomic(self.out_dir / "game_server.json", {"url": self.url, "pid": self._process.pid})
        LOG.info("ARC-AGI-3 game server starting at %s (files in %s)", self.url, self.out_dir)
        deadline = time.monotonic() + timeout_s
        while True:
            if self._process.poll() is not None:
                raise RuntimeError(f"the game server exited ({self._process.returncode}); see {self.out_dir}")
            try:
                games = requests.get(f"{self.url}/api/games", timeout=2).json()
                if len(games) == expected_games:
                    return
                self.stop()
                raise RuntimeError(f"the game server serves {len(games)} games, expected {expected_games}")
            except requests.RequestException:
                if time.monotonic() > deadline:
                    self.stop()
                    raise TimeoutError(f"the game server did not answer in {timeout_s} s; see {self.out_dir}")
                time.sleep(0.2)

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process = None

    def closed_card(self, card_id: str) -> dict[str, Any] | None:
        path = self.out_dir / "cards" / f"{card_id}.json"
        return json.loads(path.read_text()) if path.is_file() else None

    def close_card(self, card_id: str, api_key: str) -> dict[str, Any]:
        """The closed card: the server's own copy if it was closed already, else close it now (the verifier's
        close)."""
        card = self.closed_card(card_id)
        if card is not None:
            return card
        response = requests.post(
            f"{self.url}/api/scorecard/close", json={"card_id": card_id}, headers={"X-API-Key": api_key}, timeout=30
        )
        if response.status_code == 404:
            card = self.closed_card(card_id)  # closed between the two reads
            if card is not None:
                return card
            raise ValueError(f"scorecard {card_id} is unknown to the game server")
        response.raise_for_status()
        card = self.closed_card(card_id)
        if card is None:
            raise RuntimeError(f"the game server closed scorecard {card_id} but wrote no card file")
        return card

    def recording(self, card_id: str, game_id: str, guid: str) -> list[dict[str, Any]]:
        """The server's events for one game run: [{"timestamp": UTC ISO, "data": {...}}, ...]."""
        path = self.out_dir / "recordings" / card_id / f"{game_id}-{guid}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"the game server has no recording {path}")
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--environments-dir", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--scorecard-timeout-min", type=int, default=240)
    args = parser.parse_args(argv)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    serve(args.environments_dir, args.out, args.host, args.port, args.scorecard_timeout_min)


if __name__ == "__main__":
    main()
