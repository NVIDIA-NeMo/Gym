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
"""Build the ARC-AGI-3 public25 dataset (one row per game) from the environment files.

The game files are not committed. Get them one of two ways:

- ``--download-kaggle-competition DIR``: the ``environment_files/`` of the Kaggle competition
  ``arc-prize-2026-arc-agi-3`` (the files the Kaggle notebooks play offline). Needs a Kaggle token in
  ``~/.kaggle/access_token`` (or ``KAGGLE_API_TOKEN``) and the competition rules accepted.
- any existing ``environment_files`` directory (``--environments-dir``).

Every row records the game's version, its baselines and the sha256 of its files. The 25 versions are pinned in
``PUBLIC25_VERSIONS``; ``--check <jsonl>`` refuses a directory whose files differ from a dataset's sha256. The row's
``agent_context`` gives the agent session the same pinned game (``AGENT_GAME_KEYS`` of ``verifier_metadata``).

    python resources_servers/arc_agi_3/create_dataset.py --environments-dir <dir> \
        --output resources_servers/arc_agi_3/data/public25.jsonl
"""

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


COMPETITION = "arc-prize-2026-arc-agi-3"
# The pins an ARC-AGI-3 agent session receives in its agent_context ("game"): what it plays and checks before playing.
AGENT_GAME_KEYS = ("game_id", "full_game_id", "version", "files_sha256", "dispatch_index")
KAGGLE_API = "https://www.kaggle.com/api/v1"
# Kaggle answers 404 to urllib's default User-Agent (Python-urllib/x.y), even with a valid token.
KAGGLE_USER_AGENT = "nemo-gym-arc-agi-3"
# The 25 public games and their versions (the competition's environment_files, arc_agi 0.9.8).
PUBLIC25_VERSIONS = {
    "ar25": "0c556536",
    "bp35": "0a0ad940",
    "cd82": "fb555c5d",
    "cn04": "2fe56bfb",
    "dc22": "fdcac232",
    "ft09": "0d8bbf25",
    "g50t": "5849a774",
    "ka59": "38d34dbb",
    "lf52": "271a04aa",
    "lp85": "305b61c3",
    "ls20": "9607627b",
    "m0r0": "492f87ba",
    "r11l": "495a7899",
    "re86": "8af5384d",
    "s5i5": "18d95033",
    "sb26": "7fbdac44",
    "sc25": "635fd71a",
    "sk48": "d8078629",
    "sp80": "589a99af",
    "su15": "1944f8ab",
    "tn36": "ef4dde99",
    "tr87": "cd924810",
    "tu93": "0768757b",
    "vc33": "5430563c",
    "wa30": "ee6fef47",
}
# data/example.jsonl: Gym's five example rows, over the two games whose files are committed under
# data/environment_files (the other public games are not committed), so cn04 and ls20 alternate; each row is its own
# attempt (dispatch index). The CPU smoke test plays the first two.
EXAMPLE_GAMES = ("cn04", "ls20", "cn04", "ls20", "cn04")


def _kaggle_token() -> str:
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if not token:
        path = Path.home() / ".kaggle" / "access_token"
        if not path.is_file():
            raise FileNotFoundError("Set KAGGLE_API_TOKEN or write the token to ~/.kaggle/access_token")
        token = path.read_text().strip()
    return token


def _kaggle_get(url: str, token: str) -> bytes:
    """A GET of the Kaggle API with the token. The token goes to the Kaggle API only: it is an unredirected header, which
    urllib does not copy to a redirect's target (Kaggle answers a file download with a redirect to cloud storage, whose
    signed URL needs no credential)."""
    if not url.startswith(KAGGLE_API + "/"):
        raise ValueError(f"refusing to send the Kaggle token to {url}")
    request = urllib.request.Request(url, headers={"User-Agent": KAGGLE_USER_AGENT})
    request.add_unredirected_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def download_competition_environment_files(dest: Path) -> Path:
    """Download ``environment_files/`` of the competition into ``dest/environment_files``."""
    token = _kaggle_token()
    names: list[str] = []
    page_token = ""
    while True:
        url = f"{KAGGLE_API}/competitions/data/list/{COMPETITION}?pageSize=200"
        if page_token:
            url += f"&pageToken={page_token}"
        listing = json.loads(_kaggle_get(url, token))
        names += [f["name"] for f in listing.get("files", []) if f["name"].startswith("environment_files/")]
        page_token = listing.get("nextPageToken") or ""
        if not page_token:
            break
    if not names:
        raise RuntimeError(f"Kaggle competition {COMPETITION} lists no environment_files/")
    for name in names:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        query = urllib.request.quote(name, safe="")
        target.write_bytes(_kaggle_get(f"{KAGGLE_API}/competitions/data/download/{COMPETITION}/{query}", token))
    return dest / "environment_files"


def _game_dirs(environments_dir: Path) -> dict[str, dict[str, Path]]:
    """``{game: {version: directory}}``; a directory can hold several versions of a game."""
    dirs: dict[str, dict[str, Path]] = {}
    for meta_path in sorted(environments_dir.glob("*/*/metadata.json")):
        dirs.setdefault(meta_path.parent.parent.name, {})[meta_path.parent.name] = meta_path.parent
    if not dirs:
        raise FileNotFoundError(f"No */*/metadata.json under {environments_dir}")
    return dirs


def _files_sha256(game_dir: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(game_dir.iterdir())
        if p.is_file() and p.name != "metadata.json"
    }


def build_rows(environments_dir: Path, games: list[str] | None = None) -> list[dict[str, Any]]:
    """One row per game, in sorted game-id order (the row index is the harness dispatch index)."""
    dirs = _game_dirs(environments_dir)
    selected = sorted(PUBLIC25_VERSIONS) if games is None else list(games)
    missing = [g for g in selected if g not in dirs]
    if missing:
        raise KeyError(f"Games {missing} are not under {environments_dir}")
    changed = {g: sorted(dirs[g]) for g in selected if g in PUBLIC25_VERSIONS and PUBLIC25_VERSIONS[g] not in dirs[g]}
    if changed:
        raise ValueError(f"Game versions under {environments_dir} differ from the pinned public25 versions: {changed}")
    several = [g for g in selected if g not in PUBLIC25_VERSIONS and len(dirs[g]) > 1]
    if several:
        raise ValueError(f"Games {several} have several versions under {environments_dir} and no pinned one")
    rows = []
    for index, game in enumerate(selected):
        versions = dirs[game]
        game_dir = versions[PUBLIC25_VERSIONS[game]] if game in PUBLIC25_VERSIONS else next(iter(versions.values()))
        meta = json.loads((game_dir / "metadata.json").read_text())
        baseline = [int(x) for x in meta["baseline_actions"]]
        pins = {
            "game_id": game,
            "full_game_id": meta["game_id"],
            "version": game_dir.name,
            "number_of_levels": len(baseline),
            "baseline_actions": baseline,
            "tags": meta.get("tags") or [],
            "files_sha256": _files_sha256(game_dir),
            "dispatch_index": index,
        }
        rows.append(
            {
                "responses_create_params": {
                    "input": [
                        {"role": "user", "content": f"Play the ARC-AGI-3 game {game} ({meta.get('title', game)})."}
                    ]
                },
                "verifier_metadata": pins,
                "agent_context": {"agent": {"values": {"game": {k: pins[k] for k in AGENT_GAME_KEYS}}}},
            }
        )
    return rows


def check_rows(environments_dir: Path, dataset: Path) -> None:
    """Raise unless every game of ``dataset`` is under ``environments_dir`` with identical files."""
    dirs = _game_dirs(environments_dir)
    for line in dataset.read_text().splitlines():
        meta = json.loads(line)["verifier_metadata"]
        game, version = meta["game_id"], meta["version"]
        game_dir = dirs.get(game, {}).get(version)
        if game_dir is None:
            raise KeyError(f"{game} version {version} is not under {environments_dir}")
        if _files_sha256(game_dir) != meta["files_sha256"]:
            raise ValueError(f"{game}: files under {game_dir} differ from the pinned sha256")


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--environments-dir", type=Path, help="an existing environment_files directory")
    source.add_argument(
        "--download-kaggle-competition",
        type=Path,
        metavar="DIR",
        help=f"download {COMPETITION} environment_files/ into DIR",
    )
    parser.add_argument("--output", type=Path, help="public25 JSONL to write")
    parser.add_argument("--example-output", type=Path, help=f"example JSONL to write ({', '.join(EXAMPLE_GAMES)})")
    parser.add_argument("--check", type=Path, help="verify the directory against this pinned JSONL and exit")
    args = parser.parse_args(argv)

    environments_dir = args.environments_dir
    if args.download_kaggle_competition is not None:
        environments_dir = download_competition_environment_files(args.download_kaggle_competition)
        print(f"Downloaded {COMPETITION} environment_files to {environments_dir}")
    if args.check is not None:
        check_rows(environments_dir, args.check)
        print(f"{environments_dir} matches {args.check}")
        return
    if args.output is not None:
        rows = build_rows(environments_dir)
        write_jsonl(rows, args.output)
        print(f"Wrote {len(rows)} games to {args.output}")
    if args.example_output is not None:
        rows = build_rows(environments_dir, list(EXAMPLE_GAMES))
        write_jsonl(rows, args.example_output)
        print(f"Wrote {len(rows)} games to {args.example_output}")


if __name__ == "__main__":
    main()
