# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the pinned VS-Bench offline dataset into Gym request records.

The source's prompt/selection contract is from the MIT-licensed VS-Bench
https://github.com/zelaix/VS-Bench at 2c3531cdf91d1b620b4a9015126a242eb305a4d0:
``agents/prompt_agent.py``, ``models/openai.py`` and ``utils/helper.py``.
This adapter doesn't copy or change the author's benchmark implementation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq


DATASET_REPO = "VS-Bench/VS-Bench"
DATASET_REVISION = "30ef045024e1f4a778166f274767f6c8d354c676"
AUTHOR_REVISION = "2c3531cdf91d1b620b4a9015126a242eb305a4d0"
GAMES = (
    "hanabi",
    "overcooked",
    "knights_archers_zombies",
    "breakthrough",
    "kuhn_poker",
    "atari_pong",
    "simple_push",
    "coin_dilemma",
    "monster_hunt",
    "battle_of_the_colors",
)
TRACKS = ("perception", "reasoning")
FINAL_INSTRUCTION = "Do not include any extra commentary or explanation."
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "vs_bench_offline.jsonl"


def validate_source(dataset_root: Path, tracks: tuple[str, ...], games: tuple[str, ...]) -> None:
    """Reject a local dataset unless its exact source revision is identified."""
    if dataset_root.name == DATASET_REVISION:
        # Hugging Face's snapshot directory is addressed by the pinned commit.
        return
    receipt_path, manifest_path = dataset_root / "READY.json", dataset_root / "DOWNLOAD_MANIFEST.json"
    if not receipt_path.is_file() or not manifest_path.is_file():
        raise ValueError("Provide the pinned HF snapshot or a verified dataset with provenance receipts")
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (receipt.get("repo"), receipt.get("revision"), receipt.get("complete")) != (
        DATASET_REPO,
        DATASET_REVISION,
        True,
    ):
        raise ValueError("Dataset receipt does not identify the pinned release")
    if (manifest.get("repo"), manifest.get("revision")) != (DATASET_REPO, DATASET_REVISION):
        raise ValueError("Dataset manifest does not identify the pinned release")
    files = {item["rfilename"]: item for item in manifest["files"]}
    for track in tracks:
        for game in games:
            relative = f"{track}/{game}.parquet"
            entry = files[relative]
            path = dataset_root / relative
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if path.stat().st_size != entry["size"] or digest.hexdigest() != entry["lfs"]["sha256"]:
                raise ValueError(f"Dataset file does not match its receipt: {relative}")


def source_messages(prompt: dict, *, reasoning_model: bool = True) -> list[dict]:
    """Represent the upstream visual prompt as Responses API input messages."""
    images = prompt["image_path"]
    if not isinstance(images, list) or not images:
        raise ValueError("VS-Bench row must contain at least one source image")
    image_blocks = []
    for encoded in images:
        if not isinstance(encoded, str):
            raise ValueError("Expected the upstream base64 image representation")
        base64.b64decode(encoded, validate=True)
        image_blocks.append({"type": "input_image", "image_url": "data:image/png;base64," + encoded, "detail": "high"})
    question = prompt["observation"] + "\n\n" + prompt["action"] + "\n\n" + FINAL_INSTRUCTION
    if reasoning_model:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt["system"]},
                    *image_blocks,
                    {"type": "input_text", "text": question},
                ],
            }
        ]
    return [
        {"role": "system", "content": [{"type": "input_text", "text": prompt["system"]}]},
        {"role": "user", "content": [*image_blocks, {"type": "input_text", "text": question}]},
    ]


def convert_row(track: str, game: str, index: int, row: dict, *, reasoning_model: bool = True) -> dict:
    if track not in TRACKS or game not in GAMES or not 0 <= index < 400:
        raise ValueError("Unknown VS-Bench track, game or row")
    metadata = {
        "benchmark": "vs_bench",
        "track": track,
        "game": game,
        "row_index": index,
        "dataset_revision": DATASET_REVISION,
        "author_revision": AUTHOR_REVISION,
    }
    metadata["source_prompt_text"] = {key: row["prompt"][key] for key in ("system", "observation", "action")}
    if track == "reasoning":
        metadata["next_action"] = row["next_action"]
        metadata["next_legal_actions"] = json.loads(row["next_legal_actions"])
    else:
        # Preserve the stored label without replacing any upstream scorer.
        metadata["label"] = row["label"]
        metadata["source_index"] = row.get("__index_level_0__")
    return {
        "responses_create_params": {"input": source_messages(row["prompt"], reasoning_model=reasoning_model)},
        "verifier_metadata": metadata,
    }


def rows(
    dataset_root: Path,
    tracks: tuple[str, ...] = TRACKS,
    games: tuple[str, ...] = GAMES,
    limit: int = 400,
    *,
    reasoning_model: bool = True,
) -> Iterator[dict]:
    if not 1 <= limit <= 400 or any(t not in TRACKS for t in tracks) or any(g not in GAMES for g in games):
        raise ValueError("Invalid VS-Bench selection")
    for track in tracks:
        for game in games:
            source = dataset_root / track / f"{game}.parquet"
            parquet = pq.ParquetFile(source)
            expected = min(limit, 300 if track == "perception" and game == "knights_archers_zombies" else 400)
            if parquet.metadata.num_rows < expected:
                raise ValueError(f"Incomplete source data: {track}/{game}")
            index = 0
            for batch in parquet.iter_batches(batch_size=4):
                for raw in batch.to_pylist():
                    if index < expected:
                        yield convert_row(track, game, index, raw, reasoning_model=reasoning_model)
                        index += 1
                if index >= expected:
                    break


def prepare(
    dataset_root: Path | None = None,
    output_fpath: Path = DEFAULT_OUTPUT,
    *,
    tracks: tuple[str, ...] = TRACKS,
    games: tuple[str, ...] = GAMES,
    limit: int = 400,
    reasoning_model: bool = True,
) -> Path:
    if dataset_root is None:
        from huggingface_hub import snapshot_download

        dataset_root = Path(
            snapshot_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                revision=DATASET_REVISION,
                allow_patterns=[f"{t}/{g}.parquet" for t in tracks for g in games],
            )
        )
    validate_source(Path(dataset_root), tracks, games)
    output_fpath = Path(output_fpath)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    # Never turn a partial write into an apparently complete prepared dataset.
    temporary = output_fpath.with_suffix(output_fpath.suffix + ".partial")
    if output_fpath.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite prepared dataset: {output_fpath}")
    with temporary.open("x", encoding="utf-8") as stream:
        for row in rows(Path(dataset_root), tracks, games, limit, reasoning_model=reasoning_model):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.rename(output_fpath)
    return output_fpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--track", action="append", choices=TRACKS)
    parser.add_argument("--game", action="append", choices=GAMES)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--non-reasoning-model", action="store_true")
    args = parser.parse_args()
    print(
        prepare(
            args.dataset_root,
            args.output,
            tracks=tuple(args.track or TRACKS),
            games=tuple(args.game or GAMES),
            limit=args.limit,
            reasoning_model=not args.non_reasoning_model,
        )
    )
