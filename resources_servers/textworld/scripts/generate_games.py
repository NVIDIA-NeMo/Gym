# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import json
import shutil
import argparse
import tempfile
import signal
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import time

from tqdm import tqdm

import textworld
from textworld import GameOptions
from textworld.challenges import coin_collector, treasure_hunter, simple, cooking


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Game generation timed out")


def with_timeout(timeout_seconds: int):
    def decorator(func):
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator


DEFAULT_DISTRIBUTION = {
    "coin_collector": 1000,
    "treasure_hunter": 500,
    "simple": 200,
    "cooking": 2800,
    "custom": 500,
}

SPLIT_RATIOS = {
    "train": 0.80,
    "val": 0.10,
    "test": 0.10,
}


# Game Generation Functions

def generate_single_coin_collector(args: Tuple[int, int, str]) -> Dict[str, Any]:
    idx, level, output_dir = args

    temp_dir = tempfile.mkdtemp(prefix=f"tw_coin_{idx}_")

    try:
        options = GameOptions()
        options.seeds = 1000000 + level * 10000 + idx
        options.file_ext = ".z8"
        settings = {"level": level}

        game = coin_collector.make(settings, options)

        options.path = temp_dir
        game_file = textworld.generator.compile_game(game, options)

        target_name = f"coin_collector_lvl{level:03d}_seed{idx:04d}.z8"
        target_path = os.path.join(output_dir, target_name)

        if os.path.exists(game_file):
            shutil.move(game_file, target_path)
            json_file = game_file.replace(".ulx", ".json").replace(".z8", ".json")
            if os.path.exists(json_file):
                target_json = target_path.replace(".z8", ".json")
                shutil.move(json_file, target_json)

        result = {
            "file": target_name,
            "type": "coin_collector",
            "level": level,
            "seed": options.seeds,
            "objective": game.objective,
            "max_score": game.max_score,
            "metadata": game.metadata,
            "difficulty": "easy" if level <= 10 else "medium" if level <= 20 else "hard",
        }

        shutil.rmtree(temp_dir, ignore_errors=True)
        return result

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None


def generate_single_treasure_hunter(args: Tuple[int, int, str]) -> Dict[str, Any]:
    idx, level, output_dir = args

    temp_dir = tempfile.mkdtemp(prefix=f"tw_treasure_{idx}_")

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)

        options = GameOptions()
        options.seeds = 2000000 + level * 10000 + idx
        options.file_ext = ".z8"
        settings = {"level": level}

        game = treasure_hunter.make(settings, options)

        options.path = temp_dir
        game_file = textworld.generator.compile_game(game, options)

        signal.alarm(0)

        mode = game.metadata.get("mode", "unknown")
        target_name = f"treasure_hunter_lvl{level:02d}_{mode}_seed{idx:04d}.z8"
        target_path = os.path.join(output_dir, target_name)

        if os.path.exists(game_file):
            shutil.move(game_file, target_path)
            json_file = game_file.replace(".ulx", ".json").replace(".z8", ".json")
            if os.path.exists(json_file):
                target_json = target_path.replace(".z8", ".json")
                shutil.move(json_file, target_json)

        result = {
            "file": target_name,
            "type": "treasure_hunter",
            "level": level,
            "mode": mode,
            "seed": options.seeds,
            "objective": game.objective,
            "max_score": game.max_score,
            "metadata": game.metadata,
            "difficulty": mode,
        }

        shutil.rmtree(temp_dir, ignore_errors=True)
        return result

    except (TimeoutException, Exception) as e:
        signal.alarm(0)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None


def generate_single_simple(args: Tuple[int, Dict, str]) -> Dict[str, Any]:
    idx, settings, output_dir = args

    temp_dir = tempfile.mkdtemp(prefix=f"tw_simple_{idx}_")

    try:
        options = GameOptions()
        options.seeds = 3000000 + idx
        options.file_ext = ".z8"

        game = simple.make(settings, options)

        options.path = temp_dir
        game_file = textworld.generator.compile_game(game, options)

        target_name = f"simple_{settings['rewards']}_{settings['goal']}_seed{idx:04d}.z8"
        target_path = os.path.join(output_dir, target_name)

        if os.path.exists(game_file):
            shutil.move(game_file, target_path)
            json_file = game_file.replace(".ulx", ".json").replace(".z8", ".json")
            if os.path.exists(json_file):
                target_json = target_path.replace(".z8", ".json")
                shutil.move(json_file, target_json)

        result = {
            "file": target_name,
            "type": "simple",
            "rewards": settings["rewards"],
            "goal": settings["goal"],
            "seed": options.seeds,
            "objective": game.objective,
            "max_score": game.max_score,
            "metadata": game.metadata,
            "difficulty": "easy" if settings["rewards"] == "dense" else "medium" if settings["rewards"] == "balanced" else "hard",
        }

        shutil.rmtree(temp_dir, ignore_errors=True)
        return result

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None


def generate_single_cooking(args: Tuple[int, Dict, str]) -> Dict[str, Any]:
    idx, settings, output_dir = args

    temp_dir = tempfile.mkdtemp(prefix=f"tw_cooking_{idx}_")

    try:
        options = GameOptions()
        options.seeds = 4000000 + settings["recipe_seed"] * 10000 + idx
        options.file_ext = ".z8"

        game = cooking.make(settings, options)

        options.path = temp_dir
        game_file = textworld.generator.compile_game(game, options)

        skills_str = f"r{settings['recipe']}_t{settings['take']}_g{settings['go']}"
        if settings['cook']:
            skills_str += "_cook"
        if settings.get('cut', False):
            skills_str += "_cut"
        if settings.get('open', False):
            skills_str += "_open"
        if settings.get('drop', False):
            skills_str += "_drop"

        target_name = f"cooking_{settings['split']}_{skills_str}_seed{idx:04d}.z8"
        target_path = os.path.join(output_dir, target_name)

        if os.path.exists(game_file):
            shutil.move(game_file, target_path)
            json_file = game_file.replace(".ulx", ".json").replace(".z8", ".json")
            if os.path.exists(json_file):
                target_json = target_path.replace(".z8", ".json")
                shutil.move(json_file, target_json)

        complexity = settings['recipe'] + (1 if settings.get('cut') else 0) + (1 if settings.get('drop') else 0)
        difficulty = "easy" if complexity <= 2 else "medium" if complexity <= 4 else "hard"

        result = {
            "file": target_name,
            "type": "cooking",
            "split": settings["split"],
            "recipe": settings["recipe"],
            "seed": options.seeds,
            "objective": game.objective,
            "max_score": game.max_score,
            "metadata": game.metadata,
            "difficulty": difficulty,
        }

        shutil.rmtree(temp_dir, ignore_errors=True)
        return result

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None


def generate_single_custom(args: Tuple[int, Dict, str]) -> Dict[str, Any]:
    idx, config, output_dir = args

    temp_dir = tempfile.mkdtemp(prefix=f"tw_custom_{idx}_")

    try:
        options = GameOptions()
        options.seeds = 5000000 + config["seed"] + idx * 100
        options.file_ext = ".z8"
        options.nb_rooms = config["nb_rooms"]
        options.nb_objects = config["nb_objects"]
        options.quest_length = config["quest_length"]

        options.path = temp_dir
        game_file, game = textworld.make(options)

        target_name = f"custom_r{config['nb_rooms']}_o{config['nb_objects']}_q{config['quest_length']}_seed{idx:04d}.z8"
        target_path = os.path.join(output_dir, target_name)

        if os.path.exists(game_file):
            shutil.move(game_file, target_path)
            json_file = game_file.replace(".ulx", ".json").replace(".z8", ".json")
            if os.path.exists(json_file):
                target_json = target_path.replace(".z8", ".json")
                shutil.move(json_file, target_json)

        complexity = config['nb_rooms'] + config['nb_objects'] + config['quest_length']
        difficulty = "easy" if complexity <= 15 else "medium" if complexity <= 25 else "hard"

        result = {
            "file": target_name,
            "type": "custom",
            "seed": options.seeds,
            "objective": game.objective,
            "max_score": game.max_score,
            "metadata": game.metadata,
            "difficulty": difficulty,
        }

        shutil.rmtree(temp_dir, ignore_errors=True)
        return result

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None


# Dataset Generation

def generate_coin_collector_configs(num_games: int, output_dir: str) -> List[Tuple[int, int, str]]:
    configs = []

    for i in range(num_games):
        level = min(1 + (i * 300 // num_games), 300)
        configs.append((i, level, output_dir))

    return configs


def generate_treasure_hunter_configs(num_games: int, output_dir: str) -> List[Tuple[int, int, str]]:
    configs = []

    games_per_level = num_games // 30
    extra = num_games % 30

    idx = 0
    for level in range(1, 31):
        count = games_per_level + (1 if level <= extra else 0)
        for j in range(count):
            configs.append((idx, level, output_dir))
            idx += 1

    return configs


def generate_simple_configs(num_games: int, output_dir: str) -> List[Tuple[int, Dict, str]]:
    all_settings = [
        {"rewards": "dense", "goal": "detailed", "test": False},
        {"rewards": "dense", "goal": "brief", "test": False},
        {"rewards": "balanced", "goal": "detailed", "test": False},
        {"rewards": "balanced", "goal": "brief", "test": False},
        {"rewards": "sparse", "goal": "detailed", "test": False},
        {"rewards": "sparse", "goal": "brief", "test": False},
    ]

    configs = []

    if num_games < len(all_settings):
        for idx, settings in enumerate(all_settings[:num_games]):
            configs.append((idx, settings, output_dir))
    else:
        games_per_setting = num_games // len(all_settings)
        extra_games = num_games % len(all_settings)

        idx = 0
        for i, settings in enumerate(all_settings):
            count = games_per_setting + (1 if i < extra_games else 0)
            for j in range(count):
                configs.append((idx, settings, output_dir))
                idx += 1

    return configs


def generate_cooking_configs(num_games: int, output_dir: str) -> List[Tuple[int, Dict, str]]:
    room_configs = [1, 6, 6, 9, 12]
    recipe_configs = [1, 1, 2, 2, 3]

    train_count = int(num_games * 0.7)
    val_count = int(num_games * 0.15)
    test_count = num_games - train_count - val_count

    configs = []
    idx = 0

    for split, count in [("train", train_count), ("valid", val_count), ("test", test_count)]:
        for i in range(count):
            recipe = recipe_configs[i % len(recipe_configs)]
            rooms = room_configs[i % len(room_configs)]

            settings = {
                "recipe": recipe,
                "take": min(recipe + 1, 5),
                "cook": True,
                "cut": recipe >= 2,
                "open": recipe >= 1,
                "drop": recipe >= 3,
                "go": rooms,
                "split": split,
                "recipe_seed": 100 + idx,
            }
            configs.append((idx, settings, output_dir))
            idx += 1

    return configs


def generate_custom_configs(num_games: int, output_dir: str) -> List[Tuple[int, Dict, str]]:
    rooms_range = [3, 4, 5, 6, 7, 8]
    objects_range = [5, 7, 10, 12, 15]
    quest_range = [3, 4, 5, 6, 7]

    configs = []

    for i in range(num_games):
        config = {
            "seed": 5000 + i * 100,
            "nb_rooms": rooms_range[i % len(rooms_range)],
            "nb_objects": objects_range[i % len(objects_range)],
            "quest_length": quest_range[i % len(quest_range)],
        }
        configs.append((i, config, output_dir))

    return configs


def split_games_by_type(
    games: List[Dict[str, Any]],
    split_ratios: Dict[str, float]
) -> Dict[str, List[Dict[str, Any]]]:
    games_by_type = defaultdict(list)
    for game in games:
        if game:
            games_by_type[game["type"]].append(game)

    splits = {"train": [], "val": [], "test": []}

    for game_type, type_games in games_by_type.items():
        n = len(type_games)
        train_end = int(n * split_ratios["train"])
        val_end = train_end + int(n * split_ratios["val"])

        splits["train"].extend(type_games[:train_end])
        splits["val"].extend(type_games[train_end:val_end])
        splits["test"].extend(type_games[val_end:])

    return splits


def generate_dataset(
    output_dir: str = "resources_servers/textworld/games",
    target_games: int = 5000,
    distribution: Dict[str, int] = None,
    num_workers: int = None,
):
    """
    Generate TextWorld dataset with train/val/test splits.

    Args:
        output_dir: Root directory for generated games
        target_games: Target number of games
        distribution: Game type distribution
        num_workers: Number of parallel workers
    """
    if distribution is None:
        distribution = DEFAULT_DISTRIBUTION.copy()

    total_in_dist = sum(distribution.values())
    distribution = {k: int(v * target_games / total_in_dist) for k, v in distribution.items()}

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print("=" * 80)
    print(f"TextWorld game generation")
    print(f"Target: {target_games} games")
    print(f"Workers: {num_workers}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_generated_games = []
    all_errors = []
    start_time = time.time()

    for game_type, num_games in distribution.items():
        if num_games == 0:
            continue

        print(f"\n{game_type.upper()}: Generating {num_games} games")

        type_output_dir = os.path.join(output_dir, f"temp_{game_type}")
        Path(type_output_dir).mkdir(exist_ok=True)

        if game_type == "coin_collector":
            configs = generate_coin_collector_configs(num_games, type_output_dir)
            generator_func = generate_single_coin_collector
        elif game_type == "treasure_hunter":
            configs = generate_treasure_hunter_configs(num_games, type_output_dir)
            generator_func = generate_single_treasure_hunter
        elif game_type == "simple":
            configs = generate_simple_configs(num_games, type_output_dir)
            generator_func = generate_single_simple
        elif game_type == "cooking":
            configs = generate_cooking_configs(num_games, type_output_dir)
            generator_func = generate_single_cooking
        elif game_type == "custom":
            configs = generate_custom_configs(num_games, type_output_dir)
            generator_func = generate_single_custom
        else:
            continue

        with Pool(num_workers) as pool:
            games = list(tqdm(
                pool.imap(generator_func, configs),
                total=len(configs),
                desc=f"  {game_type}",
                unit="game"
            ))

        successful_games = [g for g in games if g is not None]
        num_errors = len(games) - len(successful_games)

        all_generated_games.extend(successful_games)

        for i, g in enumerate(games):
            if g is None:
                all_errors.append(f"{game_type}: config_idx={i}")

        print(f"   Successfully generated: {len(successful_games)}/{num_games} games")

    print(f"\n{'='*80}")
    print(f"Making train/val/test splits...")
    print(f"{'='*80}")

    splits = split_games_by_type(all_generated_games, SPLIT_RATIOS)

    for split_name, split_games in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        Path(split_dir).mkdir(exist_ok=True)

        for game in split_games:
            temp_type_dir = os.path.join(output_dir, f"temp_{game['type']}")
            src_z8 = os.path.join(temp_type_dir, game["file"])
            src_json = src_z8.replace(".z8", ".json")

            type_dir = os.path.join(split_dir, game["type"])
            Path(type_dir).mkdir(exist_ok=True)

            if os.path.exists(src_z8):
                shutil.move(src_z8, os.path.join(type_dir, game["file"]))
            if os.path.exists(src_json):
                shutil.move(src_json, os.path.join(type_dir, game["file"].replace(".z8", ".json")))

    for game_type in distribution.keys():
        temp_dir = os.path.join(output_dir, f"temp_{game_type}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    metadata = {
        "total_games": len(all_generated_games),
        "splits": {split: len(games) for split, games in splits.items()},
        "distribution": {
            game_type: sum(1 for g in all_generated_games if g["type"] == game_type)
            for game_type in distribution.keys()
        },
        "games": splits,
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed_time = time.time() - start_time
    elapsed_mins = elapsed_time / 60
    elapsed_hours = elapsed_mins / 60

    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total games generated: {len(all_generated_games)}")
    print(f"Failed generations: {len(all_errors)}")
    print(f"Success rate: {len(all_generated_games)/(len(all_generated_games)+len(all_errors))*100:.1f}%")
    print(f"\nTime elapsed: {elapsed_hours:.2f} hours ({elapsed_mins:.1f} minutes)")
    print(f"Generation speed: {len(all_generated_games)/elapsed_mins:.1f} games/minute")

    if all_errors:
        print(f"\nFirst 10 errors:")
        for error in all_errors[:10]:
            print(f"  - {error}")

    print(f"\nSplits:")
    for split, games in splits.items():
        print(f"  {split}: {len(games)} games")
    print(f"\nDistribution:")
    for game_type, count in metadata["distribution"].items():
        print(f"  {game_type}: {count} games")
    print(f"\nMetadata saved: {metadata_path}")
    print(f"{'='*80}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TextWorld dataset")
    parser.add_argument("--output-dir", type=str, default="resources_servers/textworld/games",
                       help="Output directory for generated games")
    parser.add_argument("--target-games", type=int, default=5000,
                       help="Target number of games to generate")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: 1)")

    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        target_games=args.target_games,
        num_workers=args.workers,
    )
