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
import json
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

import textworld

GAME_TYPE_INSTRUCTIONS = {
    "custom": "This is a procedurally-generated text adventure. Explore rooms, interact with objects, and complete the quest.",
    "coin_collector": "This is a navigation challenge. Your goal is to navigate through rooms to find and collect the coin. Focus on efficient pathfinding.",
    "treasure_hunter": "This is a treasure hunting challenge. You must find the correct treasure object while avoiding the wrong one. You may need to open containers and unlock doors.",
    "simple": "This is a house cooking challenge. You need to escape the bedroom, find a food item, and cook it by placing it on the stove.",
    "cooking": "This is a complex cooking challenge. Read the cookbook, gather ingredients (which may need cutting/cooking), and prepare the meal. Be strategic about inventory management.",
}


def get_game_type_from_filename(filename: str) -> str:
    if filename.startswith("coin_collector"):
        return "coin_collector"
    elif filename.startswith("treasure_hunter"):
        return "treasure_hunter"
    elif filename.startswith("simple_"):
        return "simple"
    elif filename.startswith("cooking_"):
        return "cooking"
    elif filename.startswith("custom_"):
        return "custom"
    else:
        return "custom"


def create_example_for_game(game_file: Path) -> Dict:
    print(f"Creating example for {game_file.name}...")

    game = textworld.Game.load(str(game_file))
    objective = game.objective
    game_type = get_game_type_from_filename(game_file.name)

    type_instructions = GAME_TYPE_INSTRUCTIONS.get(game_type, "")

    prompt_parts = [
        "You are playing a text adventure game.",
        f"\n{type_instructions}" if type_instructions else "",
        f"\n\nObjective: {objective}",
        "\n\nUse the execute_command tool to interact with the game world. You can:",
        "- Move between rooms: 'go north', 'go south', 'go east', 'go west'",
        "- Pick up objects: 'take [object]'",
        "- Interact with containers: 'open [container]', 'close [container]'",
        "- Use keys: 'unlock [door/container] with [key]'",
        "- Examine things: 'examine [object]', 'look'",
        "- Place objects: 'put [object] on [surface]'",
    ]

    if game_type == "cooking":
        prompt_parts.extend([
            "- Cook food: 'cook [food] with [appliance]'",
            "- Cut food: 'chop [food]', 'slice [food]', 'dice [food]'",
            "- Eat food: 'eat [food]'",
        ])

    content = "".join(prompt_parts)

    execute_command_tool = {
        "name": "execute_command",
        "type": "function",
        "description": "Execute a text command in the game world. Use natural language commands like 'go north', 'take apple', 'open door', etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The text command to execute in the game",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    game_file_z8 = game_file.stem + ".z8"

    example = {
        "game_file": game_file_z8,
        "game_type": game_type,
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "tools": [execute_command_tool],
        },
    }

    return example


def select_diverse_games(games_dir: str) -> List[Path]:
    '''Makes a test set of 13 games to pipeclean'''
    games_dir_path = Path(games_dir)

    selected_games = []
    # 2 custom games
    custom_games = sorted(games_dir_path.glob("custom_*.json"))
    if custom_games:
        selected_games.extend([custom_games[0], custom_games[2] if len(custom_games) > 2 else custom_games[-1]])

    # 3 coin collector games
    coin_games = {
        "coin_collector_lvl01.json": games_dir_path / "coin_collector_lvl01.json",
        "coin_collector_lvl05.json": games_dir_path / "coin_collector_lvl05.json",
        "coin_collector_lvl15.json": games_dir_path / "coin_collector_lvl15.json",
    }
    for game_name, game_path in coin_games.items():
        if game_path.exists():
            selected_games.append(game_path)

    # 3 treasure hunter games
    treasure_games = {
        "treasure_hunter_lvl02_easy.json": games_dir_path / "treasure_hunter_lvl02_easy.json",
        "treasure_hunter_lvl12_medium.json": games_dir_path / "treasure_hunter_lvl12_medium.json",
        "treasure_hunter_lvl25_hard.json": games_dir_path / "treasure_hunter_lvl25_hard.json",
    }
    for game_name, game_path in treasure_games.items():
        if game_path.exists():
            selected_games.append(game_path)

    # 2 simple games
    simple_games = {
        "simple_dense_brief.json": games_dir_path / "simple_dense_brief.json",
        "simple_sparse_brief.json": games_dir_path / "simple_sparse_brief.json",
    }
    for game_name, game_path in simple_games.items():
        if game_path.exists():
            selected_games.append(game_path)

    # 3 cooking games
    cooking_games = {
        "cooking_train_r2_t3_g6_cook_cut_open.json": games_dir_path / "cooking_train_r2_t3_g6_cook_cut_open.json",
        "cooking_valid_r2_t2_g6_cook_cut_open.json": games_dir_path / "cooking_valid_r2_t2_g6_cook_cut_open.json",
        "cooking_test_r1_t2_g6_cook_open.json": games_dir_path / "cooking_test_r1_t2_g6_cook_open.json",
    }
    for game_name, game_path in cooking_games.items():
        if game_path.exists():
            selected_games.append(game_path)

    return selected_games


def is_5k_dataset(games_dir: str) -> bool:
    """Check if this is the 5K dataset (has train/val/test subdirs)."""
    train_dir = Path(games_dir) / "train"
    return train_dir.exists() and train_dir.is_dir()


def sample_from_5k_dataset(
    games_dir: str,
    num_samples: int = None,
    split: str = "train",
    stratify_by_difficulty: bool = True,
) -> List[Path]:
    """
    Sample games from a 5K dataset.

    Args:
        games_dir: Root directory of 5K dataset
        num_samples: Number of games to sample. If None or -1, use all games.
        split: Which split to sample from ('train', 'val', or 'test')
        stratify_by_difficulty: Whether to stratify sampling by difficulty (ignored if using all games)

    Returns:
        List of game file paths
    """
    split_dir = Path(games_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    metadata_path = Path(games_dir) / "metadata.json"
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    all_games = []
    game_types = ["coin_collector", "treasure_hunter", "simple", "cooking", "custom"]

    for game_type in game_types:
        type_dir = split_dir / game_type
        if type_dir.exists():
            json_files = list(type_dir.glob("*.json"))
            all_games.extend(json_files)

    if len(all_games) == 0:
        raise FileNotFoundError(f"No games found in {split_dir}")

    if num_samples is None or num_samples == -1:
        print(f"Using all {len(all_games)} games from {split} split (no sampling)")
        return all_games

    if stratify_by_difficulty and metadata:
        from collections import defaultdict
        import random

        grouped_games = defaultdict(lambda: defaultdict(list))

        metadata_games = metadata.get("games", {}).get(split, [])
        filename_to_metadata = {g["file"]: g for g in metadata_games}

        for game_path in all_games:
            game_filename = game_path.stem + ".z8"
            game_meta = filename_to_metadata.get(game_filename, {})

            game_type = game_meta.get("type", "unknown")
            difficulty = game_meta.get("difficulty", "unknown")

            grouped_games[game_type][difficulty].append(game_path)

        total_groups = sum(len(difficulties) for difficulties in grouped_games.values())
        samples_per_group = max(1, num_samples // total_groups)

        sampled_games = []
        for game_type, difficulties in grouped_games.items():
            for difficulty, games in difficulties.items():
                sample_size = min(samples_per_group, len(games))
                sampled_games.extend(random.sample(games, sample_size))

        if len(sampled_games) < num_samples:
            remaining = num_samples - len(sampled_games)
            available = [g for g in all_games if g not in sampled_games]
            if available:
                sampled_games.extend(random.sample(available, min(remaining, len(available))))

        return sampled_games[:num_samples]

    else:
        import random
        sample_size = min(num_samples, len(all_games))
        return random.sample(all_games, sample_size)


def create_examples(
    games_dir: str = "resources_servers/textworld/games",
    output_file: str = "resources_servers/textworld/data/example.jsonl",
    use_diverse_selection: bool = True,
    num_samples: int = None,
    split: str = "train",
):
    """
    Create example dataset for NeMo Gym.

    Supports both small datasets and large (eg 5K) datasets with train/val/test splits.

    Args:
        games_dir: Directory containing generated game files
        output_file: Output JSONL file path
        use_diverse_selection: If True, select diverse representative games.
                               If False, use first 5 games.
        num_samples: Number of samples. If None, uses all games.
        split: Which split to use ('train', 'val', or 'test')
    """
    is_5k = is_5k_dataset(games_dir)

    if is_5k:
        print(f"\nUsing 5K dataset")
        # num_samples None means use ALL games (no default sampling)
        game_files = sample_from_5k_dataset(games_dir, num_samples, split, stratify_by_difficulty=True)
    else:
        print(f"\nUsing simple dataset")
        if use_diverse_selection:
            game_files = select_diverse_games(games_dir)
        else:
            game_files = sorted(Path(games_dir).glob("*.json"))[:5]

    if not game_files:
        raise FileNotFoundError(f"No .json game files found in {games_dir}. Run generate_games.py first.")

    print(f"\n{'='*60}")
    print(f"CREATING EXAMPLES")
    print(f"Processing {len(game_files)} games")
    if is_5k:
        print(f"Split: {split}")
        if num_samples is None or num_samples == -1:
            print(f"Mode: ALL games (no sampling)")
        else:
            print(f"Sampling strategy: Stratified by game type and difficulty")
    print(f"{'='*60}")

    examples = []

    for game_file in tqdm(game_files, desc="Creating examples", unit="example"):
        try:
            example = create_example_for_game(game_file)
            examples.append(example)
        except Exception as e:
            continue

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"\n{'='*60}")
    print(f"EXAMPLES CREATED")
    print(f"Total: {len(examples)} examples")
    print(f"Output: {output_file}")

    from collections import Counter
    type_counts = Counter(ex["game_type"] for ex in examples)
    print("\nBreakdown by game type:")
    for game_type, count in sorted(type_counts.items()):
        print(f"  {game_type}: {count} examples")
    print(f"{'='*60}")

    print("\nSample example:")
    print(json.dumps(examples[0], indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create training examples from TextWorld games")
    parser.add_argument("--games-dir", type=str, default="resources_servers/textworld/games",
                       help="Directory containing game files")
    parser.add_argument("--output", type=str, default="resources_servers/textworld/data/example.jsonl",
                       help="Output JSONL file")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples (for 5K datasets). Use None or -1 for ALL games.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                       help="Which split to use for 5K datasets")
    parser.add_argument("--no-diverse", action="store_true",
                       help="Disable diverse selection (use first N games)")
    parser.add_argument("--all", action="store_true",
                       help="Use ALL games instead of sampling (shorthand for --num-samples -1)")

    args = parser.parse_args()

    num_samples = args.num_samples
    if args.all:
        num_samples = -1

    create_examples(
        games_dir=args.games_dir,
        output_file=args.output,
        use_diverse_selection=not args.no_diverse,
        num_samples=num_samples,
        split=args.split,
    )
