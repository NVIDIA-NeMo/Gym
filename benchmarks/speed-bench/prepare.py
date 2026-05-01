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
"""SPEED-Bench data preparation for Gym.

Ported from
https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/speed-bench/prepare.py.

The original Skills `prepare.py` resolves turn placeholders against ~14
external HF datasets (BAMBOO, HLE, LiveCodeBench, MMLU-Pro, etc.). Re-porting
that logic verbatim is a lot of code (~600 lines), and the only thing the
*Gym* side needs to differ from Skills is the JSONL row shape — Gym wants
`responses_create_params.input` instead of `messages`.

Strategy: import Skills' helpers (`_resolve_external_data`) when available
and reuse them. If Skills is not installed (e.g. running prepare.py outside
the nemo-rl/nemo-skills container), fall back to loading
`nvidia/SPEED-Bench` directly and emitting rows with the unresolved turn
placeholders (useful for quick iteration on Gym-only changes).

Output: one JSONL per config under `data/speed_bench_<config>_benchmark.jsonl`.
Each row:

    {
      "responses_create_params": {
        "input": [
          {"role": "user", "content": "<turn 1>"},
          {"role": "user", "content": "<turn 2>"},
          ...
        ]
      },
      "verifier_metadata": {
        "src_id": "...",
        "source": "...",
        "speed_config": "qualitative" | "throughput_2k" | ...,
        "num_turns": <int>,
        "sub_category": "..." | None
      }
    }

Multi-turn rows put EVERY user turn under `responses_create_params.input`
(no interspersed assistants). The `speed_bench_agent` replays them
sequentially at rollout time.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List


LOG = logging.getLogger(__name__)

# Configs Skills supports. We prepare a configurable subset by default since
# resolving every throughput config requires re-downloading + token-padding
# all 14 external datasets.
ALL_CONFIGS = (
    "qualitative",
    "throughput_1k",
    "throughput_2k",
    "throughput_8k",
    "throughput_16k",
    "throughput_32k",
)
DEFAULT_CONFIGS = ("qualitative", "throughput_2k")


def _import_skills_helpers():
    """Try to import Skills' resolver. Returns (resolver, available_flag)."""
    try:
        from nemo_skills.dataset import speed_bench as _ns_specbench  # type: ignore  # noqa: F401
    except ImportError:
        try:
            # Skills' module is named with a hyphen on disk; may or may not
            # round-trip through the importer depending on installation. Try
            # the file-based import too.
            import importlib
            import importlib.util

            spec_module_name = "nemo_skills.dataset.speed-bench.prepare"
            mod = importlib.import_module(spec_module_name)
            return mod._resolve_external_data, True
        except Exception:
            return None, False
    # Standard import worked — the underscore-named alias exists on some
    # Skills checkouts.
    from nemo_skills.dataset.speed_bench import prepare as _prep  # type: ignore

    return _prep._resolve_external_data, True


def _resolve_external_data_via_skills_path(dataset, speed_config: str):
    """Locate Skills' resolver via the on-disk hyphenated path.

    Skills' module dir is `nemo_skills/dataset/speed-bench/` — Python's import
    system can't dot-import a hyphenated package name, so we load `prepare.py`
    by its file path. This is robust to whichever Skills checkout is mounted.
    """
    import importlib.util

    candidates = [
        # Inside the nemo-rl container (mounted Skills clone via PYTHONPATH).
        Path("/nemo_run/code/nemo_skills/dataset/speed-bench/prepare.py"),
        # Within the recipe directory (local Skills checkout).
        Path(__file__).resolve().parents[3] / "nemo-skills" / "nemo_skills" / "dataset" / "speed-bench" / "prepare.py",
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("ns_speed_bench_prepare", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod._resolve_external_data(dataset, speed_config)
    raise RuntimeError(
        f"Could not find Skills' speed-bench prepare.py to import _resolve_external_data. Tried: {candidates}"
    )


def _row_to_gym(example: dict, speed_config: str) -> dict:
    """Convert a Skills-style resolved example to a Gym JSONL row."""
    turns: List[str] = list(example["turns"])
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": turn} for turn in turns],
        },
        "verifier_metadata": {
            "src_id": example.get("src_id"),
            "source": example.get("source"),
            "speed_config": speed_config,
            "num_turns": len(turns),
            "sub_category": example.get("sub_category"),
        },
    }


def prepare_config(speed_config: str, output_dir: Path) -> Path:
    """Download SPEED-Bench, resolve external turns, and write Gym JSONL."""
    from datasets import load_dataset  # imported lazily so unit tests don't need it

    LOG.info("Loading nvidia/SPEED-Bench config %s", speed_config)
    dataset = load_dataset("nvidia/SPEED-Bench", speed_config, split="test")

    LOG.info("Resolving external dataset placeholders for config %s (%d rows)", speed_config, len(dataset))
    resolved = _resolve_external_data_via_skills_path(dataset, speed_config)

    output_path = output_dir / f"speed_bench_{speed_config}_benchmark.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output_path.open("wt", encoding="utf-8") as f:
        for example in resolved:
            row = _row_to_gym(example, speed_config)
            f.write(json.dumps(row) + "\n")
            n += 1
    LOG.info("Wrote %d rows to %s", n, output_path)
    return output_path


def prepare(configs: Iterable[str] | None = None, output_dir: Path | None = None) -> List[Path]:
    """Prepare one or more SPEED-Bench configs.

    Args:
        configs: Iterable of config names. Defaults to (qualitative, throughput_2k).
        output_dir: Output directory. Defaults to this file's `data/` sibling.
    """
    configs = tuple(configs) if configs else DEFAULT_CONFIGS
    output_dir = output_dir or (Path(__file__).resolve().parent / "data")
    paths = []
    for cfg in configs:
        if cfg not in ALL_CONFIGS:
            raise ValueError(f"Unknown speed-bench config: {cfg!r}. Allowed: {ALL_CONFIGS}")
        paths.append(prepare_config(cfg, output_dir))
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Prepare SPEED-Bench data for NeMo Gym.")
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        choices=list(ALL_CONFIGS) + ["all", "default"],
        help="Speed-bench config to prepare. Pass multiple times. 'default' = qualitative + throughput_2k.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.config:
        configs = DEFAULT_CONFIGS
    elif "all" in args.config:
        configs = ALL_CONFIGS
    elif "default" in args.config:
        configs = DEFAULT_CONFIGS
    else:
        configs = tuple(args.config)

    prepare(configs=configs, output_dir=args.output_dir)
