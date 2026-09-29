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
"""Build a task distribution over one or more dataset columns.

A *distribution* groups every task in a dataset by the value(s) of one or
more metadata columns (e.g. ``sector``, or ``sector`` + ``occupation``) and
records, for each group, the fraction of the dataset it covers and the list
of ``task_id``s that fall into it::

    {
      "Business, Finance & Operations": {"percentage": 0.05, "task_ids": ["a", "b"]},
      "Legal": {"percentage": 0.50, "task_ids": [...]},
      "Healthcare": {"percentage": 0.45, "task_ids": [...]}
    }

Datasets are the NeMo Gym Responses-API JSONL format: one task per line, with
the groupable columns living under ``responses_create_params.metadata``.

The grouping logic is intentionally separated from the CLI so the resulting
distribution can later be reused to *sample* ``task_id``s (see
``sample_task_ids``).

Usage::

    # Full defaults: the prepared GDPVal dataset (220 tasks)
    # (benchmarks/gdpval/data/gdpval_benchmark.jsonl) grouped by ``occupation``.
    # Without --output the distribution is printed to stdout.
    python -m responses_api_agents.stirrup_agent.task_distribution \
        --output occupation_distribution.json

    # --dataset defaults to the prepared GDPVal dataset when omitted.
    python -m responses_api_agents.stirrup_agent.task_distribution \
        --column sector \
        --output sector_distribution.json

    # Composite key over multiple columns, explicit dataset:
    python -m responses_api_agents.stirrup_agent.task_distribution \
        --dataset data/gdpval.jsonl --column sector --column occupation \
        --output sector_occupation_distribution.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence


# Sentinel used when a row is missing one of the requested columns.
MISSING_VALUE = "<missing>"

# Separator joining multiple column values into a single composite key.
DEFAULT_KEY_SEPARATOR = " | "

# Column grouped on when ``--column`` is not specified.
DEFAULT_COLUMN = "occupation"

# Repo root: this file is responses_api_agents/stirrup_agent/task_distribution.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Candidate GDPVal dataset locations, in priority order. The first that exists
# is used when ``--dataset`` is not given. The prepared benchmark JSONL (written
# by ``gym eval prepare --benchmark gdpval``) is preferred; the agent-local
# ``data/gdpval.jsonl`` (written by setup_scripts/gdpval.sh) is a fallback.
# The synthetic ``example.jsonl`` is intentionally *not* a default so the
# command never silently computes a distribution over a single fake task.
DEFAULT_DATASET_CANDIDATES = (
    _REPO_ROOT / "benchmarks" / "gdpval" / "data" / "gdpval_benchmark.jsonl",
    Path(__file__).resolve().parent / "data" / "gdpval.jsonl",
)


def resolve_default_dataset(
    candidates: Optional[Sequence[Path]] = None,
) -> Optional[Path]:
    """Return the first existing default GDPVal dataset, or ``None``.

    Used when the caller does not pass an explicit ``--dataset``; prefers the
    prepared benchmark JSONL and falls back to agent-local datasets.
    """
    if candidates is None:
        candidates = DEFAULT_DATASET_CANDIDATES
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _no_dataset_message() -> str:
    """Actionable error shown when no dataset is specified and no default exists."""
    searched = "".join(f"  - {c}\n" for c in DEFAULT_DATASET_CANDIDATES)
    return (
        "No dataset specified and no default GDPVal dataset was found.\n"
        f"\nSearched these default locations:\n{searched}"
        "\nTo fix this, do one of the following:\n"
        "\n  1. Prepare the GDPVal benchmark dataset (recommended). This downloads\n"
        "     the openai/gdpval dataset from HuggingFace and writes\n"
        "     benchmarks/gdpval/data/gdpval_benchmark.jsonl (220 tasks).\n"
        "\n     First activate the project virtualenv so the Gym CLI is on PATH\n"
        "     (the `gym`/`ng_*` commands live in .venv, not on your global PATH):\n"
        "\n         source .venv/bin/activate\n"
        "         export HF_TOKEN=<your-huggingface-token>\n"
        "\n     Then run the setup script (works on all installs):\n"
        "\n         bash responses_api_agents/stirrup_agent/setup_scripts/gdpval.sh\n"
        "\n     Or call a prepare CLI directly:\n"
        "\n         gym eval prepare --benchmark gdpval        # newer installs\n"
        "         ng_prepare_benchmark '+config_paths=[benchmarks/gdpval/config.yaml]'  # any install\n"
        "\n  2. Pass an explicit dataset path with --dataset <path-to.jsonl>.\n"
        "\nNote: the GDPVal dataset is gated on HuggingFace, so HF_TOKEN must be set\n"
        "and your account must have access to https://huggingface.co/datasets/openai/gdpval.\n"
    )


def iter_dataset_rows(dataset_path: str | Path) -> Iterator[Dict[str, Any]]:
    """Yield parsed JSON objects from a Responses-API JSONL dataset.

    Blank lines are skipped; malformed lines raise ``ValueError`` with the
    1-based line number so the offending row is easy to find.
    """
    path = Path(dataset_path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON line: {exc}") from exc


def extract_metadata(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the ``responses_create_params.metadata`` dict for a row.

    Falls back to a top-level ``metadata`` key (and finally the row itself)
    so the function also works on flatter dataset variants.
    """
    params = row.get("responses_create_params")
    if isinstance(params, Mapping):
        metadata = params.get("metadata")
        if isinstance(metadata, Mapping):
            return dict(metadata)
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        return dict(metadata)
    return dict(row)


def compose_key(
    metadata: Mapping[str, Any],
    columns: Sequence[str],
    *,
    separator: str = DEFAULT_KEY_SEPARATOR,
    missing_value: str = MISSING_VALUE,
) -> str:
    """Build the distribution key for a row from one or more columns.

    Each column value is stringified; missing values become ``missing_value``.
    Multiple columns are joined with ``separator`` into a composite key.
    """
    parts: List[str] = []
    for column in columns:
        value = metadata.get(column, None)
        if value is None:
            parts.append(missing_value)
        else:
            parts.append(str(value))
    return separator.join(parts)


def build_distribution(
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    task_id_column: str = "task_id",
    separator: str = DEFAULT_KEY_SEPARATOR,
    missing_value: str = MISSING_VALUE,
    precision: Optional[int] = 6,
) -> Dict[str, Dict[str, Any]]:
    """Compute the task distribution across ``columns``.

    Returns a mapping ``key -> {"percentage": float, "task_ids": [...]}`` where
    ``percentage`` is the fraction (0..1) of all tasks that share that key and
    ``task_ids`` lists every matching task in first-seen order. The mapping is
    ordered by descending ``percentage`` (ties broken by key) for readability.

    ``percentage`` values are rounded to ``precision`` decimal places when
    ``precision`` is not ``None``. Note that rounding can make the percentages
    sum to slightly more or less than 1.0; the unrounded fractions always sum
    to 1.0.
    """
    if not columns:
        raise ValueError("At least one column is required to build a distribution.")

    grouped: Dict[str, List[str]] = {}
    total = 0
    for index, row in enumerate(rows):
        metadata = extract_metadata(row)
        key = compose_key(metadata, columns, separator=separator, missing_value=missing_value)
        task_id = metadata.get(task_id_column)
        # Fall back to a positional id so every task is still counted/listed
        # even when the dataset lacks an explicit task-id column.
        task_id_str = str(task_id) if task_id is not None else f"{task_id_column}_index_{index}"
        grouped.setdefault(key, []).append(task_id_str)
        total += 1

    distribution: Dict[str, Dict[str, Any]] = {}
    for key, task_ids in grouped.items():
        fraction = (len(task_ids) / total) if total else 0.0
        percentage = round(fraction, precision) if precision is not None else fraction
        distribution[key] = {"percentage": percentage, "task_ids": task_ids}

    # Sort by descending share, then by key for stable, readable output.
    ordered = dict(
        sorted(
            distribution.items(),
            key=lambda item: (-len(item[1]["task_ids"]), item[0]),
        )
    )
    return ordered


def build_distribution_from_dataset(
    dataset_path: str | Path,
    columns: Sequence[str],
    *,
    task_id_column: str = "task_id",
    separator: str = DEFAULT_KEY_SEPARATOR,
    missing_value: str = MISSING_VALUE,
    precision: Optional[int] = 6,
) -> Dict[str, Dict[str, Any]]:
    """Convenience wrapper: read a JSONL dataset and build its distribution."""
    return build_distribution(
        iter_dataset_rows(dataset_path),
        columns,
        task_id_column=task_id_column,
        separator=separator,
        missing_value=missing_value,
        precision=precision,
    )


def sample_task_ids(
    distribution: Mapping[str, Mapping[str, Any]],
    n: int,
    *,
    rng: Optional[random.Random] = None,
    replace: bool = False,
) -> List[str]:
    """Sample ``n`` ``task_id``s in proportion to a distribution's percentages.

    Each task id is drawn by first choosing a group weighted by its
    ``percentage`` and then choosing a task id within that group. With
    ``replace=False`` (default) the same task id is never returned twice and
    ``n`` is capped at the total number of available task ids.

    This is the consumption-side counterpart to ``build_distribution`` and is
    provided so the saved distribution file can directly drive task sampling.
    """
    if n <= 0:
        return []
    rng = rng or random.Random()

    keys = list(distribution.keys())
    weights = [float(distribution[key].get("percentage", 0.0)) for key in keys]
    if not keys or sum(weights) <= 0:
        return []

    if replace:
        sampled: List[str] = []
        for _ in range(n):
            (chosen_key,) = rng.choices(keys, weights=weights, k=1)
            task_ids = list(distribution[chosen_key].get("task_ids", []))
            if not task_ids:
                continue
            sampled.append(rng.choice(task_ids))
        return sampled

    # Without replacement: track remaining ids per group and renormalise.
    remaining: Dict[str, List[str]] = {key: list(distribution[key].get("task_ids", [])) for key in keys}
    total_available = sum(len(ids) for ids in remaining.values())
    target = min(n, total_available)

    sampled = []
    while len(sampled) < target:
        live_keys = [key for key in keys if remaining[key]]
        live_weights = [float(distribution[key].get("percentage", 0.0)) for key in live_keys]
        if not live_keys or sum(live_weights) <= 0:
            break
        (chosen_key,) = rng.choices(live_keys, weights=live_weights, k=1)
        bucket = remaining[chosen_key]
        idx = rng.randrange(len(bucket))
        sampled.append(bucket.pop(idx))
    return sampled


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="task_distribution",
        description=(
            "Build a JSON distribution of tasks across one or more dataset "
            "columns (e.g. sector, occupation) from a Responses-API JSONL dataset."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Path to the input JSONL dataset (one task per line). If omitted, "
            "defaults to the prepared GDPVal dataset "
            "(benchmarks/gdpval/data/gdpval_benchmark.jsonl), falling back to "
            "the agent-local data/gdpval.jsonl or data/example.jsonl."
        ),
    )
    parser.add_argument(
        "--column",
        dest="columns",
        action="append",
        default=None,
        metavar="COLUMN",
        help=(
            "Metadata column to group by. Repeat to group by a composite key "
            "(e.g. --column sector --column occupation). "
            f"Defaults to {DEFAULT_COLUMN!r} if not specified."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to write the distribution JSON. Defaults to stdout.",
    )
    parser.add_argument(
        "--task-id-column",
        default="task_id",
        help="Metadata column holding the task id (default: task_id).",
    )
    parser.add_argument(
        "--separator",
        default=DEFAULT_KEY_SEPARATOR,
        help=f"Separator joining multiple column values into one key (default: {DEFAULT_KEY_SEPARATOR!r}).",
    )
    parser.add_argument(
        "--missing-value",
        default=MISSING_VALUE,
        help=f"Placeholder for rows missing a column (default: {MISSING_VALUE!r}).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal places to round percentages to; use -1 for no rounding (default: 6).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation for the output JSON; use -1 for compact output (default: 2).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.dataset is not None:
        dataset_path = Path(args.dataset)
        if not dataset_path.is_file():
            print(f"Dataset not found: {dataset_path}", file=sys.stderr)
            return 2
    else:
        dataset_path = resolve_default_dataset()
        if dataset_path is None:
            print(_no_dataset_message(), file=sys.stderr)
            return 2
        print(f"Using default dataset: {dataset_path}", file=sys.stderr)

    columns = args.columns
    if not columns:
        columns = [DEFAULT_COLUMN]
        print(f"No --column specified; defaulting to {DEFAULT_COLUMN!r}.", file=sys.stderr)

    precision = None if args.precision is not None and args.precision < 0 else args.precision
    indent = None if args.indent is not None and args.indent < 0 else args.indent

    distribution = build_distribution_from_dataset(
        dataset_path,
        columns,
        task_id_column=args.task_id_column,
        separator=args.separator,
        missing_value=args.missing_value,
        precision=precision,
    )

    payload = json.dumps(distribution, indent=indent, ensure_ascii=False)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
        total_tasks = sum(len(entry["task_ids"]) for entry in distribution.values())
        print(
            f"Wrote distribution over {columns} ({len(distribution)} groups, {total_tasks} tasks) to {out_path}",
            file=sys.stderr,
        )
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
