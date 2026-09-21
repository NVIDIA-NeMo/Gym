# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize the KIDBench rows a NeMo Gym run consumes.

Fetches the upstream repository at :data:`~benchmarks.kidbench.upstream_spec.UPSTREAM_REVISION`
and expands its two dataset files into Gym JSONL:

* ``single_turn.jsonl`` — 5,000 rows: the ten-cell condition grid over 500 prompts.
* ``multi_turn.jsonl`` — 200 rows: 100 scenario/child-goal pairs under both age settings.
* ``example.jsonl`` — a committed 5-row single-turn smoke sample, one per condition.
* ``example_multi_turn.jsonl`` — a committed 2-row smoke sample, one per age setting.

Nothing upstream is copied into this repository. The benchmark's use notice forbids
stripping its child-safety context (rubrics, rules, notices) from a redistribution, so
the checkout stays under ``benchmarks/kidbench/upstream/`` (gitignored) and the resources
server reads the rubrics from there at run time.

Usage::

    python -m benchmarks.kidbench.prepare
    python -m benchmarks.kidbench.prepare --output-dir resources_servers/kidbench/data
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Iterator

from benchmarks.kidbench.upstream_spec import (
    CATEGORIES,
    CONDITION_GRID,
    MULTI_TURN_AGE_SETTINGS,
    MULTI_TURN_GENERATOR_MAX_OUTPUT_TOKENS,
    MULTI_TURN_GENERATOR_TEMPERATURE,
    RISK_CLASS_OF_CATEGORY,
    SINGLE_TURN_MAX_OUTPUT_TOKENS,
    SINGLE_TURN_TEMPERATURE,
    SINGLE_TURN_TOP_P,
    UPSTREAM_REPO,
    UPSTREAM_REVISION,
    build_multi_turn_system_prompt,
    build_system_prompt,
)


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKOUT_DIR = REPO_ROOT / "benchmarks" / "kidbench" / "upstream" / "kidbench"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "resources_servers" / "kidbench" / "data"


class UpstreamError(RuntimeError):
    """The pinned upstream checkout is missing or is not at the expected revision."""


# ---------------------------------------------------------------------------
# Upstream checkout
# ---------------------------------------------------------------------------


def ensure_upstream(checkout_dir: Path = DEFAULT_CHECKOUT_DIR, *, revision: str = UPSTREAM_REVISION) -> Path:
    """Clone or update the upstream repo so ``checkout_dir`` sits exactly at ``revision``.

    Idempotent: an existing checkout already at the revision is left untouched, so this is
    safe to call from a server's startup hook or from a test's ``pytest_configure``.
    """
    git_dir = checkout_dir / ".git"
    if git_dir.exists():
        head = subprocess.run(
            ["git", "-C", str(checkout_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if head.returncode == 0 and head.stdout.strip() == revision:
            return checkout_dir
        logger.info("Updating upstream KIDBench checkout to %s", revision)
        subprocess.run(["git", "-C", str(checkout_dir), "fetch", "--depth", "1", "origin", revision], check=True)
    else:
        logger.info("Cloning %s into %s", UPSTREAM_REPO, checkout_dir)
        checkout_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", UPSTREAM_REPO, str(checkout_dir)], check=True)
        subprocess.run(["git", "-C", str(checkout_dir), "fetch", "--depth", "1", "origin", revision], check=True)

    subprocess.run(["git", "-C", str(checkout_dir), "checkout", "--quiet", revision], check=True)
    return checkout_dir


def resolve_upstream(checkout_dir: Path = DEFAULT_CHECKOUT_DIR) -> Path:
    """Return a checkout known to be at the pinned revision, without network access.

    The resources server calls this rather than :func:`ensure_upstream` so that a server
    booting inside 4k concurrent workers never races on a clone. Preparing the data is a
    deliberate, one-time step.
    """
    head = subprocess.run(
        ["git", "-C", str(checkout_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if head.returncode != 0:
        raise UpstreamError(
            f"No upstream KIDBench checkout at {checkout_dir}. Run `python -m benchmarks.kidbench.prepare` first."
        )
    if head.stdout.strip() != UPSTREAM_REVISION:
        raise UpstreamError(
            f"Upstream KIDBench checkout at {checkout_dir} is at {head.stdout.strip()}, "
            f"expected the pinned {UPSTREAM_REVISION}. Re-run `python -m benchmarks.kidbench.prepare`."
        )
    return checkout_dir


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def _responses_create_params(system_prompt: str | None, user_prompt: str) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return {
        "input": messages,
        "temperature": SINGLE_TURN_TEMPERATURE,
        "top_p": SINGLE_TURN_TOP_P,
        "max_output_tokens": SINGLE_TURN_MAX_OUTPUT_TOKENS,
    }


def single_turn_rows(upstream: Path) -> Iterator[dict[str, Any]]:
    """Expand ``kidbench_single.json`` across the ten-cell condition grid.

    Non-English rows carry ``with_cues: null`` upstream — only the no-cue prompts were
    translated — and the grid only ever asks for ``without_cues`` there, so a missing
    variant is a real defect rather than an expected gap and is raised, not skipped.
    """
    data = json.loads((upstream / "dataset" / "kidbench" / "kidbench_single.json").read_text(encoding="utf-8"))

    for cell in CONDITION_GRID:
        by_category = data[cell.language]
        system_prompt = build_system_prompt(age=cell.age, country=cell.country, language=cell.language)
        for category in CATEGORIES:
            for index, item in enumerate(by_category[category]):
                prompt = item[cell.prompt_type]
                if not prompt:
                    raise UpstreamError(
                        f"Missing '{cell.prompt_type}' prompt for {cell.language}/{category}[{index}] "
                        f"in condition cell {cell.cell_id}"
                    )
                yield {
                    "kidbench_id": f"single-{cell.cell_id}-{category}-{index:03d}",
                    "track": "single_turn",
                    "condition": cell.condition,
                    "prompt_type": cell.prompt_type,
                    "language": cell.language,
                    "age_conditioned": cell.age,
                    "country": cell.country,
                    "category": category,
                    "risk_class": RISK_CLASS_OF_CATEGORY[category],
                    "example_index": index,
                    "upstream_file": cell.upstream_file,
                    "upstream_revision": UPSTREAM_REVISION,
                    "prompt": prompt,
                    "responses_create_params": _responses_create_params(system_prompt, prompt),
                }


def multi_turn_rows(upstream: Path) -> Iterator[dict[str, Any]]:
    """Expand ``kidbench_multi.json`` across both responder age settings.

    The row carries the scenario and child goal rather than any prompt: the first child
    message is produced by the actor model at run time, so there is nothing to seed
    ``input`` with. ``responses_create_params`` therefore holds only the system prompt and
    the decoding settings the responder is held to on every turn.
    """
    data = json.loads((upstream / "dataset" / "kidbench" / "kidbench_multi.json").read_text(encoding="utf-8"))

    for setting_name, age in MULTI_TURN_AGE_SETTINGS:
        system_prompt = build_multi_turn_system_prompt(age=age)
        for category in CATEGORIES:
            for index, item in enumerate(data[category]):
                messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
                yield {
                    "kidbench_id": f"multi-{setting_name}-{category}-{index:03d}",
                    "track": "multi_turn",
                    "condition": setting_name,
                    "age_conditioned": age,
                    "language": "english",
                    "country": None,
                    "category": category,
                    "risk_class": RISK_CLASS_OF_CATEGORY[category],
                    "example_index": index,
                    "scenario": item["scenario"],
                    "child_goal": item["child_goal"],
                    "upstream_file": f"multi_turn/{setting_name}.json",
                    "upstream_revision": UPSTREAM_REVISION,
                    "responses_create_params": {
                        "input": messages,
                        "temperature": MULTI_TURN_GENERATOR_TEMPERATURE,
                        "max_output_tokens": MULTI_TURN_GENERATOR_MAX_OUTPUT_TOKENS,
                    },
                }


#: The example splits are the only KIDBench rows committed to the repository, so they are
#: drawn from upstream's benign category rather than from whichever category sorts first.
#: Single turn is "Why does it rain?" in all five condition shapes; multi turn is "Why is the
#: sky blue?" at both age settings. One base question across five conditions also shows what
#: the grid is for: the same question, answered with and without a cue that a child is asking.
EXAMPLE_CATEGORY = "benign_information_seeking"
EXAMPLE_SINGLE_INDEX = 9
EXAMPLE_MULTI_INDEX = 0


def example_rows(single: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Five single-turn smoke rows, one per condition shape.

    One row per condition keeps the sample exercising every judge path — neutral, age
    conditioned, country (which adds ``cultural_alignment``), and non-English — rather
    than five near-identical English rows.

    Multi-turn rows are deliberately excluded. They are only meaningful under the
    child-actor agent, and a multi-turn row run through ``simple_agent`` produces a
    conversation with no child message at all, which the judge sees as malformed.
    ``example_multi_turn.jsonl`` carries that track's smoke rows instead.
    """
    wanted = ("no_cue", "implicit_cue", "explicit_age", "cultural", "cross_lingual")
    return [
        next(
            row
            for row in single
            if row["condition"] == condition
            and row["category"] == EXAMPLE_CATEGORY
            and row["example_index"] == EXAMPLE_SINGLE_INDEX
        )
        for condition in wanted
    ]


def example_multi_turn_rows(multi: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Both responder age settings, so the actor loop is smoke-tested in each."""
    return [
        next(
            row
            for row in multi
            if row["condition"] == setting
            and row["category"] == EXAMPLE_CATEGORY
            and row["example_index"] == EXAMPLE_MULTI_INDEX
        )
        for setting in ("without_age", "with_age")
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %s rows to %s", len(rows), path)


def build_track(track: str, upstream: Path) -> list[dict[str, Any]]:
    """Every row of one track, with the count checked against the published protocol."""
    if track == "single_turn":
        rows, expected = list(single_turn_rows(upstream)), len(CONDITION_GRID) * 500
    elif track == "multi_turn":
        rows, expected = list(multi_turn_rows(upstream)), len(MULTI_TURN_AGE_SETTINGS) * 100
    else:
        raise ValueError(f"Unknown KIDBench track {track!r}; expected 'single_turn' or 'multi_turn'")
    if len(rows) != expected:
        raise UpstreamError(f"Expected {expected} {track} rows, built {len(rows)}")
    return rows


#: Where ``gym eval prepare`` expects each benchmark's rows; must match ``jsonl_fpath`` in the
#: benchmark's ``config.yaml``, which the CLI checks after calling :func:`prepare`.
BENCHMARK_OUTPUTS: dict[str, Path] = {
    "single_turn": REPO_ROOT / "benchmarks" / "kidbench" / "data" / "kidbench_benchmark.jsonl",
    "multi_turn": REPO_ROOT / "benchmarks" / "kidbench_multi_turn" / "data" / "kidbench_multi_turn_benchmark.jsonl",
}


def prepare(track: str = "single_turn") -> Path:
    """Entry point for ``gym eval prepare --benchmark kidbench``.

    Writes one track's rows to the path its benchmark config declares and returns it. The
    rows are the same ones :func:`main` writes for that track, so a run through the
    benchmark config and a run through the resources-server config score identical inputs.
    """
    output = BENCHMARK_OUTPUTS.get(track)
    if output is None:
        raise ValueError(f"Unknown KIDBench track {track!r}; expected one of {sorted(BENCHMARK_OUTPUTS)}")
    write_jsonl(output, build_track(track, ensure_upstream()))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout-dir", type=Path, default=DEFAULT_CHECKOUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Use an existing checkout instead of cloning or updating it.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    upstream = resolve_upstream(args.checkout_dir) if args.no_fetch else ensure_upstream(args.checkout_dir)

    single = build_track("single_turn", upstream)
    multi = build_track("multi_turn", upstream)

    write_jsonl(args.output_dir / "single_turn.jsonl", single)
    write_jsonl(args.output_dir / "multi_turn.jsonl", multi)
    write_jsonl(args.output_dir / "example.jsonl", example_rows(single))
    write_jsonl(args.output_dir / "example_multi_turn.jsonl", example_multi_turn_rows(multi))


if __name__ == "__main__":
    main()
