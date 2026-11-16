# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mix the member environments' JSONL rows into one dataset for the unified video run.

Each member is a Gym dataset whose rows already carry ``responses_create_params`` and ``verifier_metadata``
(the member's own ``prepare`` or data pipeline wrote them). This script only stamps the member's ``agent_ref``
on every row, so rollout collection and training route the row to that member's agent, and interleaves the
members in one seeded order. It never reads or rewrites media.

    python environments/unified_video_agentic/prepare.py \\
        --member video_caption_simple_agent=/data/video_caption_train.jsonl \\
        --member videomme_v1_single_turn=/data/videomme_v1_train.jsonl:0.5 \\
        --output /data/unified_video_train.jsonl

``NAME=PATH[:WEIGHT]``: ``NAME`` is the agent instance name in the Gym config, ``WEIGHT`` (default 1) scales
how many of the member's rows are kept (``round(weight * rows)``, sampled with the seed; a weight above 1
repeats rows). ``--taskset PATH[:WEIGHT]`` adds the materialized rows of a Harbor task-folder member (the
``tasks.jsonl`` a run of ``environments/harbor/agentic_vbench_train`` or ``claweval_train`` wrote): they already
name their taskset and route to its environment server, so they get no ``agent_ref``. ``--example`` writes
``data/example.jsonl`` from the JSONL members' committed example rows.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXAMPLE_MEMBERS = {
    "video_caption_simple_agent": REPO / "resources_servers/video_caption/data/example.jsonl",
    "videomme_v1_single_turn": REPO / "resources_servers/videomme_v1/data/example.jsonl",
}


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            params = row.get("responses_create_params")
            if not isinstance(params, dict) or not params.get("input"):
                raise ValueError(f"{path}:{line_number}: row has no responses_create_params.input")
            rows.append(row)
    return rows


def read_materialized_rows(path: Path) -> list[dict]:
    """The rows ``gym eval run <harbor entry>`` materialized: each names its taskset and carries its task input."""
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            taskset = row.get("task_id", {}).get("taskset") if isinstance(row.get("task_id"), dict) else None
            if not taskset or "task_input" not in row:
                raise ValueError(f"{path}:{line_number}: not a materialized task row (task_id.taskset and task_input)")
            if row.get("agent_ref"):
                raise ValueError(f"{path}:{line_number}: a materialized row routes by taskset, not agent_ref")
            rows.append(row)
    return rows


def stamp(rows: list[dict], agent: str) -> list[dict]:
    """Return the rows with ``agent_ref`` set to ``agent``; a row that already names another agent is an error."""
    stamped = []
    for row in rows:
        existing = row.get("agent_ref")
        if isinstance(existing, dict) and existing.get("name") not in (None, agent):
            raise ValueError(f"row already routes to {existing.get('name')!r}, not {agent!r}")
        stamped.append({**row, "agent_ref": {"type": "responses_api_agents", "name": agent}})
    return stamped


def take(rows: list[dict], weight: float, rng: random.Random) -> list[dict]:
    """``round(weight * len(rows))`` rows: a subsample below 1, whole copies plus a subsample above 1."""
    if weight < 0:
        raise ValueError("weight must be non-negative")
    whole, fraction = divmod(weight, 1)
    picked = rows * int(whole)
    extra = round(fraction * len(rows))
    if extra:
        picked = picked + rng.sample(rows, extra)
    return picked


def mix(members: list[tuple[str | None, Path, float]], *, seed: int = 0) -> tuple[list[dict], dict]:
    """Interleave the members in one seeded order. A member with agent ``None`` is a Harbor taskset's rows file."""
    rng = random.Random(seed)
    mixed: list[dict] = []
    counts: Counter[str] = Counter()
    for agent, path, weight in members:
        if agent is None:
            rows = take(read_materialized_rows(path), weight, rng)
            for row in rows:
                counts[f"taskset:{row['task_id']['taskset']}"] += 1
        else:
            rows = take(stamp(read_rows(path), agent), weight, rng)
            counts[agent] += len(rows)
        mixed.extend(rows)
    rng.shuffle(mixed)
    stats = {
        "rows": len(mixed),
        "per_agent": dict(counts),
        "seed": seed,
        "members": [{"agent": a, "path": str(p), "weight": w} for a, p, w in members],
    }
    return mixed, stats


def write(rows: list[dict], stats: dict | None, output: Path) -> None:
    """Write the rows; with ``stats``, also ``<output stem>_mix.json`` beside them."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if stats is not None:
        output.with_name(output.stem + "_mix.json").write_text(json.dumps(stats, indent=1) + "\n")


def _parse_weighted_path(value: str) -> tuple[Path, float]:
    path, sep, weight = value.rpartition(":")
    if not sep or not weight.replace(".", "", 1).isdigit():
        path, weight = value, "1"
    return Path(path), float(weight)


def parse_member(value: str) -> tuple[str, Path, float]:
    agent, _, rest = value.partition("=")
    if not agent or not rest:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH[:WEIGHT], got {value!r}")
    return (agent, *_parse_weighted_path(rest))


def parse_taskset(value: str) -> tuple[None, Path, float]:
    if not value or "=" in value.split(":")[0]:
        raise argparse.ArgumentTypeError(f"expected PATH[:WEIGHT], got {value!r}")
    return (None, *_parse_weighted_path(value))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--member", action="append", type=parse_member, default=[], metavar="NAME=PATH[:WEIGHT]")
    parser.add_argument(
        "--taskset",
        action="append",
        type=parse_taskset,
        default=[],
        metavar="PATH[:WEIGHT]",
        help="a Harbor member's materialized rows (results/harbor/<entry>/tasks.jsonl); routed by taskset",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--example", action="store_true", help="write data/example.jsonl from the members' examples")
    args = parser.parse_args(argv)
    if args.example:
        # 3 Video-MME rows + 2 caption rows: five rows, like every example dataset, and both routes exercised.
        members = [
            ("videomme_v1_single_turn", EXAMPLE_MEMBERS["videomme_v1_single_turn"], 0.6),
            ("video_caption_simple_agent", EXAMPLE_MEMBERS["video_caption_simple_agent"], 0.4),
        ]
        output = args.output or HERE / "data/example.jsonl"
    else:
        if not (args.member or args.taskset) or args.output is None:
            parser.error("--member or --taskset, and --output, are required unless --example is given")
        members, output = args.member + args.taskset, args.output
    rows, stats = mix(members, seed=args.seed)
    write(rows, None if args.example else stats, output)  # example data carries no sidecar beside it
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
