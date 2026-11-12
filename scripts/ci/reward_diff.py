#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-score saved rollouts with a resources server's verifier at two revisions and report reward changes.

The verifier runs in-process on rollouts the server already commits (`data/example_rollouts.jsonl` by
default), so no model, GPU or running server is involved. A verifier that calls another server, such as
an LLM judge, cannot score offline; its rows are reported as errors rather than as reward changes.

    score    Run with the server's venv from the checkout to score; writes one reward per rollout.
    compare  Render a Markdown summary of the reward changes between two score files.
    run      CI driver: build the server venv, score the base and head checkouts, print the comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_ROLLOUTS = Path("data/example_rollouts.jsonl")


def _normalize(row: dict[str, Any], index: int, required: set[str]) -> dict[str, Any]:
    """Adapt a rollout saved by an older Gym to the current request schema."""
    row = dict(row)
    # Older rollouts serialize unset Responses parameters as null, which current models reject.
    params = row.get("responses_create_params") or {}
    row["responses_create_params"] = {key: value for key, value in params.items() if value is not None}
    # Older rollouts omit the task id that some verify requests now require.
    if "id" in required and "id" not in row:
        row["id"] = index
    return row


def score(server_dir: Path, rollouts: Path) -> dict[str, Any]:
    """Score every rollout with the verifier of the `server_dir` importable from the current checkout."""
    import asyncio
    import importlib
    import inspect
    import typing
    from unittest.mock import MagicMock

    import yaml

    from nemo_gym.base_resources_server import ReverifyMode, SimpleResourcesServer
    from nemo_gym.server_utils import ServerClient

    module = importlib.import_module(f"{server_dir.parent.name}.{server_dir.name}.app")
    server_classes = [
        value
        for value in vars(module).values()
        if inspect.isclass(value) and issubclass(value, SimpleResourcesServer) and value.__module__ == module.__name__
    ]
    if len(server_classes) != 1:
        return {"skipped": f"expected one resources server class in app.py, found {len(server_classes)}"}
    server_cls = server_classes[0]
    config_cls = server_cls.model_fields["config"].annotation
    reverify_mode = config_cls.REVERIFY_MODE
    if reverify_mode == ReverifyMode.UNSUPPORTED:
        return {"skipped": "the verifier reads per-rollout session state (reverify mode: unsupported)"}

    settings: dict[str, Any] = {}
    for config_path in sorted((server_dir / "configs").glob("*.yaml")):
        for block in (yaml.safe_load(config_path.read_text()) or {}).values():
            server_block = ((block or {}).get("resources_servers") or {}).get(server_dir.name)
            if server_block and not settings:
                settings = {key: value for key, value in server_block.items() if key in config_cls.model_fields}
    # Servers resolve relative config paths (prompt templates, data) from their own directory.
    os.chdir(server_dir)
    config = config_cls(
        **{"host": "127.0.0.1", "port": 0, "entrypoint": "app.py", "name": server_dir.name, **settings}
    )
    server = server_cls(config=config, server_client=MagicMock(spec=ServerClient))
    # get_type_hints resolves string annotations from modules that use `from __future__ import annotations`.
    request_cls = typing.get_type_hints(server_cls.verify)["body"]
    required = {name for name, field in request_cls.model_fields.items() if field.is_required()}

    async def score_rows() -> list[dict[str, Any]]:
        results = []
        for index, line in enumerate(rollouts.read_text(encoding="utf-8").splitlines()):
            row = json.loads(line)
            result: dict[str, Any] = {"row": index, "stored": row.get("reward")}
            try:
                request = request_cls.model_validate(_normalize(row, index, required))
                result["reward"] = float((await server.verify(request)).reward)
            except Exception as error:  # A row that cannot be scored is reported, not fatal.
                result["error"] = f"{type(error).__name__}: {str(error)[:200]}"
            results.append(result)
        return results

    return {"reverify_mode": reverify_mode.value, "rows": asyncio.run(score_rows())}


def _mean(rewards: list[float]) -> str:
    return f"{sum(rewards) / len(rewards):.3f}" if rewards else "n/a"


def compare(server: str, rollouts: str, base: dict[str, Any] | None, head: dict[str, Any]) -> str:
    """Render the reward changes between the base and head scores of one server as Markdown."""
    lines = [f"### Reward diff: `{server}`", ""]
    if "skipped" in head:
        return "\n".join(lines + [f"Skipped: {head['skipped']}.", ""])
    head_rows = {row["row"]: row for row in head["rows"]}
    base_rows = {row["row"]: row for row in (base or {}).get("rows", [])}
    lines.append(
        f"Re-scored {len(head_rows)} saved rollouts from `{rollouts}` in-process "
        f"(reverify mode: {head['reverify_mode']}). No model was called."
    )
    lines.append("")
    if base is None or "skipped" in base:
        reason = "the server is new in this PR" if base is None else base["skipped"]
        lines.append(f"No base comparison: {reason}.")
    head_rewards = [row["reward"] for row in head_rows.values() if "reward" in row]
    base_rewards = [row["reward"] for row in base_rows.values() if "reward" in row]
    stored_rewards = [row["stored"] for row in head_rows.values() if isinstance(row.get("stored"), (int, float))]
    lines += [
        "| | stored | base | head |",
        "| -- | -- | -- | -- |",
        f"| scored | {len(stored_rewards)} | {len(base_rewards) if base_rows else 'n/a'} | {len(head_rewards)} |",
        f"| mean reward | {_mean(stored_rewards)} | {_mean(base_rewards)} | {_mean(head_rewards)} |",
        f"| errors | | {sum('error' in row for row in base_rows.values()) if base_rows else 'n/a'} "
        f"| {sum('error' in row for row in head_rows.values())} |",
        "",
    ]
    # Compare against the reward recorded at collection time: if the base verifier cannot reproduce it, the saved
    # rollouts no longer replay faithfully (schema drift or verifier inputs missing from the rollout rows).
    drift_rows = base_rows or head_rows
    drift = [row for row in drift_rows.values() if "reward" in row and row.get("stored") not in (None, row["reward"])]
    if drift:
        side = "Base" if base_rows else "Head"
        lines += [
            f"> [!WARNING]\n> {side} scores differ from the stored rewards on {len(drift)} of {len(drift_rows)} "
            "rollouts, so these rollouts may not replay faithfully. Treat the comparison below with caution.",
            "",
        ]
    changed = [
        (index, base_rows[index].get("reward"), row.get("reward"))
        for index, row in sorted(head_rows.items())
        if index in base_rows and base_rows[index].get("reward") != row.get("reward")
    ]
    if not head_rewards:
        lines.append("Could not score any rollout offline; a verifier that calls a judge model needs one.")
    elif base_rows:
        lines.append(
            f"**{len(changed)} of {len(head_rows)} rollouts changed reward.**" if changed else "No reward changes."
        )
    if changed:
        lines += ["", "| rollout | base | head |", "| -- | -- | -- |"]
        lines += [f"| {index} | {old} | {new} |" for index, old, new in changed]
    errors = [row for row in head_rows.values() if "error" in row]
    if errors:
        lines += ["", f"Head could not score {len(errors)} rollouts; first error: `{errors[0]['error']}`"]
    lines.append("")
    return "\n".join(lines)


def _score_in(checkout: Path, python: Path, server: str, rollouts: Path) -> dict[str, Any] | None:
    if not (checkout / server / "app.py").is_file():
        return None
    with tempfile.NamedTemporaryFile(suffix=".json") as output:
        command = [str(python), str(Path(__file__).resolve()), "score", server, "--rollouts", str(rollouts)]
        command += ["--output", output.name]
        env = {**os.environ, "PYTHONPATH": str(checkout)}
        result = subprocess.run(command, cwd=checkout, env=env, capture_output=True, text=True, errors="replace")
        if result.returncode != 0:
            return {"skipped": f"scoring failed: {(result.stderr.strip().splitlines() or ['no output'])[-1][:200]}"}
        return json.loads(Path(output.name).read_text())


def run(server: str, base_ref: str, rollouts: Path) -> str:
    """Build the server venv from the head checkout and compare base and head scores of the same rollouts."""
    repo = Path(__file__).resolve().parents[2]
    rollouts_path = (repo / server / rollouts).resolve()
    if not rollouts_path.is_file():
        return compare(server, str(rollouts), None, {"skipped": f"no saved rollouts at `{server}/{rollouts}`"})
    with tempfile.TemporaryDirectory(prefix="reward-diff-") as work:
        venv, base_checkout = Path(work) / "venv", Path(work) / "base"
        server_dir = repo / server
        subprocess.run(["uv", "venv", "--quiet", "--python", sys.executable, str(venv)], check=True)
        install = ["-r", "requirements.txt"] if (server_dir / "requirements.txt").is_file() else ["-e", "."]
        subprocess.run(
            ["uv", "pip", "install", "--quiet", "--python", str(venv / "bin/python"), *install],
            cwd=server_dir,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--quiet", "--detach", str(base_checkout), base_ref],
            check=True,
        )
        try:
            head = _score_in(repo, venv / "bin/python", server, rollouts_path)
            base = _score_in(base_checkout, venv / "bin/python", server, rollouts_path)
        finally:
            subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(base_checkout)], check=False)
    return compare(server, str(rollouts), base, head or {"skipped": "the server has no app.py in this PR"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("server", help="Server directory relative to the checkout, e.g. resources_servers/mcqa")
    score_parser.add_argument("--rollouts", type=Path, help=f"Rollouts JSONL (default: <server>/{DEFAULT_ROLLOUTS})")
    score_parser.add_argument("--output", type=Path, required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("server")
    compare_parser.add_argument("--base", type=Path, help="Base score file; omit for a new server")
    compare_parser.add_argument("--head", type=Path, required=True)
    compare_parser.add_argument("--rollouts", default=str(DEFAULT_ROLLOUTS))
    run_parser = commands.add_parser("run")
    run_parser.add_argument("server")
    run_parser.add_argument("--base-ref", required=True)
    run_parser.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS, help="Relative to the server directory")
    args = parser.parse_args()

    if args.command == "score":
        # The server venv installs the head checkout as an editable `nemo-gym`, and `resources_servers` is a
        # namespace package, so drop the editable finder: every Gym import must come from the checkout scored.
        checkout = os.getcwd()
        sys.meta_path[:] = [
            finder for finder in sys.meta_path if not type(finder).__module__.startswith("__editable__")
        ]
        sys.path[:] = [checkout] + [
            entry for entry in sys.path if "__editable__" not in entry and entry not in ("", checkout)
        ]
        server_dir = Path(args.server)
        output = args.output.resolve()
        result = score(server_dir, (args.rollouts or server_dir / DEFAULT_ROLLOUTS).resolve())
        output.write_text(json.dumps(result), encoding="utf-8")
        return
    if args.command == "compare":
        base = json.loads(args.base.read_text()) if args.base else None
        report = compare(args.server, args.rollouts, base, json.loads(args.head.read_text()))
    else:
        report = run(args.server, args.base_ref, args.rollouts)
    print(report)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(report + "\n")


if __name__ == "__main__":
    main()
