# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Generate NeMo Gym task rows from a ViBench checkout.

One row = one ``(app, artifact)`` pair. Its reward is the mean normalized score across
that artifact's test plans, so the app is built once and graded N times -- emitting a row
per test plan instead would rebuild the same app for every plan.

    python resources_servers/vibench/prepare.py \
        --vibench-root ~/projects/vibench/repo \
        --output resources_servers/vibench/data/example.jsonl \
        --limit 5

P0 covers ``mvp`` artifacts only. Feature artifacts (``feature1``, ``feature1-on_mvp``)
need a reference-implementation starting tree that the public ViBench repo does not ship;
``--artifacts`` accepts them so the wiring is testable once those land.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


TASK_INSTRUCTIONS = """\
You are building a web application from scratch inside this container.

The product requirements document is at {prd_path}. Read it first.

Build the complete application in {workdir}. The app must actually run: it will be started \
by an automated harness that seeds it with data through its own UI and then exercises every \
requirement in the PRD through a real browser. Anything the PRD asks for that a user cannot \
reach through the running app scores zero, no matter what the source code contains.

Do not stop until the app builds and starts cleanly."""


def artifact_test_dir(app_dir: Path, artifact: str) -> Path:
    """Test plans for ``featureN-on_mvp`` live in the base ``featureN`` folder."""
    base = artifact.split("-on_")[0]
    return app_dir / "tests" / base


def prd_chain(app_dir: Path, artifact: str) -> List[Path]:
    """PRDs the agent needs, in order.

    A feature artifact is built on top of the MVP, so the MVP PRD is prepended -- this
    mirrors how ViBench's build-feature path presents prior context.
    """
    base = artifact.split("-on_")[0]
    if base == "mvp":
        return [app_dir / "prd" / "mvp.txt"]
    return [app_dir / "prd" / "mvp.txt", app_dir / "prd" / f"{base}.txt"]


def discover_artifacts(app_dir: Path) -> List[str]:
    prd_dir = app_dir / "prd"
    if not prd_dir.is_dir():
        return []
    names = sorted(f.stem for f in prd_dir.iterdir() if f.is_file() and f.suffix in {".txt", ".md"})
    return ["mvp"] + [n for n in names if n != "mvp"] if "mvp" in names else names


def build_row(
    root: Path,
    app: str,
    artifact: str,
    workdir: str,
    system_prompt: Optional[str],
) -> Optional[Dict]:
    app_dir = root / "prds" / app

    prds = prd_chain(app_dir, artifact)
    if not all(p.exists() for p in prds):
        return None

    test_dir = artifact_test_dir(app_dir, artifact)
    if not test_dir.is_dir():
        return None
    test_plans = sorted(p for p in test_dir.iterdir() if p.is_file() and p.suffix == ".txt")
    if not test_plans:
        return None

    # Static fixtures the PRD refers to (CSV lookups and the like). test_assets/ is
    # deliberately excluded: those belong to the grader, not the builder.
    asset_dirs = [str((app_dir / "assets").relative_to(root))] if (app_dir / "assets").is_dir() else []

    prompt = TASK_INSTRUCTIONS.format(prd_path=f"{workdir}/prd.txt", workdir=workdir)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    return {
        "app": app,
        "artifact": artifact,
        "prd_files": [str(p.relative_to(root)) for p in prds],
        "test_plans": [str(p.relative_to(root)) for p in test_plans],
        "asset_dirs": asset_dirs,
        "responses_create_params": {"input": messages},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vibench-root", required=True, help="Path to a ViBench checkout")
    parser.add_argument("--output", required=True, help="Destination .jsonl")
    parser.add_argument("--apps", nargs="*", default=None, help="App names (default: every app in prds/)")
    parser.add_argument("--artifacts", nargs="*", default=["mvp"], help="Artifacts per app (default: mvp)")
    parser.add_argument("--workdir", default="/app", help="Build directory inside the sandbox")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.vibench_root).expanduser().resolve()
    prds_dir = root / "prds"
    if not prds_dir.is_dir():
        raise SystemExit(f"No prds/ directory under {root}")

    apps = args.apps or sorted(d.name for d in prds_dir.iterdir() if d.is_dir())

    rows: List[Dict] = []
    for app in apps:
        available = discover_artifacts(prds_dir / app)
        for artifact in args.artifacts:
            if artifact.split("-on_")[0] not in available:
                continue
            row = build_row(root, app, artifact, args.workdir, args.system_prompt)
            if row is not None:
                rows.append(row)

    if args.limit is not None:
        rows = rows[: args.limit]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    plans = sum(len(r["test_plans"]) for r in rows)
    print(f"Wrote {len(rows)} task(s) covering {plans} test plan(s) to {out}")


if __name__ == "__main__":
    main()
