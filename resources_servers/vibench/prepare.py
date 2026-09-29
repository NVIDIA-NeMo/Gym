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

P0 covers ``mvp`` artifacts only. ``--artifacts`` accepts feature artifacts and resolves
their PRD chain and test plans, but the environment cannot stage a starting codebase into
the build sandbox yet, which is what a feature task builds on top of.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


# ViBench's own build brief. It is a contract, not flavour text: it requires the app to ship
# setup-environment.sh and start-server.sh, and describes the environment (POSTGRES_DATABASE_URL,
# APPLICATION_PORT) the grader provides. The grading stack invokes setup-environment.sh from the
# generated seed.sh, so an app built without it fails evaluation with exit code 127 no matter how
# good the app is. Writing our own brief would also change what the benchmark measures.
CODING_PROMPT = "coding_prompt.j2"
# ViBench's own goal identifiers (_harness/runner/agent/models.py); the template branches on
# them to say "create from scratch" versus "extend what is here".
ZERO_TO_ONE = "zero-to-one"
FEATURE_BUILDING = "feature-building"

RENDER_SNIPPET = """
import sys, json
from jinja2 import Environment, FileSystemLoader, StrictUndefined
prompts_dir, prd_path, max_iterations, goal = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
env = Environment(loader=FileSystemLoader(prompts_dir), undefined=StrictUndefined, keep_trailing_newline=True)
tpl = env.get_template("coding_prompt.j2")
sys.stdout.write(tpl.render(
    goal=goal,
    prd=open(prd_path).read(),
    max_iterations=max_iterations,
    additional_instructions="",
))
"""


def vibench_python(root: Path) -> str:
    """ViBench's own interpreter, which has jinja2; fall back to this one."""
    candidate = root / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def render_task_prompt(root: Path, prd_text: str, max_iterations: int, artifact: str) -> Optional[str]:
    """Render ViBench's coding prompt, or None if the checkout cannot render it.

    ``goal`` follows the artifact: ViBench's template branches on it, and rendering
    ``zero-to-one`` for a feature artifact tells the model to build from scratch a task that
    is supposed to extend an existing codebase.
    """
    goal = ZERO_TO_ONE if artifact.split("-on_")[0] == "mvp" else FEATURE_BUILDING
    prompts_dir = root / "_harness" / "runner" / "agent" / "prompts"
    if not (prompts_dir / CODING_PROMPT).exists():
        return None

    # A fixed name inside the checkout would race between concurrent prepares and fail on a
    # read-only checkout.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(prd_text)
        tmp = Path(fh.name)
    try:
        result = subprocess.run(
            [vibench_python(root), "-c", RENDER_SNIPPET, str(prompts_dir), str(tmp), str(max_iterations), goal],
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        tmp.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"WARNING: could not render {CODING_PROMPT}: {result.stderr.strip()[-300:]}", file=sys.stderr)
        return None
    return result.stdout


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
    system_prompt: Optional[str],
    max_iterations: int,
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
    # Grader-only fixtures the evaluation agent uploads while driving the app.
    test_assets = app_dir / "test_assets"
    test_assets_dir = str(test_assets.relative_to(root)) if test_assets.is_dir() else None

    prd_text = "\n\n".join(pth.read_text() for pth in prds)
    prompt = render_task_prompt(root, prd_text, max_iterations, artifact)
    if prompt is None:
        return None
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
        "test_assets_dir": test_assets_dir,
        "responses_create_params": {"input": messages},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vibench-root", required=True, help="Path to a ViBench checkout")
    parser.add_argument("--output", required=True, help="Destination .jsonl")
    parser.add_argument("--apps", nargs="*", default=None, help="App names (default: every app in prds/)")
    parser.add_argument("--artifacts", nargs="*", default=["mvp"], help="Artifacts per app (default: mvp)")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-iterations", type=int, default=300, help="Value passed to ViBench's prompt")
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
            row = build_row(root, app, artifact, args.system_prompt, args.max_iterations)
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
