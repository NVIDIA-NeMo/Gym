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
"""Build the visual agent dataset from the task specs in data/tasks/.

    python resources_servers/visual_agent/build_dataset.py validate
    python resources_servers/visual_agent/build_dataset.py references [--only ID ...]   # needs OpenSandbox
    python resources_servers/visual_agent/build_dataset.py rows --out data/visual_agent_tasks.jsonl

Replication tasks have a hidden golden artifact under data/golden/<task_id>/. Figma tasks
generate theirs from data/assets/<task_id>/design.json. `references` renders each golden
in the task image with the grader's renderer, writes the reference images to
data/assets/<task_id>/, and records a determinism check in data/assets/<task_id>/reference_info.json.
`rows` needs no sandbox: it writes one JSONL row per task, with the reference images also
attached to the prompt as input_image parts.
"""

import argparse
import asyncio
import base64
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


SERVER_DIR = Path(__file__).parent
REPO_ROOT = SERVER_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resources_servers.visual_agent.app import VisualTask  # noqa: E402
from resources_servers.visual_agent.figma_render import count_nodes, design_to_html  # noqa: E402
from resources_servers.visual_agent.prompts import policy_prompt  # noqa: E402


DATA_DIR = SERVER_DIR / "data"
TASKS_DIR = DATA_DIR / "tasks"
GOLDEN_DIR = DATA_DIR / "golden"
ASSETS_DIR = DATA_DIR / "assets"
VENDOR_DIR = DATA_DIR / "vendor"
# Per-task render facts and determinism check, next to the rendered references.
REFERENCE_INFO_NAME = "reference_info.json"
DEFAULT_IMAGE = "apify/actor-python-playwright:3.12-1.63.0"
EXPECTED_COUNTS = {
    "website": 30,
    "interactive_app": 28,
    "game": 24,
    "3d_scene": 22,
    "slides": 24,
    "svg": 28,
    "video": 20,
    "figma": 24,
}
# Authoring-only keys that are not part of the dataset row.
AUTHORING_KEYS = {"golden"}


LOAD_ERRORS: List[str] = []


def load_specs(tasks_dir: Path = TASKS_DIR) -> List[Dict[str, Any]]:
    """Load every task spec; unparseable files are reported by `validate` instead of aborting."""
    specs: List[Dict[str, Any]] = []
    for path in sorted(tasks_dir.glob("*.json")):
        try:
            loaded = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            LOAD_ERRORS.append(f"{path.name}: invalid JSON: {exc}")
            continue
        for spec in loaded if isinstance(loaded, list) else [loaded]:
            spec["_source"] = path.name
            specs.append(spec)
    return specs


def validate(specs: List[Dict[str, Any]], *, require_references: bool = False) -> List[str]:
    problems: List[str] = list(LOAD_ERRORS)
    ids = Counter(s.get("task_id") for s in specs)
    problems += [f"duplicate task_id {i}" for i, n in ids.items() if n > 1]
    for spec in specs:
        tid = spec.get("task_id", "?")
        try:
            task = VisualTask.model_validate({k: v for k, v in spec.items() if k not in AUTHORING_KEYS | {"_source"}})
        except Exception as exc:
            problems.append(f"{tid}: schema: {exc}")
            continue
        if not str(spec.get("prompt", "")).strip():
            problems.append(f"{tid}: empty prompt")
        if '"' in spec.get("prompt", ""):
            problems.append(
                f"{tid}: prompt contains double quotes (OpenCode escapes them); use single or curly quotes"
            )
        rubric_ids = [r.id for r in task.rubric]
        if len(set(rubric_ids)) != len(rubric_ids):
            problems.append(f"{tid}: duplicate rubric ids")
        if not 4 <= len(task.rubric) <= 12:
            problems.append(f"{tid}: rubric has {len(task.rubric)} items (want 4-12)")
        if any(r.id.startswith("AUTO-") for r in task.rubric):
            problems.append(f"{tid}: rubric ids starting with AUTO- are reserved")
        for name in task.assets:
            if not (ASSETS_DIR / tid / name).is_file():
                problems.append(f"{tid}: asset {name} missing")
        if task.artifact.kind == "video" and not task.video_spec:
            problems.append(f"{tid}: video tasks need video_spec")
        if task.mode == "replication":
            if not task.reference_images:
                problems.append(f"{tid}: replication task without reference_images")
            if task.category == "figma":
                if not (ASSETS_DIR / tid / "design.json").is_file():
                    problems.append(f"{tid}: figma replication needs assets/{tid}/design.json")
            elif not (GOLDEN_DIR / tid).is_dir():
                problems.append(f"{tid}: replication task without data/golden/{tid}/")
            if task.artifact.kind == "video" and len(task.reference_frame_times or []) != len(task.reference_images):
                problems.append(f"{tid}: reference_frame_times must match reference_images")
            if task.reference_viewports is not None and (
                task.artifact.kind != "html" or len(task.reference_viewports) != len(task.reference_images)
            ):
                problems.append(f"{tid}: reference_viewports needs an html artifact and one viewport per reference")
            if require_references:
                for name in task.reference_images:
                    if not (ASSETS_DIR / tid / name).is_file():
                        problems.append(f"{tid}: reference {name} not rendered yet")
        elif task.reference_images:
            problems.append(f"{tid}: open-ended task with reference_images")
    return problems


def build_figma_goldens(specs: List[Dict[str, Any]]) -> None:
    for spec in specs:
        if spec["category"] != "figma" or spec["mode"] != "replication":
            continue
        design = json.loads((ASSETS_DIR / spec["task_id"] / "design.json").read_text())
        out = GOLDEN_DIR / spec["task_id"] / "index.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(design_to_html(design))
        print(f"{spec['task_id']}: golden from design.json {count_nodes(design)}")


# ----------------------------------------------------------------------------------------------
# References (OpenSandbox)
# ----------------------------------------------------------------------------------------------
def _sandbox_global_config(env_yaml: Path) -> Dict[str, Any]:
    from omegaconf import OmegaConf

    base = OmegaConf.load(REPO_ROOT / "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml")
    env = OmegaConf.load(env_yaml)
    merged = OmegaConf.to_container(OmegaConf.merge(base, {"sandbox": env.get("sandbox", {})}), resolve=False)
    connection = merged["sandbox"]["opensandbox"]["connection"]
    for key, value in list(connection.items()):
        if isinstance(value, str) and value.startswith("${"):
            connection.pop(key)
    return merged


async def _render_one(
    spec: Dict[str, Any], global_config: Dict[str, Any], semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    from shlex import quote

    from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider
    from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
    from resources_servers.visual_agent.opencode_runner import exec_checked, list_files, upload_tree

    tid = spec["task_id"]
    async with semaphore:
        provider = create_provider(resolve_provider_config("sandbox", global_config))
        sandbox_spec = SandboxSpec(
            image=spec.get("docker_image") or DEFAULT_IMAGE,
            ttl_s=3600,
            ready_timeout_s=1200,
            env={"EXECD_API_GRACE_SHUTDOWN": "50ms"},
            metadata=resolve_provider_metadata("sandbox", global_config) | {"benchmark": "visual-agent-build"},
            resources=SandboxResources.from_mapping({"cpu": 2, "memory_mib": 4096, "disk_gib": 30}),
            provider_options={"resource_requests": {"cpu": 0.5, "memory_mib": 1024, "disk_gib": 30}},
        )
        sandbox = AsyncSandbox(provider)
        setup = f"bash -c {quote((SERVER_DIR / 'sandbox_tools' / 'setup_sandbox.sh').read_text())}"

        async def run_setup(sb: AsyncSandbox) -> None:
            await exec_checked(sb, setup, timeout_s=900, what="setup")

        await sandbox.start_with_setup(sandbox_spec, run_setup)
        try:
            files = {"vtools.py": SERVER_DIR / "sandbox_tools" / "vtools.py"}
            files |= list_files(GOLDEN_DIR / tid, prefix="golden/")
            for lib in spec.get("vendor") or []:
                files |= list_files(VENDOR_DIR / lib, prefix=f"golden/vendor/{lib}/")
            for name in spec.get("assets") or []:
                files[f"golden/{name}"] = ASSETS_DIR / tid / name
            with tempfile.TemporaryDirectory() as tmp:
                task_json = Path(tmp) / "task.json"
                task_json.write_text(json.dumps({k: v for k, v in spec.items() if k != "_source"}))
                files["task.json"] = task_json
                await upload_tree(sandbox, files, "/build")
            output = await exec_checked(
                sandbox,
                "cd /build && python3 vtools.py reference --task task.json --golden-dir golden --out-dir refs",
                timeout_s=1200,
                what=f"render references for {tid}",
            )
            info = json.loads(output)
            await exec_checked(sandbox, "cd /build/refs && tar czf /tmp/refs.tgz .", what="pack references")
            with tempfile.TemporaryDirectory() as tmp:
                local = Path(tmp) / "refs.tgz"
                await sandbox.download("/tmp/refs.tgz", local)
                import tarfile

                with tarfile.open(local) as tar:
                    for member in tar.getmembers():
                        if member.isfile() and Path(member.name).name in spec["reference_images"]:
                            (ASSETS_DIR / tid).mkdir(parents=True, exist_ok=True)
                            data = tar.extractfile(member).read()
                            (ASSETS_DIR / tid / Path(member.name).name).write_bytes(data)
            print(f"{tid}: self-similarity {info.get('self_similarity')}", flush=True)
            return info
        finally:
            await sandbox.stop()


async def render_references(specs: List[Dict[str, Any]], env_yaml: Path, concurrency: int) -> None:
    global_config = _sandbox_global_config(env_yaml)
    semaphore = asyncio.Semaphore(concurrency)
    targets = [s for s in specs if s["mode"] == "replication"]
    results = await asyncio.gather(
        *(_render_one(s, global_config, semaphore) for s in targets), return_exceptions=True
    )
    failed = []
    for spec, result in zip(targets, results):
        if isinstance(result, BaseException):
            failed.append(f"{spec['task_id']}: {type(result).__name__}: {str(result)[:800]}")
        else:
            info_path = ASSETS_DIR / spec["task_id"] / REFERENCE_INFO_NAME
            info_path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for line in failed:
        print("FAILED", line)
    if failed:
        raise SystemExit(1)


# ----------------------------------------------------------------------------------------------
# Rows
# ----------------------------------------------------------------------------------------------
def _data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def build_row(spec: Dict[str, Any]) -> Dict[str, Any]:
    task = {k: v for k, v in spec.items() if k not in AUTHORING_KEYS | {"_source"}}
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": policy_prompt(task)}]
    for name in task.get("reference_images") or []:
        content.append(
            {"type": "input_image", "image_url": _data_url(ASSETS_DIR / task["task_id"] / name), "detail": "high"}
        )
    return {"responses_create_params": {"input": [{"role": "user", "content": content}]}, **task}


def write_rows(specs: List[Dict[str, Any]], out: Path, only: Optional[List[str]] = None) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w") as f:
        for spec in specs:
            if only and spec["task_id"] not in only:
                continue
            f.write(json.dumps(build_row(spec)) + "\n")
            count += 1
    return count


EXAMPLE_TASK_IDS = ["website-open-01", "game-open-01", "svg-rep-01", "slides-open-01", "video-open-01"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    p = sub.add_parser("references")
    p.add_argument("--only", nargs="*")
    p.add_argument("--env-yaml", default=str(REPO_ROOT / "env.yaml"))
    p.add_argument("--concurrency", type=int, default=16)
    p = sub.add_parser("rows")
    p.add_argument("--out", default=str(DATA_DIR / "visual_agent_tasks.jsonl"))
    p.add_argument("--only", nargs="*")
    p.add_argument("--example", action="store_true", help=f"write data/example.jsonl ({', '.join(EXAMPLE_TASK_IDS)})")
    args = parser.parse_args()
    # Gym's global config (used by the sandbox provider) parses sys.argv as Hydra overrides.
    sys.argv = sys.argv[:1]

    specs = load_specs()
    if args.command == "validate":
        problems = validate(specs)
        counts = Counter((s["category"], s["mode"]) for s in specs)
        for (category, mode), n in sorted(counts.items()):
            print(f"{category:16s} {mode:12s} {n}")
        by_category = Counter(s["category"] for s in specs)
        for category, expected in EXPECTED_COUNTS.items():
            if by_category.get(category, 0) != expected:
                problems.append(f"{category}: {by_category.get(category, 0)} tasks, expected {expected}")
        print(f"{len(specs)} tasks")
        for problem in problems:
            print("PROBLEM", problem)
        raise SystemExit(1 if problems else 0)
    if args.command == "references":
        targets = [s for s in specs if not args.only or s["task_id"] in args.only]
        build_figma_goldens(targets)
        asyncio.run(render_references(targets, Path(args.env_yaml), args.concurrency))
    elif args.command == "rows":
        if args.only:
            specs = [s for s in specs if s["task_id"] in args.only]
        elif args.example:
            specs = [s for s in specs if s["task_id"] in EXAMPLE_TASK_IDS]
        problems = validate(specs, require_references=True)
        if problems:
            raise SystemExit("Fix these first:\n" + "\n".join(problems))
        if args.example:
            n = write_rows(specs, DATA_DIR / "example.jsonl", EXAMPLE_TASK_IDS)
            print(f"wrote {n} rows to {DATA_DIR / 'example.jsonl'}")
        else:
            n = write_rows(specs, Path(args.out), args.only)
            print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
