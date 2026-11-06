# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapt the official BenchFlow SkillsBench task set to Harbor's local task layout."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
TASKS_DIR = DATA_DIR / "tasks"
OUTPUT_FPATH = DATA_DIR / "skillsbench_official.jsonl"
SOURCE_ENV_VAR = "SKILLSBENCH_TASKS_DIR"
SKILL_MODE_ENV_VAR = "SKILLSBENCH_SKILL_MODE"
IMAGE_MANIFEST_ENV_VAR = "SKILLSBENCH_IMAGE_MANIFEST"
AGENT_NAME = "skillsbench_harbor_agent"

# The official 87-task revision; images.json pins its published task images by digest.
SOURCE_REPO = "https://github.com/benchflow-ai/skillsbench.git"
SOURCE_REVISION = "12ebd8653093e0734c7902835cb20c054196026c"
SOURCE_CHECKOUT = DATA_DIR / "source"
GYM_ROOT = BENCHMARK_DIR.parent.parent


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _fetch_official_tasks() -> Path:
    """Sparse-clone the pinned official tasks/ tree, reusing a checkout already at that revision."""
    checkout, revision = SOURCE_CHECKOUT, SOURCE_REVISION
    if (checkout / ".git").is_dir() and _git("rev-parse", "HEAD", cwd=checkout) == revision:
        return checkout / "tasks"
    if checkout.exists():
        shutil.rmtree(checkout)
    print(f"Cloning official SkillsBench tasks at {revision} into {checkout}")
    _git("clone", "--quiet", "--filter=blob:none", "--no-checkout", SOURCE_REPO, str(checkout))
    _git("sparse-checkout", "set", "--no-cone", "/tasks/**", cwd=checkout)
    _git("checkout", "--quiet", revision, cwd=checkout)
    return checkout / "tasks"


def _task_definition(path: Path) -> tuple[dict, str]:
    content = path.read_text()
    pieces = content.split("---", 2)
    if len(pieces) != 3 or pieces[0].strip():
        raise ValueError(f"Expected YAML frontmatter in {path}")
    return yaml.safe_load(pieces[1]), pieces[2].lstrip("\r\n")


def _toml_value(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value)


def _task_toml(name: str, definition: dict) -> str:
    sandbox = definition.get("sandbox") or definition["environment"]
    verifier = definition["verifier"]
    agent = definition["agent"]
    oracle = definition.get("oracle") or {}
    metadata = definition.get("metadata") or {}
    sections = {
        "metadata": {
            "task_name": name,
            "difficulty": metadata.get("difficulty", "unknown"),
            "category": metadata.get("category", "unknown"),
        },
        "verifier": {"timeout_sec": verifier["timeout_sec"]},
        "agent": {"timeout_sec": agent["timeout_sec"]},
        "environment": {
            "build_timeout_sec": sandbox["build_timeout_sec"],
            "cpus": sandbox["cpus"],
            "memory_mb": sandbox["memory_mb"],
            "storage_mb": sandbox["storage_mb"],
            "gpus": sandbox["gpus"],
            "allow_internet": sandbox["network_mode"] != "no-network",
        },
    }
    lines = ['version = "1.0"']
    for section, fields in sections.items():
        lines.extend(("", f"[{section}]"))
        lines.extend(f"{key} = {_toml_value(value)}" for key, value in fields.items())
    for section, env in (("verifier", verifier.get("env")), ("solution", oracle.get("env"))):
        if env:
            lines.extend(("", f"[{section}.env]"))
            lines.extend(f"{json.dumps(key)} = {_toml_value(value)}" for key, value in env.items())
    return "\n".join(lines) + "\n"


def _adapt_task(source: Path, destination: Path, skill_mode: str, docker_image: str | None = None) -> None:
    definition, instruction = _task_definition(source / "task.md")
    if definition.get("schema_version") != "1.3" or definition["verifier"].get("type") != "test-script":
        raise ValueError(f"Unsupported SkillsBench task format: {source}")
    sandbox = definition.get("sandbox") or definition["environment"]
    if (
        sandbox.get("os") != "linux"
        or sandbox.get("gpus") != 0
        or sandbox["network_mode"]
        not in {
            "public",
            "no-network",
        }
    ):
        raise ValueError(f"Unsupported SkillsBench sandbox: {source}")

    destination.mkdir()
    (destination / "instruction.md").write_text(instruction)
    task_toml = _task_toml(source.name, definition)
    if docker_image:
        task_toml = task_toml.replace(
            "[environment]\n", f"[environment]\ndocker_image = {_toml_value(docker_image)}\n", 1
        )
    (destination / "task.toml").write_text(task_toml)
    shutil.copytree(source / "environment", destination / "environment")
    if skill_mode == "with-skill":
        dockerfile = destination / "environment" / "Dockerfile"
        with dockerfile.open("a") as output:
            output.write(
                "\n# Install SkillsBench skills for the agent.\n"
                "COPY skills/ /root/.agents/skills/\n"
                "COPY skills/ /root/.codex/skills/\n"
                "COPY skills/ /root/.claude/skills/\n"
                "COPY skills/ /root/.gemini/skills/\n"
                "COPY skills/ /root/.opencode/skills/\n"
            )
    shutil.copytree(source / "verifier", destination / "tests")
    shutil.copytree(source / "oracle", destination / "solution")

    test_script = destination / "tests" / "test.sh"
    test_script.rename(destination / "tests" / "test.original.sh")
    test_script.write_text(
        "#!/bin/bash\nset -e\n"
        "if [ ! -e /verifier ]; then ln -s /tests /verifier; fi\n"
        "exec bash /tests/test.original.sh\n"
    )
    test_script.chmod(0o755)

    solution_script = destination / "solution" / "solve.sh"
    solution_script.rename(destination / "solution" / "solve.original.sh")
    solution_script.write_text(
        "#!/bin/bash\nset -e\n"
        "if [ ! -e /oracle ]; then ln -s /solution /oracle; fi\n"
        "exec bash /solution/solve.original.sh\n"
    )
    solution_script.chmod(0o755)


def prepare() -> Path:
    """Build a Harbor task tree and Gym index from an official SkillsBench checkout."""
    source_override = os.environ.get(SOURCE_ENV_VAR)
    source_dir = Path(source_override).expanduser().resolve() if source_override else _fetch_official_tasks()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Set {SOURCE_ENV_VAR} to the official SkillsBench tasks/ directory: {source_dir}")
    task_dirs = sorted(path for path in source_dir.iterdir() if (path / "task.md").is_file())
    if not task_dirs:
        raise ValueError(f"No official SkillsBench task.md files found in {source_dir}")
    skill_mode = os.environ.get(SKILL_MODE_ENV_VAR, "with-skill")
    if skill_mode not in {"with-skill", "no-skill"}:
        raise ValueError(f"{SKILL_MODE_ENV_VAR} must be 'with-skill' or 'no-skill', got {skill_mode!r}")

    image_manifest_path = os.environ.get(IMAGE_MANIFEST_ENV_VAR)
    images = None
    if image_manifest_path:
        # A relative path names a file in this Gym checkout, e.g. the pinned benchmarks/skillsbench/images.json.
        manifest_file = Path(image_manifest_path).expanduser()
        if not manifest_file.is_absolute():
            manifest_file = GYM_ROOT / manifest_file
        manifest = json.loads(manifest_file.read_text())
        source_revision = manifest.get("source_revision")
        if source_revision:
            actual_revision = subprocess.check_output(
                ["git", "-C", str(source_dir.parent), "rev-parse", "HEAD"], text=True
            ).strip()
            if actual_revision != source_revision:
                raise ValueError(
                    f"Image manifest was built from SkillsBench {source_revision}, "
                    f"but {source_dir.parent} is at {actual_revision}"
                )
        images = {
            item["task_name"]: item["docker_image"]
            for item in manifest["tasks"]
            if item.get("success") and item.get("docker_image")
        }
        missing = {task.name for task in task_dirs} - images.keys()
        if missing:
            raise ValueError(f"Image manifest is missing {len(missing)} tasks: {', '.join(sorted(missing))}")

    staging = DATA_DIR / "tasks.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for source in task_dirs:
        _adapt_task(source, staging / source.name, skill_mode, images[source.name] if images else None)
    if TASKS_DIR.exists():
        shutil.rmtree(TASKS_DIR)
    staging.rename(TASKS_DIR)

    with OUTPUT_FPATH.open("w") as output:
        for source in task_dirs:
            output.write(
                json.dumps(
                    {
                        "task_name": source.name,
                        "responses_create_params": {"input": []},
                        "agent_ref": {"name": AGENT_NAME, "type": "responses_api_agents"},
                    }
                )
                + "\n"
            )
    print(f"Prepared {len(task_dirs)} official SkillsBench tasks ({skill_mode}) at {TASKS_DIR}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
