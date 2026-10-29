# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures of the image build tests: task writers, the base image record, the fake
docker, buildah and podman tools with their inspect outputs, and the digest and tag helpers
that test_harbor_image_build.py, the four build module test files, test_harbor_build_inputs.py
and test_harbor_compose_prepare.py import.
"""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

from nemo_gym.data import RECORD_VERSION
from nemo_gym.data.tasks.harbor.build_compose import compose_build_env, service_build_settings, validate_task_compose
from nemo_gym.data.tasks.harbor.build_plan import BuildPlanEntry, plan_builds
from nemo_gym.data.tasks.harbor.build_tools import ImageBuilder
from nemo_gym.data.tasks.harbor.harbor_compat import environment_content_hash
from nemo_gym.data.tasks.harbor.records import IMAGE_CONFIGS_FILE


TOML = """
schema_version = "1.4"

[task]
name = "org/hello"
version = "1.0.0"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 300.0

[environment]
cpus = 1
memory_mb = 2048
"""

BUILD_DOCKERFILE = "FROM ubuntu:24.04\nCOPY tools/ /opt/tools/\nRUN /opt/tools/setup.sh\nWORKDIR /app\nUSER app\n"
OVERLAY_DOCKERFILE = "FROM ubuntu:24.04\nRUN apt-get install -y jq\nWORKDIR /app\n"
PULL_DOCKERFILE = "FROM ubuntu:24.04\nWORKDIR /app\n"
REPO = "registry.example/team/repo"


def write_task(root: Path, *, dockerfile: str = BUILD_DOCKERFILE, toml: str = TOML, extra: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.toml").write_text(toml)
    (root / "instruction.md").write_text("Do it.\n")
    (root / "environment").mkdir(exist_ok=True)
    (root / "environment" / "Dockerfile").write_text(dockerfile)
    for name, text in (extra or {"tools/setup.sh": "#!/bin/sh\necho ok\n"}).items():
        target = root / "environment" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    return root


UBUNTU_RECORD = {
    "os": "linux",
    "architecture": "amd64",
    "image": "ubuntu:24.04",
    "config": {"Env": ["HOME=/root"]},
    "record_version": RECORD_VERSION,
}


def write_base_record(folder: Path, image: str = "ubuntu:24.04", *, inputs_digest: str | None = None) -> None:
    """Record a base image's configuration next to the task folders, as prepare does before planning builds; with
    ``inputs_digest``, the record of a built image, as the build writes it."""
    path = folder / IMAGE_CONFIGS_FILE
    recorded = json.loads(path.read_text()) if path.is_file() else {}
    recorded[image] = {**UBUNTU_RECORD, "image": image}
    if inputs_digest is not None:
        recorded[image]["inputs_digest"] = inputs_digest
    path.write_text(json.dumps(recorded))


BUILDAH_INSPECT = {
    "FromImageID": "abc123",
    "OCIv1": {
        "architecture": "amd64",
        "os": "linux",
        "config": {
            "Env": ["PATH=/usr/bin:/bin", "APP_HOME=/app"],
            "WorkingDir": "/app",
            "User": "app",
            "Cmd": ["/bin/sh"],
            "Labels": {"ignored": "yes"},
        },
    },
}
ARM_BUILDAH_INSPECT = {"FromImageID": "arm001", "OCIv1": {"architecture": "arm64", "os": "linux", "config": {}}}
DOCKER_INSPECT = [
    {
        "Id": "sha256:def456",
        "Architecture": "amd64",
        "Os": "linux",
        "Config": {"Env": ["PATH=/usr/bin"], "WorkingDir": "/srv", "User": "", "Entrypoint": ["/entry"], "Cmd": None},
        "RepoDigests": ["registry.example/team/repo@sha256:feedface"],
    }
]


class FakeTool:
    """Records every command and answers build, push and inspect the way the real tool would."""

    def __init__(
        self,
        tool: str,
        *,
        fail: str | None = None,
        inspect: dict | list | None = None,
        fail_output: str | None = None,
    ) -> None:
        self.tool = tool
        self.fail = fail
        self.fail_output = fail_output
        self.inspect = inspect
        self.calls: list[list[str]] = []
        # The environment each command ran with: ``None`` when the builder let it inherit the process environment.
        self.envs: list[dict[str, str] | None] = []

    def __call__(self, argv: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        self.envs.append(env)
        verb = argv[1] if self.tool != "buildah" else next(a for a in argv[1:] if a in ("bud", "push", "inspect"))
        if self.fail and self.fail in argv:
            return subprocess.CompletedProcess(argv, 1, "", self.fail_output or f"{verb}: boom\n")
        if verb in ("inspect", "image"):
            if self.inspect is not None:
                data = self.inspect
            else:
                data = BUILDAH_INSPECT if self.tool == "buildah" else DOCKER_INSPECT
            return subprocess.CompletedProcess(argv, 0, json.dumps(data), "")
        if verb == "push":
            if "--digestfile" in argv:
                Path(argv[argv.index("--digestfile") + 1]).write_text("sha256:pushed01\n")
                return subprocess.CompletedProcess(argv, 0, "", "")
            return subprocess.CompletedProcess(argv, 0, "latest: digest: sha256:pushed02 size: 1234\n", "")
        return subprocess.CompletedProcess(argv, 0, "built\n", "")


def local_docker() -> tuple[ImageBuilder, FakeTool]:
    tool = FakeTool("docker")
    return ImageBuilder(tool="docker", run=tool), tool


def harbor_hash(context: Path) -> str:
    """The environment id Harbor would compute for the context, straight from the vendored function."""
    return environment_content_hash(context)


def task_inputs_digest(
    task: Path, *, service: str = "main", repository: str | None = None, platform: str | None = None
) -> str:
    """The inputs digest prepare records for ``service`` of ``task``'s build context: the plan entry settled as a
    builder for ``repository`` and ``platform`` settles it."""
    from nemo_gym.data.tasks.harbor.build_plan import DEFAULT_PLATFORM
    from nemo_gym.data.tasks.harbor.build_records import build_inputs_digest, settle_entry

    entry = next(entry for entry in plan_builds(task) if entry.service == service)
    return build_inputs_digest(settle_entry(entry, repository=repository, platform=platform or DEFAULT_PLATFORM))


def tagged_buildah(tmp_path: Path, **kwargs) -> tuple[ImageBuilder, FakeTool]:
    """A buildah builder that tags for a repository but pushes nothing, so the image stays local."""
    tool = FakeTool("buildah", **kwargs)
    builder = ImageBuilder(tool="buildah", repository=REPO, push=False, storage_root=tmp_path / "store", run=tool)
    return builder, tool


def main_record(image: str, tasks: list[str], tool: str, pushed: bool) -> dict:
    """A ``built-images.json`` entry of ``main`` alone: the top-level keys older readers use plus ``services``, each
    service with its own ``tool`` and ``pushed``."""
    service = {"image": image, "tool": tool, "pushed": pushed}
    return {"image": image, "tasks": tasks, "tool": tool, "pushed": pushed, "services": {"main": service}}


def settled(entry: BuildPlanEntry, image: str = "hb__x") -> BuildPlanEntry:
    """``entry`` with ``main``'s build section applied, as ``ImageBuilder.build`` applies it once the task's Compose
    file is checked and before the build command."""
    env = compose_build_env(entry, image, None)
    document = validate_task_compose(entry.context, env, tasks=entry.tasks)
    return replace(entry, **service_build_settings(entry, document, env))


HEREDOC_DOCKERFILE = "FROM docker.io/library/alpine:3.20\nCOPY tools/ /opt/tools/\nRUN <<EOF\necho hello > /x\nEOF\n"
PROBE_TAG = "localhost/nemo-gym/heredoc-probe"


class HeredocTool:
    """A buildah or podman stand-in whose heredoc preflight passes, or fails as Ubuntu 24.04's packages fail it."""

    def __init__(self, tool: str, *, heredocs: bool) -> None:
        self.tool = tool
        self.heredocs = heredocs
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str] | None] = []
        # The Dockerfile text of every preflight, read while its temporary context still exists.
        self.probe_dockerfiles: list[str] = []

    def __call__(self, argv: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        self.envs.append(env)
        if argv[1:] == ["--version"]:
            return subprocess.CompletedProcess(argv, 0, f"{self.tool} version 1.33.7 (image-spec 1.1.0-rc.5)\n", "")
        if "-f" in argv:
            if argv[argv.index("-t") + 1] == PROBE_TAG:
                self.probe_dockerfiles.append(Path(argv[argv.index("-f") + 1]).read_text())
                if not self.heredocs:
                    stderr = 'Error: building at STEP "PROBE ": Build error: Unknown instruction: "PROBE"\n'
                    return subprocess.CompletedProcess(argv, 125, "", stderr)
            return subprocess.CompletedProcess(argv, 0, "built\n", "")
        return FakeTool(self.tool)(argv, env)
