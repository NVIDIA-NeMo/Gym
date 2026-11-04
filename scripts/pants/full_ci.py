#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a complete existing CI job directly or through Pants, preserving its contract.

This is the full-scope parity phase: Pants orchestrates entire existing jobs in
its workspace environment. Python target inference and persistent result-cache
reuse are deliberately absent until external dependencies/fixtures are pinned.
The current working directory is the checkout under test; this script may live
in a separate controller checkout (required for the untouched baseline).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
JOBS = ("core", "sandbox", "servers", "changed-servers", "session-routing", "lint", "smoke")
# Values are passed to Pants's process environment, never serialized into reports.
# GITHUB_TOKEN and other credentials are intentionally absent.
ENVIRONMENT = (
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "CI",
    "GITHUB_ACTIONS",
    "GITHUB_WORKSPACE",
    "GITHUB_SHA",
    "GITHUB_ENV",
    "GITHUB_PATH",
    "RUNNER_OS",
    "RUNNER_ARCH",
    "RUNNER_TEMP",
    "RUNNER_TOOL_CACHE",
    "AGENT_TOOLSDIRECTORY",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "JAVA_HOME",
    "JDK_HOME",
    "GOROOT",
    "GOPATH",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "CONDA",
    "CONDA_PREFIX",
    "PKG_CONFIG_PATH",
    "CMAKE_PREFIX_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "CC",
    "CXX",
    "CUDA_HOME",
    "CUDA_PATH",
    "NVM_DIR",
    "DOTNET_ROOT",
    "PNPM_HOME",
    "VCPKG_INSTALLATION_ROOT",
    "CHROME_BIN",
    "CHROMEWEBDRIVER",
    "GECKOWEBDRIVER",
    "ANDROID_HOME",
    "ANDROID_SDK_ROOT",
    "PYTHONHOME",
    "PYTHONPATH",
    "PY_COLORS",
    "NEMO_GYM_TEST_DATA_ROOT",
    "NLTK_DATA",
    "RAY_TMPDIR",
    "UV_CACHE_DIR",
    "UV_PYTHON_INSTALL_DIR",
    "UV_LINK_MODE",
    "NEMO_GYM_CONTAINER",
    "DOCKER_HOST",
    "NEMO_GYM_MULTIWORKER_E2E",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "https_proxy",
    "http_proxy",
    "no_proxy",
)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def server_types() -> tuple[str, ...]:
    """Read the sampled commit's discovery constants without importing Gym."""
    module = ast.parse(Path("nemo_gym/config/global_config.py").read_text())
    strings = {}
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                    strings[target.id] = node.value.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "SERVER_TYPE_KEYS" and isinstance(node.value, ast.Tuple):
                values = tuple(
                    strings.get(item.id)
                    if isinstance(item, ast.Name)
                    else item.value
                    if isinstance(item, ast.Constant)
                    else None
                    for item in node.value.elts
                )
                if values and all(isinstance(value, str) and value for value in values):
                    return values
    raise ValueError("Cannot read sampled main's SERVER_TYPE_KEYS; update discovery instead of reducing scope")


def source_snapshot() -> tuple[list[str], str]:
    """Fingerprint Git entries, including directory and dangling symbolic links."""
    sources = sorted(path for path in git("ls-files", "--recurse-submodules", "-z").split("\0") if path)
    digest = hashlib.sha256()
    for path in sources:
        file = Path(path)
        mode = file.lstat().st_mode
        if stat.S_ISLNK(mode):
            # Git tracks the link text, not its referent. Tracked referents are
            # fingerprinted separately, including every file in linked directories.
            content = b"symlink\0" + os.fsencode(os.readlink(file))
        elif stat.S_ISREG(mode):
            content = b"file\0" + hashlib.sha256(file.read_bytes()).digest()
        else:
            raise FileNotFoundError(f"Tracked CI input is missing or unreadable: {path}")
        digest.update(path.encode() + b"\0" + content + b"\0")
    return sources, digest.hexdigest()


def server_scope(job: str, *, shard: int, shards: int, changed_files: str) -> list[str]:
    if not 0 <= shard < shards:
        raise ValueError("Expected 0 <= shard < shards")
    if job == "changed-servers":
        types = server_types()
        servers = sorted({"/".join(path.split("/")[:2]) for path in changed_files.split()})
        if not servers or any(
            len(Path(server).parts) != 2 or Path(server).parts[0] not in types or Path(server).parts[1] in (".", "..")
            for server in servers
        ):
            raise ValueError("changed-servers requires valid changed server file paths")
        return servers
    if job == "servers":
        return sorted(
            str(path.parent)
            for server_type in server_types()
            for path in Path(server_type).glob("*/README.md")
            if "pycache" not in path.parent.name
        )[shard::shards]
    return []


def command_for(job: str, *, shard: int, shards: int, servers: list[str], output: Path) -> str:
    junit = shlex.quote(str(output / "junit"))
    coverage = shlex.quote(str(output / (f".coverage.core-{shard}" if job == "core" else ".coverage.sandbox")))
    prefix = f"export GYM_CI_JUNIT_DIR={junit}\n"
    if job == "core":
        return prefix + f"export COVERAGE_FILE={coverage}\n./scripts/ci/core_unit_tests.sh {shard} {shards}\n"
    if job == "servers":
        return prefix + f"./scripts/ci/server_tests.sh {shard} {shards}\n"
    if job == "sandbox":
        return prefix + (
            f"export COVERAGE_FILE={coverage}\n"
            "source ./scripts/ci/setup_dev.sh\n"
            "uv sync --extra dev --extra ray --extra sandbox --extra openshell --extra mlflow --extra wandb\n"
            f"pytest --cov=. --cov-report= --cov-fail-under=0 --durations=10 -m sandbox "
            f"--junitxml={shlex.quote(str(output / 'junit/sandbox.xml'))}\n"
        )
    if job == "session-routing":
        return prefix + (
            "export NEMO_GYM_MULTIWORKER_E2E=1 RAY_TMPDIR=/tmp\n"
            "source ./scripts/ci/setup_dev.sh\n"
            f"pytest tests/e2e/session_routing --junitxml={shlex.quote(str(output / 'junit/session-routing.xml'))}\n"
        )
    if job == "changed-servers":
        commands = [prefix, "source ./scripts/ci/setup_dev.sh", "status=0"]
        commands += [
            f"ng_test +entrypoint={shlex.quote(server)} +delete_venvs_after_each_test=true "
            "+should_validate_data=true || status=1"
            for server in servers
        ]
        return "\n".join([*commands, 'exit "$status"', ""])
    if job == "lint":
        return "./scripts/ci/lint.sh\n"
    if job == "smoke":
        # Verify workspace access and environment forwarding without running the CI suites.
        return "test -f pyproject.toml\ngit rev-parse --is-inside-work-tree\npython3 --version\n"
    raise ValueError(f"Unknown job: {job}")


def run_logged(command: list[str], *, log: Path, env: dict[str, str]) -> int:
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        return process.wait()


def run_pants(*, script: Path, output: Path, env: dict[str, str], sources: list[str]) -> int:
    """Execute the identical command in Pants's workspace and restore its status."""
    build_path = Path("BUILD.pants-full-ci")
    if build_path.exists():
        raise FileExistsError(f"Refusing to overwrite {build_path}; another job may be running")
    config = HERE / "full_ci.toml"
    status_file = output / "command-status.txt"
    # Always materialize failure diagnostics, then return the original failure to CI.
    shell_command = (
        f"bash {shlex.quote(str(script))}; status=$?; printf '%s\\n' \"$status\" > {shlex.quote(str(status_file))}; "
        f"cp {shlex.quote(str(status_file))} {{chroot}}/command-status.txt"
    )
    template = (HERE / "full_ci.BUILD").read_text()
    substitutions = {
        "__COMMAND__": repr(shell_command),
        # Pants globs resolve symlinks, and reject dangling links. Their exact
        # Git identity is carried by GYM_PANTS_INPUT_DIGEST instead.
        "__SOURCES__": repr([path for path in sources if not Path(path).is_symlink()]),
        "__ENVIRONMENT__": repr(sorted(env)),
    }
    for token, value in substitutions.items():
        template = template.replace(token, value)
    # Exclude runtime metadata: changes to it would otherwise dominate invalidation.
    with build_path.open("x") as stream:
        stream.write(template)
    # scie-pants discovers the version from a root pants.toml before reading CLI
    # options. Raw main has none; this untracked bootstrap file is removed below.
    bootstrap = Path("pants.toml")
    created_bootstrap = False
    pants = os.environ.get("PANTS_BIN") or shutil.which("pants") or str(Path(".pants-bin/pants").resolve())
    launcher_env = dict(env)
    for name in (
        "PANTS_GLOBAL_WORKDIR",
        "PANTS_NAMED_CACHES_DIR",
        "PANTS_LOCAL_STORE_DIR",
        "PANTS_LOCAL_EXECUTION_ROOT_DIR",
        "PANTS_VERSION",
    ):
        if name in os.environ:
            launcher_env[name] = os.environ[name]
    try:
        if not bootstrap.exists():
            with bootstrap.open("x") as stream:
                created_bootstrap = True
                stream.write(config.read_text())
        code = run_logged(
            [
                pants,
                f"--pants-config-files={json.dumps([str(config)])}",
                "--no-pantsd",
                f"--pants-distdir={output}/pants-artifact",
                "package",
                "//:job",
            ],
            log=output / "runner.log",
            env=launcher_env,
        )
        if code:
            return code
        return int(status_file.read_text().strip())
    finally:
        build_path.unlink()
        if created_bootstrap:
            bootstrap.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("baseline", "pants"), required=True)
    parser.add_argument("--job", choices=JOBS, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", "--num-shards", type=int, default=1)
    parser.add_argument("--changed-server-files", default=os.environ.get("CHANGED_SERVER_FILES", ""))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
        parser.error("Run from the root of the checkout under test")
    if not 0 <= args.shard < args.shards:
        parser.error("Expected 0 <= shard < shards")
    output_absolute = args.output.resolve()
    try:
        output = output_absolute.relative_to(root)
    except ValueError:
        parser.error("--output must be inside the checkout under test")
    if output == Path(".") or output_absolute.exists():
        parser.error("--output must be a new directory inside the checkout")
    if git("status", "--porcelain", "--untracked-files=no"):
        parser.error("Tracked source checkout must be clean before measuring CI")
    servers = server_scope(args.job, shard=args.shard, shards=args.shards, changed_files=args.changed_server_files)
    sources, source_digest = source_snapshot()
    output.mkdir(parents=True)
    (output / "junit").mkdir()
    source_sha = git("rev-parse", "HEAD")
    script = output / "command.sh"
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + command_for(
            args.job,
            shard=args.shard,
            shards=args.shards,
            servers=servers,
            output=output,
        )
    )
    manifest = {
        "job": args.job,
        "shard": args.shard,
        "shards": args.shards,
        "servers": servers,
        "junit_directory": "junit",
        "junit_required": args.job not in ("lint", "smoke"),
        "source_sha": source_sha,
        "tracked_files": len(sources),
        "source_digest": source_digest,
        "scope": "complete existing CI job",
        "command": script.read_text(),
    }
    write_json(output / "manifest.json", manifest)
    env = {name: value for name, value in os.environ.items() if name in ENVIRONMENT or name.startswith("JAVA_HOME_")}
    # The controller branch is infrastructure, not the revision under test.
    env["GITHUB_SHA"] = source_sha
    env["GYM_PANTS_INPUT_DIGEST"] = source_digest
    started_at = timestamp()
    start = time.monotonic()
    code = 1
    command_code = 1
    source_changed = False
    try:
        if args.engine == "pants":
            command_code = run_pants(script=script, output=output, env=env, sources=sources)
        else:
            command_code = run_logged(["bash", str(script)], log=output / "runner.log", env=env)
        source_changed = bool(git("status", "--porcelain", "--untracked-files=no"))
        source_changed = source_changed or source_snapshot() != (sources, source_digest)
        code = command_code or int(source_changed)
        if source_changed:
            print("Tracked source changed during CI; comparison is invalid", file=sys.stderr)
    finally:
        write_json(
            output / "metadata.json",
            {
                "engine": args.engine,
                "job": args.job,
                "shard": args.shard,
                "shards": args.shards,
                "source_sha": source_sha,
                "started_at": started_at,
                "finished_at": timestamp(),
                "elapsed_seconds": time.monotonic() - start,
                "exit_code": code,
                "command_exit_code": command_code,
                "source_changed": source_changed,
                "pants_result_cache": "session-only" if args.engine == "pants" else None,
                "implementation": "whole-job workspace execution"
                if args.engine == "pants"
                else "unchanged CI commands",
            },
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
