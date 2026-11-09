# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provision a disposable Linux runtime for the multimodal conformance probes."""

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from .ci import ROOT


def runtime_pins(version: str, architecture: str) -> dict:
    """Return reviewed publisher pins; changing the configured release requires repinning."""
    pins = json.loads(Path(__file__).with_name("multimodal_runtime_pins.json").read_text())
    if pins["version"] != version:
        raise RuntimeError(f"Update multimodal runtime pins before provisioning OpenCode {version}")
    return pins["runtimes"][architecture] | {"version": version, "plugin": pins["plugin"]}


def verify_runtime_assets(executable: Path, dependencies: Path, pins: dict) -> None:
    """Check installed assets against expected pins, independently of observed file hashes."""
    packages = json.loads((dependencies / "package-lock.json").read_text())["packages"]
    for package, integrity in (
        (pins["package"], pins["npm_integrity"]),
        (pins["plugin"]["package"], pins["plugin"]["integrity"]),
    ):
        installed = packages[f"node_modules/{package}"]
        if installed.get("version") != pins["version"] or installed.get("integrity") != integrity:
            raise RuntimeError(f"Installed package does not match reviewed integrity pin: {package}")
    with executable.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != pins["executable_sha256"]:
        raise RuntimeError("Installed OpenCode executable does not match reviewed SHA-256 pin")


def provision(directory: Path, *, bwrap_command: list[str] | None = None) -> Path:
    """Install pinned tools and validate the actual sandbox before exporting its config."""
    from responses_api_agents.opencode_multimodal_agent.eval_sandbox import EvalSandboxConfig, sandbox_command

    architecture = {"x86_64": "x64", "aarch64": "arm64"}.get(platform.machine())
    if sys.platform != "linux" or architecture is None:
        raise RuntimeError("Multimodal probes require Linux x86_64 or aarch64")
    if bwrap_command is None:
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise RuntimeError("Install bubblewrap before provisioning the multimodal runtime")
        bwrap_command = [bwrap]
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    defaults = yaml.safe_load(
        (ROOT / "responses_api_agents/opencode_multimodal_agent/configs/opencode_multimodal_agent.yaml").read_text()
    )
    version = defaults["opencode_multimodal_agent"]["responses_api_agents"]["opencode_multimodal_agent"][
        "opencode_version"
    ]
    pins = runtime_pins(str(version), architecture)
    package = pins["package"]
    dependencies = directory / "dependencies"
    subprocess.run(
        [
            "npm",
            "install",
            "--prefix",
            str(dependencies),
            "--save-exact",
            "--ignore-scripts",
            f"{package}@{version}",
            f"@opencode-ai/plugin@{version}",
        ],
        check=True,
    )
    # OpenCode initializes this even with preinstalled packages. The agent mounts
    # the shared dependency directory read-only, so complete initialization here.
    (dependencies / ".gitignore").write_text("node_modules\npackage.json\npackage-lock.json\nbun.lock\n.gitignore\n")
    executable = dependencies / "node_modules" / package / "bin/opencode"
    with executable.open("rb") as handle:
        if handle.read(4) != b"\x7fELF":
            raise RuntimeError("The sandbox requires the native OpenCode executable, not its npm launcher")
    verify_runtime_assets(executable, dependencies, pins)
    python_env = directory / "python"
    if not (python_env / "pyvenv.cfg").is_file():
        subprocess.run(["uv", "venv", "--python", sys.executable, str(python_env)], check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_env / "bin/python"),
            "Pillow==11.3.0",
            "numpy==2.2.6",
            "scipy==1.15.3",
            "opencv-python-headless==4.12.0.88",
        ],
        check=True,
    )
    runtime = EvalSandboxConfig(
        bwrap_command=bwrap_command,
        python_env=python_env,
        plugin_dependencies=dependencies,
    )
    workspace = directory / "preflight"
    workspace.mkdir(exist_ok=True)
    command = sandbox_command(
        runtime,
        workspace,
        executable,
        {"OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "4096", "OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS": "30000"},
        [
            "/bin/bash",
            "-ec",
            "python3 -c 'import PIL, numpy, scipy, cv2'; "
            "/agent-bin/opencode debug config >/dev/null; /agent-bin/opencode --version",
        ],
    )
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    if result.stdout.strip().splitlines()[-1:] != [str(version)]:
        raise RuntimeError(f"Sandbox OpenCode version mismatch: {result.stdout!r}")
    config = directory / "sandbox.json"
    config.write_text(
        json.dumps(
            {
                "command": shlex.quote(str(executable)),
                "executable_sha256": pins["executable_sha256"],
                "runtime_provenance": pins,
                "sandbox": runtime.model_dump(mode="json"),
            },
            indent=2,
        )
        + "\n"
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--github-env", type=Path)
    args = parser.parse_args()
    config = provision(args.directory)
    if args.github_env:
        with args.github_env.open("a") as handle:
            handle.write(f"GYM_MULTIMODAL_SANDBOX_CONFIG={config}\n")
            # ensure_opencode checks PATH even when the agent config supplies an absolute executable.
            executable = shlex.split(json.loads(config.read_text())["command"])[0]
            handle.write(f"PATH={Path(executable).parent}{os.pathsep}{os.environ['PATH']}\n")
            handle.write("OPENBLAS_NUM_THREADS=1\nOMP_NUM_THREADS=1\n")
    print(f"Multimodal sandbox ready: {config}")


if __name__ == "__main__":
    main()
