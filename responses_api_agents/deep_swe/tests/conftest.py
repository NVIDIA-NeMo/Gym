# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Pier adapter suite inside the exact isolated Pier interpreter."""

import asyncio
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

from responses_api_agents.deep_swe.setup_pier import (
    PIER_CONSTRAINTS_PATH,
    PIER_VERSION,
    _runtime_is_valid,
    ensure_pier_runtime,
)


_PINNED_TEST_ENV = "NEMO_GYM_DEEP_SWE_PINNED_PIER_TEST"
_PINNED_PREFIX_ENV = "NEMO_GYM_DEEP_SWE_PINNED_PIER_PREFIX"
_PINNED_GYM_ROOT_ENV = "NEMO_GYM_DEEP_SWE_PINNED_GYM_ROOT"


def _run_checked(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    process = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if process.returncode != 0:
        output = "\n".join(part for part in (process.stdout, process.stderr) if part)
        raise RuntimeError(
            f"Pinned Pier adapter test command failed with code {process.returncode}: {command[0]}\n{output[-8000:]}"
        )


def _pinned_runtime_fingerprint(runtime: Path) -> tuple[str, str]:
    marker_digest = hashlib.sha256((runtime / "runtime.json").read_bytes()).hexdigest()
    inventory = subprocess.run(
        [
            str(runtime / "bin" / "python"),
            "-I",
            "-c",
            "import json; from importlib.metadata import distributions; "
            "print(json.dumps(sorted((d.metadata.get('Name', '').lower(), d.version) for d in distributions())))",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if inventory.returncode != 0:
        raise RuntimeError("Could not inventory the pinned Pier runtime")
    return marker_digest, inventory.stdout.strip()


def _pinned_test_environment(
    *,
    runtime: Path,
    gym_root: Path,
    test_overlay: Path,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_environment is None else base_environment)
    for name in (
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
    ):
        env.pop(name, None)
    env[_PINNED_TEST_ENV] = "1"
    env[_PINNED_PREFIX_ENV] = str(runtime)
    env[_PINNED_GYM_ROOT_ENV] = str(gym_root)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(test_overlay)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return env


def pytest_configure() -> None:
    if os.environ.get(_PINNED_TEST_ENV) == "1":
        if version("datacurve-pier") != PIER_VERSION:
            raise RuntimeError(f"Pinned adapter tests require datacurve-pier=={PIER_VERSION}")
        expected_prefix = os.environ.get(_PINNED_PREFIX_ENV)
        if expected_prefix is None or Path(sys.prefix).resolve() != Path(expected_prefix).resolve():
            raise RuntimeError("Pinned adapter tests are not running inside the isolated Pier runtime")
        expected_gym_root = os.environ.get(_PINNED_GYM_ROOT_ENV)
        if expected_gym_root is None:
            raise RuntimeError("Pinned adapter tests require the intended Gym source root")
        runtime = Path(expected_prefix).resolve()
        gym_root = Path(expected_gym_root).resolve()
        import modal
        import pier

        from responses_api_agents.deep_swe import pier_sandbox_environment

        if not Path(pier.__file__).resolve().is_relative_to(runtime):
            raise RuntimeError("Pinned adapter tests imported Pier outside the isolated runtime")
        if not Path(modal.__file__).resolve().is_relative_to(runtime):
            raise RuntimeError("Pinned adapter tests imported Modal outside the isolated runtime")
        if not Path(pier_sandbox_environment.__file__).resolve().is_relative_to(gym_root):
            raise RuntimeError("Pinned adapter tests imported the adapter outside the intended Gym root")
        return

    # Importing the full agent requires Gym's server dependency set, which is
    # intentionally absent from Pier's conflicting isolated environment. Keep
    # this import in the outer Gym test process only.
    from responses_api_agents.deep_swe.app import GYM_ROOT

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Pinned Pier adapter tests require the uv executable")
    with tempfile.TemporaryDirectory(prefix="deep-swe-pier-test-runtime-") as tmp_dir:
        runtime_dir = Path(tmp_dir).resolve(strict=True) / "runtime"
        pier = asyncio.run(ensure_pier_runtime(runtime_dir, GYM_ROOT))
        runtime = pier.parent.parent
        if not _runtime_is_valid(runtime, GYM_ROOT):
            raise RuntimeError("Ephemeral Pier test runtime failed its pinned-runtime validation")
        pinned_fingerprint = _pinned_runtime_fingerprint(runtime)
        try:
            test_overlay = Path(tmp_dir) / "test-site-packages"
            _run_checked(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(runtime / "bin" / "python"),
                    "--target",
                    str(test_overlay),
                    "--constraint",
                    str(PIER_CONSTRAINTS_PATH),
                    f"pytest=={version('pytest')}",
                    f"pytest-asyncio=={version('pytest-asyncio')}",
                ],
                cwd=GYM_ROOT,
            )
            env = _pinned_test_environment(
                runtime=runtime,
                gym_root=GYM_ROOT,
                test_overlay=test_overlay,
            )
            _run_checked(
                [
                    str(runtime / "bin" / "python"),
                    "-m",
                    "pytest",
                    "-p",
                    "pytest_asyncio.plugin",
                    "responses_api_agents/deep_swe/tests/test_pier_sandbox_environment.py",
                    "-q",
                ],
                cwd=GYM_ROOT,
                env=env,
            )
        finally:
            if not _runtime_is_valid(runtime, GYM_ROOT):
                raise RuntimeError("Adapter tests changed the pinned Pier runtime")
            if _pinned_runtime_fingerprint(runtime) != pinned_fingerprint:
                raise RuntimeError("Adapter tests changed the pinned Pier package inventory or marker")
