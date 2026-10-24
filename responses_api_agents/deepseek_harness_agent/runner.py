# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned upstream SDK inside a task sandbox, independently of Gym."""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


DSH_VERSION = "0.1.5rc1"


def ensure_sdk(run_dir: Path) -> None:
    try:
        if version("deepseek-harness-sdk") == DSH_VERSION:
            return
    except PackageNotFoundError:
        pass
    # Keep dependencies out of the task repository and its Python environment.
    venv = run_dir / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = str(venv / "bin" / "python")
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--disable-pip-version-check",
            f"deepseek-harness-sdk=={DSH_VERSION}",
        ],
        check=True,
    )
    os.execv(python, [python, *sys.argv])


def run(config_path: Path) -> int:
    run_dir = config_path.parent
    result = {"finish_reason": "error", "error": None, "version": DSH_VERSION}
    try:
        ensure_sdk(run_dir)
        from deepseek_harness import DeepSeekHarness

        config = json.loads(config_path.read_text())
        config["harness"]["env"] = {
            **config["harness"].get("env", {}),
            "PKG_NATIVE_CACHE_PATH": str(run_dir / "cache"),
            "XDG_CACHE_HOME": str(run_dir / "cache"),
        }
        with (run_dir / "events.jsonl").open("w", buffering=1) as events:
            with DeepSeekHarness(
                dsh_home=str(run_dir / "home"),
                cwd=os.getcwd(),
                runtime_cwd=os.getcwd(),
                **config["harness"],
            ) as harness:
                output = harness.run(
                    config["prompt"],
                    session_id=config["session_id"],
                    on_notification=lambda notification: events.write(json.dumps(asdict(notification)) + "\n"),
                )
                result["finish_reason"] = output.finish_reason
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (run_dir / "result.json").write_text(json.dumps(result))
    return 0 if result["finish_reason"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(run(Path(sys.argv[1])))
