# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-time Python 3.10 venv builder for bigcodebench evaluation.

bigcodebench's per-task tests import 70+ pinned-version third-party libraries
(``numpy==1.21.2``, ``keras==2.11.0``, ``tensorflow==2.11.0``, ...). The
Gym ``nemo-rl`` container ships Python 3.12, where most of those pins are
unavailable. Rather than corrupt the host venv, we build an isolated
Python 3.10 venv using ``uv`` and shell out to it via ``bcb_runner.py``.

This mirrors NeMo-Skills' ``eval_bigcodebench`` approach:
  * ``pip install bigcodebench``
  * ``pip install numpy==1.26.4`` (Skills overrides the upstream pin)
  * ``pip install -r Requirements/requirements-eval.txt`` from upstream
"""

import shutil
import subprocess
from pathlib import Path


REQUIREMENTS_EVAL_URL = (
    "https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt"
)
BIGCODEBENCH_VERSION_SPEC = "bigcodebench>=0.2.5"
NUMPY_PIN = "numpy==1.26.4"


def ensure_bcb_venv(venv_path: Path, python_version: str = "3.10") -> Path:
    """Create venv_path with bigcodebench installed; return path to its python.

    Idempotent: if ``venv_path/.installed`` exists and the python binary is
    present, returns immediately. Otherwise tears down any partial venv and
    rebuilds. Raises if ``uv`` is not on PATH.
    """
    venv_path = Path(venv_path).resolve()
    python_bin = venv_path / "bin" / "python"
    sentinel = venv_path / ".installed"

    if sentinel.exists() and python_bin.exists():
        return python_bin

    if not shutil.which("uv"):
        raise RuntimeError(
            "uv is required to build the bcb_venv but was not found on PATH. "
            "Install with `pip install uv` or `pipx install uv`."
        )

    venv_path.parent.mkdir(parents=True, exist_ok=True)
    if venv_path.exists():
        shutil.rmtree(venv_path)

    # ``--seed`` populates pip in the new venv so we can use it below for the
    # bigcodebench requirements-eval install (uv's resolver chokes on the upstream
    # pin conflicts; pip handles them permissively).
    subprocess.check_call(["uv", "venv", "--seed", "--python", python_version, str(venv_path)])

    # numpy first to lock in 1.26.4 before bigcodebench's transitive deps pull a different version.
    # ``--no-deps`` on bigcodebench: the package's own ``install_requires`` pulls
    # ``vllm`` + ``torch`` (~847 MB) for its gen/serve modules, neither of which
    # we use. The resource server only calls ``bigcodebench.eval.untrusted_check``,
    # whose runtime deps come from requirements-eval.txt below. Skipping the
    # transitive install saves ~15 min on cold-start and avoids ENOSPC on
    # cluster nodes whose container ``/root/.cache`` is too small for the torch
    # wheel.
    subprocess.check_call(["uv", "pip", "install", "--python", str(python_bin), NUMPY_PIN])
    subprocess.check_call(
        ["uv", "pip", "install", "--python", str(python_bin), "--no-deps", BIGCODEBENCH_VERSION_SPEC]
    )

    # bigcodebench's upstream requirements-eval.txt has hard-conflicting pins
    # (numpy 1.21.2 vs tensorflow 2.11.0 vs keras 2.11.0 vs matplotlib 3.7.0).
    # uv refuses to resolve and silently leaves transitive deps un-installed,
    # so matplotlib ends up with cycler / kiwisolver / fonttools / etc. missing
    # and ``unsafe_execute``'s ``import matplotlib.pyplot`` raises ModuleNotFoundError
    # during ``reliability_guard`` — which then surfaces as 100% test timeouts
    # (the fork's stat.value never gets set, untrusted_check returns TIMEOUT).
    # Plain pip is more permissive about partial resolution, mirroring Skills'
    # eval_bigcodebench install path inside the nemo-skills container.
    subprocess.check_call([str(python_bin), "-m", "pip", "install", "-r", REQUIREMENTS_EVAL_URL])

    # Ensure numpy stays at 1.26.4 — requirements-eval.txt downgrades to 1.21.2
    # which is ABI-incompatible with the matplotlib wheel we just pulled.
    subprocess.check_call(["uv", "pip", "install", "--python", str(python_bin), NUMPY_PIN])

    # Smoke-test the actual chain that ``untrusted_check`` exercises.
    subprocess.check_call(
        [
            str(python_bin),
            "-c",
            "from bigcodebench.eval import untrusted_check; import matplotlib.pyplot as plt; plt.close('all')",
        ]
    )

    sentinel.touch()
    return python_bin


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--venv-path", type=Path, default=Path(__file__).parent / ".bcb_venv")
    parser.add_argument("--python-version", default="3.10")
    args = parser.parse_args()
    bin_path = ensure_bcb_venv(args.venv_path, args.python_version)
    print(f"bcb_venv ready at {bin_path}")
