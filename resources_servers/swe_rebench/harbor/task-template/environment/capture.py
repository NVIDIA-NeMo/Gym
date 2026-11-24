# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture the agent's patch in its own sandbox, as the SWE-rebench server's ``_extract_model_patch`` does.

Adapted from ``resources_servers/scale_swe/harbor/task-template/environment/capture/capture.py``. The task's
``[[verifier.collect]]`` hook runs this file once the agent is done, with the interpreter the seed recorded in
``/var/lib/nemo_gym/python_bin`` (the image's Python 3.10+, or the pinned standalone Python the seed installed). It
calls the server's own ``capture_model_patch`` (``resources_servers/swebench/patch_capture.py``, vendored unchanged
beside it) with the row's ``base_commit``, given as the first argument by the hook's command in ``task.toml`` (the
server diffs against the row's commit, not a recorded HEAD, and the agent's sandbox holds no copy of it), the files
the seed found untracked, the server's ``drop_patch_sections`` (``resources_servers/swe_rebench/verification.py``)
and the server's ``patch_capture_mode``; the module's commands run locally through ``local_sandbox.LocalSandbox``.

It writes ``/logs/artifacts/nemo_gym/model_patch.diff`` and then ``capture.json``, whose ``status`` tells the separate
verifier (``tests/nemo_gym/grade.py``) what happened, in the server's terms:

- ``ok``: the patch was read; grade it.
- ``git_error``: git exited non-zero on the repository the agent left (``CaptureCommandError`` without an
  ``error_type``), or the seed's records are gone.
- ``infra_error``: a capture command timed out, or the capture failed in any other way. The server masks these.

When this file cannot run at all the hook's fallback writes ``runner_failed``; when the hook is cut short nothing is
written. Everything here runs in the agent's sandbox, where the agent is root and owns these files, so the verifier
scores every status but ``ok`` as a completed 0 (no mask).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT_DIR = Path("/logs/artifacts/nemo_gym")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write(status: str, detail: dict, patch: str | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if patch is not None:
        tmp = OUT_DIR / "model_patch.diff.tmp"
        tmp.write_text(patch, errors="replace")
        os.replace(tmp, OUT_DIR / "model_patch.diff")
    (OUT_DIR / "capture.json").write_text(json.dumps({"status": status, **detail}, indent=2, default=str))


async def _capture(base_commit: str) -> None:
    sys.path.insert(0, str(HERE))
    from local_sandbox import LocalSandbox

    config = json.loads((HERE / "config.json").read_text())
    patch_capture = _load("nemo_gym_patch_capture", HERE / "resources_servers/swebench/patch_capture.py")
    verification = _load("nemo_gym_swe_rebench_verification", HERE / "resources_servers/swe_rebench/verification.py")

    try:
        pristine = frozenset(
            line.strip()
            for line in (HERE / "pristine_untracked.txt").read_text(errors="replace").splitlines()
            if line.strip()
        )
    except OSError as error:
        # The seed wrote it before the agent started; only the agent can have removed it.
        _write("git_error", {"error": f"seed record missing: {error}"})
        return

    try:
        capture = await patch_capture.capture_model_patch(
            LocalSandbox(),
            config["workdir"],
            base_commit,
            mode=config["patch_capture_mode"],
            pristine_untracked=pristine,
            drop_sections=verification.drop_patch_sections,
        )
    except patch_capture.CaptureCommandError as error:
        detail = {"error": str(error)[-4000:], "return_code": error.return_code, "error_type": error.error_type}
        # The server's split (app._verify): a git exit is the agent's doing, the rest is infrastructure.
        incomplete = error.error_type is not None or not isinstance(error.return_code, int)
        _write("infra_error" if incomplete else "git_error", detail)
        return
    _write(
        "ok",
        {
            "base_commit": base_commit,
            **capture.response_fields(include_patch=False),
            "warnings": capture.warnings,
            "collection_time_s": capture.collection_time_s,
        },
        patch=capture.patch,
    )


def main() -> int:
    try:
        asyncio.run(_capture(sys.argv[1]))
    except Exception as error:  # the server masks any other capture failure (VERIFIER_ERROR)
        _write("infra_error", {"error": f"{type(error).__name__}: {error}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
