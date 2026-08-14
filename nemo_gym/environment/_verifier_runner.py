# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a verifier fixture inside its resources-server environment."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nemo_gym.environment.onboarding import VerifierRunSpec, exercise_verifier_run


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _run(request_path: Path) -> dict[str, Any]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    spec_values = dict(request["spec"])
    spec_values["reward_range"] = tuple(spec_values["reward_range"])
    update_expected = request.get("update_expected", False)
    if not isinstance(update_expected, bool):
        raise TypeError("update_expected must be a boolean")

    report = asyncio.run(exercise_verifier_run(VerifierRunSpec(**spec_values), update_expected=update_expected))
    return {"ok": True, "report": report.to_dict()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=_absolute_path)
    parser.add_argument("--result", required=True, type=_absolute_path)
    args = parser.parse_args(argv)

    try:
        result = _run(args.request)
    except Exception as error:
        result = {"ok": False, "error": str(error)}

    args.result.write_text(json.dumps(result), encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
