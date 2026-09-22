# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A minimal parser-only JSON subprocess; deliberately imports no Gym/Hydra code."""

import contextlib
import importlib.metadata
import io
import json
import math
import resource
import sys
from pathlib import Path


def main() -> None:
    """Read one parse request, enforce limits, and emit exactly one JSON result."""
    # Put only the repository root on the import path so the sibling parser can
    # also be exercised directly by resource tests. It has no Gym dependencies.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from resources_servers.matharena_aime.parser import UnsafeMathExpression, parse_result

    try:
        request = json.load(sys.stdin)
        memory_bytes = request.get("memory_limit_mb", 2048) * 1024**2
        cpu_seconds = max(1, math.ceil(request.get("timeout_seconds", 5)))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        for package, expected in {
            "sympy": "1.14.0",
            "antlr4-python3-runtime": "4.11.1",
            "regex": "2026.2.28",
            "loguru": "0.7.3",
            "mpmath": "1.3.0",
        }.items():
            if importlib.metadata.version(package) != expected:
                raise RuntimeError(f"Parser dependency must be {package}=={expected}")
        with contextlib.redirect_stdout(io.StringIO()):
            result = parse_result(
                request["text"],
                strict=request["strict"],
                expected_answer=request.get("expected_answer"),
                output_tokens=request.get("output_tokens", 0),
            )
        result.update(valid=True, verifier_error=None)
    except UnsafeMathExpression as exc:
        result = {"valid": False, "verifier_error": f"unsafe_expression: {str(exc)[:512]}"}
    except Exception as exc:
        result = {"valid": False, "verifier_error": f"parser_error: {type(exc).__name__}: {str(exc)[:512]}"}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
