# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the vendored pocketshell inside its Resources-owned sandbox."""

import json
import sys
from pathlib import Path

from pocketshell import run
from shell_output import format_shell_output


def main() -> None:
    """Read a JSON command and write its bounded result; command text is never shell source."""
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        result = run(payload["command"], workspace=payload["workspace"], timeout=payload["duration"])
        output = format_shell_output(result.stdout, result.stderr, result.exit_code)
    except Exception as error:
        output = f"[bash error: {error}]\n[exit_code=-2]"
    # A lone surrogate in the output must not cost the model the whole result and its exit code.
    Path(sys.argv[2]).write_text(output, encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
