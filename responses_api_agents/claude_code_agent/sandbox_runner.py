# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch Claude Code under the shared process supervisor."""

import json
import os
import sys
from pathlib import Path


def main() -> None:
    payload = json.loads(Path(sys.argv[1]).read_text())
    os.environ.update(payload["env"])
    os.execvpe(payload["argv"][0], payload["argv"], os.environ)


if __name__ == "__main__":
    main()
