# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

import numpy as np

from resources_servers.code_gen.lcb_integration.checker import (
    check_correctness,
)


def _json_scalar(value: object) -> object:
    if isinstance(value, (np.generic, np.ndarray)):
        return value.item()
    raise TypeError(f"Unsupported harness result value: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    result, metadata = check_correctness(**request)
    temporary = args.result.with_suffix(".pending")
    temporary.write_text(json.dumps(dict(result=result, metadata=metadata), default=_json_scalar))
    temporary.replace(args.result)


if __name__ == "__main__":
    main()
