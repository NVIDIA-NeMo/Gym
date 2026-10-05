# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Inkling serving config selects the complete-response parsers; needs bash only, not vLLM."""

import subprocess
from pathlib import Path


CONFIG = Path(__file__).resolve().parents[2] / "vllm_configs" / "inkling_small.sh"


def _common_args() -> list[str]:
    script = f'source "{CONFIG}"; printf "%s\\n" "${{VLLM_COMMON_ARGS[@]}}"'
    return subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True).stdout.splitlines()


def _value_after(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_serving_config_selects_complete_response_parsers_and_their_plugin_files() -> None:
    args = _common_args()
    assert _value_after(args, "--tool-call-parser") == "inkling_complete_fast"
    assert _value_after(args, "--reasoning-parser") == "inkling_count_fast"
    for flag in ("--tool-parser-plugin", "--reasoning-parser-plugin"):
        assert Path(_value_after(args, flag)).is_file()
