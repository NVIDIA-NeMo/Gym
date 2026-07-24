# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt assets owned by the domain-generation agent."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PACKAGE_DIR / "prompts"
DOMAIN_PROMPT_PATH = PROMPTS_DIR / "domain_generation.txt"
PROMPT_ARCHIVE_DIR = PROMPTS_DIR / "archive"


def load_domain_prompt() -> str:
    return DOMAIN_PROMPT_PATH.read_text(encoding="utf-8").strip()


def archive_prompt_paths() -> tuple[Path, ...]:
    return tuple(sorted(path for path in PROMPT_ARCHIVE_DIR.iterdir() if path.is_file()))
