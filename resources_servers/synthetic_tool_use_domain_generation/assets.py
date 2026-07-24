# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Domain-generation prompt assets."""

import hashlib
from pathlib import Path


DOMAIN_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "domain_generation.txt"


def load_domain_prompt() -> str:
    return DOMAIN_PROMPT_PATH.read_text(encoding="utf-8").strip()


def domain_asset_hashes() -> dict[str, str]:
    return {"domain_prompt": hashlib.sha256(DOMAIN_PROMPT_PATH.read_bytes()).hexdigest()}
