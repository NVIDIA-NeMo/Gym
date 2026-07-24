# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt-profile loading for policy and tool generation."""

import hashlib
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel


PACKAGE_DIR = Path(__file__).resolve().parent


class PolicyToolsProfile(BaseModel):
    name: str
    golden_reference_dir: str | None = None
    policy_prompt: str
    tools_prompt: str
    policy_refine_prompt: str
    tools_refine_prompt: str
    cohesion_judge_prompt: str
    golden_judge_prompt: str


def _read_prompt(name: str) -> str:
    return (PACKAGE_DIR / "prompts" / name).read_text(encoding="utf-8").strip()


def _profile_path(name: str) -> Path:
    profile_path = PACKAGE_DIR / "profiles" / f"{name}.yaml"
    if not profile_path.is_file():
        choices = ", ".join(path.stem for path in sorted((PACKAGE_DIR / "profiles").glob("*.yaml")))
        raise ValueError(f"unknown generation profile {name!r}; available profiles: {choices}")
    return profile_path


def load_profile(name: str) -> PolicyToolsProfile:
    profile_path = _profile_path(name)
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    prompts = {key: _read_prompt(value) for key, value in raw.pop("prompts").items()}
    return PolicyToolsProfile.model_validate({**raw, **prompts})


def load_golden_pairs(profile: PolicyToolsProfile) -> list[tuple[str, str]]:
    if profile.golden_reference_dir is None:
        return []
    references_dir = PACKAGE_DIR / "references" / profile.golden_reference_dir
    pairs = []
    for policy_path in sorted(references_dir.glob("policy-*.md")):
        suffix = policy_path.stem.removeprefix("policy-")
        tools_path = references_dir / f"tools_{suffix}.jsonl"
        if not tools_path.is_file():
            raise ValueError(f"missing golden tools file for {policy_path.name}")
        pairs.append(
            (
                policy_path.read_text(encoding="utf-8").strip(),
                tools_path.read_text(encoding="utf-8").strip(),
            )
        )
    return pairs


def profile_asset_hashes(name: str) -> dict[str, str]:
    profile_path = _profile_path(name)
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    hashes = {
        field_name: hashlib.sha256((PACKAGE_DIR / "prompts" / filename).read_bytes()).hexdigest()
        for field_name, filename in raw["prompts"].items()
    }
    golden_reference_dir = raw.get("golden_reference_dir")
    if golden_reference_dir is None:
        return hashes
    references_dir = PACKAGE_DIR / "references" / golden_reference_dir
    for policy_path in sorted(references_dir.glob("policy-*.md")):
        suffix = policy_path.stem.removeprefix("policy-")
        tools_path = references_dir / f"tools_{suffix}.jsonl"
        hashes[f"golden_policy_{suffix}"] = hashlib.sha256(policy_path.read_bytes()).hexdigest()
        hashes[f"golden_tools_{suffix}"] = hashlib.sha256(tools_path.read_bytes()).hexdigest()
    return hashes
