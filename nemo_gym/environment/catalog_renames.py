# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Catalog names that were renamed, each with the catalog name that replaced it.

A catalog name is a manifest's ``name``: the folder path of its manifest under ``benchmarks/`` or ``environments/``.
It names a taskset, never the harness that runs it, so a config that only changes the harness is a run preset of its
taskset's catalog name and has no manifest of its own.

An old name keeps working: ``gym list``, ``gym env validate``, ``gym env test`` and ``gym env publish`` resolve it to
the new name with a deprecation warning, and so do ``--benchmark`` and ``--environment`` when the old name selects the
same config as the new one. An old name that now selects a run preset (another harness's config) still selects that
preset, without a warning.
"""

RENAMED_CATALOG_NAMES: dict[str, str] = {
    # Each taskset has one manifest pointing at its default config. The configs that only change the harness, or the
    # route it runs on, are its run presets.
    "automationbench/config_legacy_agent": "automationbench",
    # BIRD-Interact's releases and modes are scored separately, so each has its own catalog name.
    "bird_interact": "bird_interact/lite/c_interact",
    "deepsearchqa/configs/deepsearchqa_claude_code": "deepsearchqa/long_run",
    "deepsearchqa/configs/deepsearchqa_cline": "deepsearchqa/long_run",
    "deepsearchqa/configs/deepsearchqa_codex": "deepsearchqa/long_run",
    "deepsearchqa/configs/deepsearchqa_hermes": "deepsearchqa/long_run",
    "deepsearchqa/configs/deepsearchqa_kilocode": "deepsearchqa/long_run",
    "deepsearchqa/configs/deepsearchqa_nemo_fabric_deepagents": "deepsearchqa/long_run",
    "deepsearchqa/configs/deepsearchqa_openclaw": "deepsearchqa/long_run",
    "deepsearchqa/configs/deepsearchqa_opencode": "deepsearchqa/long_run",
    "deepsearchqa/configs/deepsearchqa_pi": "deepsearchqa/long_run",
    "deepsearchqa/configs/deepsearchqa_prime_agent": "deepsearchqa/long_run",
    "deepsearchqa/configs/deepsearchqa_simple_strands": "deepsearchqa/long_run",
    "deepswe/opencode": "deepswe",
    "skill_trigger/configs/codex": "skill_trigger",
    "skill_trigger/configs/codex_guidance": "skill_trigger/guidance",
    "skill_trigger/configs/hermes": "skill_trigger",
    "skill_trigger/configs/hermes_guidance": "skill_trigger/guidance",
    "skill_trigger/configs/pi": "skill_trigger",
    "skillsbench/harbor_opensandbox": "skillsbench/harbor",
    "swebench/multilingual/opencode": "swebench/multilingual",
    "swebench/multilingual/opencode_legacy_agent": "swebench/multilingual",
    "swebench/multimodal/opencode": "swebench/multimodal",
    "swebench/multimodal/opencode_legacy_agent": "swebench/multimodal",
    "swebench/pro/opencode": "swebench/pro",
    "swebench/pro/opencode_legacy_agent": "swebench/pro",
    "swebench/pro_v2/opencode": "swebench/pro_v2",
    "swebench/pro_v2_hard/opencode": "swebench/pro_v2_hard",
    "swebench/verified/opencode": "swebench/verified",
    "swebench/verified/opencode_legacy_agent": "swebench/verified",
    "terminal_bench_2_1/opencode": "terminal_bench_2_1",
    "terminal_bench_2_1/opencode_legacy_agent": "terminal_bench_2_1",
    "terminal_bench_2_1/terminus_2": "terminal_bench_2_1",
    "terminal_bench_2_1/terminus_2_legacy_agent": "terminal_bench_2_1",
    "terminal_bench_4/miniswe": "terminal_bench_4",
    "terminal_bench_4/miniswe_legacy_agent": "terminal_bench_4",
    "terminal_bench_science/codex": "terminal_bench_science",
    "terminal_bench_science/miniswe": "terminal_bench_science",
    "terminal_bench_science/miniswe_legacy_agent": "terminal_bench_science",
    "widesearch/configs/widesearch_claude_code": "widesearch/long_run",
    "widesearch/configs/widesearch_cline": "widesearch/long_run",
    "widesearch/configs/widesearch_codex": "widesearch/long_run",
    "widesearch/configs/widesearch_hermes": "widesearch/long_run",
    "widesearch/configs/widesearch_kilocode": "widesearch/long_run",
    "widesearch/configs/widesearch_nemo_fabric_deepagents": "widesearch/long_run",
    "widesearch/configs/widesearch_openclaw": "widesearch/long_run",
    "widesearch/configs/widesearch_opencode": "widesearch/long_run",
    "widesearch/configs/widesearch_pi": "widesearch/long_run",
    "widesearch/configs/widesearch_prime_agent": "widesearch/long_run",
    "widesearch/configs/widesearch_simple_strands": "widesearch/long_run",
}


def renamed_catalog_name_warning(old: str, new: str) -> str:
    """The deprecation warning for a use of the renamed catalog name ``old``."""
    return f"Catalog name `{old}` is deprecated: it was renamed to `{new}`. Use `{new}` instead."
