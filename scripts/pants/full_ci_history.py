# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read supported historical CI layouts without replacing them with current main."""

import re
from pathlib import Path


NATIVE_JOBS = ("core", "sandbox", "legacy-core")


def job_blocks(text: str) -> dict[str, str]:
    sections = ("\n" + text).split("\njobs:\n", 1)
    if len(sections) != 2:
        raise ValueError("Missing workflow jobs section")
    body = sections[1]
    matches = list(re.finditer(r"^  ([A-Za-z0-9_-]+):[ \t]*$", body, re.MULTILINE))
    return {
        m[1]: body[m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(body)]
        for i, m in enumerate(matches)
    }


def job_settings(block: str) -> dict:
    runner = re.search(r"^    runs-on: ([\w-]+)\s*$", block, re.MULTILINE)
    timeout = re.search(r"^    timeout-minutes: (\d+)\s*$", block, re.MULTILINE)
    if runner is None or runner[1] not in {"ubuntu-latest", "nemo-ci-aws-cpu-4-16"}:
        raise ValueError("Unsupported historical runner contract")
    return {"runner": runner[1], "timeout": int(timeout[1]) if timeout else 360}


def shards(block: str) -> list[int]:
    match = re.search(r"^        shard: \[([\d, ]+)\]\s*$", block, re.MULTILINE)
    if not match:
        return []
    values = [int(v.strip()) for v in match[1].split(",")]
    if values != list(range(len(values))):
        raise ValueError("Historical shard matrix must be a complete zero-based range")
    return values


def ci_contract(source: Path) -> dict:
    workflow = (source / ".github/workflows/unit-tests.yml").read_text()
    jobs = job_blocks(workflow)
    lint = job_blocks((source / ".github/workflows/code-linting.yml").read_text())
    setup = (source / "scripts/ci/setup_dev.sh").read_text()
    pin = re.search(r"https://astral\.sh/uv/(\d+\.\d+\.\d+)/install\.sh", setup)
    if not pin:
        raise ValueError("Sampled main has no exact recognized uv pin")
    core_shards = shards(jobs["test"])
    legacy = not core_shards
    if legacy:
        if "./scripts/ci/core_unit_tests.sh\n" not in jobs["test"] or "--cov-append" not in jobs["test"]:
            raise ValueError("Unsupported unsharded historical core/coverage contract")
        if "sandbox-tests" in jobs or "core-coverage" in jobs:
            raise ValueError("Ambiguous historical combined core layout")
    elif len(core_shards) != 4 or "sandbox-tests" not in jobs or "core-coverage" not in jobs:
        raise ValueError("Unsupported sharded historical core layout")
    servers = shards(jobs["server-suite"])
    server_count = re.search(r"^  NUM_SHARDS: [\"\']?(\d+)", workflow, re.MULTILINE)
    if not servers or not server_count or len(servers) != int(server_count[1]):
        raise ValueError("Server matrix and script shard count disagree")
    settings = {
        "legacy-core" if legacy else "core": job_settings(jobs["test"]),
        "servers": job_settings(jobs["server-suite"]),
        "session-routing": job_settings(jobs["session-routing-e2e"]),
        "lint": job_settings(lint["lint-check"]),
    }
    if not legacy:
        settings["sandbox"] = job_settings(jobs["sandbox-tests"])
        settings["changed-servers"] = job_settings(jobs["changed-server-tests"])
    return {
        "layout": "legacy-combined" if legacy else "sharded",
        "core_shards": len(core_shards) or 1,
        "server_shards": len(servers),
        "uv_version": pin[1],
        "jobs": settings,
        "coverage": job_settings(jobs["core-coverage"]) if not legacy else None,
    }


def legacy_command(source: Path) -> str:
    """Retain the sampled main's combined core/sandbox/gate shell verbatim."""
    block = job_blocks((source / ".github/workflows/unit-tests.yml").read_text())["test"]
    match = re.search(
        r"      - name: Core \+ sandbox unit tests \(coverage gate\)\n        run: \|\n((?:          .*\n|\n)+)", block
    )
    if not match:
        raise ValueError("Cannot identify historical combined coverage command")
    return (
        "\n".join(line[10:] if line.startswith("          ") else line for line in match[1].splitlines()).rstrip()
        + "\n"
    )
