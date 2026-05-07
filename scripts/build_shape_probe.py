# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""One-off: build a 5-variant gym-debug probe under /tmp/skill_probe/ and emit
a shape_probe.jsonl. Each variant is a clone of .claude/skills/gym-debug with
the same "Before you answer" checklist content inserted after the frontmatter,
using a different heading/bullet shape. Control gets no checklist. SKILL.md
differs only in that injection.

Run from repo root:
    .venv/bin/python scripts/build_shape_probe.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SRC = REPO / ".claude/skills/gym-debug"
PROBE_ROOT = Path("/tmp/skill_probe")

CHECKLIST_BODY = [
    "Confirm servers are healthy via `ng_status` before investigating code. If any server is unhealthy, the issue is infrastructure, not your benchmark.",
    "Pull one raw rollout from the JSONL and read it directly — don't rely on aggregates. Check `output_text`, `tool_calls`, and `response.usage.output_tokens`. Low `output_tokens` means the model ran out of budget. Empty `output_text` after many tool calls means the model thrashed on a tool and gave up.",
    "Distinguish infra failure from reward-wrong failure. `reward=0.0` across many rollouts can mean (a) server didn't respond, (b) model produced bad output, or (c) `verify()` rejected valid output — inspect the raw rollout before blaming the code.",
    "Check that the sandbox environment actually provides the tools your prompts assume. Missing binaries surface as empty `output_text`, not as errors.",
    "Cite specific evidence from the rollout (quote exact `tool_calls` or `output_text` substrings) before making a diagnosis.",
]

PREAMBLE = (
    "When diagnosing a NeMo Gym issue, do NOT guess from the error message alone. "
    "Work through every item below before writing your answer:\n\n"
)

VARIANTS: dict[str, tuple[str, str] | None] = {
    "control": None,
    "before-dash": ("## Before you answer — diagnostic checklist", "-"),
    "before-box": ("## Before you answer — diagnostic checklist", "- [ ]"),
    "todo-box": ("## TODO", "- [ ]"),
    "tasks-box": ("## Tasks", "- [ ]"),
}


def render_checklist(heading: str, bullet: str) -> str:
    bullets = "\n".join(f"{bullet} {item}" for item in CHECKLIST_BODY)
    return f"\n{heading}\n\n{PREAMBLE}{bullets}\n"


def patch_skill_md(original: str, injection: str | None, variant_name: str) -> str:
    lines = original.split("\n")
    closes = [i for i, line in enumerate(lines) if line.strip() == "---"]
    if len(closes) < 2:
        raise RuntimeError("expected a YAML frontmatter block")
    # Rename skill inside frontmatter so logs read clearly.
    for i, line in enumerate(lines[: closes[1]]):
        if line.startswith("name: gym-debug"):
            lines[i] = f"name: gym-debug-{variant_name}"
            break
    insert_at = closes[1] + 1
    head = "\n".join(lines[:insert_at])
    tail = "\n".join(lines[insert_at:])
    if injection is None:
        return f"{head}\n{tail}"
    return f"{head}\n{injection}\n{tail}"


def build_variant(name: str, spec: tuple[str, str] | None) -> Path:
    target = PROBE_ROOT / name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(SRC, target)
    skill_md = (target / "SKILL.md").read_text(encoding="utf-8")
    injection = render_checklist(*spec) if spec else None
    (target / "SKILL.md").write_text(patch_skill_md(skill_md, injection, name), encoding="utf-8")
    return target


def main() -> int:
    if PROBE_ROOT.exists():
        shutil.rmtree(PROBE_ROOT)
    PROBE_ROOT.mkdir(parents=True)

    for name, spec in VARIANTS.items():
        target = build_variant(name, spec)
        size = (target / "SKILL.md").stat().st_size
        print(f"  wrote {target.name:16s}  SKILL.md={size} bytes")

    out = REPO / "responses_api_agents/skill_eval_agent/data/shape_probe.jsonl"
    subprocess.run(
        [
            ".venv/bin/python",
            "scripts/build_skill_eval_jsonl.py",
            "--skills-dir",
            str(PROBE_ROOT),
            "--output",
            str(out),
        ],
        cwd=REPO,
        check=True,
    )

    seen: dict[str, str] = {}
    for line in out.read_text().splitlines():
        md = json.loads(line)["verifier_metadata"]
        seen[md["skill_name"]] = md["skill_md_sha"]
    print("\nprovenance:")
    for n, sha in sorted(seen.items()):
        print(f"  {n:28s} {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
