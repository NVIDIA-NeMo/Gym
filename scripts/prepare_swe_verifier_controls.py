# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare matched SWE resource/Harbor verifier controls from a checksum-pinned task roster.

No services or sandboxes start here. Gold and empty controls retain the original tests;
the timeout control skips installs and replaces test_cmd with a marker and bounded sleep.
Use run_swe_verifier_controls.py on the resulting directory to execute the controls.
"""

import argparse
import copy
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

from scripts.swe_rows_to_harbor.convert import convert, server_config


ROOT = Path(__file__).resolve().parents[1]
CASES = ("gold", "empty", "timeout")
FORCED_TIMEOUT_MARKER = "NEMO_GYM_FORCED_TIMEOUT_REACHED_TEST_COMMAND"


def digest(data: bytes) -> str:
    """Hash exact bytes, without newline or Unicode normalization."""
    return hashlib.sha256(data).hexdigest()


def payload_digest(row: dict) -> str:
    """Use the roster's canonical JSON convention, including ASCII escaping."""
    return digest(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())


def clean_revision(root: Path) -> str:
    """Require committed source; untracked run outputs are not source changes."""
    subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD"], check=True)
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def select_rows(
    source: Path, roster_path: Path, image_pins_path: Path, *, count: int
) -> tuple[list[dict], list[dict]]:
    """Validate the pool, ordered roster and saved image pins before changing any row."""
    raw = gzip.decompress(source.read_bytes()) if source.suffix == ".gz" else source.read_bytes()
    roster = json.loads(roster_path.read_text())
    pins = json.loads(image_pins_path.read_text())
    expected = roster["source_uncompressed_sha256"]
    if digest(raw) != expected or pins["source_pool_sha256"] != expected:
        raise ValueError("Source pool, roster and image mapping checksums must agree")
    if not 1 <= count <= len(roster["tasks"]):
        raise ValueError("count must select a nonempty prefix of the recorded roster")
    by_id = {pin["instance_id"]: pin for pin in pins["records"]}
    if len(by_id) != len(pins["records"]):
        raise ValueError("Image mapping contains duplicate task IDs")
    lines = raw.splitlines()
    rows, provenance = [], []
    for entry in roster["tasks"][:count]:
        row = json.loads(lines[entry["source_line_1based"] - 1])
        task_id = entry["instance_id"]
        if row["instance_id"] != task_id or payload_digest(row) != entry["payload_sha256"]:
            raise ValueError(f"Original payload does not match roster: {task_id}")
        pin = by_id[task_id]
        if pin["status"] != "resolved" or pin["original_image"] != row["image_name"]:
            raise ValueError(f"Image mapping does not match source row: {task_id}")
        if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", pin["image"]):
            raise ValueError(f"Image must be pinned to a manifest digest: {task_id}")
        if (pin["os"], pin["architecture"]) != ("linux", "amd64"):
            raise ValueError(f"This control recipe requires linux/amd64 task images: {task_id}")
        if not row.get("patch", "").strip():
            raise ValueError(f"Gold control requires a nonempty patch: {task_id}")
        before = payload_digest(row)
        row["image_name"] = pin["image"]
        rows.append(row)
        provenance.append(
            {
                "instance_id": task_id,
                "source_line_1based": entry["source_line_1based"],
                "original_payload_sha256": before,
                "pinned_payload_sha256": payload_digest(row),
                "original_image": pin["original_image"],
                "image": pin["image"],
                "image_config_digest": pin["config_digest"],
            }
        )
    if len({row["instance_id"] for row in rows}) != count:
        raise ValueError("Roster contains duplicate task IDs")
    return rows, provenance


def control_row(row: dict, case: str) -> dict:
    """Build an explicit control transformation without altering the original row."""
    result = copy.deepcopy(row)
    if case == "empty":
        result["patch"] = ""
    elif case == "timeout":
        result["install_config"]["install"] = []
        result["install_config"]["test_cmd"] = [f"printf '{FORCED_TIMEOUT_MARKER}\\n'; sleep 60"]
    elif case != "gold":
        raise ValueError(f"Unknown control: {case}")
    return result


def prepare(source: Path, roster: Path, image_pins: Path, parser_cache: Path, output: Path, *, count: int) -> dict:
    """Write isolated fixtures and their provenance; reject an existing output directory."""
    revision = clean_revision(ROOT)
    rows, provenance = select_rows(source, roster, image_pins, count=count)
    output.mkdir(parents=True, exist_ok=False)
    source_settings = server_config("swe_rebench")
    source_settings.update(
        log_parser_cache_dir=str(parser_cache.resolve()),
        is_verifying_golden_patch=False,
        inconclusive_verification_retries=1,
    )
    source_settings["sandbox_config"].update(ttl_s=1800, ready_timeout_s=120)
    source_settings["sandbox_config"]["provider_options"]["platform"] = {"os": "linux", "arch": "amd64"}
    fixtures = []
    for case in CASES:
        folder = output / case
        folder.mkdir()
        transformed = [control_row(row, case) for row in rows]
        row_path = folder / "rows.jsonl"
        row_path.write_text("".join(json.dumps(row) + "\n" for row in transformed))
        budget = 30 if case == "timeout" else 180
        conversion = convert(
            row_path,
            folder / "harbor",
            dataset_name=f"swe-verifier-control-{case}-{digest(row_path.read_bytes())[:12]}",
            parser_cache_dir=parser_cache,
            agent_timeout_sec=480,
            evaluation_timeout_sec=budget,
            default_source="swe_rebench",
        )
        if case == "empty":
            for item in conversion["converted"]:
                solution = folder / "harbor" / item["task"] / "solution"
                solution.mkdir()
                script = solution / "solve.sh"
                script.write_text("#!/bin/bash\n# Empty-patch control: leave the worktree unchanged.\nexit 0\n")
                script.chmod(0o755)
        settings = copy.deepcopy(source_settings)
        settings["evaluation_timeout"] = budget
        (folder / "source-settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        fixtures.append(
            {
                "case": case,
                "rows": str(row_path.resolve()),
                "row_sha256": [payload_digest(row) for row in transformed],
                "transformation": {
                    "gold": "none",
                    "empty": "patch := empty; oracle solve.sh := no-op",
                    "timeout": "install := []; test_cmd := print timeout marker, then sleep 60; eval budget := 30 seconds",
                }[case],
                "evaluation_timeout": budget,
                "expected": {
                    "reward": 1.0 if case == "gold" else 0.0,
                    "mask_sample": case == "timeout",
                    "evaluation_completed": case != "timeout",
                },
            }
        )
    manifest = {
        "gym_commit": revision,
        "source_pool_sha256": json.loads(roster.read_text())["source_uncompressed_sha256"],
        "roster_file_sha256": digest(roster.read_bytes()),
        "image_pins_file_sha256": digest(image_pins.read_bytes()),
        "lock_sha256": digest((ROOT / "uv.lock").read_bytes()),
        "count": count,
        "parser_cache": str(parser_cache.resolve()),
        "parser_commit": conversion["swe_rebench_log_parsers_commit"],
        "tasks": provenance,
        "fixtures": fixtures,
        "files_sha256": {
            str(path.relative_to(output)): digest(path.read_bytes())
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
        "coverage": "Verifier controls only. Source flat/typed resource routes and Harbor oracle; no model or old-agent /run.",
    }
    (output / "controls.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Prepare a prefix of the fixed roster; start with one task before the full matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "roster", "image-pins", "parser-cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    manifest = prepare(args.source, args.roster, args.image_pins, args.parser_cache, args.output, count=args.count)
    print(json.dumps({"prepared_tasks": manifest["count"], "manifest": str(args.output / "controls.json")}))


if __name__ == "__main__":
    main()
