# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download and prepare MMRB1 inputs for the selected judge interface."""

import json
import subprocess
import sys
from pathlib import Path
from shutil import copyfile
from urllib.request import urlopen


UPSTREAM_REVISION = "b4b92cdac0df81b2a134bcb1b91411d67b55f80f"  # pragma: allowlist secret


def prepare(source_jsonl: str | Path | None = None, *, judge_mode: str = "genrm") -> Path:
    if judge_mode not in {"genrm", "llm_as_a_judge"}:
        raise ValueError("judge_mode must be genrm or llm_as_a_judge")
    output = Path(__file__).parent / "data" / f"genrm_mmrb1_{judge_mode}_validation.jsonl"
    if source_jsonl:
        source = Path(source_jsonl).expanduser()
        try:
            with source.open() as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            matches = bool(rows) and all(
                isinstance(row, dict)
                and isinstance(row.get("verifier_metadata"), dict)
                and row["verifier_metadata"].get("judge_mode") == judge_mode
                for row in rows
            )
        except json.JSONDecodeError:
            matches = False
        if not matches:
            raise ValueError("Input must match the selected judge_mode")
        if source.resolve() != output.resolve():
            output.parent.mkdir(parents=True, exist_ok=True)
            copyfile(source, output)
    if not output.is_file():
        from benchmarks.genrm_mmrb1.build_validation import prepare as build_validation

        cache = Path(__file__).parent / ".cache" / UPSTREAM_REVISION
        completed = cache / "data/all_data.json"
        if not completed.is_file():
            # Keep upstream download logic outside Gym; pin the script and its input together.
            base_url = f"https://raw.githubusercontent.com/facebookresearch/multimodal_rewardbench/{UPSTREAM_REVISION}"
            for name in ("scripts/0_download_data.py", "data/all_data_release.json"):
                path = cache / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with urlopen(f"{base_url}/{name}", timeout=60) as response:
                    path.write_bytes(response.read())
            # The upstream downloader needs dataset-script support, removed in newer datasets.
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--isolated",
                    "--no-project",
                    "--python",
                    sys.executable,
                    "--with",
                    "datasets==3.1.0",
                    "--with",
                    "Pillow",
                    "--with",
                    "tqdm",
                    "python",
                    str(Path(__file__).with_name("run_downloader.py").resolve()),
                    "scripts/0_download_data.py",
                ],
                cwd=cache,
                check=True,
            )
        if len(json.loads(completed.read_text())) != 4711:
            raise ValueError("Expected the full 4,711-pair MMRB1 public release")
        build_validation(cache, output, genrm=judge_mode == "genrm")
    return output
