# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""compute-eval dataset prep.

Fetches the per-release problems tarball from the public NVIDIA/compute-eval
GitHub repo (``data/releases/<release>-problems.tar.gz``). This is the
same data the HuggingFace dataset ``nvidia/compute-eval`` exposes, but
served straight from the source so prepare doesn't need HF_TOKEN and
doesn't depend on the host's ``datasets`` version (``nvidia/compute-eval``'s
HF metadata references a ``Json`` feature type that was removed in
``datasets`` 4.x, which the Gym container ships).

The Skills counterpart at ``nemo_skills/dataset/compute-eval/prepare.py``
uses ``datasets.load_dataset("nvidia/compute-eval", ...)`` because Skills'
nemo-skills container ships an older ``datasets``. Both paths yield
identical problems; B3 data validation confirms parity.
"""

import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path

import orjson


DATA_DIR = Path(__file__).parent / "data"
DEFAULT_RELEASE = "2026-1"
# Pin the upstream commit so the data is reproducible. eede5ce ("Add eng diary;
# sync problems") is the rev whose tarballs match what HuggingFace currently
# serves for nvidia/compute-eval — confirmed by byte-comparing build_command,
# problem_prompt, and context_files_block fields against Skills' HF-sourced
# eval.jsonl. The earlier e01a5d2 ("Release 2026.1") tarball is stale: HF was
# updated to eede5ce's content but the pkg release wasn't bumped.
UPSTREAM_REV = "eede5ce"
TARBALL_URL_TMPL = (
    "https://raw.githubusercontent.com/NVIDIA/compute-eval/{rev}/data/releases/{release}-problems.tar.gz"
)

_CONTEXT_FILES_BLOCK_TEMPLATE = """
--- file: {path}
```{fence}
{content}
```
"""


def _fence_for_path(path: str) -> str:
    p = path.lower()
    if p.endswith((".cu", ".cuh")):
        return "cuda"
    if p.endswith((".cc", ".cpp", ".cxx")):
        return "cpp"
    if p.endswith(".c"):
        return "c"
    if p.endswith(".h") or p.endswith(".hpp"):
        return "h"
    return ""


def _format_context_files_block(context_files: list[dict[str, str]]) -> str:
    blocks: list[str] = []
    for source in context_files:
        if "path" not in source or "content" not in source:
            continue
        fence = _fence_for_path(source["path"])
        blocks.append(
            _CONTEXT_FILES_BLOCK_TEMPLATE.format(path=source["path"], fence=fence, content=source["content"])
        )
    return "".join(blocks)


def _load_problems(release: str, rev: str = UPSTREAM_REV) -> list[dict]:
    url = TARBALL_URL_TMPL.format(rev=rev, release=release)
    print(f"Fetching {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        blob = resp.read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        member = tf.getmember("problems.jsonl")
        f = tf.extractfile(member)
        if f is None:
            raise RuntimeError(f"problems.jsonl is unreadable inside {url}")
        return [json.loads(line) for line in f.read().decode("utf-8").splitlines() if line.strip()]


def prepare(
    output_path: Path = DATA_DIR / "compute_eval_benchmark.jsonl",
    release: str = DEFAULT_RELEASE,
    rev: str = UPSTREAM_REV,
) -> Path:
    problems = _load_problems(release, rev=rev)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(output_path, "wb") as f:
        for item in problems:
            out = {
                "problem_prompt": item["prompt"],
                "build_command": item["build_command"],
                "context_files_block": _format_context_files_block(item["context_files"]),
                "verifier_metadata": {
                    "task_id": item["task_id"],
                    "problem": item,
                },
            }
            f.write(orjson.dumps(out, option=orjson.OPT_SERIALIZE_NUMPY) + b"\n")
            n_written += 1

    print(f"Wrote {n_written} problems from compute-eval {release} @ {rev[:7]} to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NVIDIA/compute-eval for NeMo Gym (offline)")
    parser.add_argument(
        "--release",
        default=DEFAULT_RELEASE,
        help="Release name (e.g. '2025-1', '2025-2', '2025-3', '2026-1').",
    )
    parser.add_argument(
        "--rev",
        default=UPSTREAM_REV,
        help="Upstream NVIDIA/compute-eval git rev (commit or tag) to fetch tarball from.",
    )
    parser.add_argument("--output-path", type=Path, default=DATA_DIR / "compute_eval_benchmark.jsonl")
    args = parser.parse_args()
    prepare(output_path=args.output_path, release=args.release, rev=args.rev)
