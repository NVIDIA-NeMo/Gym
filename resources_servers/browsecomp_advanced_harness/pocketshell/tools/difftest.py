#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Differential test: pocketshell vs real GNU bash, on REAL agent-written commands.

This is the correctness gate for replacing the subprocess. Unit tests cover
semantics we thought of; this covers the semantics agents actually used.

Corpus: bash_command keystrokes extracted from recorded agent trajectories, one
command per line. Only commands the OLD guard ALLOWED are replayed -- blocked
ones never executed, so they have no ground truth.

Both engines run against an identical synthetic workspace whose page filenames
are the ones the sampled commands actually reference, filled with real page
prose. Any stdout difference is therefore a pocketshell bug, except for the
deliberate confinement divergence (bash can read /etc; pocketshell cannot),
which is reported separately.

Usage:
  difftest.py --calls-glob '/path/to/corpus/*.calls.txt' --n 20000
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pocketshell import run as ps_run  # noqa: E402


BASH = "/bin/bash"
PAGE_REF = re.compile(r"pages/[A-Za-z0-9_.*?\[\]-]+")
OUTSIDE = re.compile(r"(?<![\w/.])(?:/etc/|/home/|/root/|/proc/|/sys/|/var/|/usr/|/tmp/|~/|\.\./|/lustre/)")


def load_legacy_guard(app_py: str):
    """Load the OLD harness guard so we replay only commands that actually RAN.

    Without this the corpus still contains everything the model *tried* --
    md5sum, python3, curl -- which the deny/allow list rejected. Replaying those
    compares pocketshell against a bash that really does have python3, and
    manufactures 'mismatches' for commands that never executed in production.
    """
    src = Path(app_py).read_text()
    start = src.index("_BASH_DENY_COMMANDS = {")
    m = re.search(r"\ndef _bash_allowlisted\(keystrokes\):.*?\n(?=\n[A-Za-z_@#]|\nclass )", src[start:], re.S)
    if not m:
        raise SystemExit("could not slice the legacy guard out of app.py")
    ns = {"re": re, "Path": Path}
    exec(compile(src[start : start + m.end()], app_py, "exec"), ns)
    deny, allow = ns["_bash_denylisted"], ns["_bash_allowlisted"]
    return lambda ks: (deny(ks) or allow(ks)) is None


def load_keystrokes(globs: list[str]) -> list[str]:
    out = []
    for g in globs:
        for f in sorted(Path("/").glob(g.lstrip("/"))):
            with f.open(errors="replace") as fh:
                for line in fh:
                    _, _, payload = line.partition(":")
                    try:
                        item = json.loads(payload)
                        args = json.loads(item.get("arguments") or "{}")
                    except Exception:
                        continue
                    if not isinstance(args, dict):
                        continue
                    ks = args.get("keystrokes")
                    if isinstance(ks, str) and ks.strip():
                        out.append(ks)
    return out


def build_workspace(cmds: list[str], corpus_text: list[str]) -> Path:
    """Create a workspace containing every concrete pages/<file> the sample references."""
    ws = Path(tempfile.mkdtemp(prefix="psdiff_"))
    pages = ws / "pages"
    pages.mkdir()
    names: set[str] = set()
    for c in cmds:
        for ref in PAGE_REF.findall(c):
            name = ref.split("/", 1)[1]
            if any(ch in name for ch in "*?["):
                # materialise a couple of plausible concrete names for the glob
                stem = re.split(r"[*?\[]", name)[0]
                if stem:
                    names.add(f"{stem}glob_a.txt")
                    names.add(f"{stem}glob_b.txt")
                continue
            names.add(name)
    for i, name in enumerate(sorted(names)[:4000]):
        if not name or name in (".", ".."):
            continue
        body = "\n".join(corpus_text[(i * 7 + k) % len(corpus_text)] for k in range(60))
        try:
            (pages / name).write_text(body + "\n", errors="surrogateescape")
        except (OSError, ValueError):
            continue
    # a few always-present fixtures the agents assume exist
    for k in range(6):
        p = pages / f"{k:04d}_browse_fixture_{k}.txt"
        if not p.exists():
            p.write_text(
                "\n".join(corpus_text[(k * 13 + j) % len(corpus_text)] for j in range(80)) + "\n",
                errors="surrogateescape",
            )
    (ws / "manifest.tsv").write_text(
        "".join(f"{i:04d}\tbrowse\thttps://example{i}.org/path\tTitle {i}\t{100 + i}\n" for i in range(60)),
        errors="surrogateescape",
    )
    return ws


def run_bash(cmd: str, cwd: Path, timeout: float):
    wrapped = f"ulimit -c 0 2>/dev/null; ulimit -f 0 2>/dev/null; {cmd}"
    try:
        p = subprocess.run(
            [BASH, "-c", wrapped],
            cwd=cwd,
            capture_output=True,
            timeout=timeout,
            env={"PATH": "/usr/bin:/bin", "HOME": str(cwd), "LC_ALL": "C.UTF-8"},
        )
    except subprocess.TimeoutExpired:
        return None, None
    return p.stdout.decode("utf-8", "replace"), p.returncode


def norm(s: str) -> str:
    return s.replace("\r\n", "\n").rstrip("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls-glob", action="append", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=25)
    ap.add_argument(
        "--legacy-guard",
        default=str(Path(__file__).resolve().parents[2] / "app.py"),
        help="harness app.py to source the OLD deny/allow guard from",
    )
    args = ap.parse_args()

    ks_all = load_keystrokes(args.calls_glob)
    allowed = load_legacy_guard(args.legacy_guard)
    ks = [k for k in ks_all if allowed(k)]
    print(
        f"[corpus] {len(ks_all):,} keystrokes; {len(ks):,} passed the OLD guard (i.e. actually executed in production)"
    )
    rnd = random.Random(args.seed)
    sample = rnd.sample(ks, min(args.n, len(ks)))

    corpus_text = []
    for k in ks[:20000]:
        corpus_text.append(k)
    corpus_text += [
        "The quick brown fox jumps over the lazy dog in 1983.",
        "David Pemantle received his PhD in 1988 from MIT.",
        '{"id": "https://openalex.org/W123", "title": "A Study", "year": 2014}',
        "Table 1 | Year | Value | Notes |",
        "COST: $1,200 (approx)",
        "Retrieved 2024-01-15 from https://example.org/page",
        '  <div class="content">Some HTML text</div>',
        "Alpha Beta Gamma Delta Epsilon",
        "",
        "no match here at all",
    ]

    ws = build_workspace(sample, corpus_text)
    print(f"[workspace] {ws}  ({sum(1 for _ in (ws / 'pages').iterdir())} pages)")

    stats = Counter()
    examples: dict[str, list] = {}
    replayed = 0

    for cmd in sample:
        b_out, b_code = run_bash(cmd, ws, args.timeout)
        if b_out is None:
            stats["skipped: bash timeout"] += 1
            continue
        r = ps_run(cmd, workspace=str(ws), timeout=args.timeout)
        replayed += 1

        if norm(r.stdout) == norm(b_out):
            stats["stdout MATCH"] += 1
            if r.exit_code == b_code:
                stats["  (exit code also matches)"] += 1
            continue

        if sorted(norm(r.stdout).split("\n")) == sorted(norm(b_out).split("\n")):
            # same lines, different order -- grep -r walks readdir order in GNU
            # grep; pocketshell sorts, which is deterministic and preferable.
            stats["divergent: line ORDER only (grep -r)"] += 1
            continue

        if OUTSIDE.search(cmd):
            stats["divergent: confinement (by design)"] += 1
            key = "confinement"
        elif r.exit_code == -3:
            stats["divergent: unsupported syntax (fails closed)"] += 1
            key = "unsupported:" + (r.stderr[:60])
        else:
            stats["DIVERGENT: real mismatch"] += 1
            key = "mismatch"
        examples.setdefault(key, [])
        if len(examples[key]) < args.show:
            bl, ml = norm(b_out).split("\n"), norm(r.stdout).split("\n")
            k = 0
            while k < min(len(bl), len(ml)) and bl[k] == ml[k]:
                k += 1
            ctx = (
                f"[first diff at line {k + 1} of {len(bl)}/{len(ml)}]\n      bash: {bl[k][:150]!r}\n"
                if k < len(bl)
                else f"[bash ended at line {len(bl)}; mine has {len(ml)}]\n"
            )
            ctx += f"      mine: {ml[k][:150]!r}" if k < len(ml) else "      mine: <ended>"
            examples[key].append((cmd, ctx, "", r.stderr[:160]))

    print(f"\n[replayed] {replayed:,}")
    for k, v in stats.most_common():
        print(f"  {k:<46} {v:>8,}  {100.0 * v / max(replayed, 1):6.2f}%")

    for key, rows in examples.items():
        if key == "confinement":
            continue
        print(f"\n===== examples [{key[:70]}] =====")
        for cmd, ctx, _m, err in rows[: args.show]:
            print(f"  CMD   {cmd[:200]}")
            print(f"      {ctx}")
            if err:
                print(f"      stderr={err[:130]!r}")
            print()

    match = stats["stdout MATCH"]
    print(
        f"\nAGREEMENT (stdout, excluding by-design confinement): "
        f"{100.0 * match / max(replayed - stats['divergent: confinement (by design)'], 1):.3f}%"
    )
    print(f"workspace kept for inspection: {ws}")


if __name__ == "__main__":
    main()
