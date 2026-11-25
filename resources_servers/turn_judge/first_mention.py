# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Where a rollout first mentions a decision (stdlib only; runs in the judge sandbox as /judge/bin/first_mention.py).

    python3 /judge/bin/first_mention.py R<k> '<regex>' ['<regex>' ...] [--max N]

Lists, in trajectory order, every model call whose reasoning, messages or tool-call arguments match any regex
(case-insensitive), with the item #idx, every match offset and 200 characters of context around the first; tool
outputs are not searched (they are what the agent saw, not what it decided). Use it to find the first call that commits to a decision
(e.g. a symbol, file or phrase from the decision list), then read that call in full. Navigation only."""

import re
import sys
from pathlib import Path

HDR = re.compile(r"^===== #(-?\d+) \[call (-?\d+)\] (\S+) =====$")
SEARCHED = ("reasoning", "message", "function_call")


def items(path: Path) -> list[tuple[int, int, str, str]]:
    lines = []
    for ln in path.read_text(errors="replace").split("\n"):
        if ln.startswith("↪ ") and lines:
            lines[-1] += ln[2:]
        else:
            lines.append(ln)
    out, cur, buf = [], None, []
    for ln in lines + ["===== #-1 [call -1] end ====="]:
        m = HDR.match(ln)
        if m:
            if cur is not None:
                out.append((cur[0], cur[1], cur[2], "\n".join(buf)))
            cur, buf = (int(m.group(1)), int(m.group(2)), m.group(3)), []
        else:
            buf.append(ln)
    return out


def main() -> int:
    args = sys.argv[1:]
    max_hits = 30
    if "--max" in args:
        k = args.index("--max")
        max_hits = int(args[k + 1])
        args = args[:k] + args[k + 2 :]
    if len(args) < 2:
        print(__doc__)
        return 2
    rk, patterns = args[0], args[1:]
    path = Path("/judge/ws") / rk / "trajectory.txt"
    if not path.exists():
        print(f"no trajectory {path}")
        return 1
    try:
        rxs = [re.compile(p, re.I) for p in patterns]
    except re.error as e:
        print(f"bad regex: {e}")
        return 2
    n = 0
    for idx, call, typ, text in items(path):
        if typ not in SEARCHED or idx < 0:
            continue
        ms = sorted((m.start(), m.end()) for rx in rxs for m in rx.finditer(text))
        if not ms:
            continue
        s0, e0 = ms[0]
        ctx = re.sub(r"\s+", " ", text[max(0, s0 - 100) : e0 + 100])
        offs = ",".join(str(o) for o, _ in ms[:12]) + (",..." if len(ms) > 12 else "")
        print(f"call {call} #{idx} {typ} @{offs} of {len(text)}: ...{ctx}...")
        n += 1
        if n >= max_hits:
            print(f"(stopped after {max_hits} items; pass --max N to see more)")
            break
    if n == 0:
        print(f"no reasoning / message / tool-call item of {rk} matches {patterns!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
