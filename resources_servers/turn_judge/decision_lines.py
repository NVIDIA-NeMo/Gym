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
"""Decision statements of one model call (stdlib only; runs in the judge sandbox as /judge/bin/decision_lines.py).

    python3 /judge/bin/decision_lines.py R<k> <call>

For each item of the call: its #idx, type and length, then every sentence with decision wording ("decision", "I'll go
with", "actually", "instead", "wait", "keep", "→", ...) with its character offset in the item, the call's tool calls, a REOPENED flag when the item
reopens the choice after its last decision statement, and the end of items over 3k characters, so that a decision deep inside a very long reasoning item is not missed. Navigation only: read the
item itself before labeling."""

import re
import sys
from pathlib import Path

HDR = re.compile(r"^===== #(-?\d+) \[call (-?\d+)\] (\S+) =====$")
CUE = re.compile(
    r"\b(decision|decide[ds]?|deciding|final(?:ly)?|i'?ll go with|go with|going with|let'?s go with|"
    r"plan\b|approach\b|actually|instead|switch(?:ing)? to|keep(?:ing)?\b|wait\b|rather|choose|chose|"
    r"settle[ds]?|conclu(?:de|sion)|i'?ll (?:use|do|implement|add|change|make|leave|remove)|"
    r"i will (?:use|do|implement|add|change|make|leave|remove)|so the fix|the fix is|choice|"
    r"it is[.:]|go:|design\b|plan:|option\b|placement|to cover all|pragmatic)|[→⇒]",
    re.I,
)
TAIL = 1500  # reasoning items longer than LONG characters: also print their last TAIL characters
LONG = 3000
REOPEN = re.compile(r"\b(wait|reconsider|let me (?:check|verify|confirm)|must check|decisive)\b|\?\s*$", re.I)


def items_of(path: Path, call: int) -> list[tuple[int, str, str]]:
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
            if cur is not None and cur[1] == call:
                out.append((cur[0], cur[2], "\n".join(buf)))
            cur, buf = (int(m.group(1)), int(m.group(2)), m.group(3)), []
        else:
            buf.append(ln)
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    rk, call = sys.argv[1], int(sys.argv[2])
    path = Path("/judge/ws") / rk / "trajectory.txt"
    if not path.exists():
        print(f"no trajectory {path}")
        return 1
    items = items_of(path, call)
    if not items:
        print(f"{rk} has no model call {call}")
        return 1
    for idx, typ, text in items:
        print(f"#{idx} {typ} ({len(text)} chars)")
        if typ == "function_call":
            print(f"  tool call: {' '.join(text.split())[:120]}")
        if typ not in ("reasoning", "message"):
            continue
        sents = [(m.start(), m.group(0)) for m in re.finditer(r"[^.!?\n]+[.!?]?", text)]
        hits = [(o, s.strip()) for o, s in sents if CUE.search(s)]
        for o, s in hits:
            print(f"  @{o}: {s[:240]}")
        if hits:
            print(f"  (last decision statement at @{hits[-1][0]} of {len(text)}; read the item to its end)")
            later = [(o, x) for o, x in sents if o > hits[-1][0] and REOPEN.search(x)]
            if later:
                print(
                    f'  REOPENED after the last decision statement @{later[0][0]}: "{later[0][1].strip()[:200]}" '
                    f"(the call may end open: a lean, precision rule 1)"
                )
        if len(text) > LONG:
            print(f"  --- last {TAIL} characters of #{idx} ---\n{text[-TAIL:]}\n  ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
