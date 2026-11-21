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
"""Prompt text and image rendering of the ARC code agent.

The system prompt is built in the same steps as the original harness: a base prompt, reworded for a large token
budget and many code runs, a note on the images, the helper-function list, the sandbox-library list and the output
cap. Each rewording asserts that the sentence it replaces exists, so a change to the base text fails loudly.
"""

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image


Grid = Sequence[Sequence[int]]

SANDBOX_DIR = Path(__file__).parent / "sandbox"

# The standard ARC palette, matching the colour names in the system prompt.
PALETTE = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
}
CANVAS = (246, 246, 246)  # margin colour of the padded ("pad32") image style

TOOL_OUTPUT_CHARS = 1000  # the base prompt's output cap, replaced by the configured out_chars
BASE_TOKEN_BUDGET = 20000

TEXT_RUN_SENTENCE = (
    "- Run Python by writing code between <code> and </code>. Generation pauses after </code>, the code runs, and what it prints "
    "comes back in the next message inside <tool_response> and </tool_response>. Variables and functions you define stay defined for later runs."
)
NATIVE_RUN_SENTENCE = (
    "- Run Python by calling the `python` tool with the code in its `code` argument. The code runs, and what it prints comes back as "
    "the tool result. Variables and functions you define stay defined for later runs."
)

# The function tool offered when the model calls tools natively.
PYTHON_TOOL = {
    "type": "function",
    "name": "python",
    "description": (
        "Run Python code in the stateful sandbox and return what it prints. Variables and functions you define stay defined "
        "for later calls; the puzzle and the helper functions are preloaded."
    ),
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "The Python code to run."}},
        "required": ["code"],
    },
    "strict": False,
}

BASE_SYSTEM_PROMPT = f"""Solve this ARC-AGI puzzle by finding the hidden rule and writing the output grid for the test input. A grid is a rectangle of digits 0-9, where 0 is the background colour and the other digits are distinct colours (1 blue, 2 red, 3 green, 4 yellow, 5 grey, 6 magenta, 7 orange, 8 sky, 9 brown). Every example pair was produced by the same hidden transformation.

You have a stateful Python sandbox (numpy as np, scipy, networkx) and a tight token budget: the whole conversation, puzzle included, must fit in about {BASE_TOKEN_BUDGET} tokens.
- Run Python by writing code between <code> and </code>. Generation pauses after </code>, the code runs, and what it prints comes back in the next message inside <tool_response> and </tool_response>. Variables and functions you define stay defined for later runs.
- The puzzle is preloaded: `examples` is a list of dicts with "input" and "output" numpy arrays, and `test_inputs` is a list of arrays. The grids are also written below.
- Preloaded helpers print compact summaries:
  - `check(transform)`: runs `transform` on every example pair; prints pass/fail per pair and, for a failure, the count and first mismatched cells as (row,col):predicted>expected. This is your main feedback.
  - `objects(g)`: components as id, colour, size, bbox. `diff(a, b)`: changed cells between two grids. `crop(g, r0, c0, r1, c1)`: a small region with row numbers.
  - `predict(transform)`: prints the prediction for each test input as digit rows.
- Printed output beyond {TOOL_OUTPUT_CHARS} characters is cut off. Never print whole grids or raw arrays; the grids are already below. Writing a lot of code is fine; keep what it prints short and decision-oriented.

How to work
Your first action must be a code run. Go from the Observations to a hypothesis in a few sentences, then write the code. Do not transcribe or analyse grids cell by cell in your reasoning: that costs thousands of tokens, while `check` compares every cell of every pair in one run, and `diff`, `objects` and `crop` show the details on demand. If you notice you have been reasoning for a long time without running anything, stop and write a first candidate `transform` anyway; a failed `check` teaches more than more thinking.
1. Read the Observations and the pairs below and form a hypothesis. Rules usually have one main step plus a condition or exception. Common families: objects (move, copy, recolour, delete, select by size or count), geometry (rotate, reflect, tile, scale, crop), filling (holes, enclosed areas, lines extended until blocked), relations (nearest, inside, touching, same shape), and counting (output size or colour depends on how many of something there are).
2. Your first code run must already define a candidate `transform(grid: np.ndarray) -> np.ndarray` and run `check(transform)`. Do not explore before testing.
3. On failure, fix the first failing pair: look at its `diff`, `objects`, or a `crop` only if needed, revise the rule rather than only the code, and rerun `check`. Keep any text before a code run to three short lines: Hypothesis / Evidence against / Next test. Think concisely; long deliberation that does not change your next step wastes the budget.
4. When every pair passes, run `predict(transform)`, then write the answer. If the budget runs low, answer with your best `transform`'s prediction anyway.

Answer
Finish with the output grid for the test input inside <answer> and </answer>, one row per line, digits separated by single spaces, and nothing else between the tags. Only this block is scored, and it is correct only if every cell matches. Example for a rule that reflects the input left to right:

<answer>
2 1 0 0
0 0 3 0
</answer>"""

IMAGE_NOTE = """Each grid in the puzzle is given twice: as digits in text, and as an image placed right after its digits. The images use the standard colour scheme (0 black, 1 blue, 2 red, 3 green, 4 yellow, 5 gray, 6 magenta, 7 orange, 8 azure, 9 maroon), one square per cell, no grid lines. The digits are the exact data. Use the images to see shapes, objects, symmetry and layout at a glance, and rely on the digits and on code for exact colours, counts and positions.

"""

HELPERS_OLD = (
    "  - `objects(g)`: components as id, colour, size, bbox. `diff(a, b)`: changed cells between two grids. `crop(g, r0, c0, r1, c1)`: "
    "a small region with row numbers."
)
HELPERS_V2_DOC = (
    "  - `objects(g, bg='auto', connectivity=4, quiet=False)`: connected components, largest first. It prints a short list (id, colour, "
    "size, bbox) unless quiet=True, and returns the objects; each has `.color`, `.size` (cell count), `.bbox` (r0, c0, r1, c1 inclusive), "
    "`.cells` (list of (row, col)), `.mask` (bool array of the grid's shape), `.shape` (cells moved to the origin, so equal shapes compare "
    "equal) and `.sub` (the object cropped to its bbox). The background defaults to the grid's most common colour.\n"
    "  - `bbox(g_or_mask, color=None)`: (r0, c0, r1, c1) of a bool mask, of one colour, or of all non-background cells. `crop(g, r0, c0, r1, "
    "c1)`: a small region with row numbers. `diff(a, b)`: changed cells between two grids.\n"
    "  - `bg(g)`: the most common colour. `colors(g)`: [(colour, count), ...], most common first.\n"
    "  - `variants(g)`: the 8 rotations and reflections as [(name, grid), ...]. `norm(cells)`: a set of cells moved to the origin.\n"
    "  - `find(g, pattern, all_variants=False)`: every (row, col) where `pattern` occurs in `g` (with the variant name when "
    "all_variants=True). `holes(g, color)`: bool mask of the cells enclosed by `color`."
)
HELPERS_V3_NEW_DOC = (
    "  - `regions(mask, connectivity=4, grid=None)`: connected regions of a bool mask as objects (for example `g != bg(g)` joins "
    "touching cells of different colours); with `grid`, `.color` is the main colour and `.colors` the set of colours under each.\n"
    "  - `holes(obj_or_mask)`: the enclosed regions inside it, one object each; objects also have `.n_holes` and "
    "`.touches_border`. `inside(a, b)`: whether `a` lies in a hole of `b`.\n"
    "  - `panels(g, sep=None)`: splits a grid on full rows and columns of one colour; returns a dict with 'sep', 'rows' and 'cols' "
    "(inclusive ranges) and 'grid' (the subgrids, row-major).\n"
    "  - `runs(g, color=None, dirs=8, min_len=2)`: straight same-colour segments as (colour, start, end, direction, length). "
    "`ray(g, r, c, dr, dc, stop=None)`: the cells from (r, c) in a direction until the edge or a non-background (or `stop`) cell.\n"
    "  - `canon(x)`: a key for a shape that is the same under rotation and reflection; `variants(cells)` also works on cell sets.\n"
    "  - `fits(shape, region, used=None, variants=False)`: the offsets where a shape fits inside a region mask without touching "
    "`used`."
)

# Sandbox libraries: the sandbox also preloads these names (see HYGIENE_HEADER) and allows these extra modules.
HYGIENE_IMPORTS = ["traceback", "time", "types", "inspect", "textwrap", "sys"]
HYGIENE_RECURSION = 10000
HYGIENE_HEADER = """
from scipy import ndimage
from scipy.ndimage import label, binary_fill_holes
from collections import Counter, defaultdict, deque
import itertools
from itertools import combinations, permutations, product
"""
HYGIENE_OLD = "(numpy as np, scipy, networkx)"
HYGIENE_NEW = (
    "(numpy as np, scipy, networkx; also preloaded: `ndimage`, `label`, `binary_fill_holes`, `Counter`, `defaultdict`, `deque`, "
    "`itertools`, `combinations`, `permutations`, `product`)"
)

# Makes numbers print as 3 (not np.int64(3)) and `predict` return a short repr instead of a list of arrays.
TOOL_FIX_HEADER = """
try:
    np.set_printoptions(legacy="1.25")   # numbers print as 3, not np.int64(3)
except Exception:
    pass
class _Predictions(list):
    def __repr__(self):
        return f"<{len(self)} predicted grid(s), printed above>"
_predict_printing = predict
def predict(transform, inputs=None, timeout=CHECK_TIMEOUT_SECONDS):
    return _Predictions(_predict_printing(transform, inputs, timeout))
"""

# Harness messages.
CUT_NUDGE = (
    "[harness] Your previous turn hit the per-turn thinking limit. Stop analysing: test your current best hypothesis with a code run "
    "(`check(transform)`), or, if you are confident, give the final answer inside <answer> and </answer>."
)
NOCODE_NUDGE = (
    "[harness] That turn had no code run and no final answer. Write code now: define a candidate `transform` and run "
    "`check(transform)`, or give your final answer inside <answer> and </answer>."
)
FINAL_NUDGE = (
    "[harness] You are out of budget. Stop analysing and give your final answer now: the output grid for the test input inside "
    "<answer> and </answer>, one row per line, digits separated by single spaces, nothing else between the tags."
)
RESEND_NUDGE = (
    "[harness] Your <answer> block could not be read: {why}. Write the whole output grid again inside <answer> and </answer>, one "
    "row per line, digits separated by single spaces, every row the same length."
)


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"base prompt changed: {old[:60]!r}"
    return text.replace(old, new)


def build_system_prompt(
    budget: int,
    max_runs: int,
    out_chars: int = TOOL_OUTPUT_CHARS,
    *,
    helpers_v2: bool = True,
    helpers_v3_new: bool = True,
    sandbox_hygiene: bool = True,
    native_tools: bool = False,
) -> str:
    """The system prompt: base prompt for a large budget and many runs, image note, helper list, sandbox libraries, output cap.

    With ``native_tools`` the code-running sentence describes the `python` function tool instead of <code> blocks."""
    out = BASE_SYSTEM_PROMPT
    if native_tools:
        out = _replace_once(out, TEXT_RUN_SENTENCE, NATIVE_RUN_SENTENCE)
    out = _replace_once(
        out,
        f"a tight token budget: the whole conversation, puzzle included, must fit in about {BASE_TOKEN_BUDGET} tokens.",
        f"a large but finite token budget: the whole conversation, puzzle included, must fit in about {budget} tokens. Work in many rounds: "
        f"after each code run you see its output and can continue, up to {max_runs} runs.",
    )
    out = _replace_once(
        out,
        "Your first action must be a code run.",
        "Test early: after a short hypothesis, use code instead of long reasoning, and use the output of each run to decide the next step.",
    )
    out = _replace_once(
        out,
        "2. Your first code run must already define a candidate `transform(grid: np.ndarray) -> np.ndarray` and run "
        "`check(transform)`. Do not explore before testing.",
        "2. Define a candidate `transform(grid: np.ndarray) -> np.ndarray` and run `check(transform)` early. Exploring with `objects`, "
        "`diff` or `crop` first is fine when it helps, but come back to `check` quickly.",
    )
    anchor = "\nHow to work\n"
    assert out.count(anchor) == 1
    out = out.replace(anchor, "\n" + IMAGE_NOTE.rstrip("\n") + "\n" + anchor)
    assert helpers_v2 or not helpers_v3_new, "helpers_v3_new builds on helpers_v2 (it uses `bg`)"
    if helpers_v2:
        out = _replace_once(out, HELPERS_OLD, HELPERS_V2_DOC + ("\n" + HELPERS_V3_NEW_DOC if helpers_v3_new else ""))
    if sandbox_hygiene:
        out = _replace_once(out, HYGIENE_OLD, HYGIENE_NEW)
    if out_chars != TOOL_OUTPUT_CHARS:
        out = _replace_once(
            out,
            f"Printed output beyond {TOOL_OUTPUT_CHARS} characters is cut off.",
            f"Printed output beyond {out_chars} characters is cut off.",
        )
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Observations and text rendering


def _components(grid: Grid) -> int:
    """Count 4-connected same-colour non-zero components."""
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    count = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 or seen[r][c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r][c] = True
            colour = grid[r][c]
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and grid[ny][nx] == colour:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
    return count


def observations(pairs: Sequence[Dict[str, Any]], test_input: Grid) -> str:
    """Precomputed per-pair facts (size change, changed cells, colours, object counts)."""
    lines = ["Observations (computed):"]
    for index, pair in enumerate(pairs, start=1):
        i, o = pair["input"], pair["output"]
        ci = sorted({v for row in i for v in row})
        co = sorted({v for row in o for v in row})
        shape = f"{len(i)}x{len(i[0])}->{len(o)}x{len(o[0])}"
        if len(i) == len(o) and len(i[0]) == len(o[0]):
            changed = sum(1 for r in range(len(i)) for c in range(len(i[0])) if i[r][c] != o[r][c])
            change = f"changed {changed}/{len(i) * len(i[0])}"
        else:
            change = "size changes"
        lines.append(
            f"- pair {index}: {shape}; {change}; colours in {ci} out {co}; objects in {_components(i)} out {_components(o)}"
        )
    ct = sorted({v for row in test_input for v in row})
    lines.append(f"- test: {len(test_input)}x{len(test_input[0])}; colours {ct}; objects {_components(test_input)}")
    return "\n".join(lines)


def grid_to_text(grid: Grid) -> str:
    """One row per line, digits separated by a space."""
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


def task_header(examples: Sequence[Dict[str, Any]], test_input: Grid) -> str:
    """Python run in the sandbox after the helpers: defines ``examples`` and ``test_inputs`` as numpy arrays."""
    pairs = [
        {"input": [list(map(int, r)) for r in p["input"]], "output": [list(map(int, r)) for r in p["output"]]}
        for p in examples
    ]
    ex_literal = json.dumps(pairs, separators=(",", ":"))
    test_literal = json.dumps([[list(map(int, r)) for r in test_input]], separators=(",", ":"))
    return (
        "import numpy as np\n"
        f"examples = [{{'input': np.array(p['input'], dtype=int), 'output': np.array(p['output'], dtype=int)}} for p in {ex_literal}]\n"
        f"test_inputs = [np.array(g, dtype=int) for g in {test_literal}]\n"
    )


def sandbox_header(
    examples: Sequence[Dict[str, Any]],
    test_input: Grid,
    *,
    tool_fixes: bool = True,
    helpers_v2: bool = True,
    helpers_v3_new: bool = True,
    sandbox_hygiene: bool = True,
) -> str:
    """The full per-puzzle sandbox header: task arrays, number-printing fix, helper functions, preloaded names."""
    header = task_header(examples, test_input)
    if tool_fixes:
        header += TOOL_FIX_HEADER
    if helpers_v2:
        header += "\n" + (SANDBOX_DIR / "sandbox_helpers_v2.py").read_text()
    if helpers_v3_new:
        header += "\n" + (SANDBOX_DIR / "sandbox_helpers_v3_new.py").read_text()
    if sandbox_hygiene:
        header += "\n" + HYGIENE_HEADER
    return header


# ----------------------------------------------------------------------------------------------------------------------
# Image rendering: one image per grid


def grid_image(grid: Grid, target_px: int = 336, scale: int | None = None) -> Image.Image:
    """One square per cell; integer upscaling by ``scale`` (or so the long side is about ``target_px``), nearest neighbour."""
    rows, cols = len(grid), len(grid[0])
    scale = scale or max(1, target_px // max(rows, cols))
    img = Image.new("RGB", (cols, rows))
    px = img.load()
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            px[c, r] = PALETTE.get(int(v), PALETTE[0])
    return img.resize((cols * scale, rows * scale), Image.Resampling.NEAREST)


def pad_to_multiple(img: Image.Image, multiple: int = 32) -> Image.Image:
    """The image on a canvas whose sides are rounded up to ``multiple`` pixels; cells keep their size (padding, not resizing)."""
    size = tuple(-(-side // multiple) * multiple for side in img.size)
    if size == img.size:
        return img
    canvas = Image.new("RGB", size, CANVAS)
    canvas.paste(img, (0, 0))
    return canvas


def render_grid(
    grid: Grid, image_px: int = 336, image_scale: int | None = 12, image_style: str | None = "pad32"
) -> Image.Image:
    """Grid to image for a style: None is the plain image, "pad32" pads it to multiples of 32 pixels."""
    img = grid_image(grid, image_px, image_scale)
    if image_style is None:
        return img
    if image_style == "pad32":
        return pad_to_multiple(img, 32)
    raise ValueError(f"unknown image_style: {image_style!r}")


def png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def user_content(
    examples: Sequence[Dict[str, Any]],
    test_input: Grid,
    image_px: int = 336,
    image_scale: int | None = 12,
    image_style: str | None = "pad32",
) -> List[Dict[str, Any]]:
    """The first user message as Responses API content parts: text with each grid's digits, then that grid's image."""
    pairs = [
        {"input": [list(map(int, r)) for r in p["input"]], "output": [list(map(int, r)) for r in p["output"]]}
        for p in examples
    ]
    test = [list(map(int, r)) for r in test_input]
    parts: List[Dict[str, Any]] = []
    text: List[str] = [observations(pairs, test), ""]

    def flush() -> None:
        if text:
            parts.append({"type": "input_text", "text": "\n".join(text) + "\n"})
            text.clear()

    def grid(g: Grid, header: str) -> None:
        text.append(header)
        text.append(grid_to_text(g))
        flush()
        parts.append(
            {
                "type": "input_image",
                "image_url": png_data_url(render_grid(g, image_px, image_scale, image_style)),
                "detail": "auto",
            }
        )

    for index, pair in enumerate(pairs, start=1):
        text.append(f"Example pair {index}")
        grid(pair["input"], f"Input ({len(pair['input'])}x{len(pair['input'][0])}):")
        grid(pair["output"], f"Output ({len(pair['output'])}x{len(pair['output'][0])}):")
        text.append("")
    grid(test, f"Test input 1 ({len(test)}x{len(test[0])}):")
    flush()
    return parts
