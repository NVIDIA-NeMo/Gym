# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare long_transduction benchmark dataset.

Five prompt variants per difficulty tier:

  Arithmetic chain summing (3 variants per max_operands × N_SAMPLES):
    - "unnumbered_streaming_sum" : plain expressions in order
    - "streaming_sum"            : "[N]<expr>" in order
    - "shuffled_streaming_sum"   : "[N]<expr>" shuffled in input

  Per-line UUID sort (3 variants per uuids_per_line × N_SAMPLES):
    - "streaming_uuid_sort"          : "[N](u),(u),..." in order; model sorts
                                       UUIDs within each line by hex order.
    - "shuffled_streaming_uuid_sort" : same but line order is shuffled in input.
    - "unnumbered_uuid_sort"         : no index prefix; positional matching.

  Variable expansion (3 variants per n_variables × N_SAMPLES):
    - "streaming_var_expand"          : "[N]key1+key2" in order; model resolves
                                        each hex key against a shuffled pool of
                                        "key=word" definitions and emits
                                        "[N]word1 word2".
    - "shuffled_streaming_var_expand" : same but expression lines are shuffled;
                                        model must still emit ascending [N].
    - "unnumbered_var_expand"         : no index prefix; positional matching.

  CSV tasks (N_SAMPLES grids × difficulty levels):
    - "csv_permutation_homogeneous"   : N×N grid of 4-digit integers; model
                                        permutes rows and columns per spec.
    - "csv_permutation_heterogeneous" : N×N grid of variable-length UUID
                                        substrings (1–36 chars); same task.
    - "csv_kv_lookup"                 : N×N grid of adjective+noun key
                                        expressions (e.g. a3+v7); model
                                        resolves each cell using provided
                                        lookup tables.

Each row carries a `type` field so the resource server selects the right
parser. Sum variants share an `expressions` payload; uuid_sort variants share
a `uuid_lines` payload; CSV variants share `expected_output`, `n_rows`,
`n_cols`.

Usage:
    python prepare.py
    python prepare.py --force   # regenerate even if output already exists
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "long_transduction.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_UNNUMBERED_STREAMING_SUM = """You are a calculator. You will be given a long sequence of simple arithmetic expressions to evaluate.
Your task is to output each expression and the result of evaluating the expression.

See the example below.

```
Input:

5+6
2+4-1
4+8-3+2

Output:

5+6=11
2+4-1=5
4+8-3+2=11
```

The real sequence will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output an answer to all expressions.
Do not add whitespace.
Do not change the format.

Here is the real sequence.

Input:

{input}"""

PROMPT_STREAMING_SUM = """You are a calculator. You will be given a long sequence of simple arithmetic expressions to evaluate.
Each expression is preceded by a numeric index in brackets like [1], [2], [3], ...
Your task is to output each expression with its index and the result of evaluating the expression.

See the example below.

```
Input:

[1]5+6
[2]2+4-1
[3]4+8-3+2

Output:

[1]5+6=11
[2]2+4-1=5
[3]4+8-3+2=11
```

The real sequence will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output an answer to all expressions.
Do not add whitespace.
Do not change the format.

Here is the real sequence.

Input:

{input}"""


PROMPT_SHUFFLED_STREAMING_SUM = """You are a calculator. You will be given a long sequence of simple arithmetic expressions to evaluate.
Each expression is preceded by a numeric index in brackets like [1], [2], [3], ...
The input expressions are SHUFFLED — they appear in arbitrary order, not in numerical order.
Your task is to output each expression with its index and the result, IN ASCENDING ORDER OF INDEX, starting at [1].

See the example below.

```
Input:

[2]2+4-1
[3]4+8-3+2
[1]5+6

Output:

[1]5+6=11
[2]2+4-1=5
[3]4+8-3+2=11
```


The real sequence will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output an answer to all expressions.
Do not add whitespace.
Do not change the format.

Here is the real sequence.

Input:

{input}"""

PROMPT_STREAMING_UUID_SORT = """You will be given a long list of numbered lines. Each line has the form:

[N]hex,hex,hex,...

Where each token is an 8-character lowercase hex string (for example: a1b2c3d4).
Your task is to output each line with its index and the same hex tokens sorted in ASCENDING LEXICOGRAPHIC ORDER (compare them as plain strings).

See the example below.

```
Input:

[1]c0a8e1d2,a1b2c3d4,b1c2d3e4
[2]f0e1d2c3,01234567

Output:

[1]a1b2c3d4,b1c2d3e4,c0a8e1d2
[2]01234567,f0e1d2c3
```

The real sequence will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output every line.
Do not add whitespace.
Do not change the format.

Here is the real sequence.

Input:

{input}"""


PROMPT_SHUFFLED_STREAMING_UUID_SORT = """You will be given a long list of numbered lines. Each line has the form:

[N]hex,hex,hex,...

Where each token is an 8-character lowercase hex string (for example: a1b2c3d4).
The input lines are SHUFFLED — they appear in arbitrary order, not in numerical order.
Your task is to output each line with its index and the same hex tokens sorted in ASCENDING LEXICOGRAPHIC ORDER (compare them as plain strings), AND emit the lines themselves in ASCENDING ORDER OF [N] starting at [1].


See the example below.

```
Input:

[2]f0e1d2c3,01234567
[1]c0a8e1d2,a1b2c3d4,b1c2d3e4

Output:

[1]a1b2c3d4,b1c2d3e4,c0a8e1d2
[2]01234567,f0e1d2c3
```


The real sequence will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output every line.
Do not add whitespace.
Do not change the format.

Here is the real sequence.

Input:

{input}"""

PROMPT_UNNUMBERED_UUID_SORT = """You will be given a long list of lines. Each line has the form:

hex,hex,hex,...

Where each token is an 8-character lowercase hex string (for example: a1b2c3d4).
Your task is to output each line with the same hex tokens sorted in ASCENDING LEXICOGRAPHIC ORDER (compare them as plain strings).

See the example below:

```
Input:

c0a8e1d2,a1b2c3d4,b1c2d3e4
f0e1d2c3,01234567

Output:

a1b2c3d4,b1c2d3e4,c0a8e1d2
01234567,f0e1d2c3
```

The real sequence will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output every line.
Do not add whitespace.
Do not change the format.

Here is the real sequence.

Input:

{input}"""

PROMPT_CSV_PERMUTATION = """You will be given a CSV table and reordering instructions.

The CSV has row headers [R0], [R1], [R2], ... and column headers [C0], [C1], [C2], ...
The "New row order" lists original row headers in the order they should appear in the output.
The "New col order" lists original column headers in the order they should appear in the output.

See the example below.

```
Input:

,[C0],[C1],[C2]
[R0],a,b,c
[R1],d,e,f
[R2],g,h,i

Reordering Instructions:
New row order: [R2],[R0],[R1]
New col order: [C1],[C0],[C2]

Output:

,[C1],[C0],[C2]
[R2],h,g,i
[R0],b,a,c
[R1],e,d,f
```

The real CSV will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output every line.
Do not add whitespace.
Do not change the format.

Here is the real CSV.

Input:

{input}

Reordering Instructions:
New row order: {row_order}
New col order: {col_order}"""


PROMPT_CSV_KV_LOOKUP = """You will be given a lookup table for adjectives and nouns, followed by a CSV table.

Each cell in the CSV contains an expression of the form aN+vM, where aN is an adjective key and vM is a noun key.
Your task is to output the CSV with each expression replaced by the resolved adjective and noun, separated by a space.

See the example below.

```
Adjectives:
a0=quick
a1=lazy

Nouns:
v0=fox
v1=dog

Input:

,[C0],[C1]
[R0],a1+v0,a0+v1
[R1],a0+v0,a1+v1

Output:

,[C0],[C1]
[R0],lazy fox,quick dog
[R1],quick fox,lazy dog
```

The real lookup table and CSV will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output every line.
Do not add whitespace.
Do not change the format.

Here is the real lookup table and CSV.

Adjectives:
{adjectives}

Nouns:
{nouns}

Input:

{input}"""


PROMPT_UNNUMBERED_VAR_EXPAND = """You will be given a list of variable definitions, followed by a list of expressions.

Each variable definition has the form key=word, where key is a short hex string (for example: a3f) and word is an English word. The definitions appear in random order.

Each expression has the form key1+key2. Your task is to output, for each expression, the word for key1 and the word for key2, separated by a single space.

See the example below.

```
Variables:

a3f=big
7b2=dog
0c1=black
9e4=cat

Expressions:

a3f+7b2
0c1+9e4

Output:

big dog
black cat
```

The real input will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output an answer to every expression.
Do not add whitespace.
Do not change the format.

Here is the real input.

Variables:

{definitions}

Expressions:

{input}"""


PROMPT_STREAMING_VAR_EXPAND = """You will be given a list of variable definitions, followed by a list of expressions.

Each variable definition has the form key=word, where key is a short hex string (for example: a3f) and word is an English word. The definitions appear in random order.

Each expression is preceded by a numeric index in brackets like [1], [2], [3], ... and has the form [N]key1+key2.
Your task is to output, for each expression, its index followed by the word for key1 and the word for key2, separated by a single space, like [N]word1 word2.

See the example below.

```
Variables:

a3f=big
7b2=dog
0c1=black
9e4=cat

Expressions:

[1]a3f+7b2
[2]0c1+9e4

Output:

[1]big dog
[2]black cat
```

The real input will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output an answer to every expression.
Do not add whitespace.
Do not change the format.

Here is the real input.

Variables:

{definitions}

Expressions:

{input}"""


PROMPT_SHUFFLED_STREAMING_VAR_EXPAND = """You will be given a list of variable definitions, followed by a list of expressions.

Each variable definition has the form key=word, where key is a short hex string (for example: a3f) and word is an English word. The definitions appear in random order.

Each expression is preceded by a numeric index in brackets like [1], [2], [3], ... and has the form [N]key1+key2.
The input expressions are SHUFFLED — they appear in arbitrary order, not in numerical order.
Your task is to output, for each expression, its index followed by the word for key1 and the word for key2, separated by a single space, like [N]word1 word2, IN ASCENDING ORDER OF INDEX, starting at [1].

See the example below.

```
Variables:

a3f=big
7b2=dog
0c1=black
9e4=cat

Expressions:

[2]0c1+9e4
[1]a3f+7b2

Output:

[1]big dog
[2]black cat
```

The real input will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output an answer to every expression.
Do not add whitespace.
Do not change the format.

Here is the real input.

Variables:

{definitions}

Expressions:

{input}"""


PROMPT_TEMPLATES = {
    "streaming_sum": PROMPT_STREAMING_SUM,
    "shuffled_streaming_sum": PROMPT_SHUFFLED_STREAMING_SUM,
    "unnumbered_streaming_sum": PROMPT_UNNUMBERED_STREAMING_SUM,
    "streaming_uuid_sort": PROMPT_STREAMING_UUID_SORT,
    "shuffled_streaming_uuid_sort": PROMPT_SHUFFLED_STREAMING_UUID_SORT,
    "unnumbered_uuid_sort": PROMPT_UNNUMBERED_UUID_SORT,
    "csv_permutation_homogeneous": PROMPT_CSV_PERMUTATION,
    "csv_permutation_heterogeneous": PROMPT_CSV_PERMUTATION,
    "csv_kv_lookup": PROMPT_CSV_KV_LOOKUP,
    "unnumbered_var_expand": PROMPT_UNNUMBERED_VAR_EXPAND,
    "streaming_var_expand": PROMPT_STREAMING_VAR_EXPAND,
    "shuffled_streaming_var_expand": PROMPT_SHUFFLED_STREAMING_VAR_EXPAND,
}

SUM_TYPES = ["unnumbered_streaming_sum", "streaming_sum", "shuffled_streaming_sum"]
UUID_SORT_TYPES = ["unnumbered_uuid_sort", "streaming_uuid_sort", "shuffled_streaming_uuid_sort"]
VAR_EXPAND_TYPES = ["unnumbered_var_expand", "streaming_var_expand", "shuffled_streaming_var_expand"]
NUMBERED_TYPES = {
    "streaming_sum",
    "shuffled_streaming_sum",
    "streaming_uuid_sort",
    "shuffled_streaming_uuid_sort",
    "streaming_var_expand",
    "shuffled_streaming_var_expand",
}
PERM_FRACTIONS = [0.2, 0.4, 0.8, 1.0]
VOCAB_FRACTIONS = [1 / 8, 1 / 4, 1 / 2, 1.0]

TARGET_TOKENS_LIST = [2048, 4096, 8192, 16384, 32768, 65536]
N_SAMPLES = 5
MAX_OPERANDS_RANGE = [2, 4, 8, 16]
# Variable-expansion difficulty axis: size of the variable pool. All tiers fit
# the smallest (2048-token) budget. Names are 3 hex digits (16**3 = 4096
# possible), which comfortably covers the largest tier.
N_VARIABLES_RANGE = [8, 32, 128, 256]
VAR_HEX_DIGITS = 3


def _get_encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _rng(seed) -> random.Random:
    """Return a seeded Random instance.

    Python 3.12 only accepts None/int/float/str/bytes/bytearray as seeds;
    tuple seeds that worked via implicit hash() in 3.9 now raise TypeError.
    Converting to str() is deterministic and version-safe.
    """
    if not isinstance(seed, (type(None), int, float, str, bytes, bytearray)):
        seed = str(seed)
    return random.Random(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Arithmetic chain generation
# ─────────────────────────────────────────────────────────────────────────────


def _generate_expression(max_operands: int) -> tuple[str, int]:
    n = random.randint(2, max_operands)
    operands = [random.randint(0, 9) for _ in range(n)]
    operators = [random.choice(["+", "-"]) for _ in range(n - 1)]
    parts = [str(operands[0])]
    for op, operand in zip(operators, operands[1:]):
        parts.append(op + str(operand))
    expr = "".join(parts)
    result = operands[0]
    for op, operand in zip(operators, operands[1:]):
        result = result + operand if op == "+" else result - operand
    return expr, result


def _generate_expressions(max_operands: int, enc, target_tokens: int) -> tuple[list[dict], int]:
    """Generate as many expressions as fit a target_tokens-budget prompt.

    Budget is computed against the longest sum-prompt header (shuffled).
    """
    longest_header = max(
        (PROMPT_TEMPLATES[t].split("{input}")[0] for t in SUM_TYPES),
        key=len,
    )
    overhead = len(enc.encode(longest_header))
    budget = target_tokens - overhead

    expressions: list[dict] = []
    used_tokens = 0
    while True:
        expr, answer = _generate_expression(max_operands)
        line_tokens = len(enc.encode(f"[{len(expressions) + 1}]{expr}\n"))
        if used_tokens + line_tokens > budget:
            break
        expressions.append({"expr": expr, "answer": answer})
        used_tokens += line_tokens
    return expressions, overhead + used_tokens


def _build_sum_sample(
    expressions: list[dict],
    max_operands: int,
    sample_type: str,
    approx_prompt_tokens: int,
    rng: random.Random,
) -> dict:
    """Render one arithmetic-chain row."""
    numbered = list(enumerate(expressions, start=1))

    if sample_type == "unnumbered_streaming_sum":
        input_text = "\n".join(e["expr"] for _, e in numbered)
        expected_output = "\n".join(f"{e['expr']}={e['answer']}" for _, e in numbered)
    elif sample_type in {"streaming_sum", "shuffled_streaming_sum"}:
        if sample_type == "shuffled_streaming_sum":
            input_order = numbered.copy()
            rng.shuffle(input_order)
        else:
            input_order = numbered
        input_text = "\n".join(f"[{n}]{e['expr']}" for n, e in input_order)
        expected_output = "\n".join(f"[{n}]{e['expr']}={e['answer']}" for n, e in numbered)
    else:
        raise ValueError(f"unknown sum sample_type: {sample_type}")

    prompt = PROMPT_TEMPLATES[sample_type].replace("{input}", input_text)
    return {
        "type": sample_type,
        "question": prompt,
        "expected_output": expected_output,
        "expressions": expressions,
        "n_expressions": len(expressions),
        "max_operands": max_operands,
        "approx_prompt_tokens": approx_prompt_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UUID-sort generation
# ─────────────────────────────────────────────────────────────────────────────


def _new_uuid(rng: random.Random) -> str:
    """Deterministic 8-char lowercase hex token drawn from the provided RNG.

    We use just the first segment of a uuid4 (32 bits, 8 hex chars) so the
    tokens are short enough to fit many lines per 100K-token prompt while
    still being unique enough across the dataset (~1 in 4B collision odds).
    """
    return f"{rng.getrandbits(32):08x}"


def _generate_uuid_lines(
    uuids_per_line: int,
    enc,
    seed_key: tuple,
    target_tokens: int,
) -> tuple[list[list[str]], int]:
    """Generate UUID lines that fit a target_tokens-budget prompt.

    Each inner list is the canonical (input-presentation) UUIDs for that
    line. Token budget is computed against the longer (shuffled) header.
    """
    longest_header = max(
        (PROMPT_TEMPLATES[t].split("{input}")[0] for t in UUID_SORT_TYPES),
        key=len,
    )
    overhead = len(enc.encode(longest_header))
    budget = target_tokens - overhead

    rng = _rng(seed_key)
    lines: list[list[str]] = []
    used_tokens = 0
    while True:
        idx = len(lines) + 1
        uuids = [_new_uuid(rng) for _ in range(uuids_per_line)]
        line_str = f"[{idx}]" + ",".join(uuids) + "\n"
        line_tokens = len(enc.encode(line_str))
        if used_tokens + line_tokens > budget:
            break
        lines.append(uuids)
        used_tokens += line_tokens
    return lines, overhead + used_tokens


def _build_uuid_sample(
    uuid_lines: list[list[str]],
    uuids_per_line: int,
    sample_type: str,
    approx_prompt_tokens: int,
    rng: random.Random,
) -> dict:
    """Render one UUID-sort row.

    `uuid_lines` is the canonical (input-presentation) per-line UUIDs. The
    UUIDs are deliberately NOT pre-sorted — that's the model's task. For the
    shuffled variant, the LINE order in the input is shuffled but the model
    must still emit ascending [N].
    """
    numbered = list(enumerate(uuid_lines, start=1))

    if sample_type == "unnumbered_uuid_sort":
        input_text = "\n".join(",".join(uuids) for _, uuids in numbered)
        expected_output = "\n".join(",".join(sorted(uuids)) for _, uuids in numbered)
        prompt = PROMPT_TEMPLATES[sample_type].replace("{input}", input_text)
        return {
            "type": sample_type,
            "question": prompt,
            "expected_output": expected_output,
            "uuid_lines": uuid_lines,
            "expressions": [{"expr": list(uuids), "answer": sorted(uuids)} for uuids in uuid_lines],
            "n_lines": len(uuid_lines),
            "uuids_per_line": uuids_per_line,
            "approx_prompt_tokens": approx_prompt_tokens,
        }

    if sample_type == "streaming_uuid_sort":
        input_order = numbered
    elif sample_type == "shuffled_streaming_uuid_sort":
        input_order = numbered.copy()
        rng.shuffle(input_order)
    else:
        raise ValueError(f"unknown uuid sample_type: {sample_type}")

    def _fmt_line(n: int, uuids: list[str]) -> str:
        return f"[{n}]" + ",".join(uuids)

    input_text = "\n".join(_fmt_line(n, uuids) for n, uuids in input_order)
    expected_output = "\n".join(_fmt_line(n, sorted(uuids)) for n, uuids in numbered)
    prompt = PROMPT_TEMPLATES[sample_type].replace("{input}", input_text)

    return {
        "type": sample_type,
        "question": prompt,
        "expected_output": expected_output,
        "uuid_lines": uuid_lines,
        "expressions": [{"expr": list(uuids), "answer": sorted(uuids)} for uuids in uuid_lines],
        "n_lines": len(uuid_lines),
        "uuids_per_line": uuids_per_line,
        "approx_prompt_tokens": approx_prompt_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSV permutation generation
# ─────────────────────────────────────────────────────────────────────────────


def _new_csv_cell(rng: random.Random) -> str:
    """Random UUID string (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx) truncated to 1-36 chars."""
    hex_str = f"{rng.getrandbits(128):032x}"
    uuid_str = f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"
    return uuid_str[: rng.randint(1, 36)]


def _new_csv_cell_homogeneous(rng: random.Random) -> str:
    """Random 4-digit integer string (1000–9999)."""
    return str(rng.randint(1000, 9999))


def _build_csv_input(
    grid: list[list[str]],
    row_order: list[int],
    col_order: list[int],
    sample_type: str = "csv_permutation_heterogeneous",
) -> str:
    """Render the input CSV (original row/col order) with permutation spec and header."""
    N = len(grid)
    row_order_str = ",".join(f"[R{i}]" for i in row_order)
    col_order_str = ",".join(f"[C{j}]" for j in col_order)
    header = "," + ",".join(f"[C{j}]" for j in range(N))
    rows = [f"[R{i}]," + ",".join(grid[i][j] for j in range(N)) for i in range(N)]
    csv_str = "\n".join([header] + rows)
    return PROMPT_TEMPLATES[sample_type].format(
        row_order=row_order_str,
        col_order=col_order_str,
        input=csv_str,
    )


def _build_csv_expected_output(grid: list[list[str]], row_order: list[int], col_order: list[int]) -> str:
    """Render the expected permuted CSV."""
    header = "," + ",".join(f"[C{j}]" for j in col_order)
    rows = [f"[R{i}]," + ",".join(grid[i][j] for j in col_order) for i in row_order]
    return "\n".join([header] + rows)


def _generate_csv_grid(
    enc,
    seed_key,
    target_tokens: int,
    cell_fn=None,
    sample_type: str = "csv_permutation_heterogeneous",
) -> tuple[list[list[str]], int]:
    """Find the largest square N×N grid fitting target_tokens and return it with its token count.

    Uses a binary search: each probe regenerates the grid deterministically from seed_key
    with an identity permutation (same spec length for any permutation of the same N).
    """
    if cell_fn is None:
        cell_fn = _new_csv_cell

    def _probe(N: int) -> int:
        rng = _rng(seed_key)
        grid = [[cell_fn(rng) for _ in range(N)] for _ in range(N)]
        prompt = _build_csv_input(grid, list(range(N)), list(range(N)), sample_type)
        return len(enc.encode(prompt))

    lo, hi = 2, 300
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _probe(mid) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1

    N = lo
    rng = _rng(seed_key)
    grid = [[cell_fn(rng) for _ in range(N)] for _ in range(N)]
    return grid, _probe(N)


def _apply_perm_fraction(N: int, perm_fraction: float, rng: random.Random) -> list[int]:
    """Return a permutation of range(N) where ~perm_fraction of positions are shuffled."""
    n_permuted = max(2, int(round(N * perm_fraction)))
    positions = sorted(rng.sample(range(N), n_permuted))
    values = positions.copy()
    for _ in range(20):  # retry until non-identity shuffle
        rng.shuffle(values)
        if values != positions:
            break
    order = list(range(N))
    for pos, val in zip(positions, values):
        order[pos] = val
    return order


def _build_csv_sample(
    grid: list[list[str]],
    perm_fraction: float,
    rng: random.Random,
    approx_prompt_tokens: int,
    sample_type: str = "csv_permutation_heterogeneous",
) -> dict:
    N = len(grid)
    row_order = _apply_perm_fraction(N, perm_fraction, rng)
    col_order = _apply_perm_fraction(N, perm_fraction, rng)
    prompt = _build_csv_input(grid, row_order, col_order, sample_type)
    expected_output = _build_csv_expected_output(grid, row_order, col_order)
    return {
        "type": sample_type,
        "question": prompt,
        "expected_output": expected_output,
        "expressions": [{"expr": grid[i][j], "answer": grid[i][j]} for i in row_order for j in col_order],
        "n_rows": N,
        "n_cols": N,
        "perm_fraction": perm_fraction,
        "approx_prompt_tokens": approx_prompt_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSV key-value lookup generation
# ─────────────────────────────────────────────────────────────────────────────


def _load_word_lists() -> tuple[list[str], list[str]]:
    import wonderwords.random_word as ww_rw

    adjs = ww_rw._get_words_from_text_file("adjectivelist.txt")
    nouns = ww_rw._get_words_from_text_file("nounlist.txt")
    return adjs, nouns


def _build_csv_kv_input(adjs: list[str], nouns: list[str], grid: list[list[str]]) -> str:
    """Render the full prompt for csv_kv_lookup."""
    M = len(grid)
    adj_text = "\n".join(f"a{i}={adj}" for i, adj in enumerate(adjs))
    noun_text = "\n".join(f"v{i}={noun}" for i, noun in enumerate(nouns))
    header = "," + ",".join(f"[C{j}]" for j in range(M))
    rows = [f"[R{i}]," + ",".join(grid[i][j] for j in range(M)) for i in range(M)]
    csv_str = "\n".join([header] + rows)
    return PROMPT_TEMPLATES["csv_kv_lookup"].format(
        adjectives=adj_text,
        nouns=noun_text,
        input=csv_str,
    )


def _build_csv_kv_expected_output(adjs: list[str], nouns: list[str], grid: list[list[str]]) -> str:
    """Resolve each expression cell to 'adjective noun'."""
    M = len(grid)
    header = "," + ",".join(f"[C{j}]" for j in range(M))
    rows = []
    for i in range(M):
        cells = []
        for j in range(M):
            expr = grid[i][j]  # e.g. "a3+v7"
            adj_idx = int(expr.split("+")[0][1:])
            noun_idx = int(expr.split("+")[1][1:])
            cells.append(f"{adjs[adj_idx]} {nouns[noun_idx]}")
        rows.append(f"[R{i}]," + ",".join(cells))
    return "\n".join([header] + rows)


def _find_csv_kv_grid_size(
    enc,
    seed_key,
    target_tokens: int,
    all_adjs: list[str],
    all_nouns: list[str],
) -> tuple[int, list[str], list[str], int]:
    """Binary-search for the largest M where the kv-lookup prompt fits target_tokens.

    Uses vocab_fraction=1.0 (M adjectives and M nouns) for the upper-bound
    token estimate. Returns (M, sampled_adjs, sampled_nouns, approx_tokens).
    """
    max_m = min(len(all_adjs), len(all_nouns), 300)

    def _probe(M: int) -> int:
        rng = _rng(seed_key)
        adjs = rng.sample(all_adjs, M)
        nouns = rng.sample(all_nouns, M)
        # Sequential index grid gives a representative (slightly high) token
        # count since large indices like "a99+v99" are longer than "a0+v0".
        grid = [[f"a{i}+v{j}" for j in range(M)] for i in range(M)]
        return len(enc.encode(_build_csv_kv_input(adjs, nouns, grid)))

    lo, hi = 2, max_m
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _probe(mid) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1

    M = lo
    rng = _rng(seed_key)
    adjs = rng.sample(all_adjs, M)
    nouns = rng.sample(all_nouns, M)
    return M, adjs, nouns, _probe(M)


def _build_csv_kv_sample(
    M: int,
    vocab_fraction: float,
    adjs: list[str],
    nouns: list[str],
    rng: random.Random,
    approx_prompt_tokens: int,
) -> dict:
    """Build one csv_kv_lookup sample.

    `adjs` and `nouns` are the full M-length word lists for this grid.
    `vocab_fraction` controls how many of those words are actually used:
    n_vocab = max(2, round(M * vocab_fraction)).  Cells draw uniformly
    from [0, n_vocab), so lower fractions produce more repetition.
    """
    n_vocab = max(2, int(round(M * vocab_fraction)))
    active_adjs = adjs[:n_vocab]
    active_nouns = nouns[:n_vocab]
    grid = [[f"a{rng.randint(0, n_vocab - 1)}+v{rng.randint(0, n_vocab - 1)}" for _ in range(M)] for _ in range(M)]
    prompt = _build_csv_kv_input(active_adjs, active_nouns, grid)
    expected_output = _build_csv_kv_expected_output(active_adjs, active_nouns, grid)
    expressions = []
    for i in range(M):
        for j in range(M):
            expr = grid[i][j]
            adj_idx = int(expr.split("+")[0][1:])
            noun_idx = int(expr.split("+")[1][1:])
            expressions.append(
                {
                    "expr": expr,
                    "answer": f"{active_adjs[adj_idx]} {active_nouns[noun_idx]}",
                }
            )
    return {
        "type": "csv_kv_lookup",
        "question": prompt,
        "expected_output": expected_output,
        "expressions": expressions,
        "n_rows": M,
        "n_cols": M,
        "vocab_fraction": vocab_fraction,
        "n_vocab": n_vocab,
        "approx_prompt_tokens": approx_prompt_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Variable-expansion generation
# ─────────────────────────────────────────────────────────────────────────────


def _def_block(definitions: list[tuple[str, str]]) -> str:
    """Render the variable-definition block ('key=word' per line, one order)."""
    return "\n".join(f"{name}={word}" for name, word in definitions)


def _build_var_pool(
    n_variables: int,
    all_adjs: list[str],
    all_nouns: list[str],
    rng: random.Random,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Build a pool of n_variables hex-named variables split into adj/noun halves.

    Returns (adjectives, nouns, definitions) where each is a list of
    (hex_name, word) pairs. `definitions` is the SHUFFLED presentation order —
    adjectives and nouns interleaved randomly — since the hex names are opaque,
    this is what the model sees. Names are distinct 3-hex-digit strings.
    """
    n_adj = n_variables // 2
    n_noun = n_variables - n_adj
    max_names = 16**VAR_HEX_DIGITS
    names = [f"{v:0{VAR_HEX_DIGITS}x}" for v in rng.sample(range(max_names), n_variables)]
    adj_words = rng.sample(all_adjs, n_adj)
    noun_words = rng.sample(all_nouns, n_noun)
    adjectives = list(zip(names[:n_adj], adj_words))
    nouns = list(zip(names[n_adj:], noun_words))
    definitions = adjectives + nouns
    rng.shuffle(definitions)
    return adjectives, nouns, definitions


def _generate_var_expressions(
    adjectives: list[tuple[str, str]],
    nouns: list[tuple[str, str]],
    definitions: list[tuple[str, str]],
    rng: random.Random,
    enc,
    target_tokens: int,
) -> tuple[list[dict] | None, int]:
    """Generate expressions that fill a target_tokens-budget prompt.

    Each expression pairs a random adjective var with a random noun var (with
    replacement). The definitions block is fixed overhead; the remaining budget
    is filled with '[N]key1+key2' lines (numbered form is the longest variant,
    so all three variants fit). Returns (expressions, approx_tokens), or
    (None, overhead) when the definitions alone leave no room for expressions.
    """
    def_block = _def_block(definitions)
    longest_prefix = max(
        (PROMPT_TEMPLATES[t].split("{input}")[0].replace("{definitions}", def_block) for t in VAR_EXPAND_TYPES),
        key=len,
    )
    overhead = len(enc.encode(longest_prefix))
    budget = target_tokens - overhead
    if budget <= 0:
        return None, overhead

    expressions: list[dict] = []
    used_tokens = 0
    while True:
        adj_name, adj_word = rng.choice(adjectives)
        noun_name, noun_word = rng.choice(nouns)
        expr = f"{adj_name}+{noun_name}"
        answer = f"{adj_word} {noun_word}"
        line_tokens = len(enc.encode(f"[{len(expressions) + 1}]{expr}\n"))
        if used_tokens + line_tokens > budget:
            break
        expressions.append({"expr": expr, "answer": answer})
        used_tokens += line_tokens
    if not expressions:
        return None, overhead
    return expressions, overhead + used_tokens


def _build_var_expand_sample(
    definitions: list[tuple[str, str]],
    expressions: list[dict],
    n_variables: int,
    sample_type: str,
    approx_prompt_tokens: int,
    rng: random.Random,
) -> dict:
    """Render one variable-expansion row.

    `expressions` is canonical (1-indexed by position). For the shuffled
    variant the INPUT line order is shuffled but the expected output is always
    ascending [N]. The definition presentation order is fixed (already shuffled
    in _build_var_pool) and identical across all three variants.
    """
    def_block = _def_block(definitions)
    numbered = list(enumerate(expressions, start=1))

    if sample_type == "unnumbered_var_expand":
        input_text = "\n".join(e["expr"] for _, e in numbered)
        expected_output = "\n".join(e["answer"] for _, e in numbered)
    elif sample_type in {"streaming_var_expand", "shuffled_streaming_var_expand"}:
        if sample_type == "shuffled_streaming_var_expand":
            input_order = numbered.copy()
            rng.shuffle(input_order)
        else:
            input_order = numbered
        input_text = "\n".join(f"[{n}]{e['expr']}" for n, e in input_order)
        expected_output = "\n".join(f"[{n}]{e['answer']}" for n, e in numbered)
    else:
        raise ValueError(f"unknown var_expand sample_type: {sample_type}")

    prompt = PROMPT_TEMPLATES[sample_type].replace("{definitions}", def_block).replace("{input}", input_text)
    return {
        "type": sample_type,
        "question": prompt,
        "expected_output": expected_output,
        "expressions": expressions,
        "n_expressions": len(expressions),
        "n_variables": n_variables,
        "approx_prompt_tokens": approx_prompt_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestration
# ─────────────────────────────────────────────────────────────────────────────


def generate(force: bool = False) -> None:
    if OUTPUT_FPATH.exists() and not force:
        count = sum(1 for line in OUTPUT_FPATH.open() if line.strip())
        print(f"long_transduction benchmark already exists: {count} examples in {OUTPUT_FPATH}")
        return

    enc = _get_encoder()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Gym installs the config's prepare_dependencies before calling prepare().
    all_adjs, all_nouns = _load_word_lists()

    total = 0
    with OUTPUT_FPATH.open("w") as out:
        for target_tokens in TARGET_TOKENS_LIST:
            print(f"\n── target_tokens={target_tokens:,} ──────────────────────────────────────────")

            # Arithmetic chain: 3 variants per (max_operands, sample_idx).
            for max_operands in MAX_OPERANDS_RANGE:
                print(f"Generating sum max_operands={max_operands} ({N_SAMPLES} pairs)...")
                for i in range(N_SAMPLES):
                    expressions, n_tokens = _generate_expressions(max_operands, enc, target_tokens)
                    for sample_type in SUM_TYPES:
                        rng = _rng((target_tokens, max_operands, i, sample_type))
                        sample = _build_sum_sample(expressions, max_operands, sample_type, n_tokens, rng)
                        sample["target_tokens"] = target_tokens
                        sample["responses_create_params"] = {"max_output_tokens": target_tokens * 3 // 2}
                        out.write(json.dumps(sample) + "\n")
                        total += 1
                    print(
                        f"  sum[{i + 1:2d}/{N_SAMPLES}] "
                        f"{len(expressions)} expressions, ~{n_tokens:,} tokens, "
                        f"emitted {len(SUM_TYPES)} variants"
                    )

            # UUID sort: 3 variants per (uuids_per_line, sample_idx).
            for uuids_per_line in MAX_OPERANDS_RANGE:
                print(f"Generating uuid_sort uuids_per_line={uuids_per_line} ({N_SAMPLES} samples)...")
                for i in range(N_SAMPLES):
                    uuid_lines, n_tokens = _generate_uuid_lines(
                        uuids_per_line,
                        enc,
                        seed_key=("uuid", target_tokens, uuids_per_line, i),
                        target_tokens=target_tokens,
                    )
                    for sample_type in UUID_SORT_TYPES:
                        rng = _rng((target_tokens, uuids_per_line, i, sample_type))
                        sample = _build_uuid_sample(uuid_lines, uuids_per_line, sample_type, n_tokens, rng)
                        sample["target_tokens"] = target_tokens
                        sample["responses_create_params"] = {"max_output_tokens": target_tokens * 3 // 2}
                        out.write(json.dumps(sample) + "\n")
                        total += 1
                    print(
                        f"  uuid[{i + 1:2d}/{N_SAMPLES}] "
                        f"{len(uuid_lines)} lines × {uuids_per_line} uuids, "
                        f"~{n_tokens:,} tokens, emitted {len(UUID_SORT_TYPES)} variants"
                    )

            # CSV permutation: homogeneous and heterogeneous variants.
            # Each variant uses N_SAMPLES grids × PERM_FRACTIONS difficulties.
            for cell_type, cell_fn in [
                ("homogeneous", _new_csv_cell_homogeneous),
                ("heterogeneous", _new_csv_cell),
            ]:
                type_name = f"csv_permutation_{cell_type}"
                print(f"Generating {type_name} ({N_SAMPLES} grids × {len(PERM_FRACTIONS)} perm_fractions)...")
                for i in range(N_SAMPLES):
                    grid, n_tokens = _generate_csv_grid(
                        enc,
                        seed_key=("csv", cell_type, target_tokens, i),
                        target_tokens=target_tokens,
                        cell_fn=cell_fn,
                        sample_type=type_name,
                    )
                    N = len(grid)
                    for perm_fraction in PERM_FRACTIONS:
                        rng = _rng(("csv", cell_type, target_tokens, i, perm_fraction))
                        sample = _build_csv_sample(grid, perm_fraction, rng, n_tokens, type_name)
                        sample["target_tokens"] = target_tokens
                        sample["responses_create_params"] = {"max_output_tokens": target_tokens * 3 // 2}
                        out.write(json.dumps(sample) + "\n")
                        total += 1
                    print(
                        f"  {cell_type[:3]}[{i + 1:2d}/{N_SAMPLES}] "
                        f"{N}×{N} grid, ~{n_tokens:,} tokens, "
                        f"emitted {len(PERM_FRACTIONS)} perm_fraction variants"
                    )

            # CSV KV lookup: N_SAMPLES grids × VOCAB_FRACTIONS difficulties.
            print(f"Generating csv_kv_lookup ({N_SAMPLES} grids × {len(VOCAB_FRACTIONS)} vocab_fractions)...")
            for i in range(N_SAMPLES):
                M, sample_adjs, sample_nouns, n_tokens = _find_csv_kv_grid_size(
                    enc,
                    seed_key=("csv_kv", target_tokens, i),
                    target_tokens=target_tokens,
                    all_adjs=all_adjs,
                    all_nouns=all_nouns,
                )
                for vocab_fraction in VOCAB_FRACTIONS:
                    rng = _rng(("csv_kv", target_tokens, i, vocab_fraction))
                    sample = _build_csv_kv_sample(M, vocab_fraction, sample_adjs, sample_nouns, rng, n_tokens)
                    sample["target_tokens"] = target_tokens
                    sample["responses_create_params"] = {"max_output_tokens": target_tokens * 3 // 2}
                    out.write(json.dumps(sample) + "\n")
                    total += 1
                print(
                    f"  kv[{i + 1:2d}/{N_SAMPLES}] "
                    f"{M}×{M} grid, ~{n_tokens:,} tokens, "
                    f"emitted {len(VOCAB_FRACTIONS)} vocab_fraction variants"
                )

            # Variable expansion: 3 variants per (n_variables, sample_idx).
            for n_variables in N_VARIABLES_RANGE:
                print(f"Generating var_expand n_variables={n_variables} ({N_SAMPLES} samples)...")
                for i in range(N_SAMPLES):
                    pool_rng = _rng(("var", target_tokens, n_variables, i))
                    adjectives, nouns, definitions = _build_var_pool(n_variables, all_adjs, all_nouns, pool_rng)
                    expressions, n_tokens = _generate_var_expressions(
                        adjectives, nouns, definitions, pool_rng, enc, target_tokens
                    )
                    if expressions is None:
                        print(
                            f"  var[{i + 1:2d}/{N_SAMPLES}] "
                            f"{n_variables} vars don't fit ~{target_tokens:,} tokens, skipped"
                        )
                        continue
                    for sample_type in VAR_EXPAND_TYPES:
                        rng = _rng((target_tokens, n_variables, i, sample_type))
                        sample = _build_var_expand_sample(
                            definitions, expressions, n_variables, sample_type, n_tokens, rng
                        )
                        sample["target_tokens"] = target_tokens
                        sample["responses_create_params"] = {"max_output_tokens": target_tokens * 3 // 2}
                        out.write(json.dumps(sample) + "\n")
                        total += 1
                    print(
                        f"  var[{i + 1:2d}/{N_SAMPLES}] "
                        f"{n_variables} vars, {len(expressions)} expressions, "
                        f"~{n_tokens:,} tokens, emitted {len(VAR_EXPAND_TYPES)} variants"
                    )

    print(f"Done. Wrote {total} examples to {OUTPUT_FPATH}")


def prepare(*, force: bool = False) -> Path:
    """Generate the benchmark if needed and return its path for Gym's preparation CLI."""
    generate(force=force)
    return OUTPUT_FPATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate even if output exists")
    args = parser.parse_args()
    prepare(force=args.force)


if __name__ == "__main__":
    main()
