# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Build N rows of train/validation/example data from either:
  1) a Hugging Face dataset (streaming), or
  2) a local JSONL file (optionally .gz) with rows shaped like:
     {
       "hash_id": "...",
       "question": "...",
       "unit_tests": "{\"inputs\": [...], \"outputs\": [...]}",
       ...
     }
     (The script tolerates both proper JSON and Python-dict-style lines with single quotes.)

Each output row conforms to NeMo-Gym dataset requirements:
- responses_create_params: OpenAI Responses-compatible input
- verifier_metadata.unit_tests: {inputs: [...], outputs: [...]} (strings)

Usage examples:
  # From HF dataset (default):
  uv run python resources_servers/comp_coding/scripts/build_examples.py \
    --out resources_servers/comp_coding/data/example.jsonl \
    --count 5000

  # From a local JSONL:
  uv run python resources_servers/comp_coding/scripts/build_examples.py \
    --in-jsonl /path/to/source.jsonl \
    --out resources_servers/comp_coding/data/example.jsonl \
    --count 5000

  # Pretty sample and gzip output:
  uv run python resources_servers/comp_coding/scripts/build_examples.py \
    --in-jsonl /path/to/source.jsonl.gz \
    --out resources_servers/comp_coding/data/example.jsonl.gz \
    --count 5000 \
    --pretty-sample resources_servers/comp_coding/data/sample.json \
    --pretty-k 10

Sanity checks after writing:
  jq -c . resources_servers/comp_coding/data/example.jsonl | wc -l
  head -n 3 resources_servers/comp_coding/data/example.jsonl | jq .
  awk 'NR==1{print; exit}' resources_servers/comp_coding/data/example.jsonl | tr -d '\n' | wc -c
  grep -nP '\x{2028}|\x{2029}' resources_servers/comp_coding/data/example.jsonl || echo "No LS/PS found"
"""

import argparse
import ast
import gzip
import json
import re
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset
from tqdm import tqdm


SYSTEM_PREFIX = (
    "You are an expert competitive programmer. You will be given a problem statement "
    "and must output a complete Python solution that reads from stdin and writes to stdout."
)

CODEFENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.MULTILINE)


def _strip_codefences(s: str) -> str:
    return CODEFENCE_RE.sub("", s).strip()


def _normalize_scalar(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.rstrip("\n").strip()


def _normalize_list(lst: Any) -> List[str]:
    if lst is None:
        return []
    if isinstance(lst, (str, bytes)):
        lst = [lst]
    if not isinstance(lst, list):
        return []
    out = []
    for v in lst:
        sv = _normalize_scalar(v)
        sv = _strip_codefences(sv)
        out.append(sv)
    return out


def _safe_literal_eval(s: str) -> Any:
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def _parse_unit_tests(raw: Any) -> Dict[str, List[str]]:
    parsed: Dict[str, Any] = {}
    if isinstance(raw, dict):
        parsed = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except Exception:
            maybe = _safe_literal_eval(raw)
            if isinstance(maybe, dict):
                parsed = maybe
            else:
                parsed = {}
    else:
        parsed = {}

    return {
        "inputs": _normalize_list(parsed.get("inputs", [])),
        "outputs": _normalize_list(parsed.get("outputs", [])),
    }


def make_row(q: str, unit_tests: Dict[str, List[str]], problem_id: Optional[str] = None) -> dict:
    q_norm = _normalize_scalar(q)
    return {
        "responses_create_params": {"input": [{"role": "user", "content": f"{SYSTEM_PREFIX}\n\n{q_norm}"}]},
        "verifier_metadata": {
            "problem_id": _normalize_scalar(problem_id) if problem_id is not None else None,
            "unit_tests": {
                "inputs": unit_tests.get("inputs", []),
                "outputs": unit_tests.get("outputs", []),
            },
        },
    }


def _open_out(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8")
    return open(path, "w", encoding="utf-8")


def _open_in(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def json_safe_dumps(obj: dict) -> str:
    """
    Dump JSON compactly and escape problematic Unicode line separators.
    Ensures no raw U+2028/U+2029 appear in output (they become \\u2028/\\u2029).
    """
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    # Escape LS (U+2028) and PS (U+2029)
    return s.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def stream_dataset(ds_name: str, split: str = "train") -> Iterable[dict]:
    ds = load_dataset(ds_name, split=split, streaming=True)
    for ex in ds:
        yield ex


def _parse_jsonl_line(line: str) -> Optional[dict]:
    """
    Robustly parse a JSONL line that might be:
    - proper JSON (double quotes), or
    - a Python dict-like string with single quotes.

    Returns a dict or None if it can't be parsed.
    """
    s = line.strip()
    if not s:
        return None
    # Try JSON first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Try Python literal (single-quoted dicts, etc.)
    maybe = _safe_literal_eval(s)
    if isinstance(maybe, dict):
        return maybe
    return None


def stream_jsonl(path: str) -> Iterable[dict]:
    with _open_in(path) as f:
        for raw in f:
            obj = _parse_jsonl_line(raw)
            if obj is None:
                continue
            yield obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output .jsonl or .jsonl.gz")
    ap.add_argument("--count", type=int, default=5000, help="Number of rows to write")
    ap.add_argument("--split", default="train", help="HF split name (default: train)")
    ap.add_argument("--pretty-sample", default=None, help="Optional pretty JSON of first K rows")
    ap.add_argument("--pretty-k", type=int, default=10, help="How many rows to pretty-print")
    ap.add_argument("--ds-name", default="Nexusflow/comp_prog_filtered_no_function")
    ap.add_argument(
        "--in-jsonl",
        default=None,
        help="Optional input JSONL (.jsonl or .jsonl.gz). If provided, overrides HF dataset input.",
    )
    args = ap.parse_args()

    rows_for_pretty = []
    total = 0

    # Choose input stream
    if args.in_jsonl:
        source_iter: Iterable[dict] = stream_jsonl(args.in_jsonl)
    else:
        source_iter = stream_dataset(args.ds_name, args.split)

    with _open_out(args.out) as f:
        for ex in tqdm(islice(source_iter, args.count), total=args.count):
            q = ex.get("question", "")
            raw_ut = ex.get("unit_tests", {}) or {}
            ut = _parse_unit_tests(raw_ut)
            pid = ex.get("hash_id")

            row = make_row(q, ut, pid)
            f.write(json_safe_dumps(row) + "\n")

            if args.pretty_sample and len(rows_for_pretty) < args.pretty_k:
                rows_for_prety_limit = args.pretty_k  # alias to clarify intent
                if len(rows_for_pretty) < rows_for_prety_limit:
                    rows_for_pretty.append(row)

            total += 1

    if args.pretty_sample and rows_for_pretty:
        with open(args.pretty_sample, "w", encoding="utf-8") as ps:
            json.dump(rows_for_pretty, ps, ensure_ascii=False, indent=2)

    print(f"wrote {total} rows to {args.out}")
    if args.pretty_sample:
        print(f"wrote pretty sample ({len(rows_for_pretty)} rows) to {args.pretty_sample}")


if __name__ == "__main__":
    main()
