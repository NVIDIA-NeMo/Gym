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
Validate and (optionally) normalize comp_coding JSONL datasets **before** runtime.

What it does:
- Ensures each row has:
  - responses_create_params.input (non-empty)
  - verifier_metadata.unit_tests with:
      - "inputs":  list[str] (non-empty)
      - "outputs": list[str] (same length as inputs)
- Optionally coerces stringified unit_tests into a dict (e.g., when stored as JSON string)
- Optionally normalizes newlines by converting literal "\\n" to "\n" (inputs/outputs)
- Can write out a cleaned JSONL (dropping bad rows or failing fast)

Usage:
  uv run python resources_servers/comp_coding/scripts/validate_dataset.py \
    --in data/comp_coding/train.jsonl --fail-fast

  uv run python resources_servers/comp_coding/scripts/validate_dataset.py \
    --in data/comp_coding/train.jsonl \
    --out data/comp_coding/train.cleaned.jsonl \
    --autofix --normalize-newlines --drop-bad

CLI flags:
  --in PATH [--in PATH ...]         One or more JSONL files to validate
  --out PATH                        Where to write a cleaned JSONL (optional)
  --autofix                         Try to parse stringified unit_tests to dict
  --normalize-newlines              Replace literal "\\n" with "\n" in tests
  --fail-fast                       Stop at first error (default: keep scanning)
  --drop-bad                        When --out is set, skip invalid rows instead of failing
  --pretty-sample PATH              Write a small pretty-printed sample (first 100 ok rows)

Exit codes:
  0 on success (or successful write when --out provided)
  1 on validation error (unless --drop-bad used and output written)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _is_list_of_str(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(s, str) for s in x)


def _maybe_parse_unit_tests(ut: Any, autofix: bool) -> Dict[str, Any]:
    if isinstance(ut, dict):
        return ut
    if isinstance(ut, str) and autofix:
        # Try strict JSON first
        try:
            parsed = json.loads(ut)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        # Try lenient: strip whitespace and single quotes
        try:
            s = ut.strip().replace("'", '"')
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    raise ValueError("unit_tests must be a dict (or a JSON string if --autofix).")


def _normalize_newlines_in_tests(ut: Dict[str, Any]) -> Dict[str, Any]:
    def fix(s: str) -> str:
        return s.replace("\\n", "\n")

    inputs = ut.get("inputs", [])
    outputs = ut.get("outputs", [])
    if isinstance(inputs, list):
        inputs = [fix(s) if isinstance(s, str) else s for s in inputs]
    if isinstance(outputs, list):
        outputs = [fix(s) if isinstance(s, str) else s for s in outputs]
    ut["inputs"] = inputs
    ut["outputs"] = outputs
    return ut


def _validate_unit_tests(ut: Dict[str, Any]) -> Tuple[bool, str]:
    inputs = ut.get("inputs")
    outputs = ut.get("outputs")
    if not _is_list_of_str(inputs):
        return False, "unit_tests.inputs must be list[str] and non-empty"
    if not _is_list_of_str(outputs):
        return False, "unit_tests.outputs must be list[str]"
    if len(inputs) == 0:
        return False, "unit_tests.inputs cannot be empty"
    if len(inputs) != len(outputs):
        return False, f"inputs/outputs length mismatch: {len(inputs)} vs {len(outputs)}"
    return True, "ok"


def _validate_row(
    row: Dict[str, Any], idx: int, autofix: bool, normalize_newlines: bool
) -> Tuple[bool, Dict[str, Any], str]:
    # responses_create_params sanity
    rcp = row.get("responses_create_params")
    if not isinstance(rcp, dict):
        return False, row, "missing responses_create_params"
    input_blocks = rcp.get("input")
    if not isinstance(input_blocks, list) or len(input_blocks) == 0:
        return False, row, "responses_create_params.input must be a non-empty list"

    # unit_tests presence + structure
    vm = row.get("verifier_metadata")
    if not isinstance(vm, dict):
        return False, row, "missing verifier_metadata"
    if "unit_tests" not in vm:
        return False, row, "missing verifier_metadata.unit_tests"

    try:
        ut = _maybe_parse_unit_tests(vm["unit_tests"], autofix=autofix)
    except Exception as e:
        return False, row, f"unit_tests parse error: {e}"

    if normalize_newlines:
        ut = _normalize_newlines_in_tests(ut)

    ok, msg = _validate_unit_tests(ut)
    if not ok:
        return False, row, msg

    # If we fixed ut, write it back normalized
    vm["unit_tests"] = {"inputs": ut["inputs"], "outputs": ut["outputs"]}
    row["verifier_metadata"] = vm
    return True, row, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", dest="out", type=str, default=None)
    ap.add_argument("--autofix", action="store_true")
    ap.add_argument("--normalize-newlines", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--drop-bad", action="store_true")
    ap.add_argument("--pretty-sample", type=str, default=None)
    args = ap.parse_args()

    in_paths = [Path(p) for p in args.inputs]
    out_path = Path(args.out) if args.out else None
    sample_path = Path(args.pretty_sample) if args.pretty_sample else None

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if sample_path:
        sample_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    ok_count = 0
    bad_count = 0
    written = 0
    sample: List[Dict[str, Any]] = []

    out_f = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for in_file in in_paths:
            with open(in_file, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        row = json.loads(line)
                    except Exception as e:
                        bad_count += 1
                        msg = f"{in_file}:{line_idx} invalid JSON: {e}"
                        if args.fail_fast:
                            raise SystemExit(msg)
                        else:
                            print("ERROR:", msg)
                            continue

                    ok, fixed, msg = _validate_row(
                        row,
                        total,
                        autofix=args.autofix,
                        normalize_newlines=args.normalize_newlines,
                    )
                    if ok:
                        ok_count += 1
                        if out_f:
                            out_f.write(json.dumps(fixed, ensure_ascii=False) + "\n")
                            written += 1
                        if len(sample) < 100 and sample_path:
                            sample.append(fixed)
                    else:
                        bad_count += 1
                        if args.fail_fast and not args.drop_bad:
                            raise SystemExit(f"{in_file}:{line_idx} {msg}")
                        print("ERROR:", f"{in_file}:{line_idx}", msg)
                        if out_f and args.drop_bad:
                            # skip writing this row
                            pass
                        elif out_f and not args.drop_bad:
                            # fail the whole run if we plan to produce a cleaned file but don’t drop bad rows
                            raise SystemExit(
                                f"Refusing to write invalid row without --drop-bad: {in_file}:{line_idx} {msg}"
                            )

        if sample_path and sample:
            with open(sample_path, "w", encoding="utf-8") as s:
                json.dump(sample, s, ensure_ascii=False, indent=2)

        print(f"Scanned rows: {total} | OK: {ok_count} | Bad: {bad_count}")
        if out_f:
            print(f"Wrote cleaned rows: {written} -> {out_path}")
        if bad_count and not (out_f and args.drop_bad):
            raise SystemExit(1)
    finally:
        if out_f:
            out_f.close()


if __name__ == "__main__":
    main()
