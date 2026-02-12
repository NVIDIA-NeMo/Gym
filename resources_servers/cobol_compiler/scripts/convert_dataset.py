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

"""Convert DomainForge cobol_multipl_eval.json to NeMo-Gym JSONL format.

Usage:
    python convert_dataset.py \
        --input ~/projects/domainforge/datasets/cobol_multipl_eval.json \
        --output ../data/cobol_multipl_eval.jsonl \
        --example-output ../data/example.jsonl \
        --example-count 5
"""

import argparse
import json
import sys
from pathlib import Path


SYSTEM_PROMPT = """\
Write complete, compilable COBOL code for GnuCOBOL 3.2+. Output ONLY the COBOL program with no explanations or commentary.

## Program Structure (Required Order)
```cobol
       IDENTIFICATION DIVISION.
       PROGRAM-ID. [PROGRAM-NAME].

       ENVIRONMENT DIVISION.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       [Variable declarations with PIC clauses]

       PROCEDURE DIVISION.
       [Program logic using paragraphs]
       STOP RUN.
```

## Coding Standards
- **Variable Names**: Use WS- prefix, descriptive names with hyphens (WS-COUNTER, WS-INPUT-VALUE, WS-RESULT)
- **PIC Clauses**: Required for all elementary items
  - Integers: `PIC 9(n)` (e.g., `PIC 9(5)` for 5-digit number)
  - Decimals: `PIC 9(n)V9(m)` (e.g., `PIC 9(3)V99` for 3.2 format)
  - Text/Mixed: `PIC X(n)` (e.g., `PIC X(50)` for 50-character string)
  - Signed Numbers: `PIC S9(n)` (e.g., `PIC S9(5)` for signed integer)
- **Level Numbers**: Use 01 for main items, 05 for sub-items
- **Format**: Use free format (-free). Indentation is stylistic; columns are not enforced
- **Initialization**: Use VALUE clauses where appropriate

## Essential COBOL Constructs

### Input/Output
- **Input**: Use ACCEPT for reading from stdin: `ACCEPT WS-INPUT-VALUE`
  - **IMPORTANT**: Do NOT display prompts before ACCEPT (no "Enter input:" messages)
  - Tests provide input via stdin - just read silently
- **Output**: Use DISPLAY for output: `DISPLAY WS-RESULT`
  - Only display the final result/answer
  - Do NOT display intermediate messages or debugging output

### Parsing stdin patterns
```cobol
*> Single integer (robust)
ACCEPT WS-LINE
MOVE FUNCTION NUMVAL(FUNCTION TRIM(WS-LINE)) TO WS-N

*> Space-separated list of integers
ACCEPT WS-LINE
MOVE 1 TO WS-PTR
PERFORM UNTIL WS-PTR > FUNCTION LENGTH(WS-LINE)
    UNSTRING WS-LINE DELIMITED BY SPACE
        INTO WS-TOKEN
        WITH POINTER WS-PTR
        TALLYING WS-TALLY
    IF WS-TALLY > 0 AND WS-TOKEN NOT = SPACES
        ADD 1 TO WS-COUNT
        MOVE FUNCTION NUMVAL(WS-TOKEN) TO WS-ARR(WS-COUNT)
    END-IF
END-PERFORM
```

## Requirements (must all be true)
1. Compiles with GnuCOBOL (free format)
2. Reads input silently from stdin (no prompts)
3. Displays only the required final output (no labels/extra text)
4. Uses NUMVAL/NUMVAL-C when parsing numeric text
5. Handles empty/edge-case inputs sensibly
6. Follows proper COBOL-85+ syntax and structured programming (PERFORM, avoid GO TO)
7. Declares variables with appropriate PIC clauses
8. Implements all specified test case logic correctly

IMPORTANT: Output ONLY the COBOL code without any markdown formatting, explanations, or commentary. Do not use ``` blocks, ** formatting, or any other markdown. Start with IDENTIFICATION DIVISION."""


def convert_problem(problem: dict) -> dict:
    """Convert a single DomainForge problem to NeMo-Gym JSONL format."""
    # Build user prompt from problem description + I/O format spec
    user_content = problem["prompt"]
    if problem.get("format_specification"):
        try:
            format_spec = json.loads(problem["format_specification"])
            if "io_format_spec" in format_spec:
                user_content += "\n\n" + format_spec["io_format_spec"]
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        },
        "verifier_metadata": {
            "test_cases": problem["test_cases"],
            "task_id": problem["task_id"],
            "entry_point": problem["entry_point"],
            "category": problem.get("category", "unknown"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Convert DomainForge COBOL dataset to NeMo-Gym JSONL")
    parser.add_argument("--input", required=True, help="Path to cobol_multipl_eval.json")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--example-output", default=None, help="Path for example.jsonl subset")
    parser.add_argument("--example-count", type=int, default=5, help="Number of examples to include")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        problems = json.load(f)

    print(f"Loaded {len(problems)} problems from {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = []
    for problem in problems:
        converted.append(convert_problem(problem))

    with open(output_path, "w") as f:
        for entry in converted:
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(converted)} entries to {output_path}")

    if args.example_output:
        example_path = Path(args.example_output)
        example_path.parent.mkdir(parents=True, exist_ok=True)
        with open(example_path, "w") as f:
            for entry in converted[: args.example_count]:
                f.write(json.dumps(entry) + "\n")
        print(f"Wrote {args.example_count} example entries to {example_path}")


if __name__ == "__main__":
    main()
