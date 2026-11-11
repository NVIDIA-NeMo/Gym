# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run upstream EnterpriseOps-Gym's own verifier code (benchmark/verifier.py @ 271f2c3) on shared inputs and write the
results as golden fixtures for the Gym port: value extraction from /api/sql-runner answers and value comparison.

Usage: python -I generate_verifier_golden.py <EnterpriseOps-Gym checkout> <output.json>
"""

import json
import sys
from pathlib import Path


UPSTREAM = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.insert(0, str(UPSTREAM))
from benchmark.verifier import VerifierEngine  # noqa: E402


engine = VerifierEngine.__new__(VerifierEngine)  # extraction and comparison need no clients

answers = [
    {"success": True, "data": [{"cnt": 1}], "row_count": 1},
    {"success": True, "data": [{"cnt": 0}]},
    {"success": True, "data": [{"COUNT(*)": 3}]},
    {"success": True, "data": [{"state": "in_progress"}]},
    {"success": True, "data": [{"flag": True}]},
    {"success": True, "data": [{"value": None}]},
    {"success": True, "data": [{"amount": 12.5}]},
    {"success": True, "data": [{"id": "INC_024", "state": "closed"}]},
    {"success": True, "data": [{"id": 1}, {"id": 2}]},
    {"success": True, "data": []},
    {"success": True, "data": None},
    {"success": True, "message": "Query executed successfully. 0 rows affected.", "type": "write", "rows_affected": 0},
]
extraction = [
    {"answer": answer, "value": engine._extract_value_from_sql_result({"success": True, "result": answer})}
    for answer in answers
]

actuals = [0, 1, 2, 5, 12.5, True, False, None, "0", "1", "in_progress", "open", "2026-01-15 11:00:00", [], {"id": 1}]
expecteds = [0, 1, 3, "0", "1", "2", True, "in_progress", "check", "2026-01-15T11:00:00", None]
comparison = []
for comparison_type in ("equals", "greater_than", "less_than", "contains"):
    for actual in actuals:
        for expected in expecteds:
            result = engine._compare_values(actual, expected, comparison_type)
            comparison.append(
                {
                    "actual": actual,
                    "expected": expected,
                    "comparison_type": comparison_type,
                    "passed": result["passed"],
                }
            )

OUT.write_text(
    json.dumps(
        {
            "source": "ServiceNow/EnterpriseOps-Gym@271f2c357f763376997dfd16807fcde2474ae41b benchmark/verifier.py",
            "extraction": extraction,
            "comparison": comparison,
        },
        indent=1,
    )
    + "\n"
)
print(f"{len(extraction)} extraction and {len(comparison)} comparison cases -> {OUT}")
