# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert bird_sql task rows into bird_sql_agentic rows.

Only two things change, so single-turn vs. multi-turn results stay comparable:
the system prompt (which now mentions the tool) and the tool declaration.
The user message (question + schema) and every task field are kept as-is.

Usage:
    python resources_servers/bird_sql_agentic/create_examples.py \
        resources_servers/bird_sql/data/example.jsonl \
        resources_servers/bird_sql_agentic/data/example.jsonl
"""

import json
import sys


SYSTEM_PROMPT = (
    "You can call the execute_sql tool to run read-only SQL against the database for this question, "
    "for example to look at table columns (PRAGMA table_info(table_name)), check actual values, or test "
    "a draft query. Only one SELECT, WITH, or PRAGMA table_info statement per call; results show at most "
    'the first 20 rows. When you are confident, give your final answer within the tags "```sql" and "```". '
    "If there is more than one set of tags, the last one will be taken as your final answer."
)

EXECUTE_SQL_TOOL = {
    "type": "function",
    "name": "execute_sql",
    "description": (
        "Run one read-only SQL query (SELECT, WITH, or PRAGMA table_info) on this task's SQLite database. "
        "Returns column names, up to 20 rows, the total row count, or an error message."
    ),
    "parameters": {
        "type": "object",
        "properties": {"sql": {"type": "string", "description": "The SQL query to run."}},
        "required": ["sql"],
        "additionalProperties": False,
    },
    "strict": True,
}


def convert(row: dict) -> dict:
    row = json.loads(json.dumps(row))  # deep copy
    params = row["responses_create_params"]
    messages = [m for m in params["input"] if m.get("role") != "system"]
    params["input"] = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    params["tools"] = [EXECUTE_SQL_TOOL]
    return row


def main(src: str, dst: str) -> None:
    with open(src) as f_in, open(dst, "w") as f_out:
        for line in f_in:
            if line.strip():
                f_out.write(json.dumps(convert(json.loads(line))) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
