# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grade one BIRD answer inside the verifier sandbox: ``python3 sandbox_runner.py request.json result.json``.

Runs only in the sandbox, next to an uploaded copy of ``eval_utils.py``, and uses only the standard library. The gold
query and then the model's query run through ``eval_utils.execute_and_compare``, the code the Resources Server ran
them with before, so a query is executed, timed out and compared exactly as it was there.

``request.json`` holds ``database`` (path of the SQLite file, relative to this directory or absolute), ``gold_sql``,
``pred_sql``, ``timeout_s`` (per query), ``memory_limit_mib`` (address-space cap, or null), ``delete_database``
(remove the file afterwards, for a copy uploaded for this grade) and ``database_bytes`` (the uploaded copy's expected
size, or null). The database is opened read-only, so a query that writes fails as a query. ``result.json`` gets ``{"match": bool, "error": tag or null}``, with the tags of
``execute_and_compare``. A nonzero exit means the sandbox could not grade.
"""

import asyncio
import json
import os
import resource
import sys
from pathlib import Path

from eval_utils import execute_and_compare


async def _grade(request: dict) -> dict:
    database = Path(request["database"]).resolve()
    try:
        if not database.is_file():
            raise SystemExit(f"database {database} is missing in the sandbox")
        expected_bytes = request.get("database_bytes")
        if expected_bytes is not None and database.stat().st_size != expected_bytes:
            raise SystemExit(f"database {database} has {database.stat().st_size} bytes, expected {expected_bytes}")
        try:
            match, _gold, _pred, error = await execute_and_compare(
                db_path=database,
                gold_sql=request["gold_sql"],
                pred_sql=request["pred_sql"],
                semaphore=asyncio.Semaphore(1),
                timeout_s=request["timeout_s"],
                read_only=True,
            )
        except MemoryError:
            # A query that runs out of memory returns None from execute_sqlite; this is the comparison of two result
            # sets after both queries ran, and only the model's can be too large: the gold one is a benchmark answer.
            match, error = False, "pred_sql_error"
        return {"match": match, "error": error}
    finally:
        # Providers that share /tmp with the host (local, Apptainer without --contain) would keep each copy.
        if request.get("delete_database"):
            database.unlink(missing_ok=True)


def main() -> None:
    request_path, result_path = sys.argv[1:3]
    request = json.loads(Path(request_path).read_text())
    if request.get("memory_limit_mib"):
        # A query that needs more memory fails with MemoryError, which execute_sqlite reports as a failed query (a
        # scored zero for the model's query), instead of the sandbox being killed and the grade lost.
        limit = int(request["memory_limit_mib"]) * 1024 * 1024
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)  # a process may not raise its hard limit
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    result = asyncio.new_event_loop().run_until_complete(_grade(request))
    Path(result_path).write_text(json.dumps(result))
    # Not asyncio.run(): its teardown joins the worker thread, and a query whose deadline passed may still hold it.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
