# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the BIRD databases into a Snowflake account for the ``bird_sql`` server's ``snowflake_engine: snowflake``.

Run it once, through an admin connection, before grading on the account: it loads every ``db_id`` the account's
``BIRD`` database doesn't have yet (in the layout ``check_bird_snowflake_account`` reads) and grants the grader's
read-only role access to them. Rerun it after a reload: FUTURE grants need MANAGE GRANTS, so new tables get their
grants here. See ``resources_servers/bird_sql/README.md`` for the account setup.

Needs ``snowflake-connector-python``, ``pandas`` and ``pyarrow``, which the ``bird_sql`` server's environment has: run it
with ``resources_servers/bird_sql/.venv/bin/python``.
Run: ``python -m benchmarks.birdbench.snowflake_load_account <admin connection> --grant-to <role> [--split dev|train]
[--train-dir DIR]``.
"""

import argparse
import logging

from benchmarks.birdbench.prepare import databases_dir_for
from resources_servers.bird_sql.setup_bird_snowflake import load_bird_snowflake_account


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("connection", help="Admin connection in the connector's connections.toml, with a warehouse")
    parser.add_argument("--grant-to", required=True, help="The grader's read-only role, granted SELECT on the data")
    parser.add_argument("--split", choices=("dev", "train"), default="dev")
    parser.add_argument("--train-dir", help="--split train: an extracted BIRD train/ directory (default: download)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    load_bird_snowflake_account(databases_dir_for(args.split, args.train_dir), args.connection, args.grant_to)


if __name__ == "__main__":
    main()
