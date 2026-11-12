#!/usr/bin/env bash

set -euo pipefail

uv run /tests/grade.py 2>&1 || echo "ERROR: uv run failed with exit code $?"
