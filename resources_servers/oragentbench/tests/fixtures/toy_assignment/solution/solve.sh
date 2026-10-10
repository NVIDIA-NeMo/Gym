#!/usr/bin/env bash
set -euo pipefail
mkdir -p /app/submissions
python /solution/solve_reference.py --env-dir /app --output /app/submissions/solution.csv
