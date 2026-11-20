#!/usr/bin/env bash

set -euo pipefail

uv run /tests/grade.py 2>&1
