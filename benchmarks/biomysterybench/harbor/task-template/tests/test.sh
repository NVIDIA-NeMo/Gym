#!/usr/bin/env bash

set -euo pipefail

uv run --quiet --no-project --with litellm \
/tests/grade.py
