#!/bin/bash
# Stub for the loader test: the dataset's test.sh compares /app/selected_asin.txt with the expected product.
set -euo pipefail
mkdir -p /logs/verifier
test -s /app/selected_asin.txt && echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt
