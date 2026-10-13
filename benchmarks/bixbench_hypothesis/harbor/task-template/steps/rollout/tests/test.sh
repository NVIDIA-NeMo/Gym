#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/artifacts
mkdir -p /logs/verifier

if [[ -d .harbor/artifacts ]]; then
    mv .harbor/artifacts/* /logs/artifacts
fi

if [[ -f /logs/artifacts/reward.json ]]; then
    mv /logs/artifacts/reward.json /logs/verifier
else
    printf '%s\n' '[WARNING] /logs/artifacts/reward.json was not found; using reward 0.' >&2
    printf '0\n' > /logs/verifier/reward.txt
fi
