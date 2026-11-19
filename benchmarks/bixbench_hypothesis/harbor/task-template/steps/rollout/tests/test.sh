#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/artifacts
mkdir -p /logs/verifier

shopt -s nullglob
artifacts=(.harbor/artifacts/*)
if (( ${#artifacts[@]} )); then
    mv "${artifacts[@]}" /logs/artifacts
fi

if [[ -f /logs/artifacts/judge_error.json ]]; then
    printf '%s\n' '[ERROR] Judging failed; reward is unavailable. See /logs/artifacts/judge_error.json.' >&2
    exit 1
elif [[ -f /logs/artifacts/reward.json ]]; then
    mv /logs/artifacts/reward.json /logs/verifier
else
    printf '%s\n' '[WARNING] /logs/artifacts/reward.json was not found; using reward 0.' >&2
    printf '0\n' > /logs/verifier/reward.txt
fi
