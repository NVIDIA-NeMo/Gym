#!/bin/bash
#SBATCH --job-name=synthetic-seed-generation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out

set -euo pipefail

: "${SEED_GENERATION_CONFIG:?Set SEED_GENERATION_CONFIG to a generation YAML path}"

SEED_GENERATION_COMMAND=${SEED_GENERATION_COMMAND:-all}
DOMAIN_START=${DOMAIN_START:-}
DOMAIN_END=${DOMAIN_END:-}

args=(
  --config "$SEED_GENERATION_CONFIG"
  --resume
)

if [[ -n "$DOMAIN_START" ]]; then
  args+=(--domain-start "$DOMAIN_START")
fi
if [[ -n "$DOMAIN_END" ]]; then
  args+=(--domain-end "$DOMAIN_END")
fi

python resources_servers/synthetic_tool_use/scripts/generate_synthetic_tool_use_seeds.py \
  "${args[@]}" \
  "$SEED_GENERATION_COMMAND"
