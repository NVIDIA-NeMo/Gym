#!/usr/bin/env bash
# Given a list of changed server paths, find other servers that depend on them.
#
# Usage: echo "responses_api_models/vllm_model" | ./scripts/find_dependent_servers.sh
#   or:  ./scripts/find_dependent_servers.sh < changed_servers.txt
#
# Reads changed server paths (one per line, e.g. "resources_servers/arc_agi")
# from stdin. Outputs the full list of servers to test (changed + dependents),
# deduplicated and sorted.
#
# How it works:
#   For each changed server, converts the path to a Python import prefix
#   (e.g. "responses_api_models/vllm_model" -> "responses_api_models.vllm_model")
#   and greps all server directories for files that import from it.

set -euo pipefail

SERVER_DIRS="resources_servers responses_api_agents responses_api_models"
CHANGED_SERVERS=$(cat)

# Start with the changed servers themselves
ALL_SERVERS="$CHANGED_SERVERS"

for server in $CHANGED_SERVERS; do
  # Convert path to Python import prefix: resources_servers/arc_agi -> resources_servers.arc_agi
  IMPORT_PREFIX=$(echo "$server" | tr '/' '.')

  # Find any server that references this one (imports, string refs, configs)
  DEPENDENTS=$(grep -rl "$IMPORT_PREFIX\|$server" \
    $SERVER_DIRS 2>/dev/null \
    | grep -v __pycache__ \
    | grep -v '\.venv/' \
    | cut -d'/' -f1-2 \
    | sort -u) || true

  if [ -n "$DEPENDENTS" ]; then
    ALL_SERVERS=$(printf '%s\n%s' "$ALL_SERVERS" "$DEPENDENTS")
  fi
done

# Deduplicate and sort
echo "$ALL_SERVERS" | sort -u
