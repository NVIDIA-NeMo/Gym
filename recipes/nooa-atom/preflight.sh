#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The one check before a recipe runs anywhere: every pin matches, the config resolves, every file the
# resolved config names exists, the offline runtime validates, and (unless SKIP_ROW=1) one real row runs
# through the exact same path the full benchmark uses.
#
#   recipes/nooa-atom/preflight.sh terminal-bench-2.1
#   SKIP_ROW=1 recipes/nooa-atom/preflight.sh terminal-bench-2.1     # offline part only (no sandbox, no model)
#
# Exit codes: 2 pin mismatch, 3 missing input or environment, 4 config does not resolve, 5 the row failed.
set -euo pipefail

RECIPES=recipes/nooa-atom
[ -d "$RECIPES" ] || { echo "run this from the Gym repo root" >&2; exit 1; }
export RECIPES_DIR="$PWD/$RECIPES"
BENCH="${1:?usage: $RECIPES/preflight.sh <recipe>}"
CONFIG="$RECIPES/$BENCH.yaml"
[ -f "$CONFIG" ] || { echo "no recipe named '$BENCH'" >&2; exit 1; }
OUT="${OUT:-./results/$BENCH-preflight}"
mkdir -p "$OUT"
fail() { echo "preflight: $2" >&2; exit "$1"; }

pin() { sed -n "s/^$1:[[:space:]]*\([0-9a-f]\{40\}\).*/\1/p" "$CONFIG"; }

# 1. Pins. The recipe names them; the checkout, the agent's requirements and the task repo must agree.
GYM_COMMIT="${GYM_PIN:-$(pin gym_commit)}"
if [ "${PIN_GYM:-1}" != 0 ]; then
  [ -n "$GYM_COMMIT" ] || fail 2 "gym_commit in $CONFIG is not a 40-hex commit (recipe not frozen; PIN_GYM=0 to continue)"
  [ "$(git rev-parse HEAD)" = "$GYM_COMMIT" ] || fail 2 "checkout $(git rev-parse --short HEAD) != recipe gym_commit $GYM_COMMIT"
  [ -z "$(git status --porcelain --untracked-files=no)" ] || fail 2 "checkout has uncommitted changes"
fi
NOOA_COMMIT="$(pin nooa_commit)"
REQ=responses_api_agents/nooa_agent/requirements.txt
[ -n "$NOOA_COMMIT" ] || fail 2 "nooa_commit missing from $CONFIG"
[ "$(grep -c "$NOOA_COMMIT" "$REQ")" -ge 4 ] || fail 2 "$REQ does not pin all four NOOA packages at $NOOA_COMMIT"
grep -q "furgalep/labs-OO-Agents" "$REQ" && fail 2 "$REQ still points at a fork"
python - "$NOOA_COMMIT" <<'EOF' || fail 2 "installed NOOA packages do not match the recipe pin"
import importlib.metadata as m, sys
want = sys.argv[1]
for dist in ("nooa", "nooa-cli", "nooa-bench", "nooa-atom"):
    direct = m.distribution(dist).read_text("direct_url.json") or ""
    if want not in direct:
        print(f"{dist}: installed from {direct[:120] or 'an index'}, recipe pins {want}", file=sys.stderr)
        sys.exit(1)
EOF
SETUP="$RECIPES/$BENCH.setup.sh"
[ -f "$SETUP" ] && source "$SETUP"

# 2. Environment: names only, never values.
for v in POLICY_API_KEY OPENSANDBOX_API_KEY OPENSANDBOX_DOMAIN NOOA_RUNTIME_WHEELHOUSE; do
  [ -n "${!v:-}" ] || fail 3 "$v is not set in the environment"
done
[ -f env.yaml ] || fail 3 "./env.yaml missing (copy $RECIPES/env.yaml.example)"

# 3. Offline runtime: validate the prepared bundle against this checkout, for the pods' platform.
python -m responses_api_agents.nooa_agent.offline_runtime --output "$NOOA_RUNTIME_WHEELHOUSE" \
  --target "${NOOA_RUNTIME_TARGET:-x86_64/gnu}" --validate-only || fail 3 "offline runtime at $NOOA_RUNTIME_WHEELHOUSE does not validate"

# 4. Resolve the exact config the run will use, and check that every file it names exists.
CONFIGS=()
while IFS= read -r c; do [ -n "$c" ] && CONFIGS+=(--config "$c"); done < <(python -c 'import sys, yaml
v = (yaml.safe_load(open(sys.argv[1])) or {}).get("benchmark_config") or []
print("\n".join(v if isinstance(v, list) else [v]))' "$CONFIG")
CONFIGS+=(--config responses_api_models/vllm_model/configs/vllm_model.yaml --config "$CONFIG")
HYDRA=(hydra.run.dir="$OUT/hydra" hydra.output_subdir=null)
gym env resolve "${CONFIGS[@]}" "${HYDRA[@]}" > "$OUT/resolved.yaml" 2> "$OUT/resolve.err" || { cat "$OUT/resolve.err" >&2; fail 4 "config does not resolve"; }
python - "$OUT/resolved.yaml" <<'EOF' || fail 3 "resolved config names files that do not exist (see above)"
import os, sys, yaml
missing = []
def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{path}.{k}" if path else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, f"{path}[{i}]")
    elif isinstance(o, str):
        looks_like_file = o.endswith((".jsonl", ".py", ".yaml", ".yml", ".json", ".txt", ".sh")) and "/" in o and " " not in o
        looks_like_dir = path.endswith(("runtime_wheelhouse",))
        if (looks_like_file or looks_like_dir) and not o.startswith(("http://", "https://")) and not os.path.exists(o):
            missing.append((path, o))
walk(yaml.safe_load(open(sys.argv[1])))
for path, value in missing:
    print(f"missing: {path} = {value}", file=sys.stderr)
sys.exit(1 if missing else 0)
EOF
grep -q "agent_class: nooa_atom.gym:AtomHarness" "$OUT/resolved.yaml" || fail 4 "resolved config does not run AtomHarness"
grep -q "runtime_wheelhouse_required: true" "$OUT/resolved.yaml" || fail 4 "resolved config does not require the offline runtime"
sha256sum "$OUT/resolved.yaml" | tee "$OUT/resolved.sha256"
echo "preflight: pins, environment, runtime and config OK"

# 5. One real row through the same path as the full run. The row's own outcome is not judged here;
#    the path is: a rollout row (or a recorded failure row) must come back.
[ "${SKIP_ROW:-0}" = 1 ] && { echo "preflight: SKIP_ROW=1, done"; exit 0; }
LIMIT=1 OUT="$OUT/row" PARALLEL=1 "$RECIPES/run.sh" "$BENCH" || fail 5 "the one-row run exited non-zero"
rows=$(cat "$OUT/row/evaluator_rollouts.jsonl" 2>/dev/null | wc -l)
[ "$rows" -ge 1 ] || fail 5 "no rollout row was written (see $OUT/row)"
python - "$OUT/row/evaluator_rollouts.jsonl" <<'EOF'
import json, sys
row = json.loads(open(sys.argv[1]).readline())
resp = row.get("response") or {}
md = resp.get("metadata") or {}
print("preflight row:", {"status": resp.get("status"), "reward": row.get("reward"),
      "harness_execution": md.get("harness_execution"), "terminal_response_id": bool(md.get("terminal_response_id"))})
EOF
echo "preflight: OK"
