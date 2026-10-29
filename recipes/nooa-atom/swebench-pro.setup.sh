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
# Sourced by run.sh and preflight.sh after the benchmark is prepared: checks the roster's hash, writes the
# prepared rows of the roster's instances to a dedicated file, and makes the run collect exactly those rows.
RECIPE="$RECIPES_DIR/swebench-pro.yaml"
ROSTER="$(sed -n 's/^roster:[[:space:]]*\(.*\)$/\1/p' "$RECIPE")"
ROSTER_SHA="$(sed -n 's/^roster_sha256:[[:space:]]*\([0-9a-f]\{64\}\).*/\1/p' "$RECIPE")"
[ -f "$ROSTER" ] || { echo "swebench-pro: roster $ROSTER missing" >&2; exit 1; }
[ "$(sha256sum "$ROSTER" | cut -d' ' -f1)" = "$ROSTER_SHA" ] || { echo "swebench-pro: roster $ROSTER does not match roster_sha256" >&2; exit 1; }
PREPARED=benchmarks/swebench/data/swebench_pro_benchmark.jsonl
ROWS=benchmarks/swebench/data/swebench_pro_swe120.jsonl
if [ -f "$PREPARED" ]; then
  python - "$PREPARED" "$ROSTER" "$ROWS" <<'EOF'
import json, sys
prepared, roster, out = sys.argv[1:4]
wanted = {line.strip() for line in open(roster) if line.strip()}
rows = [json.loads(line) for line in open(prepared) if line.strip()]
selected = [r for r in rows if (r.get("verifier_metadata") or {}).get("instance_id") in wanted]
missing = wanted - {(r.get("verifier_metadata") or {}).get("instance_id") for r in selected}
if missing:
    print(f"swebench-pro: {len(missing)} roster instances are not in the prepared benchmark, e.g. {sorted(missing)[:3]}", file=sys.stderr)
    sys.exit(1)
with open(out, "w") as f:
    for r in selected:
        f.write(json.dumps(r) + "\n")
print(f"swebench-pro: {len(selected)} roster rows written to {out}")
EOF
  export INPUT="$ROWS"
else
  echo "swebench-pro: $PREPARED not prepared yet (gym eval prepare runs first; the roster rows are written on the next pass)"
fi
