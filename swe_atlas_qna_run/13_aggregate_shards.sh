#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

cd "${GYM_DIR}"
mkdir -p "$(dirname "${FULL_OUTPUT}")"
export RAY_TMPDIR

HOST_SHARD_GLOB="${SHARD_OUTPUT_DIR}/shard_*_of_${NUM_SHARDS}.jsonl"
CONTAINER_SHARD_GLOB="${SHARD_CONTAINER_OUTPUT_DIR}/shard_*_of_${NUM_SHARDS}.jsonl"
FULL_MATERIALIZED_OUTPUT="${FULL_OUTPUT%.jsonl}_materialized_inputs.jsonl"
FULL_CONTAINER_MATERIALIZED_OUTPUT="${FULL_CONTAINER_OUTPUT%.jsonl}_materialized_inputs.jsonl"

python - "${SHARD_OUTPUT_DIR}" "${NUM_SHARDS}" "${FULL_MATERIALIZED_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

shard_output_dir = Path(sys.argv[1])
num_shards = int(sys.argv[2])
merged_materialized = Path(sys.argv[3])

rollout_paths = sorted(shard_output_dir.glob(f"shard_*_of_{num_shards}.jsonl"))
if len(rollout_paths) != num_shards:
    raise SystemExit(
        f"Expected {num_shards} shard rollout files in {shard_output_dir}, found {len(rollout_paths)}: "
        + ", ".join(str(p) for p in rollout_paths)
    )

materialized_paths = []
for rollout_path in rollout_paths:
    materialized_path = rollout_path.with_name(rollout_path.stem + "_materialized_inputs.jsonl")
    if not materialized_path.exists():
        raise SystemExit(f"Missing materialized inputs for {rollout_path}: {materialized_path}")
    materialized_paths.append(materialized_path)

rows = []
seen_keys = set()
for materialized_path in materialized_paths:
    with materialized_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("_ng_task_index"), row.get("_ng_rollout_index"))
            if key in seen_keys:
                raise SystemExit(f"Duplicate materialized rollout key {key} in {materialized_path}:{line_no}")
            seen_keys.add(key)
            rows.append(row)

rows.sort(key=lambda row: (row["_ng_task_index"], row["_ng_rollout_index"]))
merged_materialized.parent.mkdir(parents=True, exist_ok=True)
with merged_materialized.open("w", encoding="utf-8") as out:
    for row in rows:
        out.write(json.dumps(row, separators=(",", ":")) + "\n")

print(f"Shard rollouts: {len(rollout_paths)} files")
print(f"Merged materialized inputs: {merged_materialized} ({len(rows)} rows)")
PY

echo "Aggregating shard rollouts matching: ${HOST_SHARD_GLOB}"

if [[ "${RUN_GYM_IN_CONTAINER}" == "true" ]]; then
  srun --overlap \
    --container-image "${GYM_IMAGE}" \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --container-workdir "${GYM_CONTAINER_DIR}" \
    bash -lc "cd \"${GYM_CONTAINER_DIR}\" && source .venv/bin/activate && export PYTHONPATH=\"${GYM_CONTAINER_DIR}:\${PYTHONPATH:-}\" && gym eval aggregate --input-glob \"${CONTAINER_SHARD_GLOB}\" --output \"${FULL_CONTAINER_OUTPUT}\" && gym eval profile --inputs \"${FULL_CONTAINER_MATERIALIZED_OUTPUT}\" --rollouts \"${FULL_CONTAINER_OUTPUT}\""
else
  srun --overlap \
    bash -lc "cd \"${GYM_DIR}\" && source .venv/bin/activate && export PYTHONPATH=\"${GYM_DIR}:\${PYTHONPATH:-}\" && gym eval aggregate --input-glob \"${HOST_SHARD_GLOB}\" --output \"${FULL_OUTPUT}\" && gym eval profile --inputs \"${FULL_MATERIALIZED_OUTPUT}\" --rollouts \"${FULL_OUTPUT}\""
fi

echo "Merged rollouts: ${FULL_OUTPUT}"
echo "Merged materialized inputs: ${FULL_MATERIALIZED_OUTPUT}"
echo "Aggregate metrics: ${FULL_OUTPUT%.jsonl}_aggregate_metrics.json"
