#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

cd "${GYM_DIR}"

INPUT="benchmarks/swe_atlas_qna/data/swe_atlas_qna_benchmark.jsonl"
HOST_INPUT="${GYM_DIR}/${INPUT}"
if [[ ! -s "${HOST_INPUT}" ]]; then
  echo "Full benchmark input missing: ${HOST_INPUT}. Run 01_prepare_smoke_slice.sh first." >&2
  exit 2
fi

mkdir -p "${SHARD_INPUT_DIR}"

python - "${HOST_INPUT}" "${SHARD_INPUT_DIR}" "${NUM_SHARDS}" <<'PY'
import json
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
num_shards = int(sys.argv[3])

if num_shards < 1:
    raise SystemExit(f"NUM_SHARDS must be >= 1, got {num_shards}")

tmp_dir = shard_dir.with_name(shard_dir.name + ".tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)
for stale in tmp_dir.glob("*.jsonl"):
    stale.unlink()

shard_paths = [tmp_dir / f"input_shard_{idx}_of_{num_shards}.jsonl" for idx in range(num_shards)]
counts = [0] * num_shards

files = [path.open("w", encoding="utf-8") for path in shard_paths]
try:
    with input_path.open(encoding="utf-8") as src:
        for task_idx, line in enumerate(src):
            if not line.strip():
                continue
            row = json.loads(line)
            # Keep task ids globally stable across shards so aggregation/profile can align rows.
            row["_ng_task_index"] = task_idx
            shard_idx = task_idx % num_shards
            files[shard_idx].write(json.dumps(row, separators=(",", ":")) + "\n")
            counts[shard_idx] += 1
finally:
    for f in files:
        f.close()

shard_dir.mkdir(parents=True, exist_ok=True)
for stale in shard_dir.glob("*.jsonl"):
    stale.unlink()
for path in shard_paths:
    path.rename(shard_dir / path.name)
tmp_dir.rmdir()

total = sum(counts)
for idx, count in enumerate(counts):
    print(f"{shard_dir / f'input_shard_{idx}_of_{num_shards}.jsonl'}: {count} rows, {count * 3} rollouts at --num-repeats 3")
print(f"Total: {total} rows, {total * 3} rollouts at --num-repeats 3")
PY

echo "Shard input directory: ${SHARD_INPUT_DIR}"
