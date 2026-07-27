#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

cd "${GYM_DIR}"
source .venv/bin/activate
mkdir -p "${RUN_DIR}" results

echo "== Preparing SWE-Atlas QnA benchmark JSONL =="
# If you have a local SWE-Atlas checkout, export SWE_ATLAS_DIR before running this script.
uv run gym eval prepare --benchmark swe_atlas_qna

echo "== Building one-task secdev/scapy smoke slice =="
python - <<'PY2'
import json
import os
from pathlib import Path
src = Path("benchmarks/swe_atlas_qna/data/swe_atlas_qna_benchmark.jsonl")
dst = Path(os.environ["SMOKE_PATH"])
dst.parent.mkdir(parents=True, exist_ok=True)
selected = None
with src.open() as f:
    for line in f:
        row = json.loads(line)
        md = row.get("verifier_metadata") or {}
        haystack = " ".join(str(md.get(k, "")) for k in ("repository", "docker_image", "sif_basename", "instance_id"))
        if "secdev_scapy" in haystack or "secdev/scapy" in haystack:
            selected = row
            break
if selected is None:
    raise SystemExit("No secdev/scapy task found in benchmark JSONL")
with dst.open("w") as out:
    out.write(json.dumps(selected) + "\n")
print(f"Wrote smoke slice: {dst}")
print("instance_id:", selected.get("instance_id"))
print("sif_basename:", (selected.get("verifier_metadata") or {}).get("sif_basename"))
PY2
