#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

cd "${GYM_DIR}"

python - <<'PY2'
import json
import os
from pathlib import Path
path = Path(os.environ["SMOKE_OUTPUT"])
if not path.exists():
    raise SystemExit(f"Missing smoke output: {path}")
for line in path.open():
    d = json.loads(line)
    rubrics = d.get("rubric_scores") or []
    passed = sum(1 for r in rubrics if (r.get("score") or {}).get("score") == "1")
    print("instance:", d.get("instance_id"))
    print("reward:", d.get("reward"), "| agg_score:", d.get("agg_score"), "| rubrics:", f"{passed}/{len(rubrics)}")
    print("counts:", {"num_rubrics": d.get("num_rubrics"), "num_scored": d.get("num_scored"), "num_unscored": d.get("num_unscored")})
    print("answer chars:", len(d.get("answer") or ""))
    print("run_error:", d.get("run_error"))
    print("verify_error:", d.get("verify_error"))
    print("verify_result_keys:", d.get("verify_result_keys"))
    if rubrics:
        print("first rubric:", rubrics[0].get("title"))
        print("first score:", rubrics[0].get("score"))
PY2
