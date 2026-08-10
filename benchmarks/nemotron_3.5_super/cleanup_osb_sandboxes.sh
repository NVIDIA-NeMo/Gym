#!/bin/bash
# Reap OpenSandbox sandboxes leaked by ONE cancelled swebench eval job — never
# anyone else's. Sandboxes are matched by the exact per-job attribution labels
# injected via default_metadata in opensandbox.yaml (slurm-job-id +
# nemo-gym.nvidia.com/user), so only sandboxes provably created by the given
# Slurm job of the given user are touched. The list API is paged (page/pageSize).
#
# Usage (from repo root):
#   bash scripts/cleanup_osb_sandboxes.sh [--user hemild] [--dry-run]
#
# NOTE: sandboxes created before the attribution labels existed (jobs <= 5704199)
# carry no slurm-job-id and are deliberately NOT matched; they expire via TTL.
set -euo pipefail

MATCH_USER="${USER:-}"
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --user) MATCH_USER="$2"; shift 2;;
        --dry-run) DRY_RUN=1; shift;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done

DOMAIN=$(awk "/^sandbox:/{s=1} s && /domain:/{print \$2; exit}" env.yaml)
API_KEY=$(awk "/^sandbox:/{s=1} s && /api_key:/{print \$2; exit}" env.yaml)

MATCH_USER="$MATCH_USER" DRY_RUN="$DRY_RUN" DOMAIN="$DOMAIN" API_KEY="$API_KEY" python3 - <<'PYEOF'
import json, os, urllib.request, urllib.error

base = os.environ["DOMAIN"].rstrip("/")
headers = {"OPEN-SANDBOX-API-KEY": os.environ["API_KEY"]}
match_user = os.environ["MATCH_USER"]
dry_run = os.environ["DRY_RUN"] == "1"


def is_mine(i):
    m = i.get("metadata", {})
    if match_user and m.get("nemo-gym.nvidia.com/user") not in (match_user,):
        return False
    return True


page, mine, total = 1, [], 0
while True:
    req = urllib.request.Request(f"{base}/v1/sandboxes?page={page}&pageSize=100", headers=headers)
    d = json.load(urllib.request.urlopen(req, timeout=30))

    mine.extend(filter(is_mine, d.get("items", [])))
    total += len(d.get("items", []))

    if not d.get("pagination", {}).get("hasNextPage"):
        break

    print(f"Found {len(mine)} of my sandboxes ({total} total). Continuing to query...")
    page += 1

for i in mine:
    sid = i["id"]
    label = i.get("metadata", {}).get("instance_id", "?")
    if dry_run:
        print(f"DRY-RUN would delete {sid} ({label}, created {i['createdAt']})")
        continue
    req = urllib.request.Request(f"{base}/v1/sandboxes/{sid}", headers=headers, method="DELETE")
    try:
        r = urllib.request.urlopen(req, timeout=30)
        print(f"deleted {sid} ({label}) -> {r.status}")
    except urllib.error.HTTPError as e:
        status = "already gone" if e.code == 404 else f"HTTP {e.code}"
        print(f"delete {sid} ({label}) -> {status}")
PYEOF
