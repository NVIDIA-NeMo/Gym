#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/scripts/nrt/load_images_from_tars.sh
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
# Load SaaS-Bench docker image tarballs into rootless Podman storage.
#
# Preferred sources:
#   1) $SAAS_IMAGE_TAR_DIR
#   2) /saas-image-tars (deployment-provided mount)
#   3) $REPO/docker/images
#
# Optional: SAAS_LOAD_ONLY="farmos grocy" to load a subset (faster smoke).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ "${SAAS_NRT_PODMAN:-}" != "1" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/setup_podman_runtime.sh"
fi

# Python used only to read RepoTags out of tar archives (stdlib only).
PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  for c in python3 /usr/bin/python3; do
    command -v "$c" >/dev/null 2>&1 && { PYTHON_BIN=$c; break; }
  done
fi

TAR_DIRS=(
  "${SAAS_IMAGE_TAR_DIR:-}"
  "/saas-image-tars"
  "${REPO_ROOT}/docker/images"
  "${SAAS_NRT_REAL_HOME:-}/saas_bench_docker_images"
)
shopt -s nullglob
tars=()
for d in "${TAR_DIRS[@]}"; do
  [[ -z "$d" || ! -d "$d" ]] && continue
  for f in "$d"/*.tar "$d"/*.tar.gz; do
    [[ -f "$f" ]] && tars+=("$f")
  done
done

# Deduplicate paths that resolve to the same file
declare -A SEEN=()
uniq_tars=()
for tar in "${tars[@]}"; do
  real=$(readlink -f "$tar" 2>/dev/null || echo "$tar")
  [[ -n "${SEEN[$real]:-}" ]] && continue
  SEEN[$real]=1
  uniq_tars+=("$tar")
done
tars=("${uniq_tars[@]}")

if [[ ${#tars[@]} -eq 0 ]]; then
  cat >&2 <<EOF
ERROR: no image tarballs found.
Download with: bash scripts/nrt/download_images_hf.sh
Set SAAS_IMAGE_TAR_DIR to the downloaded image directory.
EOF
  exit 2
fi

echo "[nrt] loading from ${#tars[@]} candidate tarballs; store=${CONTAINERS_STORAGE_CONF}"
ONLY="${SAAS_LOAD_ONLY:-}"
loaded=0

# App identifiers are normally embedded directly in the archive filename.
# Keep the small set of intentionally different names in one place so the
# preflight size estimate and the real import filter cannot diverge.  In
# particular the user-facing app is ``e-label`` while the shipped archive and
# image are ``mw-elabel``.
nrt_load_only_matches() {
  local base="$1" requested
  [[ -z "$ONLY" ]] && return 0
  for requested in $ONLY; do
    case "$requested" in
      e-label) requested="elabel" ;;
    esac
    [[ "$base" == *"$requested"* ]] && return 0
  done
  return 1
}

# Estimate how much /tmp we need for the filtered set (tar copy + unpacked ~2x).
need_bytes=0
for tar in "${tars[@]}"; do
  base=$(basename "$tar")
  nrt_load_only_matches "$base" || continue
  sz=$(stat -c%s "$tar" 2>/dev/null || echo 0)
  need_bytes=$((need_bytes + sz * 3 / 2))  # copy + partial unpack headroom
done
need_gb=$((need_bytes / 1024 / 1024 / 1024 + 1))
min_now="${SAAS_NRT_MIN_TMP_GB:-15}"
if [[ "${need_gb}" -gt "${min_now}" ]]; then
  export SAAS_NRT_MIN_TMP_GB="${need_gb}"
fi
echo "[nrt] estimated /tmp need ~${need_gb}G for selected images (copy+store headroom)"
if declare -F nrt_df_tmp >/dev/null 2>&1; then
  nrt_df_tmp
else
  df -h /tmp | tail -1
fi

# ---------------------------------------------------------------------------
# Post-load integrity check.
# Podman 3.4 on NRT can report a successful `podman load` while leaving the
# overlay store with missing layers ("layer not known" / "may be corrupted"
# on the first `podman run`). A bare tag is not enough — we must touch the
# image the same way run does (inspect + create).
# ---------------------------------------------------------------------------
nrt_image_ok() {
  local ref="$1"
  local cname="nrtv-$$-${RANDOM}"
  local out rc=0
  out=$(podman image inspect "$ref" 2>&1) || rc=$?
  if [[ $rc -ne 0 ]] || echo "$out" | grep -qiE 'layer not known|may be corrupted'; then
    echo "$out" | tail -5 >&2
    return 1
  fi
  # `create` resolves layers without starting the container (cheaper than run).
  out=$(podman create --name "$cname" --network none "$ref" 2>&1) || rc=$?
  podman rm -f "$cname" >/dev/null 2>&1 || true
  if [[ $rc -ne 0 ]] || echo "$out" | grep -qiE 'layer not known|may be corrupted'; then
    echo "$out" | tail -8 >&2
    return 1
  fi
  return 0
}

nrt_purge_refs() {
  local ref
  for ref in "$@"; do
    [[ -n "$ref" ]] || continue
    podman rmi -f "$ref" >/dev/null 2>&1 || true
  done
  # Also drop untagged leftovers that load often leaves as localhost/latest.
  podman images --format '{{.ID}} {{.Repository}}:{{.Tag}}' 2>/dev/null \
    | awk '$2=="localhost:latest" || $2=="<none>:<none>" {print $1}' \
    | sort -u \
    | while read -r id; do
        podman rmi -f "$id" >/dev/null 2>&1 || true
      done
}

nrt_load_archive() {
  # Args: local_tar is_oci -> sets load_out/load_rc in caller via echo+return
  local local_tar="$1" is_oci="$2"
  local out="" rc=1 attempt
  for attempt in 1 2 3; do
    if [[ "$is_oci" == "1" ]]; then
      out=$(podman pull "oci-archive:${local_tar}" 2>&1); rc=$?
      if [[ $rc -ne 0 ]]; then out=$(podman load -i "$local_tar" 2>&1); rc=$?; fi
    else
      out=$(podman load -i "$local_tar" 2>&1); rc=$?
      if [[ $rc -ne 0 ]]; then out=$(podman pull "docker-archive:${local_tar}" 2>&1); rc=$?; fi
    fi
    if [[ $rc -eq 0 ]]; then
      printf '%s' "$out"
      return 0
    fi
    echo "  [warn] load attempt ${attempt} failed (rc=${rc}); retrying..." >&2
    sleep 2
  done
  printf '%s' "$out"
  return 1
}

LOAD_VERIFY_ATTEMPTS="${SAAS_NRT_LOAD_VERIFY_ATTEMPTS:-3}"

for tar in "${tars[@]}"; do
  base=$(basename "$tar")
  nrt_load_only_matches "$base" || continue
  tag="${base%.tar}"
  tag="${tag%.tar.gz}"
  echo "[nrt] podman load -i $tar  ->  ${tag}:latest"
  # Copy off Lustre first — direct `podman load` from lustre is flaky on NRT
  # (often dies mid-blob ~20s). Node-local /tmp load is reliable.
  local_tar="${SAAS_NRT_SCRATCH}/images/${base}"
  mkdir -p "${SAAS_NRT_SCRATCH}/images"
  tar_bytes=$(stat -c%s "$tar" 2>/dev/null || echo 0)
  avail_kb=$(df -Pk /tmp | awk 'NR==2{print $4}')
  # Need room for the tar copy (~1x) plus unpack into store (~1x); require 2.2x.
  need_kb=$((tar_bytes * 22 / 10 / 1024))
  if [[ -n "${avail_kb}" && "${avail_kb}" -lt "${need_kb}" ]]; then
    echo "ERROR: /tmp free $((avail_kb/1024/1024))G < need ~$((need_kb/1024/1024))G for $base" >&2
    df -h /tmp >&2
    exit 1
  fi
  if [[ ! -f "$local_tar" ]]; then
    echo "[nrt] copying $base -> /tmp scratch ($(numfmt --to=iec "$tar_bytes" 2>/dev/null || echo "${tar_bytes}B"))..."
    cp -f "$tar" "$local_tar"
  fi
  # Read (image-id -> RepoTags) from the archive BEFORE loading. podman 3.4's
  # `podman load` does NOT reliably apply RepoTags (imports as localhost/latest
  # or <none>), so we tag deterministically by config image-id afterwards.
  mapfile -t id_tag_pairs < <("$PYTHON_BIN" - "$local_tar" <<'PY'
import sys, tarfile, json
p = sys.argv[1]
pairs = []
try:
    tf = tarfile.open(p)
    names = tf.getnames()
    if "manifest.json" in names:
        for e in json.load(tf.extractfile("manifest.json")) or []:
            cfg = (e.get("Config") or "")
            img_id = cfg.split("/")[-1].replace(".json", "").replace("sha256:", "")
            for t in (e.get("RepoTags") or []):
                if t and img_id:
                    pairs.append("%s\t%s" % (img_id, t))
    elif "index.json" in names:
        idx = json.load(tf.extractfile("index.json"))
        for m in idx.get("manifests", []):
            ann = m.get("annotations") or {}
            t = ann.get("org.opencontainers.image.ref.name")
            d = (m.get("digest") or "").replace("sha256:", "")
            if t and d:
                pairs.append("%s\t%s" % (d, t))
except Exception as e:
    sys.stderr.write("tag-read error: %s\n" % e)
for pr in pairs:
    print(pr)
PY
)

  is_oci=0
  tar -tf "$local_tar" 2>/dev/null | grep -qx 'oci-layout' && is_oci=1

  # Outer loop: load → tag → verify; on "layer not known" purge and retry.
  verify_ok=0
  refs_to_check=()
  set +e
  for v_attempt in $(seq 1 "${LOAD_VERIFY_ATTEMPTS}"); do
    refs_to_check=()
    load_out=$(nrt_load_archive "$local_tar" "$is_oci")
    load_rc=$?
    if [[ $load_rc -ne 0 ]]; then
      echo "  [warn] ${base}: load failed on verify-attempt ${v_attempt}/${LOAD_VERIFY_ATTEMPTS}" >&2
      echo "$load_out" | tail -8 >&2
      sleep 2
      continue
    fi
    echo "$load_out" | grep -iE 'Loaded image|Writing blob|Copying' | tail -5 || true

    if [[ ${#id_tag_pairs[@]} -gt 0 ]]; then
      applied=0
      for pair in "${id_tag_pairs[@]}"; do
        img_id="${pair%%$'\t'*}"
        rt="${pair##*$'\t'}"
        [[ "$rt" == *:* ]] || rt="${rt}:latest"
        if podman tag "$img_id" "$rt" 2>/dev/null; then
          applied=$((applied + 1))
          refs_to_check+=("$rt")
        else
          echo "  [warn] could not tag ${img_id:0:12} -> $rt" >&2
        fi
      done
      echo "[nrt] ${base}: applied ${applied}/${#id_tag_pairs[@]} tag(s) by image-id"
      if [[ ${#id_tag_pairs[@]} -eq 1 ]]; then
        first_id="${id_tag_pairs[0]%%$'\t'*}"
        if ! podman image exists "${tag}:latest" 2>/dev/null; then
          podman tag "$first_id" "${tag}:latest" 2>/dev/null && refs_to_check+=("${tag}:latest")
        else
          refs_to_check+=("${tag}:latest")
        fi
      fi
    else
      img_id=$(podman images --sort created --format '{{.ID}}' | head -1)
      if [[ -z "$img_id" ]]; then
        echo "  [warn] ${base}: no image id after load" >&2
        continue
      fi
      podman tag "$img_id" "${tag}:latest"
      refs_to_check+=("${tag}:latest")
      echo "[nrt] no manifest tags; tagged ${img_id} as ${tag}:latest"
    fi

    # Dedup refs
    mapfile -t refs_to_check < <(printf '%s\n' "${refs_to_check[@]}" | awk 'NF && !seen[$0]++')

    all_ok=1
    for ref in "${refs_to_check[@]}"; do
      echo "[nrt] verifying ${ref} ..."
      if ! nrt_image_ok "$ref"; then
        echo "  [warn] ${base}: ${ref} failed integrity check (layer store flake) — purge & reload (${v_attempt}/${LOAD_VERIFY_ATTEMPTS})" >&2
        all_ok=0
        break
      fi
      echo "[nrt] ${ref}: OK"
    done

    if [[ "$all_ok" -eq 1 && ${#refs_to_check[@]} -gt 0 ]]; then
      verify_ok=1
      break
    fi
    nrt_purge_refs "${refs_to_check[@]}" "${tag}:latest"
    sleep 2
  done
  set -e

  if [[ "$verify_ok" -ne 1 ]]; then
    echo "ERROR: ${base}: image failed load+verify after ${LOAD_VERIFY_ATTEMPTS} attempts (would hit 'layer not known' at run)" >&2
    exit 1
  fi

  loaded=$((loaded + 1))
  # Free tar copy only after a verified load (keep it for reload retries above).
  rm -f "$local_tar"
  if declare -F nrt_df_tmp >/dev/null 2>&1; then
    nrt_df_tmp || true
  else
    df -h /tmp | tail -1
  fi
done
if [[ "$loaded" -eq 0 ]]; then
  echo "ERROR: no tarballs loaded (filter SAAS_LOAD_ONLY='${ONLY}' matched nothing?)" >&2
  exit 2
fi

if podman image exists mw-code-server:latest 2>/dev/null; then
  podman tag mw-code-server:latest code-server-bundle:latest || true
fi
podman images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' | head -40 || true
echo "[nrt] load done (loaded=${loaded})"
