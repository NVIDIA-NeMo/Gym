#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Bootstrap rootless Podman so SaaS-Bench can run on NRT cpu_* nodes
# without a Docker daemon / Brev VM.
#
# Usage (inside an srun/sbatch allocation):
#   source scripts/nrt/setup_podman_runtime.sh
#
# Exports:
#   PATH           — docker shim + docker-compose ahead of system PATH
#   DOCKER_HOST    — unix socket served by `podman system service`
#   XDG_RUNTIME_DIR, CONTAINERS_STORAGE_CONF, HOME (scratch under SLURM_TMPDIR)
#   SAAS_NRT_PODMAN=1
#
# Hard requirements observed on NRT:
#   - Use node-local scratch for container storage (Lustre/home overlay breaks).
#   - Publish ports as 127.0.0.1:HOST:CTR via slirp4netns (works).
#   - `docker compose` must be routed to compose-v2 binary (podman 3.4 has no compose).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[nrt] WARNING: not inside a Slurm allocation; using /tmp scratch." >&2
fi

# Keep paths SHORT: slirp4netns rejects API sockets with path length >= 108.
# Always bind scratch to the *current* Slurm job id when available — a stale
# SAAS_NRT_SCRATCH from `sbatch --export=ALL` can point at a previous job's /tmp.
#
# WHY /tmp (not Lustre): rootless Podman must chown layer files to sub-UIDs;
# Lustre rejects that (pivot_root / permission denied). Overlay+fuse on /tmp works.
# Watch free space — image store + tar copies can be tens of GB.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRATCH_ROOT="/tmp/s${SLURM_JOB_ID}"
else
  SCRATCH_ROOT="${SAAS_NRT_SCRATCH:-/tmp/s$$}"
fi
mkdir -p "${SCRATCH_ROOT}"/{run,store/run,store/graph,home,bin,logs,cni,images}

# Keep the two storage roots as first-class values rather than relying solely
# on Podman 3.4 configuration discovery.  Some NRT nodes have ignored the
# per-job storage.conf during `podman load`, which leaves a layer index in one
# store and overlay data in another (later reported as "layer not known").
export SAAS_NRT_PODMAN_RUNROOT="${SCRATCH_ROOT}/store/run"
export SAAS_NRT_PODMAN_GRAPHROOT="${SCRATCH_ROOT}/store/graph"
# Overlay plus fuse-overlayfs is the normal, fast NRT path.  A small number
# of large archives have repeatedly produced an incomplete overlay link index
# after a successful import ("layer not known" on `podman create`).  ``vfs``
# avoids that overlay metadata path at the cost of disk space and is therefore
# an explicit per-job fallback rather than a cluster-wide default.
SAAS_NRT_PODMAN_STORAGE_DRIVER="${SAAS_NRT_PODMAN_STORAGE_DRIVER:-overlay}"
case "${SAAS_NRT_PODMAN_STORAGE_DRIVER}" in
  overlay|vfs) ;;
  *) echo "ERROR: unsupported SAAS_NRT_PODMAN_STORAGE_DRIVER=${SAAS_NRT_PODMAN_STORAGE_DRIVER}" >&2; return 2 2>/dev/null || exit 2 ;;
esac
export SAAS_NRT_PODMAN_STORAGE_DRIVER

# Resolve the real executable before adding our job-local shim to PATH.  Keep
# it exported so repeated sourcing cannot accidentally wrap the shim itself.
if [[ -z "${SAAS_NRT_PODMAN_BIN:-}" ]]; then
  SAAS_NRT_PODMAN_BIN="$(command -v podman || true)"
fi
if [[ -z "${SAAS_NRT_PODMAN_BIN}" || ! -x "${SAAS_NRT_PODMAN_BIN}" ]]; then
  echo "ERROR: podman executable not found before NRT runtime setup" >&2
  return 127 2>/dev/null || exit 127
fi
export SAAS_NRT_PODMAN_BIN

nrt_df_tmp() {
  # Print /tmp free space; exit 1 if below SAAS_NRT_MIN_TMP_GB (default 15).
  local min_gb="${SAAS_NRT_MIN_TMP_GB:-15}"
  local avail_kb avail_gb scratch="${SAAS_NRT_SCRATCH:-?}"
  avail_kb=$(df -Pk /tmp 2>/dev/null | awk 'NR==2{print $4}')
  if [[ -z "${avail_kb}" ]]; then
    echo "[nrt] WARNING: cannot read df /tmp" >&2
    return 0
  fi
  avail_gb=$((avail_kb / 1024 / 1024))
  echo "[nrt] /tmp free=${avail_gb}G (min=${min_gb}G) scratch=${scratch}"
  if [[ "${avail_gb}" -lt "${min_gb}" ]]; then
    echo "ERROR: /tmp only has ${avail_gb}G free (< ${min_gb}G). Aborting to avoid filling the node." >&2
    df -h /tmp >&2 || true
    return 1
  fi
  return 0
}
export -f nrt_df_tmp
nrt_df_tmp

export SAAS_NRT_PODMAN=1
export SAAS_NRT_SCRATCH="${SCRATCH_ROOT}"
# Preserve login home before switching HOME to node-local scratch.
export SAAS_NRT_REAL_HOME="${SAAS_NRT_REAL_HOME:-${HOME}}"
export XDG_RUNTIME_DIR="${SCRATCH_ROOT}/run"
export HOME="${SCRATCH_ROOT}/home"
# A batch environment can carry XDG_CONFIG_HOME from the submit host.  Podman
# 3.4 consults that location before $HOME/.config, so merely changing HOME is
# not sufficient: it silently falls back to ~/.local/share/containers and can
# mix its graph metadata with a different store configuration.  Pin both XDG
# homes to this allocation before Podman is ever invoked.
export XDG_CONFIG_HOME="${HOME}/.config"
export XDG_DATA_HOME="${HOME}/.local/share"

export CONTAINERS_STORAGE_CONF="${SCRATCH_ROOT}/storage.conf"
cat > "${CONTAINERS_STORAGE_CONF}" <<EOF
[storage]
driver = "${SAAS_NRT_PODMAN_STORAGE_DRIVER}"
runroot = "${SAAS_NRT_PODMAN_RUNROOT}"
graphroot = "${SAAS_NRT_PODMAN_GRAPHROOT}"
EOF
if [[ "${SAAS_NRT_PODMAN_STORAGE_DRIVER}" == "overlay" ]]; then
cat >> "${CONTAINERS_STORAGE_CONF}" <<EOF
[storage.options.overlay]
mount_program = "/usr/bin/fuse-overlayfs"
EOF
fi

# Also write at the canonical XDG location so Podman 3.4 picks it up even if
# CONTAINERS_STORAGE_CONF is ignored.
mkdir -p "${HOME}/.config/containers"
cp -f "${CONTAINERS_STORAGE_CONF}" "${HOME}/.config/containers/storage.conf"

# Override cluster-wide /etc/containers/containers.conf.d defaults, which break
# app containers on NRT:
#   10-namespaces.conf: pidns/ipcns/netns = "host"  → s6-overlay images die with
#       "s6-overlay-suexec: fatal: can only run as pid 1" (/init isn't PID 1).
#   20-env.conf: env_host = true  → host env leaks into apps.
#   30-mounts.conf: binds host /etc/passwd,/etc/group read-only → linuxserver.io
#       images lose their `abc` user ("usermod: user 'abc' does not exist").
mkdir -p "${HOME}/.config/cni/net.d" "${HOME}/.config/containers"

# Some compose stacks (pretix, ...) reference base images from Docker Hub that
# are NOT shipped in the mw-*.tar bundles (e.g. postgres:13-alpine, redis).
# NRT nodes reach Docker Hub through the cluster pull-through cache proxy
# (/etc/containers/containers.conf.d/50-proxy.conf). Allow unqualified names to
# resolve to docker.io so `postgres:13-alpine` works.
cat > "${HOME}/.config/containers/registries.conf" <<EOF
unqualified-search-registries = ["docker.io"]
short-name-mode = "permissive"
EOF

cat > "${HOME}/.config/containers/containers.conf" <<EOF
[containers]
pidns = "private"
ipcns = "private"
env_host = false
volumes = []

[engine]
cgroup_manager = "cgroupfs"
runtime = "/usr/bin/crun"

[network]
default_subnet = "10.88.0.0/16"
EOF

# docker-compose v2 standalone (cached on lustre when possible)
COMPOSE_VERSION="${SAAS_NRT_COMPOSE_VERSION:-v2.29.7}"
COMPOSE_CACHE="${SAAS_NRT_COMPOSE_CACHE:-${REPO_ROOT}/scripts/nrt/.cache}"
mkdir -p "${COMPOSE_CACHE}"
COMPOSE_BIN="${COMPOSE_CACHE}/docker-compose-${COMPOSE_VERSION}"
if [[ ! -x "${COMPOSE_BIN}" ]]; then
  echo "[nrt] downloading docker-compose ${COMPOSE_VERSION} ..."
  curl -fsSL -o "${COMPOSE_BIN}.partial" \
    "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64"
  chmod +x "${COMPOSE_BIN}.partial"
  mv "${COMPOSE_BIN}.partial" "${COMPOSE_BIN}"
fi
ln -sfn "${COMPOSE_BIN}" "${SCRATCH_ROOT}/bin/docker-compose"

# Podman 3.4 has been observed to ignore CONTAINERS_STORAGE_CONF in some
# batch environments.  Force every direct Podman invocation, including the
# service used by compose-v2, onto this allocation's graph/run roots.  This is
# deliberately a PATH shim instead of changing individual callers, so image
# loading, SlotManager, and diagnostics cannot diverge into different stores.
cat > "${SCRATCH_ROOT}/bin/podman" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec "${SAAS_NRT_PODMAN_BIN}" \\
  --root "${SAAS_NRT_PODMAN_GRAPHROOT}" \\
  --runroot "${SAAS_NRT_PODMAN_RUNROOT}" "\$@"
EOF
chmod +x "${SCRATCH_ROOT}/bin/podman"

# docker → podman shim; route `docker compose` to compose-v2
cat > "${SCRATCH_ROOT}/bin/docker" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "compose" ]]; then
  shift
  exec docker-compose "$@"
fi
exec podman "$@"
EOF
chmod +x "${SCRATCH_ROOT}/bin/docker"
export PATH="${SCRATCH_ROOT}/bin:${PATH}"

# Podman API socket so compose-v2 can talk to the engine
# Skip during image load (SAAS_NRT_SKIP_API=1) — concurrent `system service`
# has been observed to interrupt `podman load` mid-blob on NRT.
SOCK="${SCRATCH_ROOT}/podman.sock"
export DOCKER_HOST="unix://${SOCK}"
if [[ "${SAAS_NRT_SKIP_API:-}" == "1" ]]; then
  echo "[nrt] skipping podman system service (SAAS_NRT_SKIP_API=1)"
elif [[ -S "${SOCK}" ]] && [[ -n "${SAAS_NRT_PODMAN_SERVICE_PID:-}" ]] && kill -0 "${SAAS_NRT_PODMAN_SERVICE_PID}" 2>/dev/null; then
  :
elif [[ -S "${SOCK}" ]] && curl -s --unix-socket "${SOCK}" http://localhost/_ping >/dev/null 2>&1; then
  :
else
  rm -f "${SOCK}"
  podman system service --time=0 "unix://${SOCK}" \
    >"${SCRATCH_ROOT}/logs/podman-service.log" 2>&1 &
  export SAAS_NRT_PODMAN_SERVICE_PID=$!
  for _ in $(seq 1 50); do
    [[ -S "${SOCK}" ]] && break
    sleep 0.1
  done
fi

# SaaS-Bench temp / results under scratch by default (fast local disk)
export SAAS_BENCH_TMP="${SAAS_BENCH_TMP:-${SCRATCH_ROOT}/saas_tmp}"
mkdir -p "${SAAS_BENCH_TMP}"

# Force IPv4 localhost publishes in slot rewriter (see slot.py)
export SAAS_NRT_PUBLISH_BIND="${SAAS_NRT_PUBLISH_BIND:-127.0.0.1}"

echo "[nrt] scratch=${SCRATCH_ROOT}"
echo "[nrt] DOCKER_HOST=${DOCKER_HOST}"
echo "[nrt] $(docker --version 2>&1 | head -1)"
echo "[nrt] compose=$($(command -v docker-compose) version 2>&1 | head -1)"
nrt_df_tmp || true

# Best-effort cleanup of this job's /tmp scratch on exit (keeps node usable).
nrt_cleanup_scratch() {
  local rc=$?
  local scratch="${SAAS_NRT_SCRATCH:-}"
  if [[ "${SAAS_NRT_KEEP_SCRATCH:-}" == "1" ]]; then
    echo "[nrt] keeping scratch (SAAS_NRT_KEEP_SCRATCH=1): ${scratch}"
    return "${rc}"
  fi
  if [[ -n "${scratch}" && "${scratch}" == /tmp/s* ]]; then
    echo "[nrt] cleaning scratch ${scratch} ..."
    # Stop API / containers best-effort so unmounts succeed.
    [[ -n "${SAAS_NRT_PODMAN_SERVICE_PID:-}" ]] && kill "${SAAS_NRT_PODMAN_SERVICE_PID}" 2>/dev/null || true
    podman rm -af >/dev/null 2>&1 || true
    podman pod rm -af >/dev/null 2>&1 || true
    rm -rf "${scratch}" 2>/dev/null || true
    df -h /tmp 2>/dev/null | tail -1 || true
  fi
  return "${rc}"
}
trap nrt_cleanup_scratch EXIT
