#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# PID 1 for the ONLYOFFICE community server under rootless Enroot/Podman.
#
# run-community-server.sh enables ~20 systemd units, stops nginx, and then execs
# /lib/systemd/systemd to bring everything up. systemd needs a writable cgroup
# hierarchy and privileges we do not have here, so it exits and nginx is left
# stopped. Run the image's own setup, then start the units this stack needs by
# hand and stay alive as PID 1.
set -uo pipefail

configure_community_redirects() {
  local config="${1:-/etc/nginx/sites-enabled/onlyoffice}"
  [[ -f "$config" ]] || return 1
  grep -q 'absolute_redirect off;' "$config" && return 0
  # The image's permanent case-normalization rewrites otherwise synthesize
  # http://localhost/... using internal port 80 and discard the published port.
  sed -i '/^[[:space:]]*server[[:space:]]*{/a\    absolute_redirect off;' "$config"
  grep -q 'absolute_redirect off;' "$config"
}

configure_community_hostname() {
  local host="${1%%:*}" config="${2:-/etc/nginx/sites-enabled/onlyoffice}"
  [[ "$host" =~ ^[A-Za-z0-9._-]+$ && -f "$config" ]] || return 2
  if grep -Eq '^[[:space:]]*server_name[[:space:]]' "$config"; then
    sed -i -E "s/^[[:space:]]*server_name[[:space:]]+[^;]*;/    server_name ${host};/" "$config"
  else
    sed -i "/^[[:space:]]*server[[:space:]]*{/a\\    server_name ${host};" "$config"
  fi
}

configure_community_public_port() {
  local port="$1" config="${2:-/etc/nginx/sites-enabled/onlyoffice}"
  [[ "$port" =~ ^[0-9]+$ ]] && (( port > 1024 && port < 65536 )) || return 2
  [[ -f "$config" ]] || return 1
  grep -Eq "listen[[:space:]]+127[.]0[.]0[.]1:${port}[[:space:]]*;" "$config" && return 0
  # Retain port 80 for the pod's host forwarder and add its public URL inside
  # the shared network namespace for backend callbacks and editor downloads.
  local temporary
  temporary="$(mktemp)" || return 1
  awk -v port="$port" '
    { print }
    !added && /^[[:space:]]*listen[[:space:]]+(0[.]0[.]0[.]0:)?80([[:space:];]|$)/ {
      print "    listen 127.0.0.1:" port ";"; added=1
    }
    END { if (!added) exit 1 }
  ' "$config" > "$temporary" || { rm -f "$temporary"; return 1; }
  cat "$temporary" > "$config"
  local result=$?
  rm -f "$temporary"
  return "$result"
}
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi

log() { echo "[oo-init] $*"; }
rm -f /run/saas-onlyoffice-ready

# Enroot needs its uid shim; Podman provides real container users via subuids.
as_onlyoffice() {
  if [[ -x /saas-bin/sudo ]]; then
    exec /saas-bin/sudo -u onlyoffice "$@"
  else
    exec /usr/sbin/runuser -u onlyoffice -- "$@"
  fi
}
# Compose normally gates this service on its dependencies. Enroot launches
# the services concurrently; running the one-shot image setup too early can
# leave permanent database-auth/configuration errors in the Mono application.
wait_for() {
  local dependency="$1"
  shift
  local deadline=$((SECONDS + ${SAAS_OO_DEPENDENCY_TIMEOUT:-180}))
  until "$@" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      log "ERROR: $dependency did not become ready before community setup"
      exit 1
    fi
    sleep 1
  done
  log "$dependency ready"
}
mysql_ready() {
  MYSQL_PWD="${MYSQL_SERVER_PASS:?}" mysql --protocol=TCP --connect-timeout=2 \
    --host="${MYSQL_SERVER_HOST:?}" --port="${MYSQL_SERVER_PORT:-3306}" \
    --user="${MYSQL_SERVER_USER:?}" --database="${MYSQL_SERVER_DB_NAME:-onlyoffice}" \
    --execute='SELECT 1'
}
elasticsearch_ready() {
  curl -fsS --max-time 3 "http://${ELASTICSEARCH_SERVER_HOST:?}:${ELASTICSEARCH_SERVER_HTTPPORT:-9200}/_cluster/health" \
    | grep -Eq '"status"[[:space:]]*:[[:space:]]*"(green|yellow)"'
}
documentserver_ready() {
  curl -fsS --max-time 3 "${DOCUMENT_SERVER_PROTOCOL:-http}://${DOCUMENT_SERVER_HOST:?}/healthcheck" \
    | grep -Eq '^true[[:space:]]*$'
}
wait_for MySQL mysql_ready
wait_for Elasticsearch elasticsearch_ready
wait_for DocumentServer documentserver_ready

log "running image setup"
# The script detects "no container" via /proc/1/cgroup and returns instead of
# exec'ing systemd, which is what we want.
/app/run-community-server.sh
log "image setup finished (rc=$?)"
configure_community_redirects || exit 1
configure_community_hostname "${SERVER_HOST:-localhost}" || exit 1

# Rootless Podman's named-volume copy-up leaves some seeded file subdirectories
# owned by root. The Mono workers run as onlyoffice and need write access there.
# Enroot has its own ownership shim and does not support real chown to subuids.
if [[ ! -x /saas-bin/sudo ]]; then
  chown -R onlyoffice:onlyoffice /var/www/onlyoffice/Data /var/log/onlyoffice || exit 1
  if [[ -n "${SAAS_OO_PUBLIC_PORT:-}" ]]; then
    configure_community_public_port "$SAAS_OO_PUBLIC_PORT" || exit 1
  fi
fi

# monoserve Wants redis-server.service; both are plain SysV scripts here.
for svc in redis-server nginx; do
  log "starting $svc"
  service "$svc" start || log "WARNING: $svc failed to start"
done

# ExecStartPre of monoserve/monoserveApiSystem.
install -m 755 -d /var/run/onlyoffice 2>/dev/null || mkdir -p /var/run/onlyoffice
rm -f /var/run/onlyoffice/onlyoffice.socket \
      /var/run/onlyoffice/onlyofficeApiSystem.socket \
      /var/www/onlyoffice/WebStudio/mono_crash* \
      /var/www/onlyoffice/WebStudio/core.* \
      /var/www/onlyoffice/ApiSystem/mono_crash* \
      /var/www/onlyoffice/ApiSystem/core.* 2>/dev/null

# From monoserve.service / monoserveApiSystem.service Environment=.
export LD_LIBRARY_PATH=/usr/lib
export MONO_IOMAP=all
export MONO_ASPNET_WEBCONFIG_CACHESIZE=2000
export MONO_THREADS_PER_CPU=2000
export MONO_GC_PARAMS=nursery-size=128m,soft-heap-limit=512m,bridge-implementation=tarjan
export MONO_LOG_LEVEL=error
export MONO_THREADS_SUSPEND=preemptive

# The units run as User=onlyoffice. /saas-bin/sudo drops to the fake uid the
# LD_PRELOAD shim reports, so mono's own privilege checks are satisfied.
log "starting monoserve"
MONO_PATH=/var/www/onlyoffice/WebStudio/bin/ \
  as_onlyoffice /usr/bin/hyperfastcgi4 \
  /config=/etc/hyperfastcgi/onlyoffice \
  /logfile=/var/log/onlyoffice/onlyoffice.log /loglevels=Error &

log "starting monoserveApiSystem"
MONO_PATH=/var/www/onlyoffice/ApiSystem/bin/ \
  as_onlyoffice /usr/bin/hyperfastcgi4 \
  /config=/etc/hyperfastcgi/onlyofficeApiSystem \
  /logfile=/var/log/onlyoffice/onlyofficeApiSystem.log /loglevels=Error &

export NODE_OPTIONS=--dns-result-order=ipv4first
log "starting SocketIO"
(cd /var/www/onlyoffice/Services/ASC.Socket.IO/ \
  && rm -f /tmp/onlyofficeSocketIO \
  && as_onlyoffice /usr/bin/node \
       /var/www/onlyoffice/Services/ASC.Socket.IO/app.js UNIX.SERVER) &

log "starting UrlShortener"
(cd /var/www/onlyoffice/Services/ASC.UrlShortener/ \
  && rm -f /tmp/onlyofficeUrlShortener \
  && as_onlyoffice /usr/bin/node \
       /var/www/onlyoffice/Services/ASC.UrlShortener/index.js UNIX.SERVER) &

log "services launched; holding as pid 1"
touch /run/saas-onlyoffice-ready
while :; do
  sleep 3600 &
  wait $!
done
