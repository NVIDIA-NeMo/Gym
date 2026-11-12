#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/saas_bench/oo_documentserver_init.sh
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Community Server owns :80 in a shared Podman namespace. Move Docs off it
# before the image entrypoint copies/symlinks its nginx templates and starts nginx.
set -euo pipefail

configure_documentserver_port() {
  local port="$1" config_dir="$2" conf
  if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 || port == 80 )); then
    echo "Invalid shared-namespace Document Server port: $port" >&2
    return 2
  fi
  for conf in "$config_dir"/ds.conf "$config_dir"/ds.conf.tmpl; do
    [[ -f "$conf" ]] || continue
    sed -i -E "s/(listen[[:space:]]+)(0\\.0\\.0\\.0:|\\[::\\]:)?80([[:space:];])/\\1\\2${port}\\3/g" "$conf"
  done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  configure_documentserver_port "${SAAS_OO_DS_PORT:-8888}" /etc/onlyoffice/documentserver/nginx
  exec /app/ds/run-document-server.sh "$@"
fi
