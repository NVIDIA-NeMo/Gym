#!/usr/bin/env bash
set -euo pipefail

dockerd >/var/log/dockerd.log 2>&1 &
exec "$@"
