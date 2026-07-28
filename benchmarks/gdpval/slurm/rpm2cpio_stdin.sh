#!/usr/bin/env bash
set -euo pipefail

# BusyBox rpm2cpio does not accept '-' for stdin, while Apptainer's
# unprivileged installer streams RPMs through that interface.

if [[ "${1:-}" != "-" ]]; then
  exec /usr/bin/busybox rpm2cpio "$@"
fi

temporary_rpm="$(mktemp "${TMPDIR:-/tmp}/rpm2cpio.XXXXXX.rpm")"
trap 'rm -f "${temporary_rpm}"' EXIT INT TERM
/bin/cat >"${temporary_rpm}"
/usr/bin/busybox rpm2cpio "${temporary_rpm}"
