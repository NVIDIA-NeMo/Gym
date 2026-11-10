#!/usr/bin/env bash
# Build the optional wheels that the VisGym fetch_* and counting tasks need.
# Gymnasium-Robotics comes from the supplied VisGym source checkout;
# lvis is downloaded at an explicitly pinned public release.
#
# Usage: scripts/build_vendor_wheels.sh /path/to/VisGym [output-dir]
set -euo pipefail

VISGYM_REPO_ROOT="${1:?usage: build_vendor_wheels.sh /path/to/VisGym [output-dir]}"
OUT_DIR="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/vendor_wheels}"

ROBOTICS_SRC="${VISGYM_REPO_ROOT}/Gymnasium-Robotics"
if [[ ! -d "${ROBOTICS_SRC}" ]]; then
  echo "No Gymnasium-Robotics source at ${ROBOTICS_SRC}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
python -m pip wheel --no-deps --wheel-dir "${OUT_DIR}" "${ROBOTICS_SRC}"
python -m pip wheel --no-deps --wheel-dir "${OUT_DIR}" "lvis==0.5.3"

echo "Wrote wheels to ${OUT_DIR}:"
ls -1 "${OUT_DIR}"
