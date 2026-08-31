#!/usr/bin/env bash
# Build a wheel of the pinned VisGym revision into vendor_wheels/.
#
# Why a wheel instead of the git URL in requirements.txt: VisGym inherits
# Gymnasium's pyproject.toml, which declares both `classic-control` and
# `classic_control` (likewise `mujoco-py`/`mujoco_py`, `toy-text`/`toy_text`).
# PEP 685 normalizes those to one name, so uv refuses to parse the project:
#
#   TOML parse error ... duplicate normalized extra name `classic-control`
#
# NeMo-Gym builds every resource-server venv with `uv pip install`, so the
# source install fails there while pip -- which still tolerates the duplicates
# -- installs it fine. Building the wheel with pip once sidesteps the parse:
# wheel metadata is already normalized, and uv installs the result happily.
# Delete this script once VisGym drops the duplicate extras upstream and
# requirements.txt can name the git revision directly.
set -euo pipefail

VISGYM_REV="${VISGYM_REV:-927271d107ad0196ad6aa597095ca57d01c6ddbb}"
VISGYM_URL="${VISGYM_URL:-https://github.com/visgym/VIsGym.git}"
VISGYM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${VISGYM_ROOT}/vendor_wheels}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# A local checkout is used when given; otherwise the pinned revision is cloned
# into a temporary directory.
SRC_DIR="${VISGYM_REPO_ROOT:-}"
CLEANUP_DIR=""
if [[ -z "${SRC_DIR}" ]]; then
  CLEANUP_DIR="$(mktemp -d)"
  trap 'rm -rf "${CLEANUP_DIR}"' EXIT
  SRC_DIR="${CLEANUP_DIR}/VIsGym"
  git clone --quiet "${VISGYM_URL}" "${SRC_DIR}"
  git -C "${SRC_DIR}" checkout --quiet "${VISGYM_REV}"
fi

if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
  echo "pip is required to build the VisGym wheel (uv cannot parse its pyproject)." >&2
  echo "Point PYTHON_BIN at an interpreter that has pip, e.g. PYTHON_BIN=/usr/bin/python3." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
"${PYTHON_BIN}" -m pip wheel --no-deps --wheel-dir "${OUT_DIR}" "${SRC_DIR}"

echo "Wrote VisGym wheel to ${OUT_DIR}:"
ls -1 "${OUT_DIR}"/gymnasium-*.whl
