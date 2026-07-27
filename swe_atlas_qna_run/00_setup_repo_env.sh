#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

mkdir -p "${UV_CACHE_DIR}" "${TMPDIR}"
export UV_CACHE_DIR TMPDIR

cd "${GYM_DIR}"
echo "== Repo =="
git branch --show-current
git status --short

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

echo "== Python environment =="
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra dev --extra sandbox
uv pip install "mini-swe-agent==2.1.0"

echo "Done. Activate with: source ${GYM_DIR}/.venv/bin/activate"
