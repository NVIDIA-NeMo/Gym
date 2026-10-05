#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PORTABLE_PYTHON_SH="$SCRIPT_DIR/_portable_python.sh"
bash "$NEMO_GYM_ROOT/responses_api_agents/terminus_2_agent/scripts/terminus_2_agent_deps.sh"

# Terminus-2 drives a tmux session, and most task images have no tmux. A static build in the
# runtime saves harbor's per-rollout apt-get install.
TMUX_VERSION="${TMUX_VERSION:-3.5a}"
if [ "${BUILD_STATIC_TMUX:-1}" = 1 ] && [ ! -x "$DEPS_DIR/bin/tmux" ]; then
    work=$(mktemp -d)
    curl -fsSL "https://github.com/tmux/tmux/releases/download/${TMUX_VERSION}/tmux-${TMUX_VERSION}.tar.gz" | tar xz -C "$work"
    (cd "$work/tmux-${TMUX_VERSION}" && ./configure --enable-static --prefix="$work/out" >/dev/null && make -j"$(nproc)" >/dev/null && make install >/dev/null)
    cp "$work/out/bin/tmux" "$DEPS_DIR/bin/tmux"
    rm -rf "$work"
fi
"$DEPS_DIR/bin/tmux" -V
