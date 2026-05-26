#!/bin/bash
# =============================================================================
# bringup_local_host.sh — one-shot host prep for the OSWorld `docker` provider
# -----------------------------------------------------------------------------
# Mode A: controller (`ng_run`) and the docker VM host live on the SAME box.
# This script handles the bits that aren't fixable in the repo:
#
#   1. apt install: docker.io + (recording stack) ffmpeg + xvfb + tigervnc-viewer
#                   + git/curl/unzip (uv pip install osworld @ git+... needs them)
#   2. enable docker daemon + add $USER to docker group
#   3. install uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
#   4. symlink uv into /usr/local/bin so ng_run's non-interactive `bash -c`
#      subshells (which don't source ~/.bashrc, so they don't get
#      $HOME/.local/bin on PATH) can still find uv. Without this, ng_run dies
#      with `(osworld_simple_agent) /bin/bash: line 1: uv: command not found`.
#   5. pre-flight verify: docker daemon up, /dev/kvm present, uv on PATH
#
# Idempotent: re-runs are safe. apt is a no-op for installed packages,
# the docker group add and the uv installer are both idempotent.
#
# Usage:
#   bash bringup_local_host.sh
#
# After this completes, follow the agent README's Quickstart:
#   git clone https://github.com/.../Gym.git -b feature/osworld
#   cd Gym && uv venv && uv sync --extra dev
#   # then prestage Ubuntu.qcow2 (see README "First-Run Cache Acquisition" —
#   # required when concurrency > 1) and run ng_run + ng_collect_rollouts.
# =============================================================================

set -euo pipefail

ARCH="$(uname -m)"
if [ "$ARCH" != "x86_64" ]; then
    echo "ERROR: $ARCH not supported. happysixd/osworld-docker is x86_64-only." >&2
    exit 1
fi

echo "=== bringup_local_host.sh ==="
echo

# -----------------------------------------------------------------------------
# 1. apt install
# -----------------------------------------------------------------------------
echo "--- [1/5] apt install: docker + helpers + recording stack ---"
need_pkgs=()
command -v docker          >/dev/null || need_pkgs+=(docker.io)
command -v git             >/dev/null || need_pkgs+=(git)
command -v curl            >/dev/null || need_pkgs+=(curl)
command -v unzip           >/dev/null || need_pkgs+=(unzip)
command -v ffmpeg          >/dev/null || need_pkgs+=(ffmpeg)
command -v Xvfb            >/dev/null || need_pkgs+=(xvfb)
command -v xtigervncviewer >/dev/null || need_pkgs+=(tigervnc-viewer)
if [ ${#need_pkgs[@]} -gt 0 ]; then
    echo "installing: ${need_pkgs[*]}"
    sudo apt-get update -y
    sudo apt-get install -y "${need_pkgs[@]}"
else
    echo "all apt packages already present"
fi
echo

# -----------------------------------------------------------------------------
# 2. docker daemon + group
# -----------------------------------------------------------------------------
echo "--- [2/5] docker daemon + group ---"
sudo systemctl enable --now docker
if ! groups | grep -qw docker; then
    echo "adding $USER to docker group (re-login or 'newgrp docker' for it to take effect)"
    sudo usermod -aG docker "$USER"
fi
docker --version
if docker info >/dev/null 2>&1; then
    echo "  ✓ docker daemon responding"
else
    echo "  ✗ docker daemon not responding. Re-login (so docker group takes effect) and re-run." >&2
    exit 1
fi
echo

# -----------------------------------------------------------------------------
# 3. uv
# -----------------------------------------------------------------------------
echo "--- [3/5] uv ---"
# Always export — installer drops uv at $HOME/.local/bin but non-interactive
# ssh sessions don't source .bashrc, so re-runs without this would miss uv.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo "uv: $(uv --version)"
echo

# -----------------------------------------------------------------------------
# 4. uv symlink to /usr/local/bin (ng_run subshell PATH workaround)
# -----------------------------------------------------------------------------
echo "--- [4/5] symlink uv into /usr/local/bin ---"
# Why: ng_run launches each FastAPI server (`osworld_simple_agent` /
# `policy_model`) via `bash -c "uv run ..."`. Non-interactive bash doesn't
# source ~/.bashrc, so the installer's PATH edit to add $HOME/.local/bin
# isn't visible. /usr/local/bin is on the default non-interactive PATH on
# every standard Linux distro, so a symlink there is the simplest fix.
for bin in uv uvx; do
    if [ -x "$HOME/.local/bin/$bin" ] && [ ! -e "/usr/local/bin/$bin" ]; then
        sudo ln -sf "$HOME/.local/bin/$bin" "/usr/local/bin/$bin"
        echo "  ln -sf $HOME/.local/bin/$bin → /usr/local/bin/$bin"
    fi
done
# Verify the symlink works from a non-interactive shell context
/bin/bash -c "command -v uv >/dev/null && uv --version" \
    && echo "  ✓ uv visible to non-interactive bash" \
    || { echo "  ✗ uv not on system PATH for /bin/bash -c"; exit 1; }
echo

# -----------------------------------------------------------------------------
# 5. pre-flight verify
# -----------------------------------------------------------------------------
echo "--- [5/5] pre-flight verify ---"
ok=0; fail=0
chk() { printf '  %-22s %s\n' "$1:" "$2"; if [ "$3" = ok ]; then ok=$((ok+1)); else fail=$((fail+1)); fi; }
command -v docker          >/dev/null && chk "docker"        "$(docker --version)" ok || chk "docker"        "MISSING" fail
docker info >/dev/null 2>&1 &&                                chk "docker daemon"  "OK"                    ok || chk "docker daemon"  "DOWN"    fail
[ -c /dev/kvm ] &&                                            chk "kvm"            "PRESENT"               ok || chk "kvm"            "MISSING (TCG fallback, ~10× slower)" ok
command -v uv              >/dev/null && chk "uv"            "$(uv --version)"     ok || chk "uv"            "MISSING" fail
command -v ffmpeg          >/dev/null && chk "ffmpeg"        "OK"                  ok || chk "ffmpeg"        "MISSING (recording disabled)" ok
command -v Xvfb            >/dev/null && chk "Xvfb"          "OK"                  ok || chk "Xvfb"          "MISSING (recording disabled)" ok
command -v xtigervncviewer >/dev/null && chk "tigervncviewer" "OK"                 ok || chk "tigervncviewer" "MISSING (recording disabled)" ok
echo
echo "  $ok OK, $fail FAIL"

if [ $fail -gt 0 ]; then
    echo
    echo "Setup incomplete. See above + fix what's red, then re-run this script."
    exit 1
fi

cat <<EOF

✓ Host is ready for OSWorld Mode A. Next:

    git clone https://github.com/<your-fork-or-NeMo-Gym>.git
    cd <Gym-checkout>
    uv venv && uv sync --extra dev

    # fill in env.yaml at project root (see README's "Quickstart")

    # MANDATORY when concurrency > 1: prestage Ubuntu.qcow2 (~12 GB)
    # so concurrent rollouts don't race to download it. See README
    # "First-Run Cache Acquisition" for the recipe.

    # run servers + rollouts:
    ng_run "+config_paths=[\\
      responses_api_agents/osworld_agent/configs/osworld_agent.yaml,\\
      responses_api_models/openai_model/configs/openai_model.yaml]" &
    ng_collect_rollouts \\
      +agent_name=osworld_simple_agent \\
      +input_jsonl_fpath=responses_api_agents/osworld_agent/data/example.jsonl \\
      +output_jsonl_fpath=results/osworld_example.jsonl \\
      +num_repeats=1

EOF
