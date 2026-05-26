#!/bin/bash
# =============================================================================
# bringup_remote_host.sh — one-shot setup for a remote docker VM host
# -----------------------------------------------------------------------------
# Bundles 5 steps for setting up a remote machine that will host the OSWorld
# qemu VM container (used by the `remote_docker` provider — see the agent's
# README, "Deployment Modes / Mode B"):
#   1. SSH passwordless auth check (or ssh-copy-id if not yet set up)
#   2. install docker.io + ffmpeg + xvfb + tigervnc-viewer
#   3. (optional) rsync qcow2 to ~/osworld-assets/
#   4. docker pull happysixd/osworld-docker:latest
#   5. pre-flight verify (/dev/kvm, image, tools, qcow2)
#
# Run once per new remote host. Idempotent — safe to re-run; apt-get install
# is a no-op for already-present packages, docker pull resumes, rsync skips
# byte-identical files.
#
# Usage:
#   bash bringup_remote_host.sh user@host [path/to/Ubuntu.qcow2]
#
# If the qcow2 path is omitted, step 3 is skipped (assumes you stage it
# yourself, e.g. by downloading on the remote from HuggingFace).
#
# After this script finishes successfully:
#   export OSWORLD_REMOTE_HOST=user@host
# and the controller (your job runner / local shell) can use
# provider_name=remote_docker.
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 user@host [path/to/Ubuntu.qcow2]" >&2
    exit 1
fi

HOST="$1"
QCOW="${2:-}"
SSH_KEY="${OSWORLD_REMOTE_SSH_KEY:-}"

ssh_opts=( -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new )
if [ -n "$SSH_KEY" ]; then
    ssh_opts+=( -i "$SSH_KEY" )
fi

ssh_remote() { ssh "${ssh_opts[@]}" "$HOST" "$@"; }

echo "=== 1. SSH auth check (host=$HOST) ==="
if ssh_remote "echo OK" >/dev/null 2>&1; then
    echo "passwordless SSH OK"
else
    echo "passwordless SSH not working — running ssh-copy-id (you'll be prompted once)"
    if [ -n "$SSH_KEY" ]; then
        ssh-copy-id -i "${SSH_KEY}.pub" "$HOST"
    else
        ssh-copy-id "$HOST"
    fi
    ssh_remote "echo OK" >/dev/null 2>&1 || {
        echo "ERROR: ssh still failing after ssh-copy-id; abort" >&2
        exit 1
    }
    echo "SSH OK after copy-id"
fi
echo

echo "=== 2. install docker.io + ffmpeg + xvfb + tigervnc-viewer ==="
ssh_remote 'bash -s' <<'REMOTE_BOOTSTRAP'
set -e
need_install=()
command -v docker          >/dev/null || need_install+=(docker.io)
command -v ffmpeg          >/dev/null || need_install+=(ffmpeg)
command -v Xvfb            >/dev/null || need_install+=(xvfb)
command -v xtigervncviewer >/dev/null || need_install+=(tigervnc-viewer)
if [ ${#need_install[@]} -gt 0 ]; then
    echo "installing: ${need_install[*]}"
    sudo apt-get update -y
    sudo apt-get install -y "${need_install[@]}"
else
    echo "all packages already installed"
fi
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER" || true   # already-member is non-zero on some distros
echo "docker  : $(docker --version 2>&1 | head -1)"
echo "ffmpeg  : $(ffmpeg -version 2>&1 | head -1)"
echo "Xvfb    : $(Xvfb -help 2>&1 | head -1 || echo present)"
echo "tigervnc: $(xtigervncviewer --version 2>&1 | head -1 || echo present)"
REMOTE_BOOTSTRAP
echo

echo "=== 3. stage qcow2 ==="
if [ -n "$QCOW" ]; then
    if [ ! -f "$QCOW" ]; then
        echo "ERROR: qcow2 source file not found: $QCOW" >&2
        exit 1
    fi
    ssh_remote "mkdir -p ~/osworld-assets"
    sz_h=$(du -h "$QCOW" | awk '{print $1}')
    echo "rsync $QCOW ($sz_h) → $HOST:~/osworld-assets/Ubuntu.qcow2"
    rsync -avP -e "ssh ${ssh_opts[*]}" "$QCOW" "$HOST:~/osworld-assets/Ubuntu.qcow2"
else
    echo "(skipping — no qcow2 path provided)"
    echo
    echo "Stage manually later, e.g.:"
    echo "  # option A: rsync from local"
    echo "  rsync -avP /path/to/Ubuntu.qcow2 $HOST:~/osworld-assets/"
    echo "  # option B: download on the remote"
    echo "  ssh $HOST 'mkdir -p ~/osworld-assets && \\"
    echo "    curl -fL -o ~/osworld-assets/Ubuntu.qcow2.zip \\"
    echo "      https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip && \\"
    echo "    cd ~/osworld-assets && unzip Ubuntu.qcow2.zip && rm Ubuntu.qcow2.zip'"
fi
echo

echo "=== 4. docker pull happysixd/osworld-docker:latest ==="
ssh_remote "docker pull happysixd/osworld-docker:latest"
echo

echo "=== 5. pre-flight verify ==="
ssh_remote 'bash -s' <<'REMOTE_VERIFY'
ok=0
fail=0
report() { printf '  %-12s %s\n' "$1" "$2"; if [ "$3" = ok ]; then ok=$((ok+1)); else fail=$((fail+1)); fi; }
test -c /dev/kvm                                              && report "KVM:"      "OK (/dev/kvm present)"   ok || report "KVM:"      "MISSING — set OSWORLD_REMOTE_KVM=N to fall back to TCG (slow)" fail
docker image inspect happysixd/osworld-docker:latest >/dev/null 2>&1 && report "docker img:" "OK" ok || report "docker img:" "MISSING — re-run step 4" fail
command -v ffmpeg          >/dev/null && report "ffmpeg:"    "OK"          ok || report "ffmpeg:"   "MISSING (recording disabled)" fail
command -v Xvfb            >/dev/null && report "Xvfb:"      "OK"          ok || report "Xvfb:"     "MISSING (recording disabled)" fail
command -v xtigervncviewer >/dev/null && report "tigervnc:"  "OK"          ok || report "tigervnc:" "MISSING (recording disabled)" fail
test -f ~/osworld-assets/Ubuntu.qcow2  && report "qcow2:"    "OK"          ok || report "qcow2:"    "MISSING (stage step 3 first)" fail
echo
echo "$ok OK, $fail fail"
[ $fail -eq 0 ]
REMOTE_VERIFY

echo
echo "✓ Done. Next:"
echo "    export OSWORLD_REMOTE_HOST=$HOST"
echo "    # then run your sbatch / local controller with provider_name=remote_docker"
