#!/usr/bin/env bash
# Install codec-bearing packages that are excluded from the NeMo-Gym container.
#
# Run this before using VLM or audio/video benchmarks inside the container:
#
#   bash docker/install_codec_deps.sh
#
# Safe to call multiple times — exits immediately if already installed.
# Versions are read from uv.lock automatically — no manual pins to maintain.
# --no-config bypasses the project's sys_platform=='never' overrides.
set -euo pipefail

if python -c "import cv2, torchvision, torchaudio" 2>/dev/null; then
    echo "[codec-deps] Already installed, skipping."
    exit 0
fi

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$0")")"
LOCK_FILE="$REPO_ROOT/uv.lock"

# Read pinned versions from uv.lock — the [dependency-groups] codec group is
# the single source of truth so these stay in sync with uv lock automatically.
PKGS=$(python3 - "$LOCK_FILE" <<'EOF'
import re, sys
lock = open(sys.argv[1]).read()
for pkg in ["opencv-python-headless", "torchvision", "torchaudio"]:
    m = re.search(r'\nname = "' + pkg + r'"\nversion = "([^"]+)"', lock)
    if m:
        print(f"{pkg}=={m.group(1)}")
EOF
)

echo "[codec-deps] Installing codec-bearing packages..."
echo "$PKGS" | xargs uv pip install --no-config

echo "[codec-deps] Done."
