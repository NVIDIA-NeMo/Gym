# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sourced by environment/setup.sh (seed, agent sandbox) and tests/test.sh (separate verifier). Defines
# nemo_gym_python DIR: prints the path of a Python 3.10+ for the server's vendored modules. It takes the image's own
# when it has one; otherwise (SWE-rebench-V2's Dart images ship none) it installs a pinned python-build-standalone
# release into DIR, checked against its sha256, and prints that. Prints nothing and returns 1 when neither works.
NEMO_GYM_PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20251014/cpython-3.12.12%2B20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
NEMO_GYM_PBS_SHA256="c74addcd1b033a6e4d60ead3ab47fcc995569027e01d3061c4a934f363c4a0cf"  # pragma: allowlist secret

nemo_gym_python() {
  local dir=$1 candidate path archive
  for candidate in python3 python /usr/bin/python3 /usr/local/bin/python3 /opt/conda/bin/python3; do
    path=$(command -v "$candidate" 2>/dev/null) || continue
    if "$path" -I -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
      echo "$path"
      return 0
    fi
  done
  if [ "$(uname -m)" != x86_64 ]; then
    echo "[nemo_gym] no Python 3.10+ and no pinned standalone Python for $(uname -m)" >&2
    return 1
  fi
  mkdir -p "$dir" || return 1
  archive="$dir/python.tar.gz"
  # The adapter budgets 600 s for this download (PYTHON_DOWNLOAD_BUDGET_SEC, inside BUILD_TIMEOUT_SEC and
  # VERIFIER_TIMEOUT_MARGIN_SEC). Worst case with curl: 3 attempts of at most 150 s each plus 2 retry delays of 10 s
  # = 470 s; wget is stopped after 470 s in total. Either leaves 130 s of the budget.
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 2 --retry-delay 10 --max-time 150 -o "$archive" "$NEMO_GYM_PBS_URL" \
      || { echo "[nemo_gym] download failed" >&2; return 1; }
  elif command -v wget >/dev/null 2>&1; then
    timeout 470 wget -q -T 60 -O "$archive" "$NEMO_GYM_PBS_URL" || { echo "[nemo_gym] download failed" >&2; return 1; }
  else
    echo "[nemo_gym] no Python 3.10+, and neither curl nor wget to fetch one" >&2
    return 1
  fi
  if ! echo "$NEMO_GYM_PBS_SHA256  $archive" | sha256sum -c --status; then
    echo "[nemo_gym] standalone Python archive failed its sha256 check" >&2
    return 1
  fi
  tar -xzf "$archive" -C "$dir" && rm -f "$archive" || return 1
  "$dir/python/bin/python3" -I -c 'import sys; sys.exit(sys.version_info < (3, 10))' || return 1
  echo "$dir/python/bin/python3"
}
