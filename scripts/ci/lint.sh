#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ci_dir
repo_root="$(cd "${ci_dir}/../.." && pwd)"
readonly repo_root
# pre-commit is a dev-extra dependency. Resolve the pinned version and the
# lockfile-pinned wheel (URL + SHA-256) from uv.lock (the committed locked
# resolution) so the offline install enforces the recorded artifact hash.
pre_commit_version="$(awk '/^name = "pre-commit"$/{f=1} f && /^version = /{gsub(/"/, "", $3); print $3; exit}' "${repo_root}/uv.lock")"
pre_commit_wheel="$(awk '/^name = "pre-commit"$/{f=1} f && /py2.py3-none-any\.whl.*sha256:/{ if (match($0, /https:\/\/[^"]+\.whl/)) url=substr($0, RSTART, RLENGTH); if (match($0, /sha256:[0-9a-f]{64}/)) hash=substr($0, RSTART+7, 64); print url " " hash; exit }' "${repo_root}/uv.lock")"
read -r pre_commit_wheel_url pre_commit_wheel_sha256 <<<"${pre_commit_wheel}"
if [[ -z "${pre_commit_version}" || -z "${pre_commit_wheel_url}" || -z "${pre_commit_wheel_sha256}" ]]; then
    echo "Could not read the pinned pre-commit wheel (URL + SHA-256) from uv.lock" >&2
    exit 1
fi
readonly pre_commit_version
readonly tool_venv="${repo_root}/.cache/nemo-gym-ci/pre-commit-${pre_commit_version}"

# shellcheck source=scripts/ci/sanitize_env.sh
source "${ci_dir}/sanitize_env.sh"
gym_ci_sanitize_environment lint
unset -f gym_ci_sanitize_environment

cd "${repo_root}"
if command -v pre-commit >/dev/null 2>&1; then
    # Offline / container: the dev environment baked into the image (synced with
    # the dev extra) already provides pre-commit on PATH. No install needed.
    pre-commit install
    exec pre-commit run --all-files --show-diff-on-failure --color=always
fi
# Online runner: install pre-commit from the lockfile-pinned wheel, verifying the
# recorded SHA-256 before install (no bare pip version-coordinate install, which
# would not enforce the lockfile's artifact hash).
python -m venv "${tool_venv}"
wheel_file="${tool_venv}/pre_commit-${pre_commit_version}-py2.py3-none-any.whl"
curl -fLSs --retry 5 --retry-all-errors --retry-max-time 300 \
    --connect-timeout 30 --max-time 120 \
    -o "${wheel_file}" "${pre_commit_wheel_url}"
echo "${pre_commit_wheel_sha256}  ${wheel_file}" | sha256sum -c -
"${tool_venv}/bin/python" -m pip install --disable-pip-version-check "${wheel_file}"
rm -f "${wheel_file}"
"${tool_venv}/bin/pre-commit" install
exec "${tool_venv}/bin/pre-commit" run --all-files --show-diff-on-failure --color=always
