#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provider-free fingerprint cutover followed by one guarded rejudge-controller launch.
#SBATCH --job-name=gdpval-e2e-rejudge-bootstrap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append

set -euo pipefail
umask 077

: "${RUN_DIR:?set RUN_DIR}"
: "${ACTIVE_PACKAGE:?set ACTIVE_PACKAGE}"
: "${CORRECTED_RUNTIME_OVERLAY:?set CORRECTED_RUNTIME_OVERLAY}"
: "${CORRECTED_TRANSPORT_VIEW_ROOT:?set CORRECTED_TRANSPORT_VIEW_ROOT}"
: "${JUDGE_DIR_SUFFIX:?set JUDGE_DIR_SUFFIX}"
[[ ${CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS:-false} == true ]] || {
    echo "REJUDGE_BOOTSTRAP_FAIL: provider calls were not explicitly authorized" >&2
    exit 64
}

REQUESTED_ACTIVE_PACKAGE=$ACTIVE_PACKAGE
REQUESTED_RUNTIME=$CORRECTED_RUNTIME_OVERLAY
REQUESTED_VIEWS=$CORRECTED_TRANSPORT_VIEW_ROOT
REQUESTED_SUFFIX=$JUDGE_DIR_SUFFIX

# shellcheck disable=SC1090
source "$RUN_DIR/settings.env"
ACTIVE_PACKAGE=$REQUESTED_ACTIVE_PACKAGE
CORRECTED_RUNTIME_OVERLAY=$REQUESTED_RUNTIME
CORRECTED_TRANSPORT_VIEW_ROOT=$REQUESTED_VIEWS
JUDGE_DIR_SUFFIX=$REQUESTED_SUFFIX

[[ $ACCOUNT == nemotron_n3_post ]] || {
    echo "REJUDGE_BOOTSTRAP_FAIL: campaign account is not nemotron_n3_post: $ACCOUNT" >&2
    exit 64
}
[[ -f $RUN_DIR/TRANSPORT_PREBUILD_PASS_$JUDGE_DIR_SUFFIX \
    && ! -L $RUN_DIR/TRANSPORT_PREBUILD_PASS_$JUDGE_DIR_SUFFIX ]] || {
    echo "REJUDGE_BOOTSTRAP_FAIL: corrected transport receipt is missing" >&2
    exit 64
}

ACTIVE_PACKAGE=$ACTIVE_PACKAGE \
CORRECTED_RUNTIME_OVERLAY=$CORRECTED_RUNTIME_OVERLAY \
CORRECTED_TRANSPORT_VIEW_ROOT=$CORRECTED_TRANSPORT_VIEW_ROOT \
JUDGE_DIR_SUFFIX=$JUDGE_DIR_SUFFIX \
    bash "$ACTIVE_PACKAGE/prepare_rejudge_fingerprint.sh" "$RUN_DIR"

fingerprint=$(python3 -c \
    'import json,sys; value=json.load(open(sys.argv[1]))["fingerprint"]; assert len(value)==64 and all(c in "0123456789abcdef" for c in value); print(value)' \
    "$RUN_DIR/fingerprint_$JUDGE_DIR_SUFFIX.json")

ACTIVE_PACKAGE=$ACTIVE_PACKAGE \
CORRECTED_RUNTIME_OVERLAY=$CORRECTED_RUNTIME_OVERLAY \
CORRECTED_TRANSPORT_VIEW_ROOT=$CORRECTED_TRANSPORT_VIEW_ROOT \
EXPECTED_JUDGE_FINGERPRINT=$fingerprint \
JUDGE_DIR_SUFFIX=$JUDGE_DIR_SUFFIX \
CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS=true \
    bash "$ACTIVE_PACKAGE/launch_rejudge.sh" "$RUN_DIR"

printf 'REJUDGE_BOOTSTRAP_PASS fingerprint=%s\n' "$fingerprint"
