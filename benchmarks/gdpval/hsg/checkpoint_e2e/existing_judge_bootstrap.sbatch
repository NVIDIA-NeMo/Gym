#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Durable CPU bootstrap: prepare/copy an external tree, then launch its controller.
#SBATCH --job-name=gdpval-existing-bootstrap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append

set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
export PATH=/cm/local/apps/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

: "${CHECKPOINT:?set CHECKPOINT}"
: "${EXTERNAL_SOURCE:?set EXTERNAL_SOURCE}"
: "${ACTIVE_PACKAGE:?set ACTIVE_PACKAGE}"
: "${BOOTSTRAP_STATE_DIR:?set BOOTSTRAP_STATE_DIR}"
: "${AUTHORIZE_PROVIDER_CALLS:?set AUTHORIZE_PROVIDER_CALLS}"
: "${EXPECTED_IMPORT_ID:?set EXPECTED_IMPORT_ID}"
: "${EXPECTED_PACKAGE_SOURCE_SHA256:?set EXPECTED_PACKAGE_SOURCE_SHA256}"
: "${EXPECTED_RUN_ID:?set EXPECTED_RUN_ID}"
: "${EXPECTED_REFERENCE_OVERLAY_SHA256:?set EXPECTED_REFERENCE_OVERLAY_SHA256}"
: "${CHECKPOINT_E2E_OWNER_ROOT:?set CHECKPOINT_E2E_OWNER_ROOT}"
: "${CHECKPOINT_E2E_AAV2_ROOT:?set CHECKPOINT_E2E_AAV2_ROOT}"
: "${CHECKPOINT_E2E_DATASET:?set CHECKPOINT_E2E_DATASET}"
: "${CHECKPOINT_E2E_REFERENCE_OVERLAY:?set CHECKPOINT_E2E_REFERENCE_OVERLAY}"
: "${CHECKPOINT_E2E_ENV_FILE:?set CHECKPOINT_E2E_ENV_FILE}"
: "${CHECKPOINT_E2E_EXISTING_ROOT:?set CHECKPOINT_E2E_EXISTING_ROOT}"
: "${CHECKPOINT_E2E_MODEL_NAME:?set CHECKPOINT_E2E_MODEL_NAME}"
: "${GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS:?set GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS}"
: "${CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION:?set CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION}"

fail() { echo "EXISTING_BOOTSTRAP_FAIL: $*" >&2; exit 64; }
[[ $AUTHORIZE_PROVIDER_CALLS == true || $AUTHORIZE_PROVIDER_CALLS == false ]] \
    || fail "authorization must be true or false"
[[ $CHECKPOINT_E2E_MODEL_NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] \
    || fail "model name is unsafe"
[[ $GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS =~ ^[1-4]$ ]] \
    || fail "Gemini concurrency must be an integer from 1 through 4"
[[ $CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION == true \
    || $CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION == false ]] \
    || fail "persistent judge session must be true or false"
[[ $EXPECTED_IMPORT_ID =~ ^import-[0-9a-f]{24}$ \
    && $EXPECTED_PACKAGE_SOURCE_SHA256 =~ ^[0-9a-f]{64}$ \
    && $EXPECTED_RUN_ID =~ ^[A-Za-z0-9._-]+-[0-9a-f]{16}$ \
    && $EXPECTED_REFERENCE_OVERLAY_SHA256 =~ ^[0-9a-f]{64}$ ]] \
    || fail "bootstrap content identity is invalid"
for path in "$CHECKPOINT_E2E_REFERENCE_OVERLAY" "$CHECKPOINT_E2E_ENV_FILE"; do
    [[ $path == /* && $path != *,* && $path != *$'\n'* && $path != *$'\r'* \
        && -f $path && ! -L $path \
        && $(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$path") == "$path" ]] \
        || fail "bootstrap file path is unsafe: $path"
done
[[ $(sha256sum "$CHECKPOINT_E2E_REFERENCE_OVERLAY" | awk '{print $1}') \
    == "$EXPECTED_REFERENCE_OVERLAY_SHA256" ]] \
    || fail "reference overlay changed after bootstrap submission"
gym_override_count=0
[[ -z ${CHECKPOINT_E2E_GYM_ROOT:-} ]] || gym_override_count=$((gym_override_count + 1))
[[ -z ${CHECKPOINT_E2E_EXPECTED_GYM_REVISION:-} ]] || gym_override_count=$((gym_override_count + 1))
(( gym_override_count == 0 || gym_override_count == 2 )) \
    || fail "Gym root and expected revision overrides must be supplied together"
[[ -d $ACTIVE_PACKAGE && ! -L $ACTIVE_PACKAGE \
    && -x $ACTIVE_PACKAGE/run_existing_judge.sh ]] || fail "active package is invalid"
[[ -d $BOOTSTRAP_STATE_DIR && ! -L $BOOTSTRAP_STATE_DIR ]] \
    || fail "bootstrap state directory is invalid"

output=$BOOTSTRAP_STATE_DIR/prepare.${SLURM_JOB_ID:-manual}.out
CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS="$AUTHORIZE_PROVIDER_CALLS" \
CHECKPOINT_E2E_OWNER_ROOT="$CHECKPOINT_E2E_OWNER_ROOT" \
CHECKPOINT_E2E_AAV2_ROOT="$CHECKPOINT_E2E_AAV2_ROOT" \
CHECKPOINT_E2E_DATASET="$CHECKPOINT_E2E_DATASET" \
CHECKPOINT_E2E_REFERENCE_OVERLAY="$CHECKPOINT_E2E_REFERENCE_OVERLAY" \
CHECKPOINT_E2E_ENV_FILE="$CHECKPOINT_E2E_ENV_FILE" \
CHECKPOINT_E2E_EXISTING_ROOT="$CHECKPOINT_E2E_EXISTING_ROOT" \
CHECKPOINT_E2E_MODEL_NAME="$CHECKPOINT_E2E_MODEL_NAME" \
GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS="$GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS" \
CHECKPOINT_E2E_GYM_ROOT="${CHECKPOINT_E2E_GYM_ROOT:-}" \
CHECKPOINT_E2E_EXPECTED_GYM_REVISION="${CHECKPOINT_E2E_EXPECTED_GYM_REVISION:-}" \
    bash "$ACTIVE_PACKAGE/run_existing_judge.sh" prepare "$CHECKPOINT" "$EXTERNAL_SOURCE" \
    | tee "$output"

prepared_run=$(sed -n 's/^RUN_DIR=//p' "$output" | tail -1)
[[ $prepared_run == /* && -d $prepared_run && ! -L $prepared_run ]] \
    || fail "prepare did not publish a valid RUN_DIR"
[[ ${prepared_run##*/} == "$EXPECTED_RUN_ID" ]] \
    || fail "checkpoint changed between bootstrap validation and snapshot"
prepared_import_id=$(sed -n 's/^EXISTING_PREPARED import_id=//p' "$output" | tail -1)
[[ $prepared_import_id == "$EXPECTED_IMPORT_ID" ]] \
    || fail "external source changed between bootstrap validation and snapshot"
prepared_package_sha=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["judge_package_source_inventory_sha256"])' \
    "$prepared_run/existing_import_receipt.json") \
    || fail "prepared package identity receipt is unreadable"
[[ $prepared_package_sha == "$EXPECTED_PACKAGE_SOURCE_SHA256" ]] \
    || fail "active package changed between bootstrap validation and snapshot"
(
    # Generated owner-only settings containing validated absolute paths.
    # shellcheck disable=SC1090
    source "$prepared_run/settings.env"
    [[ $REFERENCE_OVERLAY == "$CHECKPOINT_E2E_REFERENCE_OVERLAY" \
        && $REFERENCE_OVERLAY_SHA256 == "$EXPECTED_REFERENCE_OVERLAY_SHA256" \
        && $ENV_FILE == "$CHECKPOINT_E2E_ENV_FILE" ]]
) || fail "prepare did not preserve the bootstrap reference/environment inputs"

run_receipt=$BOOTSTRAP_STATE_DIR/PREPARED_RUN_DIR
temporary=$run_receipt.tmp.$$
printf '%s\n' "$prepared_run" > "$temporary"
chmod 0400 "$temporary"
if [[ -e $run_receipt || -L $run_receipt ]]; then
    [[ -f $run_receipt && ! -L $run_receipt ]] || fail "prepared-run receipt is invalid"
    cmp -s "$temporary" "$run_receipt" || fail "prepared-run receipt drift"
    rm -f "$temporary"
else
    mv "$temporary" "$run_receipt"
fi

CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS="$AUTHORIZE_PROVIDER_CALLS" \
CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION="$CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION" \
GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS="$GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS" \
    bash "$prepared_run/existing_judge_package/run_existing_judge.sh" submit "$prepared_run"

marker=$BOOTSTRAP_STATE_DIR/BOOTSTRAP_COMPLETE
temporary=$marker.tmp.$$
: > "$temporary"
chmod 0400 "$temporary"
if [[ -e $marker || -L $marker ]]; then
    [[ -f $marker && ! -L $marker && ! -s $marker ]] || fail "bootstrap marker drift"
    rm -f "$temporary"
else
    mv "$temporary" "$marker"
fi
printf 'EXISTING_BOOTSTRAP_PASS run_dir=%s authorized=%s\n' \
    "$prepared_run" "$AUTHORIZE_PROVIDER_CALLS"
