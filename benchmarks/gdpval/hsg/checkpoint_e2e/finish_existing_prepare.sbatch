#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Finish an interrupted import snapshot, then launch the MARS-local controller.
#SBATCH --job-name=gdpval-existing-mars-finish
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --open-mode=append

set -euo pipefail
umask 077
SAFE_PATH=/cm/local/apps/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH=$SAFE_PATH
export PYTHONDONTWRITEBYTECODE=1

: "${RUN_DIR:?set RUN_DIR}"
: "${CHECKPOINT_E2E_EXECUTION_PACKAGE:?set CHECKPOINT_E2E_EXECUTION_PACKAGE}"
: "${CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS:?set authorization}"
: "${CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION:?set persistent-session policy}"
: "${GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS:?set Gemini concurrency}"

fail() { echo "EXISTING_MARS_FINISH_FAIL: $*" >&2; exit 64; }
[[ $CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS == true \
    && $CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION == true \
    && $GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS =~ ^[1-4]$ ]] \
    || fail "unsafe judging policy"
[[ $RUN_DIR == /lustre/* && -d $RUN_DIR && ! -L $RUN_DIR \
    && -f $RUN_DIR/settings.env && ! -L $RUN_DIR/settings.env \
    && -f $RUN_DIR/existing_import_receipt.json && ! -L $RUN_DIR/existing_import_receipt.json \
    && -f $RUN_DIR/existing_import_receipt.json.sha256 \
    && ! -L $RUN_DIR/existing_import_receipt.json.sha256 \
    && -d $RUN_DIR/existing_judge_package && ! -L $RUN_DIR/existing_judge_package \
    && -d $RUN_DIR/judge_runtime_overlay && ! -L $RUN_DIR/judge_runtime_overlay ]] \
    || fail "interrupted prepared run is incomplete"

RUN_ID=$(sed -n 's/^RUN_ID=//p' "$RUN_DIR/settings.env")
[[ $RUN_ID =~ ^[A-Za-z0-9._-]+-[0-9a-f]{16}$ ]] || fail "invalid run id"
JOB_USER=${SLURM_JOB_USER:-${USER:-}}
[[ $JOB_USER =~ ^[A-Za-z0-9._-]+$ ]] || fail "invalid Slurm user"
BOOTSTRAP_ROOT=/raid/scratch/${JOB_USER}/gdpval-e2e-bootstrap-${SLURM_JOB_ID}-finish
install -d -m 0700 "$BOOTSTRAP_ROOT"
[[ $(stat -f -c '%T' "$BOOTSTRAP_ROOT") != lustre ]] \
    || fail "bootstrap root resolves to Lustre"
cp -- "$CHECKPOINT_E2E_EXECUTION_PACKAGE/mars_node_local.sh" "$BOOTSTRAP_ROOT/mars_node_local.sh"
chmod 0500 "$BOOTSTRAP_ROOT/mars_node_local.sh"
# shellcheck disable=SC1090
source "$BOOTSTRAP_ROOT/mars_node_local.sh"
mars_init "$RUN_ID" finish "$RUN_DIR"
mars_stage_package "$CHECKPOINT_E2E_EXECUTION_PACKAGE"
mars_stage_file "$RUN_DIR/settings.env" "$MARS_JOB_ROOT/settings.env"
# shellcheck disable=SC1090
source "$MARS_JOB_ROOT/settings.env"
[[ $ACCOUNT == nemotron_n3_post && $CPU_PARTITION == cpu && $CPU_QOS == cpu-normal ]] \
    || fail "unexpected Slurm routing"

runtime=$RUN_DIR/judge_runtime_overlay_existing
if [[ ! -e $runtime && ! -L $runtime ]]; then
    temporary=$RUN_DIR/.judge_runtime_overlay_existing.${SLURM_JOB_ID}.$$
    cp -a -- "$RUN_DIR/judge_runtime_overlay" "$temporary"
    mv "$temporary" "$runtime"
fi
[[ -d $runtime && ! -L $runtime && -f $runtime/runtime_manifest.json ]] \
    || fail "durable corrected runtime snapshot is invalid"

sha256sum -c --status "$RUN_DIR/existing_import_receipt.json.sha256" \
    || fail "immutable import receipt sidecar check failed"
import_id=$(jq -er '.import_id | strings' "$RUN_DIR/existing_import_receipt.json") \
    || fail "could not parse import receipt identity"
[[ $import_id =~ ^import-[0-9a-f]{24}$ ]] || fail "import receipt has an invalid identity"
import_receipt_sha=$(sha256sum "$RUN_DIR/existing_import_receipt.json" | awk '{print $1}')
[[ $import_receipt_sha =~ ^[0-9a-f]{64}$ ]] || fail "could not hash import receipt"
existing_env=$RUN_DIR/existing_judge.env
temporary=$existing_env.tmp.${SLURM_JOB_ID}.$$
printf '%s=%q\n' \
    IMPORT_ID "$import_id" \
    IMPORT_RECEIPT_SHA256 "$import_receipt_sha" \
    ACTIVE_PACKAGE "$RUN_DIR/existing_judge_package" \
    CORRECTED_RUNTIME_OVERLAY "$runtime" \
    CORRECTED_TRANSPORT_VIEW_ROOT "$RUN_DIR/judge_transport_views_existing" \
    JUDGE_DIR_SUFFIX existing \
    GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS "$GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS" \
    > "$temporary"
chmod 0400 "$temporary"
if [[ -e $existing_env || -L $existing_env ]]; then
    [[ -f $existing_env && ! -L $existing_env ]] || fail "existing settings are unsafe"
    cmp -s "$temporary" "$existing_env" || fail "existing settings drift"
    rm -f "$temporary"
else
    mv "$temporary" "$existing_env"
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/existing_controller_submissions"
# shellcheck disable=SC1090
source "$MARS_PACKAGE/slurm_receipts.sh"
receipt=$RUN_DIR/existing_controller_submissions/attempt_1.jobid
job_name="gdp-${RUN_ID:0:20}-exctl"
job=$(slurm_submit_or_adopt "$receipt" mars-existing-controller-a1 "$job_name" \
    --parsable -J "$job_name" -A "$ACCOUNT" -p "$CPU_PARTITION" --qos="$CPU_QOS" \
    -t 12:00:00 --cpus-per-task=2 --mem=8G \
    -o "$RUN_DIR/logs/%j_existing_controller.out" \
    -e "$RUN_DIR/logs/%j_existing_controller.err" \
    --export=RUN_DIR="$RUN_DIR",CHECKPOINT_E2E_EXECUTION_PACKAGE="$CHECKPOINT_E2E_EXECUTION_PACKAGE",CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS=true,CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION=true,GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS="$GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS",PATH="$SAFE_PATH" \
    "$MARS_PACKAGE/existing_judge_controller.sbatch") \
    || fail "could not submit MARS-local controller"
slurm_publish_job_receipt "$RUN_DIR/EXISTING_CONTROLLER.jobid" "$job" true >/dev/null \
    || fail "could not publish controller receipt"
printf 'EXISTING_MARS_FINISH_PASS run_dir=%s controller=%s\n' "$RUN_DIR" "$job"
