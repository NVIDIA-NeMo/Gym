#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit or safely adopt one corrected sibling-judge controller.

set -euo pipefail
umask 077

[[ $# == 1 ]] || { echo "usage: $0 RUN_DIR" >&2; exit 64; }
RUN_DIR=$(cd -P -- "$1" && pwd -P)
: "${ACTIVE_PACKAGE:?set ACTIVE_PACKAGE}"
: "${CORRECTED_RUNTIME_OVERLAY:?set CORRECTED_RUNTIME_OVERLAY}"
: "${CORRECTED_TRANSPORT_VIEW_ROOT:?set CORRECTED_TRANSPORT_VIEW_ROOT}"
: "${EXPECTED_JUDGE_FINGERPRINT:?set EXPECTED_JUDGE_FINGERPRINT}"
REQUESTED_ACTIVE_PACKAGE=$ACTIVE_PACKAGE
REQUESTED_CORRECTED_RUNTIME_OVERLAY=$CORRECTED_RUNTIME_OVERLAY
REQUESTED_CORRECTED_TRANSPORT_VIEW_ROOT=$CORRECTED_TRANSPORT_VIEW_ROOT
REQUESTED_EXPECTED_JUDGE_FINGERPRINT=$EXPECTED_JUDGE_FINGERPRINT
REQUESTED_JUDGE_DIR_SUFFIX=${JUDGE_DIR_SUFFIX:-nested_refs_v1}

[[ $REQUESTED_ACTIVE_PACKAGE == /* && -d $REQUESTED_ACTIVE_PACKAGE \
    && ! -L $REQUESTED_ACTIVE_PACKAGE ]] \
    || { echo "invalid ACTIVE_PACKAGE" >&2; exit 64; }
[[ $REQUESTED_JUDGE_DIR_SUFFIX =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ \
    && ${#REQUESTED_JUDGE_DIR_SUFFIX} -le 32 ]] \
    || { echo "invalid JUDGE_DIR_SUFFIX" >&2; exit 64; }
[[ $REQUESTED_EXPECTED_JUDGE_FINGERPRINT =~ ^[0-9a-f]{64}$ ]] \
    || { echo "invalid EXPECTED_JUDGE_FINGERPRINT" >&2; exit 64; }

# shellcheck disable=SC1090
source "$RUN_DIR/settings.env"
CAMPAIGN_E2E_DIR=$E2E_DIR
ACTIVE_PACKAGE=$REQUESTED_ACTIVE_PACKAGE
CORRECTED_RUNTIME_OVERLAY=$REQUESTED_CORRECTED_RUNTIME_OVERLAY
CORRECTED_TRANSPORT_VIEW_ROOT=$REQUESTED_CORRECTED_TRANSPORT_VIEW_ROOT
EXPECTED_JUDGE_FINGERPRINT=$REQUESTED_EXPECTED_JUDGE_FINGERPRINT
JUDGE_DIR_SUFFIX=$REQUESTED_JUDGE_DIR_SUFFIX
E2E_DIR=$ACTIVE_PACKAGE
# shellcheck disable=SC1090
source "$E2E_DIR/slurm_receipts.sh"

fail() { echo "REJUDGE_LAUNCH_FAIL: $*" >&2; exit 64; }

[[ $ACTIVE_PACKAGE != "$CAMPAIGN_E2E_DIR" ]] \
    || fail "corrected package aliases the original campaign package"
[[ $CORRECTED_RUNTIME_OVERLAY == "$RUN_DIR"/* && -d $CORRECTED_RUNTIME_OVERLAY \
    && ! -L $CORRECTED_RUNTIME_OVERLAY ]] \
    || fail "corrected runtime must be a sibling below the campaign run"
[[ $CORRECTED_TRANSPORT_VIEW_ROOT == "$RUN_DIR"/* && -d $CORRECTED_TRANSPORT_VIEW_ROOT \
    && ! -L $CORRECTED_TRANSPORT_VIEW_ROOT ]] \
    || fail "corrected transport view must be a sibling below the campaign run"
[[ $JUDGE_DIR_SUFFIX != e2e ]] || fail "corrected rejudge cannot target judge_e2e"

job_liveness() {
    local job=$1 queue accounting state
    queue=$(squeue -h -j "$job" -o '%A' 2>/dev/null) || queue=
    if [[ -n $queue ]] && grep -qx "$job" <<<"$queue"; then
        printf 'live\n'
        return
    fi
    accounting=$(sacct -X -j "$job" --format=JobIDRaw,State -n -P 2>/dev/null) \
        || { printf 'unknown\n'; return; }
    state=$(awk -F '|' -v wanted="$job" '$1 == wanted {print $2; exit}' <<<"$accounting")
    state=${state%%+*}; state=${state%% *}
    case "$state" in
        COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)
            printf 'terminal\n' ;;
        *) printf 'unknown\n' ;;
    esac
}

submission_root=$RUN_DIR/rejudge_controller_${JUDGE_DIR_SUFFIX}_submissions
mkdir -p "$submission_root"
latest=0
for path in "$submission_root"/attempt_[0-9]*.jobid; do
    [[ -e $path || -L $path ]] || continue
    [[ -f $path && ! -L $path ]] || fail "controller receipt is not a regular file: $path"
    name=${path##*/}; number=${name#attempt_}; number=${number%.jobid}
    [[ $number =~ ^[1-9][0-9]*$ ]] || fail "invalid controller receipt name: $name"
    (( number > latest )) && latest=$number
done
for (( expected=1; expected<=latest; expected++ )); do
    [[ -f $submission_root/attempt_${expected}.jobid \
        && ! -L $submission_root/attempt_${expected}.jobid ]] \
        || fail "controller receipt sequence has a gap before attempt $expected"
done
current_receipt=$RUN_DIR/REJUDGE_CONTROLLER_${JUDGE_DIR_SUFFIX}.jobid
if (( latest > 0 )); then
    latest_job=$(slurm_read_job_receipt "$submission_root/attempt_${latest}.jobid") \
        || fail "latest controller receipt is invalid"
    case "$(job_liveness "$latest_job")" in
        live)
            slurm_publish_job_receipt "$current_receipt" "$latest_job" true >/dev/null \
                || fail "could not refresh current controller receipt"
            printf 'REJUDGE_ALREADY_RUNNING=%s\nRUN_DIR=%s\nJUDGE_DIR=%s/judge_%s\n' \
                "$latest_job" "$RUN_DIR" "$RUN_DIR" "$JUDGE_DIR_SUFFIX"
            exit 0
            ;;
        terminal) ;;
        *) fail "controller $latest_job has no live or terminal Slurm evidence" ;;
    esac
fi

max_submissions=${CHECKPOINT_E2E_REJUDGE_CONTROLLER_MAX_SUBMISSIONS:-4}
[[ $max_submissions =~ ^[1-9][0-9]*$ && $max_submissions -le 8 ]] \
    || fail "invalid controller submission bound"
attempt=$((latest + 1))
(( attempt <= max_submissions )) || fail "controller submission bound reached: $latest/$max_submissions"
job_name="gdp-${RUN_ID:0:20}-nrctl"
receipt=$submission_root/attempt_${attempt}.jobid
job=$(slurm_submit_or_adopt "$receipt" "rejudge-${JUDGE_DIR_SUFFIX}-controller-a${attempt}" "$job_name" \
    --parsable -J "$job_name" -A "$ACCOUNT" -p "$CPU_PARTITION" --qos="$CPU_QOS" \
    -t 12:00:00 --cpus-per-task=2 --mem=8G \
    -o "$RUN_DIR/logs/%j_rejudge_controller.out" \
    -e "$RUN_DIR/logs/%j_rejudge_controller.err" \
    --export=ALL,RUN_DIR="$RUN_DIR",ACTIVE_PACKAGE="$ACTIVE_PACKAGE",CORRECTED_RUNTIME_OVERLAY="$CORRECTED_RUNTIME_OVERLAY",CORRECTED_TRANSPORT_VIEW_ROOT="$CORRECTED_TRANSPORT_VIEW_ROOT",EXPECTED_JUDGE_FINGERPRINT="$EXPECTED_JUDGE_FINGERPRINT",JUDGE_DIR_SUFFIX="$JUDGE_DIR_SUFFIX",CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS=true \
    "$E2E_DIR/rejudge_controller.sbatch")
slurm_publish_job_receipt "$current_receipt" "$job" true >/dev/null \
    || fail "could not publish current controller receipt"
printf 'REJUDGE_CONTROLLER=%s\nRUN_DIR=%s\nJUDGE_DIR=%s/judge_%s\n' \
    "$job" "$RUN_DIR" "$RUN_DIR" "$JUDGE_DIR_SUFFIX"
