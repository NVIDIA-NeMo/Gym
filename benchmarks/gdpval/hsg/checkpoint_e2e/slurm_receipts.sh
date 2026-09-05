#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Crash-safe Slurm submission receipts for the checkpoint evaluator.

# This file is sourced by both the login-node launcher and the controller.  A
# submission intent is published before `sbatch`; if the caller dies after
# Slurm accepts the job but before its receipt is published, the next owner
# locates that exact job by its deterministic Slurm comment and adopts it.

slurm_receipt_error() {
    printf 'CHECKPOINT_E2E_SLURM_RECEIPT_FAIL: %s\n' "$*" >&2
    return 64
}

slurm_normalize_job_id() {
    local raw=${1:-} job
    job=${raw%%;*}
    job=${job//$'\n'/}
    job=${job//$'\r'/}
    [[ $job =~ ^[1-9][0-9]*$ ]] || {
        slurm_receipt_error "invalid Slurm job id: $raw"
        return
    }
    printf '%s\n' "$job"
}

slurm_read_job_receipt() {
    local receipt=$1 raw line_count
    [[ -f $receipt && ! -L $receipt ]] || {
        slurm_receipt_error "job receipt is not a regular file: $receipt"
        return
    }
    IFS= read -r raw < "$receipt" || {
        slurm_receipt_error "cannot read job receipt: $receipt"
        return
    }
    line_count=$(awk 'END {print NR + 0}' "$receipt")
    [[ $line_count == 1 ]] || {
        slurm_receipt_error "job receipt must contain exactly one line: $receipt"
        return
    }
    slurm_normalize_job_id "$raw"
}

slurm_publish_job_receipt() {
    local receipt=$1 raw=$2 replace=${3:-false} job temporary
    job=$(slurm_normalize_job_id "$raw") || return
    [[ $replace == true || $replace == false ]] || {
        slurm_receipt_error "invalid receipt replacement policy: $replace"
        return
    }
    mkdir -p "$(dirname "$receipt")"
    if [[ -e $receipt || -L $receipt ]]; then
        [[ -f $receipt && ! -L $receipt ]] || {
            slurm_receipt_error "job receipt is not a regular file: $receipt"
            return
        }
        if [[ $replace == false ]]; then
            [[ $(slurm_read_job_receipt "$receipt") == "$job" ]] || {
                slurm_receipt_error "immutable job receipt drift: $receipt"
                return
            }
            printf '%s\n' "$job"
            return
        fi
    fi
    temporary="${receipt}.tmp.$$"
    printf '%s\n' "$job" > "$temporary"
    chmod 0400 "$temporary"
    mv -f "$temporary" "$receipt"
    printf '%s\n' "$job"
}

slurm_submission_comment() {
    local role=$1 digest
    [[ -n ${RUN_DIR:-} && $role =~ ^[A-Za-z0-9._:-]+$ ]] || {
        slurm_receipt_error "unsafe or incomplete Slurm submission identity: $role"
        return
    }
    digest=$(printf '%s\n%s\n' "$RUN_DIR" "$role" | sha256sum | awk '{print $1}') || return
    [[ $digest =~ ^[0-9a-f]{64}$ ]] || {
        slurm_receipt_error "could not derive Slurm submission identity"
        return
    }
    printf 'gdp-e2e-%s\n' "${digest:0:32}"
}

slurm_publish_intent() {
    local intent=$1 role=$2 job_name=$3 comment=$4
    local temporary created_at actual
    [[ $role =~ ^[A-Za-z0-9._:-]+$ && $job_name =~ ^[A-Za-z0-9._-]+$ \
        && $comment =~ ^gdp-e2e-[0-9a-f]{32}$ ]] || {
        slurm_receipt_error "unsafe Slurm intent fields for role=$role"
        return
    }
    if [[ -e $intent || -L $intent ]]; then
        [[ -f $intent && ! -L $intent ]] || {
            slurm_receipt_error "submission intent is not a regular file: $intent"
            return
        }
        actual=$(sed -n '1,5p' "$intent")
        [[ $(awk 'END {print NR + 0}' "$intent") == 5 \
            && $actual == $'schema=gdpval.slurm-submit-intent.v1\n'"role=$role"$'\n'"job_name=$job_name"$'\n'"comment=$comment"$'\n'created_at=* ]] || {
            slurm_receipt_error "submission intent drift: $intent"
            return
        }
        created_at=$(sed -n 's/^created_at=//p' "$intent")
        [[ $created_at =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || {
            slurm_receipt_error "submission intent has an invalid timestamp: $intent"
            return
        }
        SLURM_INTENT_CREATED=false
        SLURM_INTENT_START=$created_at
        return 0
    fi
    created_at=$(date '+%Y-%m-%dT%H:%M:%S')
    [[ $created_at =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || {
        slurm_receipt_error "could not timestamp Slurm submission intent"
        return
    }
    mkdir -p "$(dirname "$intent")"
    temporary="${intent}.tmp.$$"
    printf 'schema=gdpval.slurm-submit-intent.v1\nrole=%s\njob_name=%s\ncomment=%s\ncreated_at=%s\n' \
        "$role" "$job_name" "$comment" "$created_at" > "$temporary"
    chmod 0400 "$temporary"
    if [[ -e $intent || -L $intent ]]; then
        rm -f "$temporary"
        slurm_receipt_error "submission intent appeared concurrently: $intent"
        return
    fi
    mv "$temporary" "$intent"
    SLURM_INTENT_CREATED=true
    SLURM_INTENT_START=$created_at
}

slurm_matching_jobs() {
    local job_name=$1 comment=$2 start_time=$3 queue accounting candidate detail
    local combined=""
    queue=$(squeue -h -n "$job_name" -o '%A' 2>/dev/null) || return 75
    while IFS= read -r candidate; do
        [[ -n $candidate ]] || continue
        candidate=$(slurm_normalize_job_id "$candidate") || return 64
        if detail=$(scontrol show job -o "$candidate" 2>/dev/null) \
            && [[ " $detail " == *" Comment=$comment "* ]]; then
            combined+="$candidate"$'\n'
        fi
    done <<< "$queue"

    accounting=$(sacct -X -S "$start_time" --name "$job_name" \
        --format=JobIDRaw,Comment -n -P 2>/dev/null) || return 75
    while IFS='|' read -r candidate recorded_comment _rest; do
        [[ $candidate =~ ^[1-9][0-9]*$ && $recorded_comment == "$comment" ]] || continue
        combined+="$candidate"$'\n'
    done <<< "$accounting"
    printf '%s' "$combined" | awk 'NF && !seen[$0]++ {print $0}'
}

slurm_adopt_existing_intent() {
    local job_name=$1 comment=$2 start_time=$3
    local grace=${CHECKPOINT_E2E_SLURM_ADOPTION_GRACE_SECONDS:-60}
    local poll=${CHECKPOINT_E2E_SLURM_ADOPTION_POLL_SECONDS:-5}
    local attempts attempt matches count joined query_success=0
    [[ $grace =~ ^[1-9][0-9]*$ && $poll =~ ^[1-9][0-9]*$ \
        && $grace -le 300 && $poll -le 60 ]] || {
        slurm_receipt_error "invalid Slurm adoption grace/poll: $grace/$poll"
        return
    }
    attempts=$(( (grace + poll - 1) / poll + 1 ))
    for (( attempt=1; attempt<=attempts; attempt++ )); do
        if matches=$(slurm_matching_jobs "$job_name" "$comment" "$start_time"); then
            query_success=1
            count=$(awk 'NF {n++} END {print n+0}' <<< "$matches")
            if (( count == 1 )); then
                printf '%s\n' "$matches"
                return 0
            fi
            if (( count > 1 )); then
                joined=$(paste -sd, <<< "$matches")
                slurm_receipt_error \
                    "ambiguous Slurm adoption for job_name=$job_name comment=$comment ids=$joined"
                return
            fi
        else
            case $? in
                64) return 64 ;;
                *) query_success=0 ;;
            esac
        fi
        (( attempt == attempts )) || sleep "$poll"
    done
    (( query_success == 1 )) || {
        slurm_receipt_error "Slurm state remained unavailable while adopting comment=$comment"
        return
    }
    slurm_receipt_error \
        "persisted submission intent has no matching Slurm job after ${grace}s; refusing a duplicate-prone resubmit: comment=$comment"
}

slurm_sbatch_without_parent_memory_limits() {
    # A controller can submit recovery work from inside its own small CPU
    # allocation. With --export=ALL, Slurm's allocation-derived memory values
    # otherwise leak into the child job and can cap an inner srun at the
    # controller's memory even when the child allocation is much larger. Let
    # Slurm repopulate the SLURM_* values from the child allocation, and prevent
    # ambient SBATCH_* request defaults from overriding the explicit request or
    # the batch script.
    env \
        -u SLURM_MEM_PER_NODE \
        -u SLURM_MEM_PER_CPU \
        -u SLURM_MEM_PER_GPU \
        -u SBATCH_MEM \
        -u SBATCH_MEM_PER_NODE \
        -u SBATCH_MEM_PER_CPU \
        -u SBATCH_MEM_PER_GPU \
        sbatch "$@"
}

slurm_submit_or_adopt() {
    local receipt=$1 role=$2 job_name=$3
    shift 3
    local intent="$(dirname "$receipt")/.slurm_submit_intents/$(basename "$receipt").intent"
    local comment raw job
    [[ $# -gt 0 ]] || {
        slurm_receipt_error "no sbatch arguments supplied for role=$role"
        return
    }
    if [[ -e $receipt || -L $receipt ]]; then
        slurm_read_job_receipt "$receipt"
        return
    fi
    comment=$(slurm_submission_comment "$role") || return
    slurm_publish_intent "$intent" "$role" "$job_name" "$comment" || return
    if [[ $SLURM_INTENT_CREATED == false ]]; then
        job=$(slurm_adopt_existing_intent "$job_name" "$comment" "$SLURM_INTENT_START") || return
        printf 'ADOPTED_SLURM_JOB role=%s job=%s comment=%s\n' "$role" "$job" "$comment" >&2
        slurm_publish_job_receipt "$receipt" "$job" >/dev/null || return
        printf '%s\n' "$job"
        return
    fi
    raw=$(slurm_sbatch_without_parent_memory_limits --comment="$comment" "$@") || {
        slurm_receipt_error "sbatch submission failed for role=$role"
        return
    }
    job=$(slurm_normalize_job_id "$raw") || return
    slurm_publish_job_receipt "$receipt" "$job" >/dev/null || return
    printf '%s\n' "$job"
}
