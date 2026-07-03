#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/submit_jet_log.sh [OPTIONS] RESULTS_DIR

Create and upload a JET custom-workload log for one Gym evaluation job.
RESULTS_DIR is the experiment directory, for example results/<EXP_NAME>.

Options:
  --job-id ID          Slurm job ID. Inferred from SLURM_JOB_ID or the output
                       JSONL when the results directory contains one job.
  --exit-code CODE     Job exit code (default: query sacct, then 0).
  --duration SECONDS   Job duration (default: query sacct, then 0).
  --script NAME        Workload script metadata (default: eval.sub).
  --archive PATH       Output JET zip path.
  --dry-run            Validate the upload without sending it.
  --no-upload          Only create the local JET zip.
  --fill-environment   Fill hardware/system fields on this machine.
  --no-fill            Do not fill hardware/system fields.
  -h, --help           Show this help.

Environment:
  JET_ORIGIN           Log origin (default: mlperf-manual).
  JET_ASSET_TTL        Uploaded user-asset TTL in days (default: 90).
  JET_USER             JET user/maintainer (default: USER).
  JET_FRAMEWORK        Workload framework (default: vllm).
  JET_BIN              JET executable (default: jet).

By default environment fields are filled only while running inside the same
Slurm job. For a later manual submission they would describe the login node,
not the evaluation allocation; use --fill-environment to opt in explicitly.
EOF
}

results_dir=
job_id=${SLURM_JOB_ID:-}
exit_code=${JET_EXIT_CODE:-}
duration=${JET_DURATION:-}
workload_script=${JET_WORKLOAD_SCRIPT:-eval.sub}
jet_archive=${JET_ARCHIVE:-}
upload=1
dry_run=0
fill_environment=auto

while (( $# > 0 )); do
    case "$1" in
        --job-id)
            job_id=${2:?--job-id requires a value}
            shift 2
            ;;
        --exit-code)
            exit_code=${2:?--exit-code requires a value}
            shift 2
            ;;
        --duration)
            duration=${2:?--duration requires a value}
            shift 2
            ;;
        --script)
            workload_script=${2:?--script requires a value}
            shift 2
            ;;
        --archive)
            jet_archive=${2:?--archive requires a value}
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --no-upload)
            upload=0
            shift
            ;;
        --fill-environment)
            fill_environment=1
            shift
            ;;
        --no-fill)
            fill_environment=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n "$results_dir" ]]; then
                echo "Only one RESULTS_DIR may be supplied" >&2
                usage >&2
                exit 2
            fi
            results_dir=$1
            shift
            ;;
    esac
done

if (( $# > 0 )); then
    if [[ -n "$results_dir" || $# -ne 1 ]]; then
        echo "Only one RESULTS_DIR may be supplied" >&2
        usage >&2
        exit 2
    fi
    results_dir=$1
fi

if [[ -z "$results_dir" || ! -d "$results_dir" ]]; then
    echo "RESULTS_DIR must be an existing directory" >&2
    usage >&2
    exit 2
fi

results_dir=$(cd "$results_dir" && pwd -P)
exp_name=${results_dir##*/}

if [[ -z "$job_id" ]]; then
    declare -A inferred_job_ids=()
    shopt -s nullglob
    for output in "$results_dir/$exp_name."*.jsonl; do
        filename=${output##*/}
        candidate=${filename#"$exp_name."}
        candidate=${candidate%.jsonl}
        [[ "$candidate" == *_materialized_inputs ]] && continue
        inferred_job_ids["$candidate"]=1
    done
    shopt -u nullglob

    if (( ${#inferred_job_ids[@]} == 1 )); then
        for candidate in "${!inferred_job_ids[@]}"; do
            job_id=$candidate
        done
    elif (( ${#inferred_job_ids[@]} == 0 )); then
        echo "Could not infer a job ID; pass --job-id" >&2
        exit 2
    else
        echo "The results directory contains multiple jobs; pass --job-id" >&2
        printf '  %s\n' "${!inferred_job_ids[@]}" >&2
        exit 2
    fi
fi

if [[ -z "$exit_code" ]] && command -v sacct >/dev/null 2>&1; then
    exit_code=$(sacct -j "$job_id" -X --format=ExitCode --noheader 2>/dev/null \
        | awk 'NF {split($1, code, ":"); print code[1]; exit}') || true
fi
exit_code=${exit_code:-0}
if [[ ! "$exit_code" =~ ^[0-9]+$ ]]; then
    echo "Exit code must be a non-negative integer: $exit_code" >&2
    exit 2
fi

if [[ -z "$duration" ]] && command -v sacct >/dev/null 2>&1; then
    duration=$(sacct -j "$job_id" -X --format=ElapsedRaw --noheader 2>/dev/null \
        | awk 'NF {print $1; exit}') || true
fi
duration=${duration:-0}

node_count=${JET_NODE_COUNT:-}
if [[ -z "$node_count" && -n "${SLURM_JOB_ID:-}" && "$SLURM_JOB_ID" == "$job_id" ]]; then
    node_count=${SLURM_NNODES:-}
fi
if [[ -z "$node_count" ]] && command -v sacct >/dev/null 2>&1; then
    node_count=$(sacct -j "$job_id" -X --format=NNodes --noheader 2>/dev/null \
        | awk 'NF {print $1; exit}') || true
fi
node_count=${node_count:-0}

output_prefix="$results_dir/$exp_name.$job_id"
trajectory_archive="$results_dir/trajectories-$job_id.tar.gz"
if [[ -z "$jet_archive" ]]; then
    jet_archive="$results_dir/jet-$exp_name.$job_id.zip"
elif [[ "$jet_archive" != /* ]]; then
    jet_archive="$PWD/$jet_archive"
fi

# New jobs keep their SWE run directories under a job-specific root. Fall back
# to the legacy flat layout so older result directories can still be submitted.
trajectory_root=$results_dir
if [[ -d "$results_dir/slurm-$job_id" ]]; then
    trajectory_root="$results_dir/slurm-$job_id"
elif find "$results_dir" -maxdepth 1 -type d -name 'swebench_results_*' -print -quit | grep -q .; then
    echo "Warning: using the legacy flat SWE results layout; all trajectories in" >&2
    echo "         $results_dir will be included because they cannot be attributed to one job." >&2
fi

trajectory_dirs=()
while IFS= read -r -d '' directory; do
    trajectory_dirs+=("${directory#"$results_dir"/}")
done < <(find "$trajectory_root" -type d -name trajectories -print0)

if (( ${#trajectory_dirs[@]} > 0 )); then
    tar -C "$results_dir" -czf "$trajectory_archive" -- "${trajectory_dirs[@]}"
    echo "Packed ${#trajectory_dirs[@]} trajectories directories into $trajectory_archive"
else
    echo "Warning: no trajectories directories found under $trajectory_root" >&2
fi

asset_args=()
for asset in \
    "$output_prefix.jsonl" \
    "${output_prefix}_aggregate_metrics.json" \
    "${output_prefix}_materialized_inputs.jsonl" \
    "$results_dir/slurm-$job_id.out"; do
    if [[ -f "$asset" ]]; then
        asset_args+=(--asset "$asset")
    else
        echo "Warning: output asset not found: $asset" >&2
    fi
done
if (( ${#trajectory_dirs[@]} > 0 )); then
    asset_args+=(--asset "$trajectory_archive")
fi

if (( ${#asset_args[@]} == 0 )); then
    echo "No output assets found for job $job_id" >&2
    exit 1
fi

jet_bin=${JET_BIN:-jet}
if ! command -v "$jet_bin" >/dev/null 2>&1; then
    echo "JET CLI not found: $jet_bin" >&2
    exit 127
fi

origin=${JET_ORIGIN:-nemo-gym}
jet_user=${JET_USER:-${USER:?USER is not set}}
framework=${JET_FRAMEWORK:-vllm}
asset_ttl=${JET_ASSET_TTL:-90}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)

create_args=(
    "$jet_archive"
    --fill-source-code "$repo_root"
    --data type=workload
    --data "origin=$origin"
    --data "user=$jet_user"
    --data workload.type=custom
    --data "workload.key=custom/$exp_name"
    --data "workload.maintainers[0]=$jet_user"
    --data "workload.spec.name=$exp_name"
    --data "workload.spec.script=$workload_script"
    --data "workload.spec.nodes=$node_count"
    --data "workload.spec.framework=$framework"
    --data "duration=$duration"
    --data "exit_code=$exit_code"
    --data "slurm.job=$job_id"
    --data "slurm.exit_code=$exit_code"
)

if [[ ! -f "$jet_archive" ]]; then
    create_args+=(--generate-id)
fi

if [[ "$fill_environment" == auto ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" && "$SLURM_JOB_ID" == "$job_id" ]]; then
        fill_environment=1
    else
        fill_environment=0
    fi
fi
if [[ "$fill_environment" == 1 ]]; then
    create_args+=(--fill-gpu --fill-cpu --fill-cluster --fill-system --fill-libraries)
fi

if (( exit_code == 0 )); then
    create_args+=(--data status.code=0 --data status.name=success --data status.message=Success)
else
    create_args+=(--data status.code=1.1 --data "status.name=error > workload")
    create_args+=(--data "status.message=Workload exited with code $exit_code")
fi

if [[ -n "${MODEL:-}" ]]; then
    create_args+=(--data "workload.spec.model=$MODEL")
fi
if [[ -n "${JET_SOURCE_IMAGE:-}" ]]; then
    create_args+=(--data "source_image.name=$JET_SOURCE_IMAGE")
fi
create_args+=("${asset_args[@]}")

"$jet_bin" logs create "${create_args[@]}"
echo "Created JET log archive: $jet_archive"

if (( upload == 0 )); then
    exit 0
fi

upload_args=("$jet_archive" --user-assets-ttl "$asset_ttl")
if (( dry_run == 1 )); then
    upload_args+=(--dry-run)
fi
"$jet_bin" logs upload "${upload_args[@]}"
