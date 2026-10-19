#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Prefill profiling (set here via environment or in VLLM_CONFIG):
# PROFILE_PREFILL=1 captures 20 worker steps, 300 seconds after API readiness.
# PREFILL_PROFILE_STEPS, PREFILL_PROFILE_DELAY_SECONDS (or manual), and
# PREFILL_PROFILE_DIR (default: results/nsys under SLURM_SUBMIT_DIR) override this.
# Detailed tracing adds overhead; use separate runs for throughput measurements.
# API-triggered captures exclude startup/graph construction. For those call sites,
# enable cuBLAS logging before launch or collect a separate startup trace.

# Mooncake disk offload is opt-in: ENABLE_MOONCAKE_DISK_OFFLOAD=1 also enables
# Mooncake unless ENABLE_MOONCAKE=0 was explicitly set (an error).
# Required only with disk offload: MOONCAKE_SSD_PATH=/absolute/node-local/ssd/dir.
# Per-node defaults: MOONCAKE_CPU_CACHE_GIB=600, MOONCAKE_DISK_QUOTA_GIB=512.
# One standalone owner per node replaces the embedded worker pools. SSD data is
# job-local and removed on orderly shutdown, and excluded from warmed images.
# SIGKILL/node failure may leave job directories. Without disk offload, Mooncake
# retains its embedded 150GB worker pools and needs no SSD mount or owner process.
# MOONCAKE_DEFAULT_KV_LEASE_TTL overrides either default: 180000 ms without disk
# offload, 30000 ms with it. Owner/master logs go to slurm-logs/%j-%x.log.
# MOONCAKE_CACHE_TRACE=1 enables hash-removal summaries and master INFO logs.

# Input arguments and validation
# PD (default): NUM_PREFILL_NODES=<P> NUM_DECODE_NODES=<D>
# Aggregated: VLLM_MODE=aggregated NUM_NODES=<replicas> (defaults to one)
# Two TP2 replicas per node: VLLM_MODE=aggregated VLLM_ENGINES_PER_NODE=2 NUM_NODES=1
# Four TP1 replicas per node: VLLM_MODE=aggregated VLLM_ENGINES_PER_NODE=4 NUM_NODES=1
# Disaggregated pools also support two TP2 replicas per node with VLLM_ENGINES_PER_NODE=2.
# Disaggregated TP1 pools: VLLM_ENGINES_PER_NODE=4 NUM_PREFILL_NODES=1 NUM_DECODE_NODES=1
# SAVE_WARMED_CONTAINER=/shared/images/warmed.sqsh runs startup only (omit eval args),
# collects all nodes' FlashInfer autotuning caches, compiled JIT modules, and
# vLLM startup plans inside one image, exports it, and exits. No cache mount is used.
# Later runs with CONTAINER pointing to that image automatically load the baked
# caches. Keep the same GPU type and
# serving config; changed kernels/configs may need fresh tuning. Mounted model
# weights, GPU memory and CUDA graphs are not captured. Output must not already exist.
# Opt-in debug inventory: 0 disables all scans; 10 samples every ten seconds.
VLLM_CACHE_INVENTORY_INTERVAL="${VLLM_CACHE_INVENTORY_INTERVAL:-0}"
if [[ ! "$VLLM_CACHE_INVENTORY_INTERVAL" =~ ^(0|[1-9][0-9]{1,5})$ ]]; then
    echo "ERROR: VLLM_CACHE_INVENTORY_INTERVAL must be 0 or an integer from 10 to 999999." >&2
    exit 1
fi
VLLM_MODE="${VLLM_MODE:-pd}"
VLLM_ENGINES_PER_NODE="${VLLM_ENGINES_PER_NODE:-1}"
case "$VLLM_MODE" in
    pd)
        NUM_PREFILL_NODES=${NUM_PREFILL_NODES:?Required in PD mode}
        NUM_DECODE_NODES=${NUM_DECODE_NODES:?Required in PD mode}
        NUM_NODES=$((NUM_PREFILL_NODES + NUM_DECODE_NODES))
        ;;
    aggregated)
        NUM_NODES=${NUM_NODES:-1}
        NUM_PREFILL_NODES=0
        NUM_DECODE_NODES=0
        ;;
    *)
        echo "VLLM_MODE must be pd or aggregated" >&2
        exit 1
        ;;
esac
case "$VLLM_ENGINES_PER_NODE" in
    1 | 2 | 4) ;;
    *)
        echo "ERROR: VLLM_ENGINES_PER_NODE must be 1, 2, or 4." >&2
        exit 1
        ;;
esac
MODEL=$MODEL
MODEL_NAME="${MODEL_NAME:-$MODEL}"
CONTAINER=$CONTAINER
MOUNTS=$MOUNTS
VLLM_CONFIG=$VLLM_CONFIG
SAVE_WARMED_CONTAINER="${SAVE_WARMED_CONTAINER:-}"
warm_cache_helper=$(cat "$(dirname "${BASH_SOURCE[0]}")/save_warmed_vllm_container.py")
if [[ -n "$SAVE_WARMED_CONTAINER" ]]; then
    if (( $# > 0 )); then
        echo "ERROR: SAVE_WARMED_CONTAINER runs startup only; omit evaluation arguments." >&2
        exit 1
    fi
    mkdir -p "$(dirname "$SAVE_WARMED_CONTAINER")" slurm-logs
    SAVE_WARMED_CONTAINER="$(cd "$(dirname "$SAVE_WARMED_CONTAINER")" && pwd -P)/$(basename "$SAVE_WARMED_CONTAINER")"
fi
ENABLE_MOONCAKE_DISK_OFFLOAD=${ENABLE_MOONCAKE_DISK_OFFLOAD:-0}
case "$ENABLE_MOONCAKE_DISK_OFFLOAD" in
    0 | 1) ;;
    *)
        echo "ERROR: ENABLE_MOONCAKE_DISK_OFFLOAD must be 0 or 1." >&2
        exit 1
        ;;
esac
ENABLE_MOONCAKE=${ENABLE_MOONCAKE:-$ENABLE_MOONCAKE_DISK_OFFLOAD}
MOONCAKE_CACHE_TRACE=${MOONCAKE_CACHE_TRACE:-0}
case "$MOONCAKE_CACHE_TRACE" in
    0 | 1) ;;
    *)
        echo "ERROR: MOONCAKE_CACHE_TRACE must be 0 or 1." >&2
        exit 1
        ;;
esac
worker_mounts=$MOUNTS
mooncake_mode=embedded
mooncake_global_segment_size='"150GB"'
mooncake_enable_offload=false
mooncake_eviction_high_watermark=0.9
mooncake_minloglevel=${MOONCAKE_MINLOGLEVEL:-1}
mooncake_metric_reporting=false
mooncake_default_lease_ttl=180000
if (( ENABLE_MOONCAKE_DISK_OFFLOAD )); then
    if [[ "$ENABLE_MOONCAKE" != 1 ]]; then
        echo "ERROR: Disk offload requires ENABLE_MOONCAKE=1." >&2
        exit 1
    fi
    MOONCAKE_SSD_PATH=${MOONCAKE_SSD_PATH:?Set to an existing node-local SSD directory on every worker}
    # Pyxis mount specifications use commas/colons; constrain the path before it
    # is embedded in the generated batch script. Check its filesystem on the node.
    if [[ ! "$MOONCAKE_SSD_PATH" =~ ^/[a-zA-Z0-9_./-]+$ || "/$MOONCAKE_SSD_PATH/" == *"/../"* ]]; then
        echo "ERROR: MOONCAKE_SSD_PATH must be an absolute path without spaces, mount separators, or '..'." >&2
        exit 1
    fi
    MOONCAKE_CPU_CACHE_GIB=${MOONCAKE_CPU_CACHE_GIB:-600}
    MOONCAKE_DISK_QUOTA_GIB=${MOONCAKE_DISK_QUOTA_GIB:-512}
    for budget in "$MOONCAKE_CPU_CACHE_GIB" "$MOONCAKE_DISK_QUOTA_GIB"; do
        if [[ ! "$budget" =~ ^[1-9][0-9]{0,5}$ ]]; then
            echo "ERROR: Mooncake CPU/disk budgets must be positive integer GiB values (at most 999999)." >&2
            exit 1
        fi
    done
    worker_mounts="$MOUNTS,$MOONCAKE_SSD_PATH:/mooncake-ssd"
    mooncake_mode=standalone-store
    mooncake_global_segment_size=0
    mooncake_enable_offload=true
    mooncake_eviction_high_watermark=0.7
    mooncake_minloglevel=0
    mooncake_metric_reporting=true
    mooncake_default_lease_ttl=30000
else
    # Values are expanded into the generated worker command even when disabled.
    MOONCAKE_CPU_CACHE_GIB=0
    MOONCAKE_DISK_QUOTA_GIB=0
fi
MOONCAKE_DEFAULT_KV_LEASE_TTL=${MOONCAKE_DEFAULT_KV_LEASE_TTL:-$mooncake_default_lease_ttl}
if [[ "$VLLM_MODE" == aggregated && "$ENABLE_MOONCAKE" != 0 ]]; then
    echo "ENABLE_MOONCAKE requires VLLM_MODE=pd" >&2
    exit 1
fi
# Independent mode starts one TP replica, two TP2 replicas, or four TP1 replicas per node. Coupled mode
# forms one multi-node DP/EP engine per tier for models that cannot fit per node.
VLLM_PD_DEPLOYMENT_MODE="${VLLM_PD_DEPLOYMENT_MODE:-independent}"
SLURM_COMMENT="${SLURM_COMMENT:-}"
OPENSANDBOX_DOMAIN="${OPENSANDBOX_DOMAIN:-}"
OPENSANDBOX_API_KEY="${OPENSANDBOX_API_KEY:-}"
OPENSANDBOX_PROTOCOL="${OPENSANDBOX_PROTOCOL:-http}"

case "$VLLM_PD_DEPLOYMENT_MODE" in
    independent | coupled)
        ;;
    *)
        echo "ERROR: VLLM_PD_DEPLOYMENT_MODE must be independent or coupled; got '$VLLM_PD_DEPLOYMENT_MODE'." >&2
        exit 1
        ;;
esac

if [[ "$VLLM_MODE" == aggregated && "$VLLM_PD_DEPLOYMENT_MODE" == coupled ]]; then
    echo "ERROR: VLLM_MODE=aggregated does not support VLLM_PD_DEPLOYMENT_MODE=coupled." >&2
    exit 1
fi
if [[ "$VLLM_ENGINES_PER_NODE" != 1 && "$VLLM_PD_DEPLOYMENT_MODE" == coupled ]]; then
    echo "ERROR: VLLM_ENGINES_PER_NODE=$VLLM_ENGINES_PER_NODE requires VLLM_PD_DEPLOYMENT_MODE=independent." >&2
    exit 1
fi

should_run_eval=$(( $# > 0 ))
if (( should_run_eval )); then
    EXPERIMENT_NAME=$EXPERIMENT_NAME

    EXPORT_TO_CSV=${EXPORT_TO_CSV:-0}
    EXPORT_CSV_TO_MODEL_DIR=${EXPORT_CSV_TO_MODEL_DIR:-0}
else
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-vllm_only}"

    EXPORT_TO_CSV=0
    EXPORT_CSV_TO_MODEL_DIR=0
fi

# Fixed vLLM Port configurations
PREFILL_VLLM_NIXL_SIDE_CHANNEL_PORT=5600
DECODE_VLLM_NIXL_SIDE_CHANNEL_PORT=5700

ROUTER_SERVER_PORT=8000
ROUTER_METRICS_PORT=29000
WORKER_SERVER_PORT=8001

PREFILL_DP_RPC_PORT=13345
DECODE_DP_RPC_PORT=13346

ROUTER_PREFILL_POLICY="${ROUTER_PREFILL_POLICY:-cache_aware}"
ROUTER_DECODE_POLICY="${ROUTER_DECODE_POLICY:-cache_aware}"
ROUTER_POLICY="${ROUTER_POLICY:-cache_aware}"
# Shared startup timeout for vLLM engines, router workers, and Gym endpoint readiness.
ROUTER_WORKER_STARTUP_TIMEOUT_SECONDS="${ROUTER_WORKER_STARTUP_TIMEOUT_SECONDS:-1800}"
VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-$ROUTER_WORKER_STARTUP_TIMEOUT_SECONDS}"
ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE="${ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE:-1}"
if [[ "$VLLM_ENGINES_PER_NODE" != 1 && "$ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE" != 1 ]]; then
    echo "ERROR: Multiple independent engines require ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE=1." >&2
    exit 1
fi
# Optional whitespace-separated flags, e.g. ROUTER_ARGS="--balance-abs-threshold 32 --balance-rel-threshold 1.1".

# The generated command is parsed by Bash again on the evaluation node.
# Preserve each argument, including Hydra dictionaries, as literal shell data.
eval_arguments=""
if (( should_run_eval )); then
    printf -v eval_arguments '%q ' "$@"
fi

eval_command=$(cat <<EOF
set -euo pipefail

# Activate environment in container and cd into Gym. The Gym path here may be mounted.
source /opt/Gym_venv/bin/activate
cd /opt/Gym

export NEMO_GYM_RUN_ID="\$SLURM_JOB_ID"
export NEMO_GYM_USER="\${NEMO_GYM_USER:-\$SLURM_JOB_USER}"

source "$VLLM_CONFIG"

gym eval prepare $eval_arguments +use_cached_prepared_benchmarks=true

experiment_name=$EXPERIMENT_NAME/slurm_job_id_\$SLURM_JOB_ID/date_\$(date +%Y%m%d_%H%M%S)
# export_to_csv.py derives <base>_aggregate_metrics.json from this, so the
# default timestamped name makes the aggregate unfindable to anything that
# did not watch the job run. Override it when results/ is already per-run.
rollouts_fpath=\${ROLLOUTS_FPATH:-results/\$experiment_name.jsonl}

# Scrape each API server directly; the router only exposes its own metrics.
gym_config_args=(
    --config benchmarks/nemotron_3.5_super/sandbox_utils.yaml
    --config benchmarks/nemotron_3.5_super/policy_model_override.yaml
)
inference_metrics_config="results/\$experiment_name/inference-metrics.yaml"
mkdir -p "\$(dirname "\$inference_metrics_config")"
read -r -a nodes <<< "\$ALL_NODES"
{
    printf 'inference_metrics:\n  enabled: true\n  endpoints:\n'
    for node_index in "\${!nodes[@]}"; do
        # Coupled tiers expose one API server each; other ranks are headless.
        if [[ "$VLLM_PD_DEPLOYMENT_MODE" == coupled ]] && \
            (( node_index != 0 && node_index != $NUM_PREFILL_NODES )); then
            continue
        fi
        for (( engine_index = 0; engine_index < $VLLM_ENGINES_PER_NODE; engine_index++ )); do
            replica_index=\$(( node_index * $VLLM_ENGINES_PER_NODE + engine_index ))
            worker_port=\$(( $WORKER_SERVER_PORT + engine_index ))
            printf '    node%s: "http://%s:%s/metrics"\n' \
                "\$replica_index" "\${nodes[node_index]}" "\$worker_port"
        done
    done
    if (( $VLLM_CACHE_INVENTORY_INTERVAL > 0 )); then
        printf '  cache_inventory: true\n  cache_inventory_interval_s: %s\n' "$VLLM_CACHE_INVENTORY_INTERVAL"
    fi
    printf '  router_endpoints:\n    main: "http://%s:$ROUTER_METRICS_PORT/metrics"\n' "\$ROUTER_NODE"
    if (( $ENABLE_MOONCAKE )); then
        printf '  mooncake_endpoint: "http://%s:9003/metrics"\n' "\${nodes[0]}"
    fi
} > "\$inference_metrics_config"
gym_config_args+=(--config "\$inference_metrics_config")

# +uv_venv_dir=/opt/uv_venvs is from the container.
# +skip_venv_if_present=true will reuse the venvs baked into the container if possible.
# ++use_absolute_ip=true: Necessary for communication between harness in sandbox and Gym model servers
# ++upload_rollouts=false: Rollouts file is massive. We leave on the cluster.
# global_aiohttp_connector_limit_per_host: 16k concurrent requests should be enough. We can raise further if our inference is efficient enough to support.
# port_range_low, port_range_high: Move into ephemeral ports
# We add the sandbox_utils and policy_model_override yamls so users don't need to add them on every invocation
gym eval run \
    $eval_arguments \
    "\${gym_config_args[@]}" \
    +wandb_project=$USER-gym-eval \
    +wandb_name=\$experiment_name \
    +uv_venv_dir=/opt/uv_venvs \
    +nemo_gym_log_dir=results/\$experiment_name/logs \
    +skip_venv_if_present=true \
    ++output_jsonl_fpath=\$rollouts_fpath \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++use_absolute_ip=true \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=http://\$(getent hosts "\$ROUTER_NODE" | awk 'NR == 1 {print \$1}'):$ROUTER_SERVER_PORT/v1 \
    ++policy_api_key=dummy_api_key \
    ++policy_model_name=$MODEL_NAME \
    ++upload_rollouts=false \
    ++global_aiohttp_connector_limit_per_host=16384 \
    ++port_range_low=63000 \
    ++port_range_high=64000 \
    "\${GYM_MODEL_PARAMS[@]}" \
    ++model_endpoint_readiness_timeout_seconds=$ROUTER_WORKER_STARTUP_TIMEOUT_SECONDS


if (( $EXPORT_TO_CSV )); then
    python benchmarks/nemotron_3.5_super/export_to_csv.py \
        --model-path $MODEL \
        --jsonl-fpath-base \$(realpath "\${rollouts_fpath%.jsonl}")

    if (( $EXPORT_CSV_TO_MODEL_DIR )); then
        cp "\${rollouts_fpath%.jsonl}_export.csv" $MODEL/export.csv
    fi
fi

EOF
)

vllm_command=$(cat <<EOF
#!/bin/bash

set -euo pipefail

# Generic vLLM environment variables.
export VLLM_USE_FASTOKENS=1
export VLLM_CACHE_INVENTORY_INTERVAL=$VLLM_CACHE_INVENTORY_INTERVAL
export VLLM_ENGINE_READY_TIMEOUT_S="$VLLM_ENGINE_READY_TIMEOUT_S"

# @bxyu-nvidia: This timeout keep_alive helps reduce connection reset errors between vllm-router and the prefill/decode instances.
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=180

# TODO @bxyu-nvidia: Unfortunately there's an accuracy issue with the rust frontend in vLLM 0.29.0, around 1-2% delta on SWE Verified.
# export VLLM_USE_RUST_FRONTEND=1

# NIXL uses UCX for cross-node KV transfer. Explicitly enable UCX's CUDA
# transports and the GB200 InfiniBand interface; otherwise UCX treats VRAM as
# host memory and NIXL KV-cache registration fails with NIXL_ERR_BACKEND.
export UCX_TLS=rc_x,rc,dc_x,dc,cuda_copy,cuda_ipc
export UCX_RNDV_SCHEME=get_zcopy
export UCX_RNDV_THRESH=0
# On OCI-HSG, only these NICs are available
export UCX_NET_DEVICES=mlx5_0,mlx5_1,mlx5_3,mlx5_4

# Helpful NCCL env vars to set on modern clusters.
export NCCL_CUMEM_ENABLE=1
export NCCL_MNNVL_ENABLE=1
export NCCL_NVLS_ENABLE=1

export ENABLE_MOONCAKE=$ENABLE_MOONCAKE
export MOONCAKE_CACHE_TRACE=$MOONCAKE_CACHE_TRACE
export VLLM_ENGINES_PER_NODE=$VLLM_ENGINES_PER_NODE
source "$VLLM_CONFIG"

if [[ -f /opt/super-vl-evals/flashinfer-autotune-cache/warmed-manifest.json ]]; then
    python3 -c "\$warm_cache_helper" inspect
fi

baked_cache=/opt/super-vl-evals/flashinfer-autotune-cache
if [[ -n "$SAVE_WARMED_CONTAINER" || -f "\$baked_cache/.baked" ]]; then
    export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="\$baked_cache"
    mkdir -p "\$baked_cache"
    echo "Using image-local FlashInfer autotune cache: \$baked_cache on \$(hostname)"
fi
if [[ -n "$SAVE_WARMED_CONTAINER" || -f "\$baked_cache/.startup-plans-baked" ]]; then
    export VLLM_CACHE_ROOT=/opt/super-vl-evals/vllm-cache
    export VLLM_ENABLE_STARTUP_PLAN="\${VLLM_ENABLE_STARTUP_PLAN:-1}"
    echo "Using image-local vLLM startup plans: \$VLLM_CACHE_ROOT/startup_plan on \$(hostname)"
fi
if [[ -n "$SAVE_WARMED_CONTAINER" ]]; then
    PROFILE_PREFILL=0
    export VLLM_ENABLE_STARTUP_PLAN=1
    # Keep every worker's runtime JIT output inside its named container. The
    # export step collects these libraries from all nodes and installs them as AOT.
    export FLASHINFER_WORKSPACE_BASE=/opt/super-vl-evals/flashinfer-warmup-workspace
    touch "\$baked_cache/.warmup-start"
fi

# Custom images contain patched AOT kernels and their build-time smoke check.
# Do this once on every node before any prefill/decode/headless workers fork.
flashinfer_manifest=/opt/super-vl-evals/flashinfer-build.env
if [[ -f "\$flashinfer_manifest" ]]; then
    if ! grep -qx 'FLASHINFER_REPLAY_VERIFIED=1' "\$flashinfer_manifest"; then
        echo "ERROR: Rebuild this image with the updated custom_flashinfer_version.sh (unverified ReplaySSM kernels)." >&2
        exit 1
    fi
    # Retain the caller's cache root, but never reuse a different image build's
    # kernels under the unchanged public FlashInfer version (e.g. 0.7.0.post1).
    flashinfer_build_id=\$(sha256sum "\$flashinfer_manifest")
    flashinfer_build_id=\${flashinfer_build_id%% *}
    export FLASHINFER_WORKSPACE_BASE="\${FLASHINFER_WORKSPACE_BASE:-/tmp}/flashinfer-builds/\$flashinfer_build_id"
    mkdir -p "\$FLASHINFER_WORKSPACE_BASE"
    unset FLASHINFER_DISABLE_JIT
    echo "FlashInfer build=\$flashinfer_build_id workspace=\$FLASHINFER_WORKSPACE_BASE on \$(hostname)"
    FLASHINFER_DISABLE_JIT=1 python3 /opt/super-vl-evals/verify-flashinfer.py \\
        --verify /opt/super-vl-evals/flashinfer-warmup.env
fi

# Increase the number of file descriptors to 65k
if [[ \$(ulimit -Hn) == "unlimited" ]] || [[ 65535 -lt \$(ulimit -Hn) ]]; then
  ulimit -Sn 65535
fi

this_node_hostname=\$(hostname)
read -r -a nodes <<< "\$ALL_NODES"

# The model config can opt prefill into a short, API-triggered GPU trace.
# Keep profiler setup here so decode, evaluation, and aggregated workers are unaffected.
profiler_pid=""
trigger_pid=""
cleanup_prefill_profile() {
    if [[ -n "\$trigger_pid" ]]; then
        kill "\$trigger_pid" 2>/dev/null || true
        wait "\$trigger_pid" 2>/dev/null || true
        trigger_pid=""
    fi
    if [[ -n "\$profiler_pid" ]]; then
        kill "\$profiler_pid" 2>/dev/null || true
        wait "\$profiler_pid" 2>/dev/null || true
        profiler_pid=""
    fi
}

trigger_prefill_profile() {
    local url="http://\$this_node_hostname:\$profile_port"
    local deadline=\$(( SECONDS + 1200 ))
    until curl --noproxy '*' --silent --fail --max-time 5 "\$url/health" >/dev/null; do
        if (( SECONDS >= deadline )); then
            echo "WARNING: Timed out waiting for \$url/health; no prefill profile captured." >&2
            return 1
        fi
        sleep 1
    done
    echo "Prefill profiling: \$url ready; capturing in \$profile_delay seconds." >&2
    # Short sleeps let teardown cancel the trigger without leaving a long-lived sleep process.
    deadline=\$(( SECONDS + 10#\$profile_delay ))
    while (( SECONDS < deadline )); do sleep 1; done
    if curl --noproxy '*' --silent --show-error --fail --max-time 120 \
        -X POST "\$url/start_profile" >/dev/null; then
        echo "Prefill profiling: capture started on \$url (worker-step limit: \$steps)." >&2
    else
        echo "WARNING: Could not start prefill profile on \$url; serving continues." >&2
        return 1
    fi
}

ensure_nsys() {
    if command -v nsys >/dev/null 2>&1; then return; fi
    if [[ \$(id -u) != 0 ]] || ! command -v apt-get >/dev/null 2>&1; then
        echo "ERROR: Installing Nsight Systems requires root inside an Ubuntu container; otherwise preinstall nsys in the image." >&2
        return 1
    fi
    local ID="" VERSION_ID="" arch
    source /etc/os-release
    if [[ "\$ID" != ubuntu ]]; then
        echo "ERROR: Automatic Nsight Systems installation supports Ubuntu; preinstall nsys for \$ID." >&2
        return 1
    fi
    arch=\$(dpkg --print-architecture)
    if [[ "\$arch" != arm64 && "\$arch" != amd64 ]]; then
        echo "ERROR: Unsupported Nsight Systems architecture: \$arch." >&2
        return 1
    fi
    echo "Prefill profiling: installing Nsight Systems CLI for Ubuntu \$VERSION_ID / \$arch." >&2
    # NVIDIA's devtools repository provides both the Arm SBSA and x86 CLI.
    # Install only in this worker's container; existing nsys installations are reused.
    if ! (
        export DEBIAN_FRONTEND=noninteractive
        apt-get update &&
        apt-get install -y --no-install-recommends wget gnupg ca-certificates &&
        mkdir -p /usr/share/keyrings &&
        wget -qO- --timeout=30 --tries=3 \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
            | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-devtools-keyring.gpg &&
        printf '%s\\n' "deb [signed-by=/usr/share/keyrings/nvidia-devtools-keyring.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu\${VERSION_ID//./}/\$arch/ /" \
            | tee /etc/apt/sources.list.d/nvidia-devtools.list >/dev/null &&
        apt-get update &&
        apt-get install -y --no-install-recommends nsight-systems-cli curl
    ); then
        echo "ERROR: Nsight Systems installation failed; check container write permissions and access to the Ubuntu/NVIDIA package repositories." >&2
        return 1
    fi
    hash -r
    if ! command -v nsys >/dev/null 2>&1; then
        echo "ERROR: Nsight Systems installation completed but nsys is still missing from PATH." >&2
        return 1
    fi
    nsys --version
}

serve_prefill() {
    local profile_port=\$1
    shift
    if [[ "\${PROFILE_PREFILL:-0}" != 1 ]]; then
        vllm serve "\$@"
        return
    fi
    local steps="\${PREFILL_PROFILE_STEPS:-20}"
    if [[ ! "\$steps" =~ ^[1-9][0-9]*\$ ]]; then
        echo "ERROR: PREFILL_PROFILE_STEPS must be a positive integer." >&2
        return 1
    fi
    local profile_delay="\${PREFILL_PROFILE_DELAY_SECONDS:-300}"
    if [[ "\$profile_delay" != manual && ! "\$profile_delay" =~ ^[0-9]+\$ ]]; then
        echo "ERROR: PREFILL_PROFILE_DELAY_SECONDS must be a nonnegative integer or manual." >&2
        return 1
    fi
    local api_worker=1 arg
    for arg in "\$@"; do
        if [[ "\$arg" == --headless ]]; then api_worker=0; fi
    done
    ensure_nsys || return 1
    if [[ "\$profile_delay" != manual ]] && (( api_worker )) && ! command -v curl >/dev/null 2>&1; then
        echo "ERROR: Automatic prefill profiling requires curl in the vLLM container." >&2
        return 1
    fi
    local profile_dir="\${PREFILL_PROFILE_DIR:-\${SLURM_SUBMIT_DIR:-\$PWD}/results/nsys}"
    mkdir -p "\$profile_dir"
    # Background coupled heads and independent engines own their profiler/trigger.
    # The parent shell owns router and Mooncake cleanup.
    if (( BASH_SUBSHELL > 0 )); then
        trap cleanup_prefill_profile EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM
    fi
    # Trace graph nodes and child TP workers. 'stop' ends collection without
    # shutting down the serving process; this captures one window per launch.
    VLLM_WORKER_MULTIPROC_METHOD=spawn nsys profile \
        --trace=cuda,nvtx,osrt,nccl,cublas,cublas-verbose \
        --nccl-trace=all --pytorch=functions-trace-shapes \
        --sample=process-tree --cpuctxsw=none \
        --cudabacktrace=all --python-backtrace=cuda --python-sampling=true \
        --trace-fork-before-exec=true --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --output "\$profile_dir/prefill-\$SLURM_JOB_ID-\$this_node_hostname-port\$profile_port-%p" \
        vllm serve "\$@" \
        --profiler-config "{\"profiler\":\"cuda\",\"max_iterations\":\$steps}" &
    profiler_pid=\$!
    # Coupled headless ranks are profiled through their tier's API head.
    if [[ "\$profile_delay" != manual ]] && (( api_worker )); then
        trigger_prefill_profile &
        trigger_pid=\$!
    fi
    local status=0
    wait "\$profiler_pid" || status=\$?
    profiler_pid=""
    cleanup_prefill_profile
    return "\$status"
}

mooncake_pid=""
mooncake_owner_pid=""
mooncake_disk_dir=""
cleanup_worker_services() {
    cleanup_prefill_profile
    if [[ -n "\$mooncake_owner_pid" ]]; then
        kill "\$mooncake_owner_pid" 2>/dev/null || true
        wait "\$mooncake_owner_pid" 2>/dev/null || true
        mooncake_owner_pid=""
    fi
    if [[ -n "\$mooncake_disk_dir" ]]; then
        rm -rf -- "\$mooncake_disk_dir"
        mooncake_disk_dir=""
    fi
    if [[ -n "\$mooncake_pid" ]]; then
        kill "\$mooncake_pid" 2>/dev/null || true
        wait "\$mooncake_pid" 2>/dev/null || true
    fi
}

# Cover setup failures and headless workers before serving-mode traps take over.
trap cleanup_worker_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if (( ENABLE_MOONCAKE )); then
    export MOONCAKE_CONFIG_PATH="\${MOONCAKE_CONFIG_PATH:-/tmp/mooncake-\$SLURM_JOB_ID/config.json}"
    mkdir -p "\$(dirname "\$MOONCAKE_CONFIG_PATH")"
    cat > "\$MOONCAKE_CONFIG_PATH" <<MOONCAKE_CONFIG
{
  "mode": "$mooncake_mode",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "\${nodes[0]}:50051",
  "global_segment_size": $mooncake_global_segment_size,
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "mlx5_0,mlx5_1,mlx5_3,mlx5_4",
  "enable_offload": $mooncake_enable_offload
}
MOONCAKE_CONFIG

    # @bxyu-nvidia: Specific to OCI-HSG, as with the device_names above. Otherwise we get errors like:
    # W0924 18:25:48.906106 1830563 worker_pool.cpp:477] Worker: Cannot make connection for endpoint: 10.109.24.46:15741@mlx5_2, pausing peer rail and retrying through an alternate peer RNIC
    export MC_TE_FILTERS=mlx5_0,mlx5_1,mlx5_3,mlx5_4

    # custom_mooncake_version.sh bakes the fork and its native dependencies into
    # the image. The PyPI distribution owns the same files and would overwrite it.
    mooncake_manifest=/opt/super-vl-evals/mooncake-build.env
    if [[ -f "\$mooncake_manifest" ]]; then
        echo "Using image-local Mooncake build:"
        cat "\$mooncake_manifest"
    else
        uv pip install --system mooncake-transfer-engine-cuda13
    fi
    python3 -c 'from mooncake.engine import TransferEngine; from mooncake.store import MooncakeDistributedStore'

    # @bxyu-nvidia: Need these for mooncake connector on GB200 https://github.com/vllm-project/vllm/blob/main/docs/features/mooncake_connector_usage.md#environment-variables
    export WITH_NVIDIA_PEERMEM=0

    if (( SLURM_PROCID == 0 )); then
        # Removal summaries use LOG(INFO), including without disk offload.
        mooncake_log_level=$mooncake_minloglevel
        if [[ "\${MOONCAKE_CACHE_TRACE:-0}" == 1 ]]; then
            mooncake_log_level=0
        fi
        echo "Starting mooncake_master on \${nodes[0]}"
        mooncake_master \
            -rpc_port=50051 \
            -enable_offload=$mooncake_enable_offload \
            -rpc_thread_num=4 \
            -metrics_port=9003 \
            -eviction_high_watermark_ratio=$mooncake_eviction_high_watermark \
            -eviction_ratio=0.1 \
            -minloglevel=\$mooncake_log_level \
            -enable_metric_reporting=$mooncake_metric_reporting \
            -default_kv_lease_ttl=$MOONCAKE_DEFAULT_KV_LEASE_TTL \
            -logtostderr &
        mooncake_pid=\$!
    fi

    mooncake_health_url="http://\${nodes[0]}:9003/health"
    mooncake_deadline=\$(( SECONDS + 120 ))
    echo "Waiting for Mooncake master at \$mooncake_health_url"
    until curl --fail --silent --noproxy '*' --connect-timeout 2 --max-time 5 \
        --output /dev/null "\$mooncake_health_url"; do
        if [[ -n "\$mooncake_pid" ]] && ! kill -0 "\$mooncake_pid" 2>/dev/null; then
            echo "mooncake_master exited during startup" >&2
            exit 1
        fi
        if (( SECONDS >= mooncake_deadline )); then
            echo "Timed out waiting for Mooncake master at \$mooncake_health_url" >&2
            exit 1
        fi
        sleep 1
    done
    echo "Mooncake master is ready"

    if (( $ENABLE_MOONCAKE_DISK_OFFLOAD )); then
        # Fail before allocating the owner pool if the bind mount is not disk-backed.
        disk_fs=\$(findmnt -n -T /mooncake-ssd -o FSTYPE)
        if [[ "\$disk_fs" =~ ^(tmpfs|ramfs|overlay|nfs|nfs4|lustre)\$ ]]; then
            echo "ERROR: /mooncake-ssd must be node-local disk storage; found \$disk_fs." >&2
            exit 1
        fi
        disk_quota_bytes=\$(($MOONCAKE_DISK_QUOTA_GIB * 1024 * 1024 * 1024))
        python3 - "\$disk_quota_bytes" <<'DISK_CHECK'
import shutil
import sys
free = shutil.disk_usage('/mooncake-ssd').free
quota = int(sys.argv[1])
if quota > free:
    raise SystemExit(f'Mooncake SSD quota {quota} bytes exceeds free space {free}; lower MOONCAKE_DISK_QUOTA_GIB')
DISK_CHECK
        mooncake_disk_dir=\$(mktemp -d "/mooncake-ssd/mooncake-\$SLURM_JOB_ID-\${SLURM_RESTART_COUNT:-0}.XXXXXX")
        export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="\$mooncake_disk_dir"
        export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend
        # Drain pending SSD persistence promptly so RAM replicas become evictable.
        export MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS=1
        export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
        export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE="\$disk_quota_bytes"
        export MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES="\$disk_quota_bytes"
        export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES="\$disk_quota_bytes"
        # Match the connector's 1280 MiB DirectIO staging budget.
        export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=1342177280
        owner_ip=\$(python3 -c 'from vllm.utils.network_utils import get_ip; print(get_ip())')
        # The preferred segment is the TransferEngine address, NOT the client RPC port.
        export MOONCAKE_PREFERRED_SEGMENT="\$owner_ip:50053"
        echo "Starting Mooncake SSD owner on \$this_node_hostname: segment=\$MOONCAKE_PREFERRED_SEGMENT, CPU=$MOONCAKE_CPU_CACHE_GIB GiB, disk=\$mooncake_disk_dir, quota=\$disk_quota_bytes bytes"
        mooncake_client \\
            --host="\$MOONCAKE_PREFERRED_SEGMENT" \\
            --port=50052 \\
            --master_server_address="\${nodes[0]}:50051" \\
            --metadata_server=P2PHANDSHAKE \\
            --protocol=rdma \\
            --device_names=mlx5_0,mlx5_1,mlx5_3,mlx5_4 \\
            --global_segment_size=$((MOONCAKE_CPU_CACHE_GIB * 1024 * 1024 * 1024)) \\
            --enable_offload=true \\
            --threads=4 \\
            --logtostderr &
        mooncake_owner_pid=\$!
        # The client RPC server starts only after pool/SSD initialization succeeds.
        python3 - "\$owner_ip" "\$mooncake_owner_pid" <<'OWNER_READY'
import os
import socket
import sys
import time
host, pid = sys.argv[1], int(sys.argv[2])
deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit('Mooncake SSD owner exited during startup; inspect its logs above')
    try:
        with socket.create_connection((host, 50052), timeout=1):
            break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit('Mooncake SSD owner did not become ready within 180 seconds')
OWNER_READY
        kill -0 "\$mooncake_owner_pid"
        echo "Mooncake SSD owner is ready on \$this_node_hostname"
    fi
fi

router_common_args=(--log-level error --prometheus-host 0.0.0.0 --prometheus-port $ROUTER_METRICS_PORT)
read -r -a router_extra_args <<< $(printf '%q' "${ROUTER_ARGS:-}")
if (( \${#router_extra_args[@]} )); then
    router_common_args+=("\${router_extra_args[@]}")
fi

if [[ "$VLLM_MODE" == pd && "$VLLM_PD_DEPLOYMENT_MODE" == coupled ]]; then
    PREFILL_HEAD=\${nodes[0]}
    DECODE_HEAD=\${nodes[$NUM_PREFILL_NODES]}

    set_headless_args() {
        # Model configs tune API processes for serving ranks. Headless ranks
        # reject this option, whether supplied as two arguments or with '='.
        headless_args=()
        while (( \$# )); do
            if [[ "\$1" == --api-server-count ]]; then
                shift 2
            elif [[ "\$1" == --api-server-count=* ]]; then
                shift
            else
                headless_args+=("\$1")
                shift
            fi
        done
    }

    if (( SLURM_PROCID == 0 )); then
        # The first prefill rank owns its tier's API server. The remaining
        # prefill ranks run headless so expert parallelism spans the tier.
        VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
        VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_VLLM_NIXL_SIDE_CHANNEL_PORT \
        serve_prefill $WORKER_SERVER_PORT "$MODEL" --served-model-name "$MODEL_NAME" "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_PREFILL_ARGS[@]}" \
            --host \$this_node_hostname \
            --port $WORKER_SERVER_PORT \
            --data-parallel-size $NUM_PREFILL_NODES \
            --data-parallel-address \$PREFILL_HEAD \
            --data-parallel-rpc-port $PREFILL_DP_RPC_PORT \
            --api-server-count 1 \
            &
        prefill_pid=\$!
        coupled_pids=("\$prefill_pid")
        cleanup_coupled_head() {
            local status=\$?
            trap - EXIT INT TERM
            # Signal both local services without delaying failure propagation;
            # the enclosing srun tears down the remaining distributed workers.
            kill "\${coupled_pids[@]}" 2>/dev/null || true
            cleanup_worker_services
            exit "\$status"
        }
        trap cleanup_coupled_head EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM

        vllm-router \
            --prefill-policy $ROUTER_PREFILL_POLICY \
            --decode-policy $ROUTER_DECODE_POLICY \
            --vllm-pd-disaggregation \
            --prefill "http://\$PREFILL_HEAD:$WORKER_SERVER_PORT" \
            --decode "http://\$DECODE_HEAD:$WORKER_SERVER_PORT" \
            --host \$PREFILL_HEAD \
            --port $ROUTER_SERVER_PORT \
            --intra-node-data-parallel-size $ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE \
            --request-timeout-secs 86400 \
            --worker-startup-timeout-secs "$ROUTER_WORKER_STARTUP_TIMEOUT_SECONDS" \
            "\${router_common_args[@]}" &
        router_pid=\$!
        coupled_pids+=("\$router_pid")

        # Monitor both services after readiness. Polling also catches children that
        # exited before monitoring started, which wait -n can otherwise miss.
        while kill -0 "\$prefill_pid" 2>/dev/null && kill -0 "\$router_pid" 2>/dev/null; do
            sleep 1
        done
        failed_role=prefill
        failed_pid=\$prefill_pid
        if kill -0 "\$prefill_pid" 2>/dev/null; then
            failed_role=router
            failed_pid=\$router_pid
        fi
        failed_status=0
        wait "\$failed_pid" || failed_status=\$?
        # Neither service should exit by itself, even with a zero exit status.
        (( failed_status != 0 )) || failed_status=1
        echo "ERROR: \$failed_role process exited after startup (status=\$failed_status)." >&2
        exit "\$failed_status"
    elif (( SLURM_PROCID < $NUM_PREFILL_NODES )); then
        set_headless_args "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_PREFILL_ARGS[@]}"
        VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
        VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_VLLM_NIXL_SIDE_CHANNEL_PORT \
        serve_prefill $WORKER_SERVER_PORT "$MODEL" --served-model-name "$MODEL_NAME" "\${headless_args[@]}" \
            --headless \
            --data-parallel-size $NUM_PREFILL_NODES \
            --data-parallel-start-rank \$SLURM_PROCID \
            --data-parallel-address \$PREFILL_HEAD \
            --data-parallel-rpc-port $PREFILL_DP_RPC_PORT
    elif (( SLURM_PROCID == $NUM_PREFILL_NODES )); then
        # Decode mirrors prefill with one API rank and headless ranks across
        # the other decode nodes.
        VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
        VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_VLLM_NIXL_SIDE_CHANNEL_PORT \
        vllm serve "$MODEL" --served-model-name "$MODEL_NAME" "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_DECODE_ARGS[@]}" \
            --host \$this_node_hostname \
            --port $WORKER_SERVER_PORT \
            --data-parallel-size $NUM_DECODE_NODES \
            --data-parallel-address \$DECODE_HEAD \
            --data-parallel-rpc-port $DECODE_DP_RPC_PORT \
            --api-server-count 1
    else
        set_headless_args "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_DECODE_ARGS[@]}"
        VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
        VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_VLLM_NIXL_SIDE_CHANNEL_PORT \
        vllm serve "$MODEL" --served-model-name "$MODEL_NAME" "\${headless_args[@]}" \
            --headless \
            --data-parallel-size $NUM_DECODE_NODES \
            --data-parallel-start-rank \$(( SLURM_PROCID - $NUM_PREFILL_NODES )) \
            --data-parallel-address \$DECODE_HEAD \
            --data-parallel-rpc-port $DECODE_DP_RPC_PORT
    fi
else
    router_pid=""
    worker_pids=()

    cleanup_vllm() {
        local status=\$? child_pid
        local -a child_pids=()
        trap - EXIT INT TERM
        # A signal can arrive after a child starts but before its PID is recorded.
        # Bash's job table includes that child, as well as the router and Mooncake.
        while IFS= read -r child_pid; do
            child_pids+=("\$child_pid")
        done < <(jobs -p)
        if (( \${#child_pids[@]} )); then
            kill "\${child_pids[@]}" 2>/dev/null || true
            wait "\${child_pids[@]}" 2>/dev/null || true
        fi
        cleanup_worker_services
        exit "\$status"
    }
    trap cleanup_vllm EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    serve_independent_engines() {
        local nixl_base_port=\$1
        shift
        local arg skip_value engine_index worker_port service_pid failed_status gpu_start gpu_mask
        local tensor_parallel_size=$(( 4 / VLLM_ENGINES_PER_NODE ))
        local -a visible_gpus engine_args service_pids serve_command
        # Respect the allocation's GPU order, including UUID-based masks.
        IFS=',' read -r -a visible_gpus <<< "\${CUDA_VISIBLE_DEVICES-0,1,2,3}"
        if (( \${#visible_gpus[@]} < 4 )); then
            echo "ERROR: $VLLM_ENGINES_PER_NODE independent engines require at least four visible GPUs." >&2
            exit 1
        fi

        # Model configs may specify TP/PP/DP for a full-node replica. Replace
        # those settings in both CLI forms, retaining unrelated model tuning.
        engine_args=()
        skip_value=0
        for arg in "\$@"; do
            if (( skip_value )); then
                skip_value=0
                continue
            fi
            case "\$arg" in
                (--tensor-parallel-size|--pipeline-parallel-size|--data-parallel-size|--data-parallel-size-local|--api-server-count|-tp|-pp|-dp)
                    skip_value=1 ;;
                (--tensor-parallel-size=*|--pipeline-parallel-size=*|--data-parallel-size=*|--data-parallel-size-local=*|--api-server-count=*|-tp=*|-pp=*|-dp=*)
                    ;;
                (*) engine_args+=("\$arg") ;;
            esac
        done
        engine_args+=(--tensor-parallel-size "\$tensor_parallel_size" --pipeline-parallel-size 1
            --data-parallel-size 1 --data-parallel-size-local 1 --api-server-count 1)
        # Install once before concurrent engines can race on the package manager.
        if [[ "$VLLM_MODE" == pd && "\${PROFILE_PREFILL:-0}" == 1 ]] && \
            (( SLURM_PROCID < $NUM_PREFILL_NODES )); then
            ensure_nsys || return 1
        fi
        for (( engine_index = 0; engine_index < $VLLM_ENGINES_PER_NODE; engine_index++ )); do
            worker_port=\$(( $WORKER_SERVER_PORT + engine_index ))
            serve_command=(vllm serve)
            if [[ "$VLLM_MODE" == pd ]] && (( SLURM_PROCID < $NUM_PREFILL_NODES )); then
                serve_command=(serve_prefill "\$worker_port")
            fi
            gpu_start=\$(( engine_index * tensor_parallel_size ))
            printf -v gpu_mask '%s,' "\${visible_gpus[@]:gpu_start:tensor_parallel_size}"
            gpu_mask=\${gpu_mask%,}
            echo "Starting TP\$tensor_parallel_size engine \$engine_index on GPUs \$gpu_mask, port \$worker_port"
            CUDA_VISIBLE_DEVICES="\$gpu_mask" \
            VLLM_NIXL_SIDE_CHANNEL_HOST="\$this_node_hostname" \
            VLLM_NIXL_SIDE_CHANNEL_PORT=\$(( nixl_base_port + engine_index )) \
            "\${serve_command[@]}" "$MODEL" --served-model-name "$MODEL_NAME" "\${engine_args[@]}" \
                --host "\$this_node_hostname" --port "\$worker_port" &
            worker_pids+=("\$!")
        done

        # Detect even an immediate successful exit, and stop all local peers.
        service_pids=("\${worker_pids[@]}")
        if [[ -n "\$router_pid" ]]; then
            service_pids+=("\$router_pid")
        fi
        while true; do
            for service_pid in "\${service_pids[@]}"; do
                if ! kill -0 "\$service_pid" 2>/dev/null; then
                    failed_status=0
                    wait "\$service_pid" || failed_status=\$?
                    (( failed_status != 0 )) || failed_status=1
                    echo "ERROR: Independent engine or router process \$service_pid exited (status=\$failed_status)." >&2
                    exit "\$failed_status"
                fi
            done
            sleep 1
        done
    }

    if (( SLURM_PROCID == 0 )); then

        # Set a super long request timeout since some reasoning requests may take a long time to generate.
        # Don't manually wait as vllm-router will wait for the URLs to come up
        # Set a longer worker startup timeout since some models e.g. DSv4 take > 10 mins to load.
        router_args=( \
            --host \$this_node_hostname \
            --port $ROUTER_SERVER_PORT \
            --intra-node-data-parallel-size $ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE \
            --request-timeout-secs 86400 \
            --worker-startup-timeout-secs "$ROUTER_WORKER_STARTUP_TIMEOUT_SECONDS" \
            "\${router_common_args[@]}"
        )

        if [[ "$VLLM_MODE" == pd ]]; then
            router_args+=( \
                --prefill-policy $ROUTER_PREFILL_POLICY \
                --decode-policy $ROUTER_DECODE_POLICY \
                --vllm-pd-disaggregation
            )
            for (( i = 0; i < $NUM_PREFILL_NODES; i++ )); do
                for (( engine_index = 0; engine_index < $VLLM_ENGINES_PER_NODE; engine_index++ )); do
                    worker_port=\$(( $WORKER_SERVER_PORT + engine_index ))
                    router_args+=(--prefill "http://\${nodes[i]}:\$worker_port")
                done
            done
            for (( i = 0; i < $NUM_DECODE_NODES; i++ )); do
                node_idx=\$(( $NUM_PREFILL_NODES + i ))
                for (( engine_index = 0; engine_index < $VLLM_ENGINES_PER_NODE; engine_index++ )); do
                    worker_port=\$(( $WORKER_SERVER_PORT + engine_index ))
                    router_args+=(--decode "http://\${nodes[node_idx]}:\$worker_port")
                done
            done
        else
            router_args+=(--policy $ROUTER_POLICY --worker-urls)
            for node in "\${nodes[@]}"; do
                for (( engine_index = 0; engine_index < $VLLM_ENGINES_PER_NODE; engine_index++ )); do
                    worker_port=\$(( $WORKER_SERVER_PORT + engine_index ))
                    router_args+=("http://\$node:\$worker_port")
                done
            done
        fi

        vllm-router "\${router_args[@]}" &

        router_pid=\$!

        sleep 5
        if ! kill -0 "\$router_pid" 2>/dev/null; then
            echo "vllm-router exited during startup" >&2
            exit 1
        fi
    fi

    if [[ "$VLLM_MODE" == aggregated ]]; then
        # Reuse prefill tuning by default. Configs can supply VLLM_AGGREGATED_ARGS
        # instead (including an empty array) for independent tuning.
        if declare -p VLLM_AGGREGATED_ARGS &>/dev/null; then
            worker_args=("\${VLLM_COMMON_ARGS[@]}" "\${VLLM_AGGREGATED_ARGS[@]}")
        else
            worker_args=("\${VLLM_COMMON_ARGS[@]}" "\${VLLM_PREFILL_ARGS[@]}")
        fi
        # A complete replica must not wait for KV transfers from a prefill worker.
        # Strip both CLI forms, including connectors set in VLLM_COMMON_ARGS.
        aggregated_args=()
        skip_value=0
        for arg in "\${worker_args[@]}"; do
            if (( skip_value )); then
                skip_value=0
                continue
            fi
            if [[ "\$arg" == --kv-transfer-config ]]; then
                skip_value=1
            elif [[ "\$arg" != --kv-transfer-config=* ]]; then
                aggregated_args+=("\$arg")
            fi
        done
        if (( $VLLM_ENGINES_PER_NODE > 1 )); then
            serve_independent_engines $PREFILL_VLLM_NIXL_SIDE_CHANNEL_PORT "\${aggregated_args[@]}"
        else
            vllm serve "$MODEL" --served-model-name "$MODEL_NAME" "\${aggregated_args[@]}" \
                --host \$this_node_hostname \
                --port $WORKER_SERVER_PORT
        fi
    elif (( $VLLM_ENGINES_PER_NODE > 1 )); then
        if (( SLURM_PROCID < $NUM_PREFILL_NODES )); then
            serve_independent_engines $PREFILL_VLLM_NIXL_SIDE_CHANNEL_PORT \
                "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_PREFILL_ARGS[@]}"
        else
            serve_independent_engines $DECODE_VLLM_NIXL_SIDE_CHANNEL_PORT \
                "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_DECODE_ARGS[@]}"
        fi
    elif (( SLURM_PROCID < $NUM_PREFILL_NODES )); then
        # Prefill
        VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
        VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_VLLM_NIXL_SIDE_CHANNEL_PORT \
        serve_prefill $WORKER_SERVER_PORT "$MODEL" --served-model-name "$MODEL_NAME" "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_PREFILL_ARGS[@]}" \
            --host \$this_node_hostname \
            --port $WORKER_SERVER_PORT
    else
        # Decode
        VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
        VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_VLLM_NIXL_SIDE_CHANNEL_PORT \
        vllm serve "$MODEL" --served-model-name "$MODEL_NAME" "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_DECODE_ARGS[@]}" \
            --host \$this_node_hostname \
            --port $WORKER_SERVER_PORT
    fi
fi
EOF
)

batch_command=$(cat <<EOF
set -euo pipefail

nodes=(\$(scontrol show hostnames "\$SLURM_JOB_NODELIST"))
# Pyxis ignores --container-image when a named container already exists.
# Scope names to a job attempt so a new warmed image is actually imported.
container_attempt="\$SLURM_JOB_ID-\${SLURM_RESTART_COUNT:-0}"
worker_container="vllm-\$container_attempt"
if [[ -n "$SAVE_WARMED_CONTAINER" ]]; then
    worker_container="vllm-warm-\$container_attempt"
fi
echo "Starting vLLM image $CONTAINER in container \$worker_container"

ALL_NODES="\${nodes[*]}" \
srun --nodes=$NUM_NODES --ntasks=$NUM_NODES --ntasks-per-node=1 --kill-on-bad-exit=1 \
    --container-image=$CONTAINER \
    --container-name="\$worker_container" \
    --container-writable \
    --container-mounts=$worker_mounts \
    --container-workdir=\$SLURM_SUBMIT_DIR \
    --no-container-mount-home \
    bash -c '
        set -euo pipefail
        cd "\$SLURM_SUBMIT_DIR"
        exec "\$@"
    ' bash bash -c "\$vllm_command" &
server_step=\$!
cache_warmup_step=""
cache_export_dir=""

cleanup_server() {
    job_status=\$?
    trap - EXIT INT TERM
    set +e
    if [[ -n "\$cache_warmup_step" ]]; then
        kill "\$cache_warmup_step" 2>/dev/null || true
        wait "\$cache_warmup_step" 2>/dev/null || true
    fi
    if [[ -n "\$server_step" ]]; then
        kill "\$server_step" 2>/dev/null || true
        wait "\$server_step" 2>/dev/null || true
    fi
    if [[ -n "\$cache_export_dir" ]]; then
        rm -rf -- "\$cache_export_dir"
    fi
    exit "\$job_status"
}
trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -n "$SAVE_WARMED_CONTAINER" ]]; then
    cache_urls=()
    for (( node_index = 0; node_index < \${#nodes[@]}; node_index++ )); do
        if [[ "$VLLM_MODE" == pd && "$VLLM_PD_DEPLOYMENT_MODE" == coupled ]] && \
            (( node_index != 0 && node_index != $NUM_PREFILL_NODES )); then
            continue
        fi
        for (( engine_index = 0; engine_index < $VLLM_ENGINES_PER_NODE; engine_index++ )); do
            cache_urls+=("http://\${nodes[node_index]}:\$(( $WORKER_SERVER_PORT + engine_index ))")
        done
    done
    python3 -c "\$warm_cache_helper" wait \
        --timeout "$ROUTER_WORKER_STARTUP_TIMEOUT_SECONDS" "\${cache_urls[@]}" &
    cache_warmup_step=\$!
    completed_pid=""
    completed_status=0
    wait -n -p completed_pid "\$server_step" "\$cache_warmup_step" || completed_status=\$?
    if [[ "\$completed_pid" == "\$server_step" ]]; then
        echo "ERROR: vLLM exited before cache warmup completed (status=\$completed_status)." >&2
        (( completed_status != 0 )) || completed_status=1
    fi
    (( completed_status == 0 )) || exit "\$completed_status"
    cache_warmup_step=""

    # Write archives directly to the shared export directory. Mount only this
    # temporary transfer directory; runtime caches stay inside each container.
    # Include both PD roles and headless workers' compiled modules/startup plans.
    cache_export_dir=\$(mktemp -d "$(dirname "$SAVE_WARMED_CONTAINER")/.vllm-warm.XXXXXX")
    for (( node_index = 0; node_index < \${#nodes[@]}; node_index++ )); do
        dump_args=(--output "\$cache_export_dir/caches.tar")
        if [[ "$VLLM_MODE" == pd && "$VLLM_PD_DEPLOYMENT_MODE" == coupled ]] && \
            (( node_index != 0 && node_index != $NUM_PREFILL_NODES )); then
            dump_args+=(--allow-empty)
        fi
        srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus=0 \
            --nodelist="\${nodes[node_index]}" --container-name="\$worker_container" \
            --container-writable --no-container-mount-home --container-workdir=/ \
            --container-mounts="\$cache_export_dir:\$cache_export_dir" \
            python3 -c "\$warm_cache_helper" dump "\${dump_args[@]}"
    done

    # Stop the serving step before exporting. The named rootfs persists for the
    # final single-node step; do not set --container-save on the multi-node step.
    kill "\$server_step"
    wait "\$server_step" || true
    server_step=""
    # A GPU is needed to verify that the installed modules load with JIT disabled.
    srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus=1 \
        --nodelist="\${nodes[0]}" --container-name="\$worker_container" \
        --container-writable --no-container-mount-home --container-workdir=/ \
        --container-mounts="\$cache_export_dir:\$cache_export_dir" \
        --container-save="\$cache_export_dir/warmed.sqsh" \
        python3 -c "\$warm_cache_helper" install --input "\$cache_export_dir/caches.tar"
    test -s "\$cache_export_dir/warmed.sqsh"
    # Verify the exported image in a fresh, unnamed container. Checking the
    # mutable warmup rootfs alone cannot prove the squashfs contains the caches.
    srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus=1 \
        --nodelist="\${nodes[0]}" --container-image="\$cache_export_dir/warmed.sqsh" \
        --no-container-mount-home --container-workdir=/ \
        python3 -c "\$warm_cache_helper" verify
    # Publish only a successful export; a concurrent job must not overwrite it.
    ln "\$cache_export_dir/warmed.sqsh" "$SAVE_WARMED_CONTAINER"
    echo "Saved warmed vLLM image: $SAVE_WARMED_CONTAINER"
    exit 0
fi

if (( $should_run_eval )); then
    # No need to wait for endpoint since Gym will wait for model endpoints to spin up before proceeding.

    # @bxyu-nvidia: Put the Gym servers on a separate node from the one running vllm-router.
    # This helps relieve so much network traffic on one node.
    if [[ -v 'nodes[1]' ]]; then
        EVAL_NODE=\${nodes[1]}
    else
        EVAL_NODE=\${nodes[0]}
    fi

    # @bxyu-nvidia: We need --cpus-per-task=SLURM_CPUS_ON_NODE, otherwise we run into a lot of ServerDisconnectedError and ConnectionResetByPeer errors from Gym servers and vLLM. Not sure what the correlation is
    ROUTER_NODE="\${nodes[0]}" \
    ALL_NODES="\${nodes[*]}" \
    srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=\$SLURM_CPUS_ON_NODE --nodelist="\$EVAL_NODE" --gpus=0 \
        --container-image=$CONTAINER \
        --container-name="eval-\$container_attempt" \
        --container-mounts=$MOUNTS \
        --container-workdir="\$SLURM_SUBMIT_DIR" \
        --no-container-mount-home \
        bash -c '
            set -euo pipefail
            cd "\$SLURM_SUBMIT_DIR"
            exec bash -c "\$eval_command"
        ' &
    eval_step=\$!

    completed_pid=""
    completed_status=0
    wait -n -p completed_pid "\$server_step" "\$eval_step" || completed_status=\$?

    if [[ "\$completed_pid" == "\$server_step" ]]; then
        if (( completed_status == 0 )); then
            completed_status=1
        fi
        echo "vLLM server step exited unexpectedly with status \$completed_status" >&2
        kill "\$eval_step" 2>/dev/null || true
        wait "\$eval_step" 2>/dev/null || true
        exit "\$completed_status"
    fi

    exit "\$completed_status"
fi

wait "\$server_step"
EOF
)

# --segment > 0 otherwise the engine will hang on the second or third engine step.
SEGMENT=${SEGMENT:-$NUM_NODES}

submit_dir=$(pwd -P)
# An exported connection is sent as arguments; otherwise env.yaml is read.
if [[ -n "$OPENSANDBOX_DOMAIN" ]]; then
    cleanup_connection=(--domain "$OPENSANDBOX_DOMAIN" --api-key "$OPENSANDBOX_API_KEY" --protocol "$OPENSANDBOX_PROTOCOL")
else
    cleanup_connection=(--connection-config "$submit_dir/env.yaml")
fi
cleanup_user=${NEMO_GYM_USER:-$USER}
mooncake_memory_arg=""
if (( ENABLE_MOONCAKE )); then
    mooncake_memory_arg="--mem=0"
fi
main_job_id=$(
    NEMO_GYM_USER="$cleanup_user" \
    vllm_command="$vllm_command" \
    eval_command="$eval_command" \
    batch_command="$batch_command" \
    warm_cache_helper="$warm_cache_helper" \
    sbatch \
        --parsable \
        --nodes=$NUM_NODES \
        --time="${SBATCH_TIMELIMIT:-04:00:00}" \
        --job-name=gym-$EXPERIMENT_NAME-$USER \
        --output=slurm-logs/%j-%x.log \
        --ntasks-per-node=1 \
        --comment="$SLURM_COMMENT" \
        --exclusive \
        ${mooncake_memory_arg:+"$mooncake_memory_arg"} \
        --segment=$SEGMENT \
        ${NODELIST:+--nodelist="$NODELIST"} \
        --wrap 'exec bash -c "$batch_command"'
)
main_job_id=${main_job_id%%;*}

if (( should_run_eval )); then
    # @bxyu-nvidia: Don't run cleanup job in reservation
    unset SBATCH_RESERVATION
    if ! cleanup_job_id=$(
        sbatch \
            --parsable \
            --dependency=afterany:"$main_job_id" \
            --partition=cpu \
            --qos=cpu-normal \
            --gres=none \
            --gpus-per-node=0 \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=1 \
            --mem=256M \
            --time=00:30:00 \
            --job-name="gym-cleanup-$main_job_id" \
            --output="$submit_dir/slurm-logs/%j-gym-cleanup-$main_job_id.log" \
            "$submit_dir/nemo_gym/sandbox/providers/opensandbox/cleanup_sandboxes.py" \
            "${cleanup_connection[@]}" \
            --run-id "$main_job_id" \
            --user "$cleanup_user" \
            --reap
    ); then
        echo "Submitted batch job $main_job_id"
        echo "Failed to submit the sandbox-cleanup job for batch job $main_job_id;" \
            "it is running and its sandboxes will need reaping by hand" >&2
        exit 0
    fi
    cleanup_job_id=${cleanup_job_id%%;*}
fi

echo "Submitted batch job $main_job_id"
if (( should_run_eval )); then
    echo "Submitted cleanup job $cleanup_job_id for batch job $main_job_id"
fi
