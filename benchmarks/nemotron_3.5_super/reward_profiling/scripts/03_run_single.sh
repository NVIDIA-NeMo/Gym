#!/bin/bash
# 03 - Reward profiling in one Slurm job: a vLLM prefill/decode endpoint plus the Gym sweep driver.
#
# Forked from ../sbatch_external_vllm.sh; the vLLM half is unchanged, the eval half runs a sweep
# instead of a benchmark split. Run 01_materialize.sh first to produce SWEEP_DIR. Profiles inside
# the job when collection finishes, so a completed job leaves a profile behind; use 05_profile.sh
# to profile a run that was killed, or to re-profile after a merge.
#
# USAGE
#   MODEL=<ckpt> CONTAINER=<eval sqsh> SANDBOX_CONTAINER=<sandbox sqsh> \
#     SWEEP_DIR=<out>/<nickname> bash $R/scripts/03_run_single.sh
#
# REQUIRED
#   SWEEP_DIR         the OUT_DIR/<nickname> directory 01 wrote
#   MODEL             served checkpoint path, also used as policy_model_name
#   CONTAINER         reward-profiling sqsh (../../build_eval_container.sh with SKIP_PREPARE=1)
#                     may come from the manifest's srun block instead, as may SANDBOX_CONTAINER
#                     and MOUNTS; SBATCH_* may come from its sbatch block. An env var set here wins
#
# Does not resubmit itself. For retries on a single job use 03_run_sharded.sh with NUM_SHARDS=1.
#
# OPTIONAL - shape
#   NUM_PREFILL_NODES / NUM_DECODE_NODES   1 / 2. Nodes = P + D, capped ~16 by --segment
#   WALLTIME                               manifest sbatch.timelimit, else 04:00:00
#   SBATCH_*                               anything sbatch reads from the environment. The
#                                          manifest's sbatch block sets these; setting one here
#                                          overrides it. Falls back to nemotron_n4_post / gpu:4 /
#                                          interactive when neither names them
#
# OPTIONAL - sandbox lane (ns_tools, math_formal_lean)
#   SANDBOX_CONTAINER  a nemo-skills sandbox sqsh. Without one those entries read
#                      NEMO_SKILLS_SANDBOX_HOST/PORT, fall back to 127.0.0.1:6000 where nothing
#                      listens, and fail with a bare 500 per rollout rather than at startup
#   SANDBOX_PORT / SANDBOX_WORKERS         6000 / 32
#
# OPTIONAL - throughput
#   NUM_SAMPLES_IN_PARALLEL   512 x decode_nodes, or the manifest's value if it sets one
#   MAX_NUM_SEQS_PER_DECODE_ENGINE         512
#   ALLOW_PARTIAL_ROLLOUTS                 True
#   ENV_PORT_RANGE_LOW / _HIGH             unset; the manifest's gym_env_start sets the range
#   SERVERS_READY_TIMEOUT_S / ENV_START_ATTEMPTS   1800 / 4
#   ENV_YAML / MOUNTS / LOG_DIR            $PWD/env.yaml / /lustre:/lustre / SWEEP_DIR/slurm-logs
#
# The sandbox image runs nginx over N uWSGI workers on ONE node, hashing X-Session-ID so a
# stateful IPython session stays pinned to a worker. It does not span nodes and
# NEMO_SKILLS_SANDBOX_HOST takes a single host, so sandbox capacity is one node's worth --
# measured at 483 rollouts/hr on 32 workers. Scaling past that needs a balancer in front.
set -euo pipefail

# SWEEP_DIR first and on its own: it is the path to the manifest's own output, so unlike MODEL and
# CONTAINER it cannot be supplied by the manifest.
if [[ -z "${SWEEP_DIR:-}" ]]; then
    echo "ERROR: SWEEP_DIR is required. See the header of $0 for the full argument list." >&2
    exit 2
fi

# Apply the manifest's sbatch/srun keys only where this environment leaves them unset, keeping
# precedence at manifest -> env var -> command line. Free-form, so any SBATCH_* works.
while IFS='=' read -r _k _v; do
    [[ -n "$_k" ]] || continue
    [[ -n "${!_k:-}" ]] || export "$_k=$_v"
done < <(python - "$SWEEP_DIR" <<'PY_SBATCH'
import json, sys
from pathlib import Path
try:
    doc = json.loads((Path(sys.argv[1]) / "sweep_report.json").read_text())
except OSError:
    doc = {}
for block in ("sbatch", "srun", "vllm", "gym_eval_profile"):
    for key, value in (doc.get(block) or {}).items():
        print(f"{key}={value}")
PY_SBATCH
)


# Checked after the manifest has had its say, so a manifest that names its own container satisfies
# this rather than tripping it.
for _required in MODEL CONTAINER; do
    if [[ -z "${!_required:-}" ]]; then
        echo "ERROR: $_required is required, and the manifest's srun block does not set it." >&2
        echo "       See the header of $0 for the full argument list." >&2
        exit 2
    fi
done

# One prefill node feeds several decode nodes; decode is the throughput-limiting side. P4D12 is the
# largest configuration tested, and 16 nodes still fits inside one 18-node NVL72 rack, which
# --segment requires.
NUM_PREFILL_NODES=${NUM_PREFILL_NODES:-1}
NUM_DECODE_NODES=${NUM_DECODE_NODES:-2}

VLLM_CONFIG=${VLLM_CONFIG:-benchmarks/nemotron_3.5_super/vllm_configs/nemotron_3.5_super.sh}
MOUNTS=${MOUNTS:-/lustre:/lustre}

# Logs live inside SWEEP_DIR, so a sharded run keeps each shard's job logs separate.
LOG_DIR=${LOG_DIR:-$SWEEP_DIR/slurm-logs}
mkdir -p "$LOG_DIR"

# Empty disables the sandbox sidecar, which is correct for the no-judge and judge lanes.
SANDBOX_CONTAINER=${SANDBOX_CONTAINER:-}
SANDBOX_PORT=${SANDBOX_PORT:-6000}
# One worker per core starves the driver, which shares the node. 32 was measured; raise it when the
# sandbox gets a node to itself.
SANDBOX_WORKERS=${SANDBOX_WORKERS:-32}
if [[ -n "$SANDBOX_CONTAINER" && ! -f "$SANDBOX_CONTAINER" ]]; then
    echo "ERROR: SANDBOX_CONTAINER=$SANDBOX_CONTAINER does not exist." >&2
    exit 2
fi

# Secrets reach Gym through env.yaml, which it auto-loads from its working directory. The judge
# lane needs it: the judge config_overlay interpolates ${nv_inference_api_key}, which env.yaml resolves
# from the shell. Without the mount the config fails to parse rather than failing at judge time.
ENV_YAML=${ENV_YAML:-$PWD/env.yaml}
if [[ -f "$ENV_YAML" ]]; then
    MOUNTS="$MOUNTS,$ENV_YAML:/opt/Gym/env.yaml"
else
    echo "WARNING: no env.yaml at $ENV_YAML; judge environments will fail to resolve their API key." >&2
fi

# Scale concurrency with decode capacity: each decode engine schedules up to max_num_seqs
# sequences (512 in vllm_configs/nemotron_3.5_super.sh), so this saturates the engines instead of
# leaving them idle. A measured run at 128 against D2 sat at 12.5% of capacity, with the client
# semaphore rather than the GPUs as the limit. Stays under the 16k per-host aiohttp connector cap
# set below until roughly D32. Lower it if the driver process becomes the bottleneck.
MAX_NUM_SEQS_PER_DECODE_ENGINE=${MAX_NUM_SEQS_PER_DECODE_ENGINE:-512}
# Settings layer, lowest to highest: manifest defaults -> these env vars -> anything you add on
# the gym command line. Each is unset by default, so nothing here overrides what the manifest
# declared unless you ask for it; if the manifest is silent too, Gym's own default applies.
#
# A ++ override beats a config file, so passing one unconditionally -- which this used to do --
# silently ignores the manifest. Set a variable and it wins; leave it and the manifest wins.
GLOBAL_AIOHTTP_CONNECTOR_LIMIT_PER_HOST=${GLOBAL_AIOHTTP_CONNECTOR_LIMIT_PER_HOST:-}
RUN_PORT_RANGE_LOW=${RUN_PORT_RANGE_LOW:-}
RUN_PORT_RANGE_HIGH=${RUN_PORT_RANGE_HIGH:-}
# gym env start probes for a free port then binds, so 63 servers racing each other can lose the
# port between probe and bind -- job 6563714 died with "lc_judge ... 19730: address already in
# use" after it had reported startup complete. The manifest sets a wide range clear of the Linux
# ephemeral range (32768-60999); set these only to override it for one run.
ENV_PORT_RANGE_LOW=${ENV_PORT_RANGE_LOW:-}
ENV_PORT_RANGE_HIGH=${ENV_PORT_RANGE_HIGH:-}
ALLOW_PARTIAL_ROLLOUTS=${ALLOW_PARTIAL_ROLLOUTS:-True}

# Returns 0 either way: under `set -e` a bare [[ -n "" ]] would abort the script.
_override() {
    if [[ -n "$2" ]]; then printf ' ++%s=%s' "$1" "$2"; fi
    return 0
}

RUN_OVERRIDES=""
RUN_OVERRIDES+=$(_override global_aiohttp_connector_limit_per_host "$GLOBAL_AIOHTTP_CONNECTOR_LIMIT_PER_HOST")
RUN_OVERRIDES+=$(_override port_range_low "$RUN_PORT_RANGE_LOW")
RUN_OVERRIDES+=$(_override port_range_high "$RUN_PORT_RANGE_HIGH")

ENV_OVERRIDES=""
ENV_OVERRIDES+=$(_override port_range_low "$ENV_PORT_RANGE_LOW")
ENV_OVERRIDES+=$(_override port_range_high "$ENV_PORT_RANGE_HIGH")

# Settings the manifest declared for this command, rendered as ++ overrides. An env var of the same
# name in upper case wins, so precedence is manifest -> env var -> anything you add on the command
# line. num_samples_in_parallel is excluded: the launcher computes it from the job's shape below.
MANIFEST_OVERRIDES=$(python - "$SWEEP_DIR" <<'PY_MANIFEST'
import json, os, sys
from pathlib import Path
try:
    doc = json.loads((Path(sys.argv[1]) / "sweep_report.json").read_text())
except OSError:
    doc = {}
out = []
for key, value in (doc.get("gym_eval_run") or {}).items():
    if key == "num_samples_in_parallel":
        continue
    if os.environ.get(key.upper()):
        continue  # an explicit env var overrides the manifest
    out.append(f"++{key}={value}")
print(" ".join(out))
PY_MANIFEST
)

# num_samples_in_parallel is the exception: it depends on the job's shape, which only the launcher
# knows. 512 per decode engine matches max_num_seqs in the vLLM config. An explicit env var still
# wins over the computed value.
_manifest_concurrency=$(python - "$SWEEP_DIR" <<'PY_CONC'
import json, sys
from pathlib import Path
try:
    doc = json.loads((Path(sys.argv[1]) / "sweep_report.json").read_text())
except OSError:
    doc = {}
print((doc.get("gym_eval_run") or {}).get("num_samples_in_parallel", ""))
PY_CONC
)
NUM_SAMPLES_IN_PARALLEL=${NUM_SAMPLES_IN_PARALLEL:-${_manifest_concurrency:-$((MAX_NUM_SEQS_PER_DECODE_ENGINE * NUM_DECODE_NODES))}}

EXPERIMENT_NAME="${EXPERIMENT_NAME:-reward-profiling}"

# Falls back to the manifest's sbatch.timelimit before the hardcoded default. sbatch reads
# SBATCH_TIMELIMIT from the environment, but --time is passed on the command line below and a
# flag beats an env var -- so without this the manifest's timelimit is silently ignored.
WALLTIME=${WALLTIME:-${SBATCH_TIMELIMIT:-04:00:00}}

# sbatch picks these up from the environment. Without an account it refuses the job outright, and
# without a GRES request a non-CPU partition rejects it -- both are hard errors at submit, so they
# get defaults rather than being discovered one failed submission at a time.
export SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-nemotron_n4_post}
export SBATCH_GRES=${SBATCH_GRES:-gpu:4}
# interactive schedules far sooner than normal on this cluster, and a profiling sweep is
# restartable -- it resumes from the materialized inputs -- so latency to first rollout matters
# more than protection from preemption.
export SBATCH_QOS=${SBATCH_QOS:-interactive}

# 63 servers came up in ~4 min from baked venvs; the ceiling is for a cold container that has to
# install them at runtime.
SERVERS_READY_TIMEOUT_S=${SERVERS_READY_TIMEOUT_S:-1800}
# No ++use_absolute_ip=true here, though upstream sets it. It binds every server to the node IP,
# which puts the head server somewhere other than 127.0.0.1:11000 -- where gym eval run --no-serve
# looks for it. Upstream gets away with it because it serves end-to-end and never has to discover
# its own servers. Jobs 6563122 and 6563743 both died on "Could not connect to the head server"
# because of this; bench_e2e.py has never passed it, which is why the same split works there.
#
ENV_START_ATTEMPTS=${ENV_START_ATTEMPTS:-4}

# 1, always: 01_materialize.sh already wrote num_repeats copies of every row, so anything higher
# would multiply them again.
NUM_REPEATS=${NUM_REPEATS:-1}

# env.yaml interpolates ${oc.env:VAR} for judge keys and similar. An unset one is not caught until
# gym env start parses the config inside the container, which costs a full spin-up and every retry
# before failing -- jobs 6606132/6606139 burned 4 attempts each on a missing NVI_KEY_EVALUATOR.
# A bare `VAR=value` in .bashrc is the usual cause: it is a shell variable, not an environment one,
# so sbatch never sees it.
if [[ -f "$ENV_YAML" ]]; then
    _missing=$(python - "$ENV_YAML" <<'PY_ENVVARS'
import os, re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
names = sorted(set(re.findall(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)", text)))
print(" ".join(n for n in names if not os.environ.get(n)))
PY_ENVVARS
)
    if [[ -n "$_missing" ]]; then
        echo "ERROR: $ENV_YAML interpolates environment variables that are not exported:" >&2
        for _v in $_missing; do echo "         $_v" >&2; done
        echo "       Export them before submitting. A bare assignment in .bashrc is not enough --" >&2
        echo "       it must be 'export VAR=...' for sbatch to pass it into the container." >&2
        exit 2
    fi
fi

# Same shadowing as WALLTIME: --comment is a flag, so read the manifest's sbatch.comment first.
SLURM_COMMENT="${SLURM_COMMENT:-${SBATCH_COMMENT:-}}"

PREFILL_VLLM_NIXL_SIDE_CHANNEL_PORT=5600
DECODE_VLLM_NIXL_SIDE_CHANNEL_PORT=5700

ROUTER_SERVER_PORT=8000
WORKER_SERVER_PORT=8001

eval_command=$(cat <<EOF
# Activate the container's Gym venv. SWEEP_DIR holds the artifacts 01_materialize.sh wrote.
source /opt/Gym_venv/bin/activate
cd /opt/Gym

# Serve every environment in the sweep from one deployment; rollout collection routes each row
# to the agent named in its own agent_ref.
env_start_log=$SWEEP_DIR/env_start.log

# Retry the whole spin-up. Gym probes for a free port and then binds, so 63 servers coming up at
# once occasionally lose the race: 6563714 lost lc_judge on :19730 and 6563785 lost
# abstention_simple_agent on :22661, both immediately after reporting startup complete. It is
# stochastic -- 6563167 brought all 63 up on the same config -- so a retry converts an occasional
# hard failure into a slower start, which is the right trade for a multi-hour sweep.
for _attempt in \$(seq 1 $ENV_START_ATTEMPTS); do
    : > "\$env_start_log"
    gym env start --config $SWEEP_DIR/sweep_config.yaml \\
        +uv_venv_dir=/opt/uv_venvs \\
        +skip_venv_if_present=true \\
       $ENV_OVERRIDES \\
        ++policy_base_url=http://\$(getent hosts "\$ROUTER_NODE" | awk 'NR == 1 {print \$1}'):$ROUTER_SERVER_PORT/v1 \
        ++policy_api_key=dummy_api_key \\
        ++policy_model_name=$MODEL > "\$env_start_log" 2>&1 &
    gym_servers_pid=\$!
    trap 'kill \$gym_servers_pid 2>/dev/null || true' EXIT

    echo "attempt \$_attempt/$ENV_START_ATTEMPTS: waiting for Gym servers (log: \$env_start_log) ..."
    servers_up=0
    for _i in \$(seq 1 $((SERVERS_READY_TIMEOUT_S / 10))); do
        if grep -q "servers ready!" "\$env_start_log" 2>/dev/null; then
            grep -m1 -oE "All [0-9]+ / [0-9]+ servers ready!" "\$env_start_log"
            servers_up=1
            break
        fi
        if ! kill -0 "\$gym_servers_pid" 2>/dev/null; then
            echo "gym env start exited on attempt \$_attempt:" >&2
            grep -E "address already in use|finished unexpectedly" "\$env_start_log" | tail -3 >&2 || true
            break
        fi
        sleep 10
    done

    if (( servers_up )); then
        break
    fi
    kill "\$gym_servers_pid" 2>/dev/null || true
    wait "\$gym_servers_pid" 2>/dev/null || true
    if (( _attempt == $ENV_START_ATTEMPTS )); then
        echo "ERROR: servers never came up in $ENV_START_ATTEMPTS attempts; see \$env_start_log" >&2
        tail -30 "\$env_start_log" >&2 || true
        exit 1
    fi
    echo "retrying spin-up ..." >&2
    sleep 10
done

# --resume is load-bearing: Gym reads the pre-expanded inputs instead of re-expanding them
# (~100 min single-threaded for a full sweep), and a walltime kill continues where it stopped.
gym eval run --no-serve --resume \\
    --input $SWEEP_DIR/rollouts_materialized_inputs.jsonl \\
    --output $SWEEP_DIR/rollouts.jsonl \\
    ++num_repeats=$NUM_REPEATS \\
    ++num_samples_in_parallel=$NUM_SAMPLES_IN_PARALLEL \\
    $MANIFEST_OVERRIDES \\
    +nemo_gym_log_dir=$SWEEP_DIR/logs \\
    +uv_venv_dir=/opt/uv_venvs \\
    +skip_venv_if_present=true \\
    $RUN_OVERRIDES

# Split back out to one directory per manifest entry, then profile each separately. agent_ref
# cannot do this -- the three ns_tools entries share ns_tools_simple_agent -- so the split keys on
# _ng_task_index against the task_index_range materialize recorded per entry.
python -m nemo_gym.sweep split $SWEEP_DIR

for _label_dir in $SWEEP_DIR/by_label/*/; do
    _label=\$(basename "\$_label_dir")
    [ -s "\$_label_dir/rollouts.jsonl" ] || { echo "skipping \$_label: no rollouts"; continue; }
    echo "=== profiling \$_label ==="
    gym eval profile \\
        --inputs "\$_label_dir/rollouts_materialized_inputs.jsonl" \\
        --rollouts "\$_label_dir/rollouts.jsonl" \\
        ++allow_partial_rollouts=$ALLOW_PARTIAL_ROLLOUTS > "\$_label_dir/profile.txt" 2>&1 \
        || echo "  profile failed for \$_label; see \$_label_dir/profile.txt" >&2
    tail -5 "\$_label_dir/profile.txt" || true
done

# Sweep-wide summary as well. allow_partial_rollouts so a walltime kill still yields a profile.
gym eval profile \\
    --inputs $SWEEP_DIR/rollouts_materialized_inputs.jsonl \\
    --rollouts $SWEEP_DIR/rollouts.jsonl \\
    ++allow_partial_rollouts=$ALLOW_PARTIAL_ROLLOUTS
EOF
)

pd_command=$(cat <<EOF
#!/bin/bash

set -euo pipefail

# Nemotron's three-read Mamba SSM state must use the dimension-sequence layout when KV transfer is enabled.
# Not used when the model has no Mamba layers.
export VLLM_SSM_CONV_STATE_LAYOUT=DS

export VLLM_USE_FASTOKENS=1

# NIXL uses UCX for cross-node KV transfer. Explicitly enable UCX's CUDA
# transports and the GB200 InfiniBand interface; otherwise UCX treats VRAM as
# host memory and NIXL KV-cache registration fails with NIXL_ERR_BACKEND.
export UCX_TLS=rc_x,rc,cuda_copy,cuda_ipc
export UCX_NET_DEVICES=mlx5_0:1
export UCX_IB_ADDR_TYPE=eth
export UCX_RNDV_SCHEME=get_zcopy
export UCX_RNDV_THRESH=0

source "$VLLM_CONFIG"

if [[ \$(ulimit -Hn) == "unlimited" ]] || [[ 65535 -lt \$(ulimit -Hn) ]]; then
  ulimit -Sn 65535
fi

this_node_hostname=\$(hostname)
if (( SLURM_PROCID == 0 )); then
    read -r -a nodes <<< "\$ALL_NODES"

    # @bxyu-nvidia: for --intra-node-data-parallel-size: Not sure what to set this to other than 1. I can't tell from the docs what is appropriate and 1 seems to work fine.
    # Set a super long request timeout since some reasoning requests may take a long time to generate.
    # Don't manually wait as vllm-router will wait for the URLs to come up
    router_args=( \
        --prefill-policy cache_aware \
        --decode-policy cache_aware \
        --vllm-pd-disaggregation \
        --host \$this_node_hostname \
        --port $ROUTER_SERVER_PORT \
        --intra-node-data-parallel-size 1 \
        --request-timeout-secs 86400 \
        --log-level error
    )

    for (( i = 0; i < $NUM_PREFILL_NODES; i++ )); do
        router_args+=(--prefill "http://\${nodes[i]}:$WORKER_SERVER_PORT")
    done
    for (( i = 0; i < $NUM_DECODE_NODES; i++ )); do
        node_idx=\$(( $NUM_PREFILL_NODES + i ))
        router_args+=(--decode "http://\${nodes[node_idx]}:$WORKER_SERVER_PORT")
    done

    vllm-router "\${router_args[@]}" &

    router_pid=\$!
    trap 'kill "\$router_pid" 2>/dev/null || true' EXIT
fi

# Split nodes here by index
if (( SLURM_PROCID < $NUM_PREFILL_NODES )); then
    # Prefill
    VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_VLLM_NIXL_SIDE_CHANNEL_PORT \
    vllm serve "$MODEL" "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_PREFILL_ARGS[@]}" \
        --host \$this_node_hostname \
        --port $WORKER_SERVER_PORT
else
    # Decode
    VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_VLLM_NIXL_SIDE_CHANNEL_PORT \
    vllm serve "$MODEL" "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_DECODE_ARGS[@]}" \
        --host \$this_node_hostname \
        --port $WORKER_SERVER_PORT
fi
EOF
)

NUM_NODES=$((NUM_PREFILL_NODES + NUM_DECODE_NODES))
batch_command=$(cat <<EOF
set -euo pipefail

# The container ships its own environment. A PYTHONPATH set on the submitting host -- the
# GYM_SITE_PACKAGES guard in 03_run_sharded.sh exports one -- propagates through sbatch and
# shadows it for every step, vLLM included. Job 6601167 died in vLLM startup with "cannot import
# name 'is_offline_mode' from huggingface_hub" resolved out of the host venv.
unset PYTHONPATH

nodes=(\$(scontrol show hostnames "\$SLURM_JOB_NODELIST"))

ALL_NODES="\${nodes[*]}" \
srun --nodes=$NUM_NODES --ntasks=$NUM_NODES --ntasks-per-node=1 \
    --container-image=$CONTAINER \
    --container-name=container-on-node \
    --container-mounts=$MOUNTS \
    --container-workdir=\$SLURM_SUBMIT_DIR \
    --no-container-mount-home \
    bash -lc '
        set -euo pipefail
        cd "\$SLURM_SUBMIT_DIR"
        exec "\$@"
    ' bash bash -lc "\$vllm_command" &
server_step=\$!

cleanup_server() {
    kill "\$server_step" 2>/dev/null || true
    wait "\$server_step" 2>/dev/null || true
}
trap cleanup_server EXIT INT TERM

# No need to wait for endpoint since Gym will wait for model endpoints to spin up before proceeding.

    # @bxyu-nvidia: Put the Gym servers on a separate node than the PREFILL_HEAD which is also running the vllm-router
    # This helps relieve so much network traffic on one node.
    if [[ -v 'nodes[1]' ]]; then
        EVAL_NODE=\${nodes[1]}
    else
        EVAL_NODE=\${nodes[0]}
    fi

    # Sandbox sidecar on the eval node, so ns_tools reaches it over loopback. Started before the
    # driver and waited on: if the driver starts first it burns rollouts against a dead port.
    if [[ -n "$SANDBOX_CONTAINER" ]]; then
        echo "starting sandbox on \$EVAL_NODE (${SANDBOX_WORKERS} workers, port ${SANDBOX_PORT})"
        srun --overlap --nodes=1 --ntasks=1 --nodelist="\$EVAL_NODE" --gpus=0 \
            --cpus-per-task=${SANDBOX_WORKERS} \
            --container-image=$SANDBOX_CONTAINER \
            --container-mounts=$MOUNTS \
            --no-container-mount-home \
            bash -lc 'export UWSGI_PROCESSES=${SANDBOX_WORKERS} NUM_WORKERS=${SANDBOX_WORKERS} \
                      LISTEN_PORT=${SANDBOX_PORT} NGINX_PORT=${SANDBOX_PORT}; exec /start-with-nginx.sh' \
            > "$LOG_DIR/\${SLURM_JOB_ID}-sandbox.log" 2>&1 &
        sandbox_step=\$!

        SANDBOX_IP=\$(getent hosts "\$EVAL_NODE" | awk 'NR==1 {print \$1}')
        echo "waiting for sandbox at \$SANDBOX_IP:${SANDBOX_PORT}/health ..."
        for _i in \$(seq 1 60); do
            if curl -sf -m 3 "http://\$SANDBOX_IP:${SANDBOX_PORT}/health" >/dev/null 2>&1; then
                echo "sandbox healthy after \${_i}0s"; break
            fi
            if ! kill -0 "\$sandbox_step" 2>/dev/null; then
                echo "ERROR: sandbox exited during startup; see $LOG_DIR/\${SLURM_JOB_ID}-sandbox.log" >&2
                exit 1
            fi
            sleep 10
        done
        if ! curl -sf -m 3 "http://\$SANDBOX_IP:${SANDBOX_PORT}/health" >/dev/null 2>&1; then
            echo "ERROR: sandbox never became healthy; see $LOG_DIR/\${SLURM_JOB_ID}-sandbox.log" >&2
            exit 1
        fi
        # Exported, not prefixed onto srun: bash decides which words are assignments at parse
        # time, so an expanded "VAR=value" would become the command name instead. srun propagates
        # the environment into the container, which is how ROUTER_NODE already reaches it.
        export NEMO_SKILLS_SANDBOX_HOST="\$SANDBOX_IP"
        export NEMO_SKILLS_SANDBOX_PORT="${SANDBOX_PORT}"
    fi

    # @bxyu-nvidia: We need --cpus-per-task=SLURM_CPUS_ON_NODE, otherwise we run into a lot of ServerDisconnectedError and ConnectionResetByPeer errors from Gym servers and vLLM. Not sure what the correlation is
    ROUTER_NODE="\${nodes[0]}" \
    srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=\$SLURM_CPUS_ON_NODE --nodelist="\$EVAL_NODE" --gpus=0 \
        --container-image=$CONTAINER \
        --container-name=eval-container-on-node \
        --container-mounts=$MOUNTS \
        --container-workdir="\$SLURM_SUBMIT_DIR" \
        --no-container-mount-home \
        bash -lc '
            set -euo pipefail
            cd "\$SLURM_SUBMIT_DIR"
            # set -e is not inherited through a fresh bash -lc, so a failing gym command
            # inside eval_command would otherwise exit 0 and look like a successful run.
            exec bash -lc "set -euo pipefail; \$eval_command"
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
        trap - EXIT INT TERM
        exit "\$completed_status"
    fi

    cleanup_server
    trap - EXIT INT TERM
    exit "\$completed_status"
EOF
)

# --segment > 0 otherwise the engine will hang on the second or third engine step.
vllm_command="$pd_command" \
eval_command="$eval_command" \
batch_command="$batch_command" \
sbatch \
    --nodes=$NUM_NODES \
    --time=$WALLTIME \
    --job-name=gym-$EXPERIMENT_NAME-$USER \
    --output="$LOG_DIR/%j-%x.log" \
    --ntasks-per-node=1 \
    --comment="$SLURM_COMMENT" \
    --exclusive \
    --segment=$NUM_NODES \
    --wrap 'exec bash -lc "$batch_command"'
