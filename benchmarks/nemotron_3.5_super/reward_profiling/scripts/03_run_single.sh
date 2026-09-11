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
#   SANDBOX_PORT / SANDBOX_WORKERS         6000 / 64 (workers is per node)
#   SANDBOX_NODES                          every node in the job
#   SANDBOX_LB_PORT                        6001; unused when SANDBOX_NODES is 1
#
# OPTIONAL - throughput
#   NUM_SAMPLES_IN_PARALLEL   512 x decode_nodes, or the manifest's value if it sets one
#   MAX_NUM_SEQS_PER_DECODE_ENGINE         512
#   ALLOW_PARTIAL_ROLLOUTS                 True
#   ENV_PORT_RANGE_LOW / _HIGH             unset; the manifest's gym_env_start sets the range
#   RUN_PORT_RANGE_LOW / _HIGH             unset; the driver's own range, separate from the above
#   GLOBAL_AIOHTTP_CONNECTOR_LIMIT_PER_HOST  unset; Gym's 1024/num_workers applies. Per Gym server
#                                          process, so ~63k in aggregate -- see the note below
#   SERVERS_READY_TIMEOUT_S / ENV_START_ATTEMPTS   1800 / 4
#   NUM_REPEATS                            the manifest's gym_eval_run.num_repeats. Collection runs
#                                          at 1: repeats are already written into the inputs by
#                                          01_materialize.sh, so raising this MULTIPLIES them
#   ENV_YAML / MOUNTS / LOG_DIR            $PWD/env.yaml / /lustre:/lustre / SWEEP_DIR/slurm-logs
#   VLLM_CONFIG                            the manifest's vllm.config; the arg script that is sourced
#   EXPERIMENT_NAME / SLURM_COMMENT        job-name prefix / --comment
#
# OPTIONAL - vllm-router (the single process every Gym server talks to)
#   ROUTER_REQUEST_TIMEOUT_S               3600
#   ROUTER_HEALTH_TIMEOUT_S / _INTERVAL_S  60 / 120
#   ROUTER_HEALTH_FAILURES                 10
#   ROUTER_MAX_CONCURRENT / ROUTER_QUEUE_SIZE / ROUTER_QUEUE_TIMEOUT_S
#                                          32768 / NUM_SAMPLES_IN_PARALLEL / 600
#   ROUTER_CIRCUIT_BREAKER                 0 (off)
#   ROUTER_LOG_LEVEL                       info
#
# Sandbox capacity is SANDBOX_NODES x SANDBOX_WORKERS concurrent executions. Each sandbox is nginx
# over N single-process uWSGI workers, consistent-hashing X-Session-ID so a stateful IPython
# session pins to one worker; the image forces one process per worker, so the worker count *is*
# the concurrency. With SANDBOX_NODES > 1 the launcher fronts them with an nginx balancer that
# hashes the same header the same way, so session -> node -> worker stays deterministic.
set -euo pipefail

RP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
for block in ("sbatch", "srun", "vllm", "vllm_router", "gym_eval_profile"):
    for key, value in (doc.get(block) or {}).items():
        print(f"{key}={value}")
PY_SBATCH
)


# Checked after the manifest has had its say, so a manifest that names its own container satisfies
# this rather than tripping it.
for _required in MODEL CONTAINER; do
    if [[ -z "${!_required:-}" ]]; then
        # MODEL comes from the manifest's vllm block, CONTAINER from its srun block.
        _blk=$([[ "$_required" == "MODEL" ]] && echo vllm || echo srun)
        echo "ERROR: $_required is required, and the manifest's $_blk block does not set it." >&2
        echo "       See the header of $0 for the full argument list." >&2
        exit 2
    fi
done

# ---- job shape -------------------------------------------------------------------
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
# Workers per sandbox node, and the real concurrency number. The image sets STATEFUL_SANDBOX=1,
# which pins UWSGI_PROCESSES to 1 so each worker serves exactly one execution at a time -- passing
# UWSGI_PROCESSES is a no-op, NUM_WORKERS is the knob. 32 on one node meant 32 concurrent
# executions against a driver concurrency of 4,096 in job 6706202, which is where the 1,168
# session timeouts came from.
SANDBOX_WORKERS=${SANDBOX_WORKERS:-64}
# How many of the job's nodes run a sandbox. Sandboxes ride along on the vLLM nodes with --overlap
# and --gpus=0: they are CPU work, and a GB200 node has 144 cores that the engines do not use.
# Total sandbox concurrency is SANDBOX_NODES x SANDBOX_WORKERS. Defaults to every node in the job.
SANDBOX_NODES=${SANDBOX_NODES:-$((NUM_PREFILL_NODES + NUM_DECODE_NODES))}
# Port the balancer listens on. Only used when SANDBOX_NODES > 1; with a single sandbox the driver
# talks to it directly and there is no balancer process.
SANDBOX_LB_PORT=${SANDBOX_LB_PORT:-6001}
if [[ -n "$SANDBOX_CONTAINER" && ! -f "$SANDBOX_CONTAINER" ]]; then
    echo "ERROR: SANDBOX_CONTAINER=$SANDBOX_CONTAINER does not exist." >&2
    exit 2
fi
if (( SANDBOX_NODES > NUM_PREFILL_NODES + NUM_DECODE_NODES )); then
    echo "ERROR: SANDBOX_NODES=$SANDBOX_NODES exceeds the job's $((NUM_PREFILL_NODES + NUM_DECODE_NODES)) nodes." >&2
    echo "       Sandboxes ride along on the vLLM nodes; they do not add any." >&2
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

# Scale concurrency with decode capacity. Note this is *half* the engines' own
# --max-num-seqs 1024 in vllm_configs/nemotron_3.5_super.sh; the two do not have to match, and
# measurement says neither is the binding constraint. Engines actually ran p50 28-33 concurrent
# sequences, p99 66-210, max 435 (jobs 7060112 and 6706202), because an agentic rollout spends
# most of its wall time in tool calls, sandbox execution and judging rather than generating.
#
# Do not lower this to 128: a run at 128 against D2 sat at 12.5% of capacity with the client
# semaphore, not the GPUs, as the limit (128/(2x1024) = 6.25% of engine capacity), and 128 would
# clip the p99 bursts that are the only
# moments the driver has real work queued. Raising it to 1024 to match the engines is harmless
# but unsupported -- P2D8 requested 4,096 and never had more than ~264 in flight.
#
# Do NOT raise NUM_SAMPLES_IN_PARALLEL above this derived default without re-testing: 8192 (2x)
# on P2D8 killed a decode engine ~30 min into collection with `assert num_new_tokens > 0` in
# vllm/v1/core/sched/scheduler.py:914, which cascaded to NIXL_ERR_REMOTE_DISCONNECT on the other
# engines (job 7062901). The same shape at 4096 ran 1 h 37 m clean with zero connection errors,
# at ~29,941 rollouts/hr sustained (job 7061265; a favourable 553 s window hit 45,452, which is not
# the rate to plan with). More in-flight work reaches a vLLM scheduler edge case long before it
# reaches any capacity limit.
#
# GLOBAL_AIOHTTP_CONNECTOR_LIMIT_PER_HOST is *not* the ceiling here, despite looking like one.
# Gym defaults it to 1024 / num_workers (server_utils.py:96,163) and aiohttp keys limit_per_host on
# (host, port), but the client that talks to the router lives in each Gym agent server process --
# 63 of them -- so the aggregate is ~63k connections, not 1024. It has never been close to binding:
# 4,096 slots produced ~352 concurrent requests (job 7061265).
#
# The ceilings that could actually bind, in order of likelihood:
#   1. the driver's single asyncio event loop, which is unmeasured
#   2. the sandbox tier, SANDBOX_NODES x SANDBOX_WORKERS concurrent executions
#   3. engine capacity, max_num_seqs x decode_nodes = 8,192 at P2D8
# (Two earlier versions of this comment were wrong: one claimed a 16k default, the next claimed
# the 1024 was a global cap. Neither is true.)
MAX_NUM_SEQS_PER_DECODE_ENGINE=${MAX_NUM_SEQS_PER_DECODE_ENGINE:-512}
# Settings layer, lowest to highest: manifest defaults -> these env vars -> anything you add on
# the gym command line. Each is unset by default, so nothing here overrides what the manifest
# declared unless you ask for it; if the manifest is silent too, Gym's own default applies.
#
# ---- Gym overrides: unset means the manifest wins ---------------------------------
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

# ---- manifest-declared ++ overrides -----------------------------------------------
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

# ---- submit-time guards -----------------------------------------------------------
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

# ---- vllm-router ----------------------------------------------------------------
# One process fronts every engine, so every Gym server's traffic crosses it. Job 6706202 logged
# ~483,000 ClientOSErrors here against 25,494 completed rollouts while the engines idled at 4.5%
# KV cache -- read the "Hit N global" counter in env_start.log, not the line count, because Gym
# prints one line per 100 errors.
#
# Left at their upstream defaults the router health-checks each engine every 60 s with a 5 s
# timeout and ejects it after 3 misses -- which is what the settings below exist to override.
# A busy engine misses that easily, and an ejection drops its in-flight connections, so the
# defaults turn load into apparent failure. These are deliberately generous: a slow /health means
# the engine is working.
ROUTER_HEALTH_TIMEOUT_S=${ROUTER_HEALTH_TIMEOUT_S:-60}
ROUTER_HEALTH_INTERVAL_S=${ROUTER_HEALTH_INTERVAL_S:-120}
ROUTER_HEALTH_FAILURES=${ROUTER_HEALTH_FAILURES:-10}
# Finite, and shorter than the job. The previous 86400 meant "never" inside a 4 h walltime, so a
# wedged request held a driver concurrency slot until the job died -- observed elapsed_s reached
# 7,995 s. A request that has not returned in an hour is not coming back.
ROUTER_REQUEST_TIMEOUT_S=${ROUTER_REQUEST_TIMEOUT_S:-3600}
# Admission control. The router's own default queue is 100 deep, which overflows to 429 the moment
# a burst exceeds it, so size the queue to the driver's concurrency instead.
ROUTER_MAX_CONCURRENT=${ROUTER_MAX_CONCURRENT:-32768}
ROUTER_QUEUE_SIZE=${ROUTER_QUEUE_SIZE:-$NUM_SAMPLES_IN_PARALLEL}
ROUTER_QUEUE_TIMEOUT_S=${ROUTER_QUEUE_TIMEOUT_S:-600}
# 1 keeps the router's circuit breaker, 0 disables it. Off by default: the engines are a fixed
# fleet inside one job, so there is nowhere to fail over to, and opening the breaker only turns a
# slow engine into a dead one.
ROUTER_CIRCUIT_BREAKER=${ROUTER_CIRCUIT_BREAKER:-0}
# `error` hid every worker-health transition in 6706202, which is exactly what explains a stall.
ROUTER_LOG_LEVEL=${ROUTER_LOG_LEVEL:-info}

eval_command=$(cat <<EOF
# The 63 Gym servers hold a socket per in-flight request to the router, so the default soft
# limit is the wrong order of magnitude at NUM_SAMPLES_IN_PARALLEL=4096. Only the vLLM shell
# raised this before, which left the client side -- the side that reported the ClientOSErrors in
# 6706202 -- on the default.
if [[ \$(ulimit -Hn) == "unlimited" ]] || [[ 65535 -lt \$(ulimit -Hn) ]]; then
  ulimit -Sn 65535
else
  ulimit -Sn "\$(ulimit -Hn)"
fi

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
PYTHONPATH="$RP_DIR${PYTHONPATH:+:$PYTHONPATH}" python -m infra split $SWEEP_DIR

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
    # Request timeout is deliberately finite now (ROUTER_REQUEST_TIMEOUT_S, 3600) rather than the
    # 86400 this line used to argue for: inside a 4 h job 'never' means a wedged request holds a
    # driver concurrency slot until the walltime kills it.
    # Don't manually wait as vllm-router will wait for the URLs to come up
    router_args=( \
        --prefill-policy cache_aware \
        --decode-policy cache_aware \
        --vllm-pd-disaggregation \
        --host \$this_node_hostname \
        --port $ROUTER_SERVER_PORT \
        --intra-node-data-parallel-size 1 \
        --request-timeout-secs $ROUTER_REQUEST_TIMEOUT_S \
        --health-check-timeout-secs $ROUTER_HEALTH_TIMEOUT_S \
        --health-check-interval-secs $ROUTER_HEALTH_INTERVAL_S \
        --health-failure-threshold $ROUTER_HEALTH_FAILURES \
        --max-concurrent-requests $ROUTER_MAX_CONCURRENT \
        --queue-size $ROUTER_QUEUE_SIZE \
        --queue-timeout-secs $ROUTER_QUEUE_TIMEOUT_S \
        --log-level $ROUTER_LOG_LEVEL
    )
    if (( $ROUTER_CIRCUIT_BREAKER == 0 )); then
        router_args+=(--disable-circuit-breaker)
    fi

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
# --kill-on-bad-exit=1 so one dead engine ends the step. Without it srun keeps the step alive
# waiting on the surviving tasks, the `wait -n` below never fires, and the job runs on with no
# endpoint: job 7062901 lost an engine to a vLLM scheduler assertion and then burned 1 h 40 m of
# 10 nodes retrying against a router that had already shut down (6.45M ClientOSErrors, zero
# rollouts). Failing fast lets 03_run_sharded.sh's watcher resubmit and --resume carry the work.
srun --kill-on-bad-exit=1 --nodes=$NUM_NODES --ntasks=$NUM_NODES --ntasks-per-node=1 \
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

    # Sandbox tier. One sandbox per node for the first SANDBOX_NODES nodes, fronted by an nginx
    # balancer when there is more than one. Started before the driver and waited on: if the driver
    # starts first it burns rollouts against a dead port.
    #
    # Each sandbox is nginx over NUM_WORKERS single-process uWSGI workers, consistent-hashing
    # X-Session-ID so a stateful IPython session pins to one worker. The balancer hashes the same
    # header the same way, so the two tiers compose: session -> node -> worker, deterministically.
    # Round-robin here would break sessions, which is why this is not a plain proxy.
    if [[ -n "$SANDBOX_CONTAINER" ]]; then
        sandbox_nodes=()
        for (( i = 0; i < $SANDBOX_NODES; i++ )); do
            sandbox_nodes+=("\${nodes[i]}")
        done
        echo "starting $SANDBOX_NODES sandbox(es), ${SANDBOX_WORKERS} workers each, on \${sandbox_nodes[*]}"

        sandbox_steps=()
        for _sb_node in "\${sandbox_nodes[@]}"; do
            srun --overlap --nodes=1 --ntasks=1 --nodelist="\$_sb_node" --gpus=0 \
                --cpus-per-task=${SANDBOX_WORKERS} \
                --container-image=$SANDBOX_CONTAINER \
                --container-mounts=$MOUNTS \
                --no-container-mount-home \
                bash -lc 'export NUM_WORKERS=${SANDBOX_WORKERS} \
                          LISTEN_PORT=${SANDBOX_PORT} NGINX_PORT=${SANDBOX_PORT}; exec /start-with-nginx.sh' \
                > "$LOG_DIR/\${SLURM_JOB_ID}-sandbox-\${_sb_node}.log" 2>&1 &
            sandbox_steps+=(\$!)
        done

        # Wait for every sandbox. A missing one is fatal rather than degraded: the balancer hashes
        # over a fixed server list, so a dead member silently black-holes its share of sessions.
        for _idx in "\${!sandbox_nodes[@]}"; do
            _sb_node="\${sandbox_nodes[_idx]}"
            _sb_pid="\${sandbox_steps[_idx]}"
            _sb_ip=\$(getent hosts "\$_sb_node" | awk 'NR==1 {print \$1}')
            echo "waiting for sandbox at \$_sb_ip:${SANDBOX_PORT}/health ..."
            for _i in \$(seq 1 60); do
                if curl -sf -m 3 "http://\$_sb_ip:${SANDBOX_PORT}/health" >/dev/null 2>&1; then
                    echo "sandbox on \$_sb_node healthy after \${_i}0s"; break
                fi
                if ! kill -0 "\$_sb_pid" 2>/dev/null; then
                    echo "ERROR: sandbox on \$_sb_node exited during startup; see $LOG_DIR/\${SLURM_JOB_ID}-sandbox-\${_sb_node}.log" >&2
                    exit 1
                fi
                sleep 10
            done
            if ! curl -sf -m 3 "http://\$_sb_ip:${SANDBOX_PORT}/health" >/dev/null 2>&1; then
                echo "ERROR: sandbox on \$_sb_node never became healthy; see $LOG_DIR/\${SLURM_JOB_ID}-sandbox-\${_sb_node}.log" >&2
                exit 1
            fi
        done

        EVAL_NODE_IP=\$(getent hosts "\$EVAL_NODE" | awk 'NR==1 {print \$1}')
        if (( $SANDBOX_NODES == 1 )); then
            # Single sandbox: talk to it directly, no balancer hop.
            SANDBOX_ENDPOINT_IP=\$(getent hosts "\${sandbox_nodes[0]}" | awk 'NR==1 {print \$1}')
            SANDBOX_ENDPOINT_PORT=${SANDBOX_PORT}
        else
            # Balancer on the eval node, so the driver reaches it over loopback.
            _lb_conf="$SWEEP_DIR/sandbox-lb.conf"
            {
                echo "events { worker_connections 16384; }"
                echo "http {"
                # Log the upstream, not just the client: without it every line shows the eval
                # node and there is no way to tell whether the hash is actually spreading load.
                echo "    log_format lb '\\\$remote_addr \\\$status upstream: \\\$upstream_addr session: \\\$http_x_session_id';"
                echo "    access_log /dev/stdout lb;"
                # Empty header means no session -- fall back to the per-request id so those
                # requests spread rather than all piling onto one node.
                echo "    map \\\$http_x_session_id \\\$hash_key { \"\"  \\\$request_id; default \\\$http_x_session_id; }"
                echo "    upstream sandbox_nodes {"
                echo "        hash \\\$hash_key consistent;"
                for _sb_node in "\${sandbox_nodes[@]}"; do
                    _sb_ip=\$(getent hosts "\$_sb_node" | awk 'NR==1 {print \$1}')
                    echo "        server \$_sb_ip:${SANDBOX_PORT} max_fails=0;"
                done
                echo "    }"
                echo "    server {"
                echo "        listen ${SANDBOX_LB_PORT};"
                echo "        client_max_body_size 10M;"
                echo "        location / {"
                echo "            proxy_pass http://sandbox_nodes;"
                echo "            proxy_set_header X-Session-ID \\\$http_x_session_id;"
                # Match the per-node nginx: long code executions, no buffering, and never retry
                # elsewhere -- a retry on another node would land on a different IPython session.
                echo "            proxy_connect_timeout 1200s;"
                echo "            proxy_send_timeout 1200s;"
                echo "            proxy_read_timeout 1200s;"
                echo "            proxy_buffering off;"
                echo "            proxy_next_upstream off;"
                echo "        }"
                echo "    }"
                echo "}"
            } > "\$_lb_conf"

            echo "starting sandbox balancer on \$EVAL_NODE:${SANDBOX_LB_PORT} over $SANDBOX_NODES nodes"
            srun --overlap --nodes=1 --ntasks=1 --nodelist="\$EVAL_NODE" --gpus=0 \
                --container-image=$SANDBOX_CONTAINER \
                --container-mounts=$MOUNTS \
                --no-container-mount-home \
                bash -lc "exec nginx -c '\$_lb_conf' -g 'daemon off;'" \
                > "$LOG_DIR/\${SLURM_JOB_ID}-sandbox-lb.log" 2>&1 &
            sandbox_lb_step=\$!

            echo "waiting for sandbox balancer at \$EVAL_NODE_IP:${SANDBOX_LB_PORT}/health ..."
            for _i in \$(seq 1 30); do
                if curl -sf -m 3 "http://\$EVAL_NODE_IP:${SANDBOX_LB_PORT}/health" >/dev/null 2>&1; then
                    echo "sandbox balancer healthy after \${_i}0s"; break
                fi
                if ! kill -0 "\$sandbox_lb_step" 2>/dev/null; then
                    echo "ERROR: sandbox balancer exited during startup; see $LOG_DIR/\${SLURM_JOB_ID}-sandbox-lb.log" >&2
                    exit 1
                fi
                sleep 10
            done
            if ! curl -sf -m 3 "http://\$EVAL_NODE_IP:${SANDBOX_LB_PORT}/health" >/dev/null 2>&1; then
                echo "ERROR: sandbox balancer never became healthy; see $LOG_DIR/\${SLURM_JOB_ID}-sandbox-lb.log" >&2
                exit 1
            fi
            SANDBOX_ENDPOINT_IP="\$EVAL_NODE_IP"
            SANDBOX_ENDPOINT_PORT=${SANDBOX_LB_PORT}
        fi

        # Exported, not prefixed onto srun: bash decides which words are assignments at parse
        # time, so an expanded "VAR=value" would become the command name instead. srun propagates
        # the environment into the container, which is how ROUTER_NODE already reaches it.
        export NEMO_SKILLS_SANDBOX_HOST="\$SANDBOX_ENDPOINT_IP"
        export NEMO_SKILLS_SANDBOX_PORT="\$SANDBOX_ENDPOINT_PORT"
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
    --job-name="${SBATCH_JOB_NAME:-gym-$EXPERIMENT_NAME-$USER}" \
    --output="$LOG_DIR/%j-%x.log" \
    --ntasks-per-node=1 \
    --comment="$SLURM_COMMENT" \
    --exclusive \
    --segment=$NUM_NODES \
    --wrap 'exec bash -lc "$batch_command"'
