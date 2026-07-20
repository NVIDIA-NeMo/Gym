# OSWorld public tools

Each first-time deployment check and runtime step has one checked-in entry
point. `prepare.py` remains at the benchmark root because it is the benchmark
configuration entry point; host checks and lifecycle wrappers live here:

```text
model host       -> probe_model_endpoint.py
environment host -> check_environment.sh
agent/control     -> prepare.py -> start_control.sh -> run_eval.sh
abnormal recovery -> cleanup_run.sh
```

| Tool | Purpose |
| --- | --- |
| `probe_model_endpoint.py` | Require the configured model identity and optionally exercise the one- or three-image chat-completions request shape |
| `check_environment.sh` | Validate local or SSH-reached Linux/Docker/KVM/qcow2 environment-host readiness |
| `start_control.sh` | Supervisor-friendly wrapper around `gym env start` |
| `run_eval.sh` | Supervisor-friendly wrapper around `gym eval run --no-serve` |
| `cleanup_run.sh` | Recovery-only cleanup for stale processes or labeled Sandbox containers after abnormal termination |
| `prepare_osworld_vm.sh` | Download and verify the pinned OSWorld qcow2 baseline |

Model serving itself belongs to the selected model's deployment project;
`probe_model_endpoint.py` is the provider-neutral contract check used before
Gym starts. For example:

```bash
python3 benchmarks/osworld/tools/probe_model_endpoint.py \
  --base-url http://MODEL_HOST:8000/v1 \
  --api-key local-vllm \
  --model SERVED_MODEL_NAME \
  --image-count 3
```

Validate an environment host locally, or stream the same checked-in checker to
a remote host that does not have a Gym checkout:

```bash
# On the environment host:
bash benchmarks/osworld/tools/check_environment.sh /absolute/path/to/Ubuntu.qcow2

# Or, from its paired agent/control host:
bash benchmarks/osworld/tools/check_environment.sh \
  --ssh REMOTE_USER@ENV_HOST_REACHABLE_IP \
  /same/absolute/path/on/both/hosts/Ubuntu.qcow2
```

The checker pins the public OSWorld disk size and SHA-256 by default. Set
`EXPECTED_VM_SHA256`, `EXPECTED_VM_SIZE`, or `MIN_FREE_GIB` only when the
deployment intentionally uses a different verified baseline or capacity
threshold.

Both runtime wrappers require `OSWORLD_RUN_ID`. Set
`NEMO_GYM_CONTROL_HOST` when the control services must advertise a non-loopback
address. Their optional positional argument selects the root for logs and
results; it defaults to the Gym repository root.

For a split-host Gym Docker deployment, run the wrappers on the agent/control
host and point its normal Docker CLI at the OSWorld environment host. The
identical qcow2 path and non-interactive SSH authorization are one-time host
preparation; they are not repeated for each benchmark run:

```bash
export DOCKER_HOST=ssh://REMOTE_USER@ENV_HOST_REACHABLE_IP
export OSWORLD_SANDBOX_PUBLISH_HOST=ENV_HOST_REACHABLE_IP
bash benchmarks/osworld/tools/check_environment.sh \
  --ssh REMOTE_USER@ENV_HOST_REACHABLE_IP \
  /same/absolute/path/on/both/hosts/Ubuntu.qcow2
docker info  # validates Docker's SSH transport from this host
```

`start_control.sh` revalidates both variables and the Docker connection before
starting Gym. The qcow2 path written by `prepare.py --vm-path` must exist at
the same absolute path on the Docker host. Once `docker info` succeeds, each
run uses only `prepare.py`, `start_control.sh`, and `run_eval.sh`. Export the
same `DOCKER_HOST` when running `cleanup_run.sh` so cleanup remains scoped to
the correct daemon.

Only when normal termination fails or stale run-owned entities block recovery,
stop that run while preserving its logs and results:

```bash
export OSWORLD_RUN_ID=my-osworld-run
bash benchmarks/osworld/tools/cleanup_run.sh /absolute/run/root
```

The runtime wrappers record PIDs under `RUN_ROOT/run/osworld/RUN_ID/`. Cleanup
validates both the PID environment and command before signaling it. Docker
cleanup requires the Sandbox, OSWorld workload, and run-ID labels to match; it
does not remove unlabeled or other-run containers. Model services are outside
this lifecycle and are never stopped by this tool.
