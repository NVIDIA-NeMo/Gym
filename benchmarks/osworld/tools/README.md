# OSWorld optional tools

The canonical runtime path does not depend on this directory:

```text
prepare.py
  -> gym env start
  -> gym eval run --no-serve
```

The tracked files here are the public runtime wrappers and the one VM
preparation helper required by a first-time Sandbox deployment.

| Tool | Purpose |
| --- | --- |
| `start_control.sh` | Supervisor-friendly wrapper around `gym env start` |
| `run_eval.sh` | Supervisor-friendly wrapper around `gym eval run --no-serve` |
| `cleanup_run.sh` | Stop one run's recorded Gym processes and remove only its labeled Sandbox containers |
| `prepare_osworld_vm.sh` | Download and verify the pinned OSWorld qcow2 baseline |

Model serving belongs to the deployment layer. Model-specific benchmark
selection belongs to `prepare.py --profile`; neither is implemented here.

Both runtime wrappers require `OSWORLD_RUN_ID`. Set
`NEMO_GYM_CONTROL_HOST` when the control services must advertise a non-loopback
address. Their optional positional argument selects the root for logs and
results; it defaults to the Gym repository root.

After an evaluation, stop only that run while preserving its logs and results:

```bash
export OSWORLD_RUN_ID=my-osworld-run
bash benchmarks/osworld/tools/cleanup_run.sh /absolute/run/root
```

The runtime wrappers record PIDs under `RUN_ROOT/run/osworld/RUN_ID/`. Cleanup
validates both the PID environment and command before signaling it. Docker
cleanup requires the Sandbox, OSWorld workload, and run-ID labels to match; it
does not remove unlabeled or other-run containers. Model services are outside
this lifecycle and are never stopped by this tool.
