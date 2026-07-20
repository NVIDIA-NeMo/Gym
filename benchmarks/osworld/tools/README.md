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
| `prepare_osworld_vm.sh` | Download and verify the pinned OSWorld qcow2 baseline |

Model serving belongs to the deployment layer. Model-specific benchmark
selection belongs to `prepare.py --profile`; neither is implemented here.

Both runtime wrappers require `OSWORLD_RUN_ID`. Set
`NEMO_GYM_CONTROL_HOST` when the control services must advertise a non-loopback
address. Their optional positional argument selects the root for logs and
results; it defaults to the Gym repository root.
