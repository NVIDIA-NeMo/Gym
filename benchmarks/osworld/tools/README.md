# OSWorld optional tools

The canonical runtime path does not depend on this directory:

```text
prepare.py
  -> gym env start
  -> gym eval run --no-serve
```

The tracked files here are optional host, VM, and dataset preparation tools.
They do not start Gym or proxy runtime traffic.

| Tool | Purpose |
| --- | --- |
| `start_control.sh` | Supervisor-friendly wrapper around `gym env start` |
| `run_eval.sh` | Supervisor-friendly wrapper around `gym eval run --no-serve` |
| `bringup_local_host.sh` | Install Docker, `uv`, and optional video tools on a Linux x86_64 Gym host |
| `check_host_prerequisites.sh` | Audit an existing Gym host without changing it |
| `prepare_osworld_vm.sh` | Download and verify the pinned OSWorld qcow2 baseline |
| `convert_osworld_tasks.py` | Convert upstream OSWorld manifests and tasks to Gym JSONL |
| `check_task_input_parity.py` | Compare materialized task definitions across two inputs |

Model serving belongs to the deployment layer. Model-specific benchmark
selection belongs to `prepare.py --profile`; neither is implemented here.

Both runtime wrappers require `OSWORLD_RUN_ID`. Set
`NEMO_GYM_CONTROL_HOST` when the control services must advertise a non-loopback
address. Their optional positional argument selects the root for logs and
results; it defaults to the Gym repository root.
