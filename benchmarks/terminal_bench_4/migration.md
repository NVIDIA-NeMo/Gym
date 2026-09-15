# TB4 agent/resources migration (historical reference)

The earlier Harbor-backed split established separate agent and resources servers.
The current implementation uses Gym's native TB4 lifecycle; see
[native lifecycle notes](native-lifecycle.md) for its behavior and validation.

## Retained implementation

- `nemo_gym/sandbox/handoff.py` defines the seed, termination, and verify contract.
  Borrowed connections release their transport; resources own teardown.
- `resources_servers/terminal_bench_4` owns pinned task resolution, lifecycle,
  authoritative budgets, official grading, artifacts, and persisted retries.
- `miniswe_sandboxed_agent` runs mini-SWE 2.1.0 `DefaultAgent`, bridges its
  synchronous loop to Gym's async model and sandbox clients, and joins the loop
  before verification. The existing SWE-bench agent remains separate.
- The Compose normalizer lives in `resources_servers/terminal_bench_4/compose_config.py`.
  TB4 no longer requires changes to the general Harbor agent or its dependencies.

The 66 task pins, category membership, dataset digest, and Compose image metadata
are unchanged. Provider renewal, request/limit mapping, shared memory, offline
verifier policy, task overlays, and endpoint routing remain required by mini-SWE.

## Mini-SWE reference validation

The historical mini-SWE sample covered five CPU tasks, all 11 Compose tasks,
and all three GPU tasks. It recorded 5/5, 8/11, and 3/3 healthy, respectively.
These measured results remain in [health-baseline.json](health-baseline.json).
The later full native CPU sweep is recorded in
[health-native-miniswe-cpu.json](health-native-miniswe-cpu.json).

Historical run directories under `../artifacts/` include
`split-resources-cpu-miniswe-v2`, `split-resources-cpu-miniswe-mcp`,
`split-resources-compose-miniswe`, and `split-resources-gpu-miniswe`.
Despite the CPU directory's historical name, those sampled CPU packages declare
no MCP tools. The official MCP task is `medical-claims-processing`; its existing
Compose deployment blocker prevents live MCP certification.

Capped smokes use three agent steps and a 900-second cap. Healthy results require
model output, official grading, and no infrastructure error. They are not
benchmark scores. See the [current smoke commands](README.md#validation) for
reproduction and the native lifecycle notes for subsequent fixes and limitations.

## Automated checks

```sh
MSWEA_GLOBAL_CONFIG_DIR=/tmp/tb4-miniswe MSWEA_SILENT_STARTUP=1 pytest -q \
  resources_servers/terminal_bench_4/tests \
  responses_api_agents/miniswe_sandboxed_agent/tests \
  --cov=resources_servers.terminal_bench_4 --cov=nemo_gym.sandbox.handoff \
  --cov=nemo_gym.sandbox.agent --cov=responses_api_agents.miniswe_sandboxed_agent \
  --cov-report=term-missing
```
