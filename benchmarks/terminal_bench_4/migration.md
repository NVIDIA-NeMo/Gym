# TB4 agent/resources migration

## Review baseline

This branch starts at `flafrance/terminal-bench-4`,
`66efd1b7a7c0240d998d735e335eef1ed340ca77`. Review the additional change with
`git diff flafrance/terminal-bench-4...HEAD`.

The plan inspected `53ee40725c0090e1c8fbe64fbe8e92e837bc17f2`.
The actual base additionally contains `9dc7293d6` (offline verifier egress) and
`66efd1b7a` (Compose runtime requirements). Those fixes are prerequisites and
are outside this migration's diff. The inspected `origin/pr-3332` is
`bb03de6ef5aef3587a7f2334b1fca5471c2dd716`; its merge base with the task branch is
`4402098b145ee2985e760e33d46f9adc321cc64b`. Comparing against that merge base would
include inherited Compose infrastructure and unrelated base history.

The 66 task pins, category membership, dataset digest, and Compose image metadata
are unchanged. Provider renewal, request/limit mapping, shared memory, offline
verifier policy, task overlays, and endpoint routing reuse the existing adapter.

## Implementation boundary

- `nemo_gym/sandbox/handoff.py` defines the typed seed, termination, and verify
  contract. Borrowed connections release their transport; resources own teardown.
- `resources_servers/terminal_bench_4` owns trusted task resolution, lifecycle,
  authoritative budgets, official grading, artifacts, and persisted retries.
- `opencode_sandboxed_agent` retains its existing model-routing and conversion
  code, adding a borrowed-environment execution path. Its legacy path is unchanged.
- `miniswe_sandboxed_agent` runs the real pinned mini-SWE `DefaultAgent`, bridges
  its synchronous loop to Gym's async model and sandbox clients, and joins it
  before verification. The existing SWE-bench agent remains separate.

Harbor **0.23.0** provides a custom-agent extension, but no supported public
pause/resume interface for an externally executing harness. The resources server
therefore retains its public `Trial.run()` lifecycle with `ExternalAgent` as a
rendezvous adapter. That adapter reports readiness and waits; it does not invoke
`HarborAgent.run_job()`, OpenCode, or mini-SWE. Harbor still performs task setup,
health checks, declared main/sidecar artifact transfer, separate verification,
and cleanup. A separate 360-second setup handshake prevents installer time from
consuming the task's agent budget.

Sessions are isolated by the caller identity and rollout ID. Duplicate seed and
verify calls share one attempt and its first submitted response. Completed
verification can be replayed from disk. Restarted active episodes reject resume;
provider TTL remains the fallback after abrupt owner death. Use one resources
worker per artifact directory. Phase and termination records are persisted before
verification, and teardown remains with the resources owner.

## Evaluation profile

OpenCode stays at **1.17.11**. mini-SWE is **2.1.0**, profile
`tb4-miniswe-text-v1`: one bash action per response, an explicit submission
convention, task skills in the instruction, and a persistent task-local MCP CLI.
The CLI pins `mcp==1.29.0` and `httpx-aiohttp==0.2.0`, preserves tool session state,
and passes image results as multimodal model inputs. These prompt and tool
changes are part of the mini-SWE evaluation profile.

Live smoke runs use `gpt-5.4-mini-2026-03-17`, three harness steps, and a
900-second cap that can only shorten the official agent budget. Setup and
verifier budgets retain their official values. A healthy smoke needs actual
model output plus official grading and no infrastructure error. Reward zero is
expected in these short runs and is not a benchmark score. Missing submissions
may return a grade before deeper verifier checks execute.

The standalone smoke runner uses real HTTP Gym agent/resources servers. mini-SWE
uses the Gym model server. Remote OpenCode uses the public OpenAI endpoint in
this standalone mode; remote Gym model routing and semantic-turn capture are
covered by regression tests but are **not certified by these live runs**.

## Validation record

Validation proceeds CPU, then Compose, then GPU against the existing
`../artifacts/tb4-all-task-health-report.md` baseline (57 healthy, 9 blocked).
Accepted run directories live under `../artifacts/` relative to the repository.
Each retains health results, agent output, Harbor trial artifacts, and logs.
Recent runs also record source hashes in `run.json`.

CPU OpenCode covers all 52 tasks: `split-resources-cpu-opencode-v4` covers
formal-crypto, interleaved-vigenere, and ks-solver-cpp; `split-resources-cpu-all`
covers the other 49. The result is **50 healthy / 52 attempted**. The two failures
remain risk-scorer-replay and rs-archive-clone, whose non-root images cannot perform
the required root setup on this deployment. The former readiness failures
cad-model, fin-saccr-rwa, and ontology-kg-querying are healthy in this run.

mini-SWE CPU checks pass on five tasks: the same three initial tasks in
`split-resources-cpu-miniswe-v2`, then freecad-impeller and ontology-kg-querying in
`split-resources-cpu-miniswe-mcp`. Despite the latter directory's historical name,
those task packages declare no MCP tools. Only medical-claims-processing declares
nonempty MCP configuration in this pinned dataset; its deployment status controls
whether the official MCP task is supported.

Compose checks complete at **8 healthy / 11 attempted for each harness**, in
`split-resources-compose-opencode` and `split-resources-compose-miniswe`.
The three baseline blockers remain: kv-live-surgery requires `SYS_PTRACE`;
medical-claims-processing and payments-pipeline-fix cannot perform required root
execution on this deployment. The official MCP task therefore remains unsupported.
The healthy cases include shared-network-mode tasks, startup health dependencies,
and main/sidecar state collection before separate verification. This is declared
TCP-forwarding support, not full namespace equivalence.

OpenCode GPU checks finish at **3 healthy / 3 attempted**, in
`split-resources-gpu-opencode`. fp8-rmsnorm-gemm recovers after verifier allocation
retries: the required CUDA source transfers successfully and compiles. Its
optional default `/logs/artifacts` copy fails and remains recorded in the artifact
manifest; the task-declared CUDA file is present. A read-only hardware probe in a
running GPU verifier reports **NVIDIA H100 80GB HBM3**.

The deterministic parity run in `split-resources-parity/comparison.json` submits
the same six official ks-solver-cpp solution files through the new resources path
and Harbor's Oracle path. Both receive **1.0**, with task pin
`sha256:d4eef6bb296028e57be0abb4b19dab71cf04b807cbe22fad4bc64174f9380cc0`.
The comparison retains hashes of every submitted solution file. This establishes
one file-state/separate-verifier parity case, not general stateful-service parity.

The deliberate one-second ks-solver-cpp deadline canary in
`split-resources-timeout-canary-v2` returns `termination=timeout`, an official
zero, and no infrastructure error, while retaining worker artifact references.
It intentionally has no OpenCode export/model output and is excluded from the
healthy model matrix.

Early runs before the `setsid --wait` and export-download fixes are retained for
diagnosis and excluded from accepted results. Valid official grades from runs
without captured model output do not count as healthy model smoke checks.

### Automated checks

The existing sandbox, OpenSandbox, Compose, and Harbor suite passed 340 tests.
Six legacy TB2.1 and mini-SWE sandbox checks also pass.
The focused handoff/agent/environment/preparation suite passed 94 tests with **96.86%**
coverage across the new resources, handoff/lifecycle, mini-SWE, and borrowed
OpenCode modules. Tests cover pins, duplicate requests, caller isolation,
restart/shutdown, preparation rollback, cancellation, timeout grading,
missing grades/exports, borrowed ownership, trajectory/usage conversion, and
MCP transports. A real MCP stdio server confirms persistent state across calls.

```sh
MSWEA_GLOBAL_CONFIG_DIR=/tmp/tb4-miniswe MSWEA_SILENT_STARTUP=1 pytest -q \
  resources_servers/terminal_bench_4/tests \
  responses_api_agents/miniswe_sandboxed_agent/tests \
  responses_api_agents/opencode_sandboxed_agent/tests \
  responses_api_agents/harbor_agent_general/tests/test_sandbox_environment.py \
  --cov=resources_servers.terminal_bench_4 --cov=nemo_gym.sandbox.handoff \
  --cov=nemo_gym.sandbox.agent --cov=responses_api_agents.miniswe_sandboxed_agent \
  --cov=responses_api_agents.opencode_sandboxed_agent.borrowed --cov-report=term-missing
```

See the [smoke command](README.md#validation) for reproduction. Run each category
explicitly; the runner stops before the next category if any task is unhealthy.
Inspect and classify failures before continuing. Live artifacts and resolved
configuration remain local; provider credentials do not appear in handoffs or
prepared rows. The resources configuration remains `verified: false`.
