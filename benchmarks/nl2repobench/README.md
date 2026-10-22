# NL2RepoBench in Gym: datasets x harnesses x network modes

One resources server (`resources_servers/nl2repobench`) serves every combination below; only the config file and the `agent_network_mode` override change.

## Datasets (pick one config family)
| Dataset | Tasks | Configs | Notes |
|---|---|---|---|
| Upstream NL2RepoBench | 104 | `benchmarks/nl2repobench/{opencode,openhands}.yaml` | pinned per-task grading images (ghcr); tests run in the agent's own workspace |
| vahid_1016 held-out, **verified-50** (recommended) | 50 | `benchmarks/nl2repobench_vahid1016_v50/{opencode,openhands}.yaml` | self-contained (tasks + jsonl in the folder); every task's reference code passes 100% of its tests in the per-task image; see its README |
| vahid_1016 held-out, 75-task (legacy) | 75 | `benchmarks/nl2repobench_vahid1016_testset/opencode.yaml` | superseded by verified-50: ~20 tasks cannot score above 0 in its verify image |

## Harnesses (pick one config per dataset)
- **opencode** (`responses_api_agents/opencode_sandboxed_agent`): auto-compacts its context near the declared window (524K here).
- **OpenHands SDK 1.50** (`responses_api_agents/openhands_sdk_sandboxed_agent`): native tool calls, no condenser (no compaction). Needs an offline bundle (`build_bundle.sh`, then
  `++nl2repobench_opencode_sandboxed_agent.responses_api_agents.openhands_sdk_sandboxed_agent.bundle_path=...`) for `block_target_v2` and `no_internet_preinstalled`.

## Network modes (`++nl2repobench_opencode_resources_server.resources_servers.nl2repobench.agent_network_mode=<mode>`)
| Mode | Agent sandbox | Egress | Reference-lookup protection |
|---|---|---|---|
| `official` (default) | generic image | open | none: treat results as an upper bound |
| `block_target` | generic image | open | legacy pip/git/curl wrappers matching the task id only (kept unchanged for reproducing old runs; ineffective on vahid_1016 ids) |
| `block_target_v2` | generic image | source-hosting domains denied (GitHub etc.), PyPI open | `guard_v2.py`: real package/repo identifiers, subcommand-aware pip/git/curl/uv/apt wrappers, python/node code inspection, no `-m venv` |
| `no_internet_preinstalled` | per-task image with dependencies preinstalled | deny-all except the model host | reference repo wiped before the first turn; its editable-install metadata and same-named cargo crates scrubbed |

On vahid_1016 tasks the guard identifiers come from the clone URL in `test_commands.json`; upstream tasks have no clone URL, so `block_target_v2` falls back to the task id there (the egress deny rules still apply).

## Combinations and status (what has actually been run)
- **Verified-50** x {opencode, OpenHands} x {`block_target_v2`, `no_internet_preinstalled`}: full runs (Lightning final, 4 SFT checkpoints, Qwen3.8-27B); `block_target_v2` results were regraded with the final verifier.
- **75-task legacy** testset: opencode and OpenHands under `block_target` / `block_target_v2` / `no_internet_preinstalled` (graded with the old verifier; superseded).
- **Upstream** x opencode: `official`, `block_target` and `no_internet_preinstalled` runs predate this change; `block_target_v2` has only a partial Qwen run (5 of 104 tasks, timed out).
- **Upstream** x OpenHands: the config resolves and the agent has run on the vahid_1016 tasks, but there is **no completed upstream x OpenHands run yet**: treat it as untested end to end until a smoke run is recorded.
- Parent-repo launchers: `launch_upstream_nl2repo_eval.sh` (opencode), `launch_upstream_nl2repo_eval_openhands.sh`, `launch_v50_{opencode,openhands}.sh`.

## Grading notes
verified-50 / vahid_1016 grading: reference clone + install, (optional) test-only dependency step, **purge of the reference implementation**, overlay of the agent's workspace, held-out tests; every mode grades in
the per-task image for verified-50. Upstream grading is unchanged (tests in the agent's workspace on the pinned per-task image).
Background, leak audit, residual risks: `sft_runs/lightning_vahid1016/testset/LEAK_AUDIT.md` in the parent repo.
