# NL2RepoBench in Gym: datasets x harnesses x network modes

One resources server (`resources_servers/nl2repobench`) serves every combination below; only the config file and the `agent_network_mode` override change.

## Datasets (pick one config family)
| Dataset | Tasks | Configs | Notes |
|---|---|---|---|
| Upstream NL2RepoBench | 104 | `benchmarks/nl2repobench/{opencode,openhands}.yaml` | pinned per-task grading images (ghcr); tests run in the agent's own workspace |
| Internal held-out sets (verified-50, legacy 75-task) | 50 / 75 | not in Gym: kept in the internal nl2repo repo (`benchmarks/`), linked into `Gym/benchmarks/` by its launchers | same resources server and configs layout |

## Harnesses (pick one config per dataset)
- **opencode** (`responses_api_agents/opencode_sandboxed_agent`): no compaction and no pruning (`compaction.auto: false`, `prune: false` in `opencode.yaml`), so a conversation that outgrows the declared 524288-token window (`opencode_max_context_window`) ends the run with a context error instead of being summarised. It continues an idle session with `idle_nudge` (finish with `TASK_COMPLETE`), ends a looping one with `stuck_detector`, and holds OpenCode's 500-step cap as one budget across resumes.
- **OpenHands SDK 1.50** (`responses_api_agents/openhands_sdk_sandboxed_agent`): native tool calls, no condenser (no compaction). Needs an offline bundle (`build_bundle.sh`, then
  `++nl2repobench_opencode_sandboxed_agent.responses_api_agents.openhands_sdk_sandboxed_agent.bundle_path=...`) for `block_target_v2` and `no_internet_preinstalled`.

## Network modes (`++nl2repobench_opencode_resources_server.resources_servers.nl2repobench.agent_network_mode=<mode>`)
| Mode | Agent sandbox | Egress | Reference-lookup protection |
|---|---|---|---|
| `official` (default) | generic image | open | none: treat results as an upper bound |
| `block_target` | generic image | open | legacy pip/git/curl wrappers matching the task id only (kept unchanged for reproducing old runs; ineffective on vahid_1016 ids) |
| `block_target_v2` | generic image | source-hosting domains denied (GitHub etc.), PyPI open | `guard_v2.py`: real package/repo identifiers, subcommand-aware pip/git/curl/uv/apt wrappers, python/node code inspection, no `-m venv` |
| `no_internet_preinstalled` | per-task image with dependencies preinstalled | deny-all except the model host | reference repo wiped before the first turn; its editable-install metadata and same-named cargo crates scrubbed |

For `no_internet_preinstalled` on the upstream 104, also set `scrub_workspace_installs=true` and `scrub_target_installs=true` (both off by default). An audit of all 104 images (2026-11-02) found the real implementation in 13 of them beyond what the default wipe reaches, and agents copied it into `/workspace`:
- pip/uv caches holding the target's wheel (arxiv-mcp-server): `scrub_workspace_installs`.
- target installs whose distribution name is the task name (databases, gitingest, justext, stamina, wifiphisher): `scrub_workspace_installs` empties them and `scrub_target_installs` removes the leftover dirs.
- installs from a local checkout under another name (paillier -> `phe`, pytorch-grad-cam -> `grad_cam`), a `setup.py install` egg (pytz), and checkouts left at the root (`/project` in asteval, `/app` in paillier, `/jinja` with the held-out tests): `scrub_target_installs`.

Four targets ship vendored inside pip or setuptools in their images (six, tenacity, typing_extensions, more-itertools) and cannot be removed without breaking pip; exclude them when a leak-free score matters. Images that run as a non-root user without a writable home (more-itertools, pyautogui) also need `++...opencode_sandboxed_agent.opencode_data_home_in_tmp=true`.

**Official vs clean score.** With `grading_protocol: upstream`, every row carries both. `reward` is NL2RepoBench's own scoring: every "N passed" line of the pytest output summed, over the stored test count, capped at 1. Use it to compare with published numbers. `clean_reward` counts only each pytest command's outer summary line, over the larger of the stored count and the outer run's passed+failed+errors. It is `None` for `clean_exclude_tasks`, which defaults to the four vendored targets above, so `mean/clean_reward` in the aggregate metrics is the clean score over the rest. The two differ where tests run pytest themselves (pytest-cov: 510 "passed" against 184 tests, 1.00 official, 0.81 clean) or the stored count is stale (parse: 96 stored, 131 run). On a 99-row DeepSeek V4.1 Flash run, 13 rows changed, all downward.

On vahid_1016 tasks the guard identifiers come from the clone URL in `test_commands.json`; upstream tasks have no clone URL, so `block_target_v2` falls back to the task id there (the egress deny rules still apply).

## OpenCode runtime in the sandbox
`opencode.yaml` sets no runtime paths, so the OpenCode agent downloads the installer from `opencode.ai`. That works in the modes with open egress (`official`, `block_target`, `block_target_v2`) and needs no staging.

`no_internet_preinstalled` has no egress, so supply a pinned runtime the sandbox can read and make the agent refuse to start without it, so nothing is downloaded and a missing runtime fails at startup, before any sandbox is seeded or model called:

```
++nl2repobench_opencode_sandboxed_agent.responses_api_agents.opencode_sandboxed_agent.require_opencode_runtime=true
++nl2repobench_opencode_sandboxed_agent.responses_api_agents.opencode_sandboxed_agent.remote_opencode_install_script_path=<install.sh path in the sandbox>
++nl2repobench_opencode_sandboxed_agent.responses_api_agents.opencode_sandboxed_agent.remote_opencode_binary_path=<opencode binary path in the sandbox>
```

Use `preinstalled_opencode=true` instead of the two paths for an image that already has `opencode` on `PATH`. A deployment that sets the two paths without `require_opencode_runtime` (a cached runtime on an open-egress mode) works as before.

## Long runs: client timeout
Leave `global_aiohttp_client_timeout_s` unset (the default: no total or read limit on the shared aiohttp client). The shared client also carries each whole-rollout `/run` call through the `legacy_agent` relay, which here lasts up to the 28800 s episode deadline (21600 s agent budget, then export, collection and grading), so a global limit below that cuts off healthy runs. A model call is already bounded by the agent itself (`opencode_model_call_timeout` for OpenCode, `model_call_timeout_s` for OpenHands). Set the option only to bound a peer that accepts a connection and never answers, and then to more than 28800 s, for example `+global_aiohttp_client_timeout_s=32400`.

## Combinations and status (what has actually been run)
- **Verified-50** x {opencode, OpenHands} x {`block_target_v2`, `no_internet_preinstalled`}: full runs (Lightning final, 4 SFT checkpoints, Qwen3.8-27B); `block_target_v2` results were regraded with the final verifier.
- **75-task legacy** testset: opencode and OpenHands under `block_target` / `block_target_v2` / `no_internet_preinstalled` (graded with the old verifier; superseded).
- **Upstream** x opencode: `official`, `block_target` and `no_internet_preinstalled` runs predate this change; `block_target_v2` has only a partial Qwen run (5 of 104 tasks, timed out).
- **Upstream** x OpenHands: the config resolves and the agent has run on the vahid_1016 tasks, but there is **no completed upstream x OpenHands run yet**: treat it as untested end to end until a smoke run is recorded.
- Parent-repo launchers: `launch_upstream_nl2repo_eval.sh` (opencode), `launch_upstream_nl2repo_eval_openhands.sh`, `launch_v50_{opencode,openhands}.sh`.

## Grading notes
verified-50 / vahid_1016 grading: reference clone + install, (optional) test-only dependency step, **purge of the reference implementation**, overlay of the agent's workspace, held-out tests; every mode grades in
the per-task image for verified-50. Upstream 104 grades with `grading_protocol: upstream` (since 2026-10-27), which follows NL2RepoBench's
own post-processor: the agent's packaging files (setup.py, pyproject.toml, requirements*, ...) and its files at the held-out test paths
are deleted, the rest is overlaid on the pinned per-task image's /workspace, PYTHONPATH=/workspace, every command runs, and the pytest
commands' "N passed" lines are summed. Before that, runs used `gym_strict`: the agent's own packaging was installed as-is and the first
failing command zeroed the row (about half of all rows), and agent-written tests at the held-out paths could inflate a score.
Background, leak audit, residual risks: `sft_runs/lightning_vahid1016/testset/LEAK_AUDIT.md` in the parent repo.
