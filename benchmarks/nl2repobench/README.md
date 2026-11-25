# NL2RepoBench in Gym: datasets x harnesses x network modes

One resources server (`resources_servers/nl2repobench`) serves every combination below; only the config file and the `agent_network_mode` override change.

## Datasets (pick one config family)
| Dataset | Tasks | Configs | Notes |
|---|---|---|---|
| Upstream NL2RepoBench | 104 | `benchmarks/nl2repobench/opencode.yaml` (`opencode_sandboxed_agent.yaml` to reproduce published runs) | pinned per-task grading images (ghcr); tests run in the agent's own workspace |
| Internal held-out sets (verified-50, legacy 75-task) | 50 / 75 | not in Gym | same resources server and configs layout |

## Harnesses (pick one config per dataset)
- **opencode** (`opencode.yaml`, `responses_api_agents/opencode_agent` on agent sessions): the resources server lends the
  agent sandbox and `single_agent_turn_legacy` runs the episode. Same step cap (500) and no compaction as below, but one
  activation: `opencode_agent` has no idle nudge or stuck detector yet, so a reply without a tool call ends the run.
  The restricted modes need the OpenCode runtime staged in the lent sandbox, which this preset does not do yet: run it in
  `official` (or another open-egress mode) until it does.
- **opencode, legacy relay** (`opencode_sandboxed_agent.yaml`, deprecated `responses_api_agents/opencode_sandboxed_agent`
  behind `legacy_agent`): kept to reproduce the runs below. No compaction and no pruning (`compaction.auto: false`, `prune: false` in `opencode_sandboxed_agent.yaml`), so a conversation that outgrows the declared 524288-token window (`opencode_max_context_window`) ends the run with a context error instead of being summarised. It continues an idle session with `idle_nudge` (finish with `TASK_COMPLETE`), ends a looping one with `stuck_detector`, and holds OpenCode's 500-step cap as one budget across resumes.

## Network modes (`++nl2repobench_opencode_resources_server.resources_servers.nl2repobench.agent_network_mode=<mode>`)
| Mode | Agent sandbox | Egress | Reference-lookup protection |
|---|---|---|---|
| `official` (default) | generic image | open | none: treat results as an upper bound |
| `block_target` | generic image | open | legacy pip/git/curl wrappers matching the task id only (kept unchanged for reproducing old runs; ineffective on task sets whose ids are not real package names) |
| `block_target_v2` | generic image | source-hosting domains denied (GitHub etc.), PyPI open | `guard_v2.py`: real package/repo identifiers, subcommand-aware pip/git/curl/uv/apt wrappers, python/node code inspection, no `-m venv` |
| `no_internet_preinstalled` | per-task image with dependencies preinstalled | deny-all except the model host | reference repo wiped before the first turn; its editable-install metadata and same-named cargo crates scrubbed |

The modes that deny egress (`no_internet_preinstalled`, and `official` with `enforce_agent_no_network: true`) need the
`opensandbox` provider, which applies the policy, and a `sandbox_model_server`. With any other provider, or without a
model server, the seed is refused (HTTP 422) instead of running the agent with open egress or with no model.

For `no_internet_preinstalled` on the upstream 104, also set `scrub_workspace_installs=true` and `scrub_target_installs=true` (both off by default). An audit of all 104 images (2026-11-02) found the real implementation in 13 of them beyond what the default wipe reaches, and agents copied it into `/workspace`:
- pip/uv caches holding the target's wheel (arxiv-mcp-server): `scrub_workspace_installs`.
- target installs whose distribution name is the task name (databases, gitingest, justext, stamina, wifiphisher): `scrub_workspace_installs` empties them and `scrub_target_installs` removes the leftover dirs.
- installs from a local checkout under another name (paillier -> `phe`, pytorch-grad-cam -> `grad_cam`), a `setup.py install` egg (pytz), and checkouts left at the root (`/project` in asteval, `/app` in paillier, `/jinja` with the held-out tests): `scrub_target_installs`.

Four targets ship vendored inside pip or setuptools in their images (six, tenacity, typing_extensions, more-itertools) and cannot be removed without breaking pip; exclude them when a leak-free score matters. Images that run as a non-root user without a writable home (more-itertools, pyautogui) also need `++...opencode_sandboxed_agent.opencode_data_home_in_tmp=true`.

**Official vs clean score.** With `grading_protocol: upstream`, every row carries both. `reward` is NL2RepoBench's own scoring: every "N passed" line of the pytest output summed, over the stored test count, capped at 1. Use it to compare with published numbers. `clean_reward` counts only each pytest command's outer summary line, over the larger of the stored count and the outer run's passed+failed+errors. It is `None` for `clean_exclude_tasks`, which defaults to the four vendored targets above, so `mean/clean_reward` in the aggregate metrics is the clean score over the rest. The two differ where tests run pytest themselves (pytest-cov: 510 "passed" against 184 tests, 1.00 official, 0.81 clean) or the stored count is stale (parse: 96 stored, 131 run). On a 99-row DeepSeek V4.1 Flash run, 13 rows changed, all downward.

On tasks whose `test_commands.json` has a clone URL, the guard identifiers come from it; upstream tasks have no clone URL, so `block_target_v2` falls back to the task id there (the egress deny rules still apply).

## OpenCode runtime in the sandbox (legacy relay preset)
`opencode_sandboxed_agent.yaml` sets no runtime paths, so the OpenCode agent downloads the installer from `opencode.ai`. That works in the modes with open egress (`official`, `block_target`, `block_target_v2`) and needs no staging.

`no_internet_preinstalled` has no egress, so supply a pinned runtime the sandbox can read and make the agent refuse to start without it, so nothing is downloaded and a missing runtime fails at startup, before any sandbox is seeded or model called:

```
++nl2repobench_opencode_sandboxed_agent.responses_api_agents.opencode_sandboxed_agent.require_opencode_runtime=true
++nl2repobench_opencode_sandboxed_agent.responses_api_agents.opencode_sandboxed_agent.remote_opencode_install_script_path=<install.sh path in the sandbox>
++nl2repobench_opencode_sandboxed_agent.responses_api_agents.opencode_sandboxed_agent.remote_opencode_binary_path=<opencode binary path in the sandbox>
```

Use `preinstalled_opencode=true` instead of the two paths for an image that already has `opencode` on `PATH`. A deployment that sets the two paths without `require_opencode_runtime` (a cached runtime on an open-egress mode) works as before.

## Combinations and status (what has actually been run)
- **Verified-50** x opencode x {`block_target_v2`, `no_internet_preinstalled`}: full runs (Lightning final, 4 SFT checkpoints, Qwen3.8-27B); `block_target_v2` results were regraded with the final verifier.
- **75-task legacy** testset: opencode under `block_target` / `block_target_v2` / `no_internet_preinstalled` (graded with the old verifier; superseded).
- **Upstream** x opencode: `official`, `block_target` and `no_internet_preinstalled` runs predate this change; `block_target_v2` has only a partial Qwen run (5 of 104 tasks, timed out).

## Grading notes
Grading of task sets with a reference clone (the internal sets): reference clone + install, (optional) test-only dependency step, **purge of the reference implementation**, overlay of the agent's workspace, held-out tests; every mode grades in
the per-task image for verified-50. Upstream 104 grades with `grading_protocol: upstream` (since 2026-10-27), which follows NL2RepoBench's
own post-processor: the agent's packaging files (setup.py, pyproject.toml, requirements*, ...) and its files at the held-out test paths
are deleted, the rest is overlaid on the pinned per-task image's /workspace, PYTHONPATH=/workspace, every command runs, and the pytest
commands' "N passed" lines are summed. Before that, runs used `gym_strict`: the agent's own packaging was installed as-is and the first
failing command zeroed the row (about half of all rows), and agent-written tests at the held-out paths could inflate a score.
Residual risks of every mode other than `official` (open-egress grading, the guards' own limits) are listed under
"Limits of the scores" in [`resources_servers/nl2repobench/README.md`](../../resources_servers/nl2repobench/README.md).
