# SwarmBench

> Internal name. It collides with unrelated public benchmarks (arXiv 2505.04364, arXiv 2608.30661), so do not cite
> it publicly as "SwarmBench" until it is renamed or disambiguated.

SwarmBench v2.1 has 8 tracks x 3 tiers (easy, medium, hard). Each tier is an `(n, wall-clock budget)` setting. There
are 208 test instances (10 easy, 8 medium and 8 hard per track) and 24 dev instances (1 per track and tier). The
tracks, tiers, budget rule, tools-only rule and reporting are described in the
[resources server README](../../resources_servers/swarmbench/README.md).

`manifest_v2.json` is the frozen v2.1 manifest. For every instance it records `id`, `track`, `tier`, `split`, `n`,
`seed`, `budget_s`, `grace_s`, `track_version`, and a SHA-256 over the prompt (with the canary) and every generated
world and gold file. `prepare.py` regenerates each instance with the server's track code, fails on any hash
mismatch, and writes one row per instance. A row holds the prompt (canary line, task, budget line), the track's
tools in `responses_create_params.tools` (for agents that read tools from the row), and the manifest entry as
`verifier_metadata`. The server rebuilds the world for each rollout. `instances.json` is the v1.0 manifest
(4 tracks, no tiers or budget), kept so v1 numbers stay reproducible.

```bash
gym eval prepare --benchmark swarmbench/opencode_solo                                # test split (about 3 min)
gym eval prepare --benchmark swarmbench/opencode_solo +prepare_script_args.split=dev  # dev split
gym eval prepare --benchmark swarmbench/opencode_solo \
  +prepare_script_args.manifest=benchmarks/swarmbench/instances.json                  # v1.0
```

`manifest_types.json` freezes the 20 promoted pilot types (see the server README): 180 test instances (3 per type and
tier) and 60 dev instances (1 per type and tier), hashed the same way. Prepare it with
`+prepare_script_args.manifest=benchmarks/swarmbench/manifest_types.json`. `prepare.build_types_manifest()` rebuilds
it after a type changes.

Every split is written to `data/swarmbench_benchmark.jsonl`, so prepare the split you want to run right before
running it.

## Solo vs swarm (OpenCode)

Both configs run `opencode_sandboxed_agent` in OpenSandbox. The world tools reach the sandbox over MCP, egress is
limited to the model server and the SwarmBench server (`network_access: model_and_tools`), and shell, file
and web tools are denied (the tools-only rule).

| config | OpenCode `task` tool | prompt |
|---|---|---|
| `swarmbench/opencode_solo` | denied | the task prompt |
| `swarmbench/opencode_task` | allowed (native subagents, same model) | the task prompt plus "Use subagents." |

Both keep automatic compaction on and set no practical step cap. The run timeout is 2700 s (the largest budget plus
grace, plus startup). A rollout is scored on what it left behind (the edited repo, or the answer), so
`execution_failure_reward_zero` is off.

```bash
gym eval run \
  --benchmark swarmbench/opencode_solo \
  --split benchmark \
  --model-type vllm_model \
  --model "$POLICY_MODEL" \
  --model-url "$POLICY_BASE_URL" \
  --model-api-key "$POLICY_API_KEY" \
  --output results/swarmbench_opencode_solo.jsonl
```

Replace `opencode_solo` with `opencode_task` for the swarm arm. For pass@1 over k repeats, set the dataset's
`num_repeats` to k.

- Set `OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY` and `OPENCODE_SANDBOX_IMAGE`. The sandbox has no internet, so
  OpenCode must be preinstalled in the image or installed by `remote_opencode_install_script_path` from a host the
  sandbox may reach. Pass `+swarmbench_opencode_preinstalled=false` in the second case.
- The SwarmBench server must be reachable from the sandboxes. The OpenCode benchmark config sets
  `use_absolute_ip: true`.
- `opencode_max_context_window` is 245760, below a 262144 served context. Lower it for models served with less.

Any other Gym agent can run the same rows. See "Running any Gym agent" in the server README.
