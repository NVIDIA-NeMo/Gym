# Apex Agents rollout agent

This agent runs the same Stirrup 0.1 harness family used by Gym's GDPval agent inside a pinned Archipelago image.
Archipelago owns the world, workplace MCP servers, and `/apps` + `/mcp/` gateway. Stirrup owns the model/tool loop.
The Apex harness repository is not cloned or imported.

Each rollout:

1. restores the cached world ZIP into `/filesystem` and `/.apps_data`;
2. starts Archipelago's gateway and configures the MCP servers packaged in the image;
3. saves the initial local-grader ZIP snapshot;
4. runs Stirrup for at most 200 model turns;
5. saves the final ZIP snapshot; and
6. sends completed submissions, their final answer, and both snapshots to the resources server for grading.

The initial Stirrup toolbelt contains `list_tools`, `inspect_tool`, `add_tool`, `remove_tool`, `todo_write`, and
`finish`. MCP tools are absent until the model adds them. Adding/removing a tool changes the schemas sent on the next
model turn. MCP calls time out after 60 seconds. Long text output uses a 20,000-character head plus 5,000-character
tail excerpt when it exceeds a 24,000-token budget, estimated at four characters per token; Stirrup compresses image
blocks to about one megapixel when serializing them for the model.

`finish` is the only submission mechanism. A completed submission is rejected while any todo is pending or in
progress. A max-turn exit or `status="incomplete"` does not call the grader; its initial/final snapshots are still
saved under the configured artifact output directory for inspection.

The Archipelago image is reused when present. If it is missing and automatic building is enabled, Gym exports the
configured pinned Archipelago commit and builds the SIF once. The lightweight Stirrup runtime is pinned in
`stirrup-requirements.txt`, built once inside that image, and cached under `deps/`; it does not vendor either source
repository into Gym.

## Input row modes

The default row format is the public Apex Agents benchmark format. Those rows omit `runtime_mode`, include
`task_input_files`, and run as `runtime_mode: "world_zip"`. Gym downloads or restores the world ZIP, seeds the
session, and then starts the standard Apex/Archipelago gateway path. This default keeps the original benchmark data
schema backward-compatible.

Private prebuilt deliveries use `runtime_mode: "prebuilt_world"`. These rows do not carry `task_input_files`; instead
they must include a `task_slug`, and the agent must be configured with `prebuilt_world_manifest`. The manifest maps the
row's `world_id` to a trusted local SIF image. The prebuilt path uploads only the lightweight Stirrup runtime helpers,
bootstraps them inside the task SIF, and runs the delivered startup command for that task slug.

## Fixed-port worlds and network isolation

Some prebuilt world images start their gateway, app services, and MCP servers on fixed ports (for example
8000 and 8100-8107). Several rollouts on one node then collide in the shared host network namespace. The prebuilt-world
launcher can give each sandbox its own private Apptainer network namespace:

```bash
++apex_agent_apptainer_extra_start_args='[--net,"--network=none",--writable-tmpfs,--cleanenv,--pid,--no-mount,"home,tmp,bind-paths",--home,/root]'
```

Unprivileged Apptainer only offers the `none` network, so the sandbox loses its route to the model server, and the
Stirrup client runs inside the sandbox. With `policy_egress_relay: auto` (the default) the agent enables the relay when
the apptainer start arguments contain `--net` or a `--network` option (`always` and `never` force it), opens
a unix socket on the host that forwards to the model server, binds it into the sandbox at `/egress/policy.sock`, and
`run_stirrup_rollout` points the client at a loopback listener that forwards to that socket. Unix sockets ignore
network namespaces. Only plain-HTTP model endpoints are supported.
