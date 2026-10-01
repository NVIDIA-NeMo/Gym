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

Stirrup summarizes the conversation once it reaches 70% of the client's context window. Stirrup 0.1 reads that window
from the same `max_tokens` value it sends as `max_completion_tokens`, so by default summarization starts at 70% of
`max_output_tokens`. Set `context_window_tokens` (`apex_agent_context_window_tokens`) to the served model's context
length to summarize at 70% of the real window instead; requests keep `max_output_tokens` as their output cap.

`finish` is the only submission mechanism. A completed submission is rejected while any todo is pending or in
progress. A max-turn exit or `status="incomplete"` does not call the grader; its initial/final snapshots are still
saved under the configured artifact output directory for inspection.

The Archipelago image is reused when present. If it is missing and automatic building is enabled, Gym exports the
configured pinned Archipelago commit and builds the SIF once. The lightweight Stirrup runtime is pinned in
`stirrup-requirements.txt`, built once inside that image, and cached under `deps/`; it does not vendor either source
repository into Gym.

## Mid-rollout checkpoint and resume

A rollout can outlive a cluster allocation. Gym's own resume only skips rows that finished; a rollout killed
mid-flight is re-dispatched from turn zero. Set `apex_agent_resume_checkpoint_dir` to a host directory that
outlives the node (a shared filesystem) to continue such rollouts instead:

- The agent bind-mounts `<dir>/<task_id>/t<task_index>_r<rollout_index>_a<attempt_index>` into the sandbox at
  `/checkpoint` (apptainer provider). Requests without Gym's dispatch stamps have no identity to resume under and
  run without checkpoints. The unified Slurm harness sets the directory to `<run dir>/resume_checkpoints` by
  default (`APEX_RESUME_CHECKPOINT_DIR`); an empty value disables it.
- At the start of a turn's model call, the one point where every tool call of the previous turn has landed, the
  runtime writes Stirrup's cache state (messages, history groups, per-turn tool metadata), the Apex state Stirrup does
  not track (active toolbelt, todo list, model-client counters), a world snapshot of `/filesystem` and
  `/.apps_data`, and the first segment's initial snapshot, then a manifest with a size and SHA-256 per file. Files are
  named by a generation counter and the previous generation is pruned only after the new manifest exists, so a
  crash mid-write keeps the previous checkpoint. Writes are throttled to one per
  `resume_checkpoint_interval_seconds` (default 60) and held back for two minutes after an MCP tool call times out
  client-side, since the server may still be writing. A heartbeat every 30 seconds records time spent since the last
  checkpoint.
- Gym re-dispatches a killed rollout with the same task, rollout, and attempt indices, so it lands in the same
  directory: the world is restored from the checkpoint, Stirrup continues from the recorded turn, and the initial
  snapshot is carried forward so grading still diffs against the pre-agent world. A retry after a classed failure
  carries the next attempt index and starts clean in its own directory.
- Time already spent, including the heartbeat, is subtracted from the per-task `timeout`. When less than
  `resume_min_remaining_seconds` remains, the rollout is reported as `timeout_exceeded` with its checkpointed
  trajectory instead of starting a segment.
- Any row written for the attempt, graded or a classed failure, deletes its checkpoint directory. A manifest that
  fails verification is retried once and otherwise ignored, never deleted; the next checkpoint supersedes it.

Rows carry `apex_resume_segments`, `apex_resumed_from_turn`, and `apex_resume_checkpoints` on the response. A
resumed segment replays from the last checkpoint, so the turns since it are lost: at most one turn when a
checkpoint was written at every boundary, more when the interval throttle skipped some. In-memory state of the MCP
server processes is not captured; their durable state lives under the two snapshotted roots.

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
