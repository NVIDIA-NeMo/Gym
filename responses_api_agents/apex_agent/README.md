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

A run that returns normally is scored only when its tools were still there at the end. After Stirrup returns, the
runner fails the rollout when the gateway process has exited (ZIP worlds) or the gateway no longer answers
`/health`. When the run's last MCP calls timed out or raised, with no answered call after them, it also opens a new
MCP session through the gateway and lists the tools; the rollout fails only if that does not answer within 60
seconds. Timed-out calls alone never fail it, because a slow command the policy started times out the same way. A
failed rollout exits the sandbox non-zero and the row is a retried `sandbox_error` instead of the policy's zero. The
number of MCP calls that timed out or raised is reported as `apex_mcp_unanswered_calls` on the response, also for a
rollout that ended on its timeout.

On a failed or timed-out rollout the agent recovers the trajectory from a partial result the sandbox can write. It
first makes a copy inside the sandbox, cut at `max_partial_result_bytes` (256 MiB by default), and downloads only
that copy. When the sandbox can no longer run that command, as after a timeout with `direct_exec`, the agent
downloads the file itself. The download refuses anything that is not a regular file, but it copies a file of any
size to the host before the limit is applied; a copy over the limit is then deleted and the row carries no partial
trajectory. This is the same exposure the final snapshot has while `max_snapshot_bytes` is unset.

`max_snapshot_bytes` (off by default) bounds both snapshots. A final snapshot over it is the policy's own output and
is scored 0 as `apex_agents:final_snapshot_too_large` without calling the grader. An initial snapshot over it is a
dataset problem and stays a terminal `snapshot_too_large` failure.

The Archipelago image is reused when present. If it is missing and automatic building is enabled, Gym exports the
configured pinned Archipelago commit and builds the SIF once. The lightweight Stirrup runtime is pinned in
`stirrup-requirements.txt`, built once inside that image, and cached under `deps/`; it does not vendor either source
repository into Gym.

## Tool-call latency in ZIP worlds

Archipelago's gateway opens a new backend MCP session for every tool call by default. The packaged tool servers
speak stdio, so each call starts a new server process, which costs about 10 seconds per call regardless of the tool.
Set `mcp_session_affinity: true` (`apex_agent_mcp_session_affinity`) to keep one session per tool server for the
whole rollout. It is off by default, so existing numbers stay comparable; wall-clock timeouts make runs with and
without it non-comparable. It applies to ZIP worlds only: prebuilt world images configure their own gateway.

## Mid-rollout checkpoint and resume

A rollout can outlive a cluster allocation. Gym's own resume only skips rows that finished; a rollout killed
mid-flight is re-dispatched from turn zero. Set `apex_agent_resume_checkpoint_dir` to a host directory that
outlives the node (a shared filesystem) to continue such rollouts instead:

- The agent bind-mounts `<dir>/<run_id>/<task_id>/t<task_index>_r<rollout_index>_a<attempt_index>` into the sandbox
  at `/checkpoint` (apptainer provider), where `<run_id>` is the collector's `_ng_run_id`. Evaluations that share a
  checkpoint directory therefore never see each other's checkpoints, and each manifest also names the run that
  wrote it: a checkpoint of another run is not restored. Requests without Gym's dispatch stamps, the run id
  included, have no identity to resume under and run without checkpoints; the agent logs one warning the first time
  a request arrives with indices but no run id. Checkpoints written under an earlier directory layout, without the
  run id level or without a run id in the manifest, are not picked up.
- At the start of a turn's model call, the one point where every tool call of the previous turn has landed, the
  runtime writes Stirrup's cache state (messages, history groups, per-turn tool metadata), the Apex state Stirrup does
  not track (active toolbelt, todo list, model-client counters), a world snapshot of `/filesystem` and
  `/.apps_data`, and the first segment's initial snapshot, then a manifest with a size and SHA-256 per file. Files are
  named by a generation counter and the previous generation is pruned only after the new manifest exists, so a
  crash mid-write keeps the previous checkpoint. Writes are throttled to one per
  `resume_checkpoint_interval_seconds` (default 60) and held back for two minutes after an MCP tool call times out
  client-side, since the server may still be writing. A heartbeat every 30 seconds records time spent since the last
  checkpoint. It names the manifest it extends, so a resumed segment keeps charging its time against the checkpoint
  it started from until it writes a new one. A fresh execution replaces the stored initial snapshot on its first
  checkpoint; a resumed one keeps the verified snapshot of the first segment.
- A request that arrives again with the same run id and the same task, rollout, and attempt indices lands in the
  same directory: the world is restored from the checkpoint, Stirrup continues from the recorded turn, and the
  initial snapshot is carried forward so grading still diffs against the pre-agent world. A retry after a classed
  failure carries the next attempt index and starts clean in its own directory.
- Known limit: a collector that is restarted reserves a new execution number for a rollout it had dispatched but
  got no row for, so that dispatch carries a new attempt index, looks in a new directory, and starts from turn zero
  with a full time budget. The checkpoint of the interrupted execution is left unused under its old index.
- Time already spent, including the heartbeat, is subtracted from the per-task `timeout`. When less than
  `resume_min_remaining_seconds` remains, the rollout is scored 0 as `agent_timeout` with its checkpointed
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
8000 and 8100-8107). Several rollouts on one node then collide in the shared host network namespace. This override
gives each sandbox its own private Apptainer network namespace:

```bash
++apex_agent_apptainer_extra_start_args='[--net,"--network=none",--writable-tmpfs,--cleanenv,--pid,--no-mount,"home,tmp,bind-paths",--home,/root]'
```

Unprivileged Apptainer only offers the `none` network, so the sandbox loses its route to the model server, and the
Stirrup client runs inside the sandbox. With `policy_egress_relay: auto` (the default) the agent enables the relay when
the apptainer start arguments contain `--net` or a `--network` option (`always` and `never` force it), opens
a unix socket on the host that forwards to the model server, binds it into the sandbox at `/egress/policy.sock`, and
`run_stirrup_rollout` points the client at a loopback listener that forwards to that socket. Unix sockets ignore
network namespaces. Only plain-HTTP model endpoints are supported.
