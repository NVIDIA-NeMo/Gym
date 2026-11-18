# Recorded-context ordering evaluation

The stock Gymnasium agent drives the existing PartnerIF Resources implementation
through `/reset` and `/step`; `legacy_agent` relays each complete episode. No Gym
agent, Environment Server or `nemo_gym/server_runtime` implementation is changed.
The shared `partner_if` verifier remains stateless for ordinary tasks.

`recorded_context_app.py` owns each frozen simulator session and semantic grading.
Its helpers under `resources_servers/partner_if/recorded_context` provide bounded
virtual Read, Bash (including bounded searches), Skill, Write, Edit and Explore behavior. Data snapshots
are checked against pinned asset hashes and never imported or executed. Known
missing files differ from unknown snapshot coverage; unsupported operations and
invalid tool calls record a coverage gap and mask the sample.

After each complete response/tool batch, the shared LLM judge assesses the full
fresh capture. A failed ordering criterion ends the episode with zero reward.
A passed semantic judgment becomes decisive only after a returned,
identity-validated sibling probe. All assistant calls precede all results in a
batch. Neither labels in model output nor a probe batched with an earlier-required
prescription can manufacture a passing receipt. Pending safe actions continue.

Resources authorizes at most six parent turns and eight policy calls total,
including Explore's read-only child calls. It checks the remaining budget before
authorizing another parent. Temperature is 0.7, top_p 0.95, and each call has an
8,192-token cap. Policy/provider transport retries are disabled. Unknown coverage,
judge/infrastructure faults and undecided cutoffs are masked, with diagnostics.

## Input and results

Reset returns no extra observation, and `text_only_history: false` preserves the
Responses input and tools. Gym's stock Model Server converts Responses to Chat
for the provider. That conversion can normalize null or whitespace-only assistant
content and `strict: false` tool fields; it does not promise byte-for-byte equality
with the old custom Chat request. This is an explicit harness revision, which the
local 2.1.8 release candidate records as `gymnasium-v1` / `gym-responses-to-chat`.
Frozen row and asset actor parameters are checked before execution.

Stock Gymnasium omits terminal tool output from its own response and counts only
parent usage. Resources therefore supplies a canonical flat report under terminal
`info.partner_if_recorded_context_result`, including terminal results, parent and
Explore usage, original task/actor binding, grading fields and verifier history.
The existing dispatcher normalizes only a host-configured recorded-context route,
validates that report and retains transport failures/cleanup warnings. Collector
IDs and attempts travel in host-owned reporting metadata outside the actor input,
so an older report cannot be rebound to another capture or attempt. It does
not control execution or ask the judge again.

Sessions and completed step replies are held in memory. Repeated step IDs return
the same completed reply. An interrupted, uncertain step is not replayed as new
model work. `/close` supports the stock agent's cancellation cleanup. There is no
disk receipt store, saved-prefix policy loop or cross-process continuation;
restart an interrupted rollout after process failure.

## Assets, release and local verification

The published private 2.1.7 source release remains immutable and its version-2
asset envelope includes `runtime.json`. The local release staging command in the
benchmark README writes a proposed 2.1.8 dataset plus unchanged assets and their
digests without uploading anything. Publication and repinning are pending.

`config_recorded_context.yaml` is included by the full benchmark configuration.
Set `recorded_context_package_dir` in a private override to select another pinned
package. An older version-1 envelope also needs
`recorded_context_runtime_contract_path` and
`recorded_context_runtime_contract_sha256`. Credentials belong in ignored config.

Public synthetic fixtures under `benchmarks/partner_if/tests/fixtures/replay`
exercise the actual stock agent, relay and Resources HTTP boundaries with scripted
model responses and semantic judgments. These tests verify integration and limits;
they do not establish semantic judge quality. A fresh real-model rollout is still
needed before submitting this harness revision.
