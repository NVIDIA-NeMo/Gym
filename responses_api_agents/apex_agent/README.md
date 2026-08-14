# Apex Agents agent

This agent composes two upstream pieces rather than reimplementing either one:

- Archipelago's OCI image provides Linux, Office tooling, `sandbox_fs.so`, the world directories, and the MCP
  server implementations.
- Mercor's `apex-agent-harness` provides `ArchipelagoMCPEnvironment`, `ApexAgent`, model drivers, MCP subprocess
  orchestration, snapshots, and the rollout loop.

Gym provides the per-task lifecycle, model-server routing, held-out dataset boundary, artifact transfer, and
verification.

## Runtime boundary

```text
trusted Gym agent process
  seed session + fetch world from resources server
  create one sandbox from the configured Archipelago OCI image
  inject the cached, git-pinned apex-agent-harness runtime
  upload only world.zip + task-visible prompt
       |
       v
per-task sandbox
  upstream ApexAgent policy loop ---> Gym policy model server
       |
       +-- upstream ArchipelagoMCPEnvironment
       |     +-- stdio MCP subprocesses from /app/mcp_servers
       |     +-- /filesystem + /.apps_data world state
       |
       +-- final answer + changed-artifact snapshot
       v
trusted Gym agent process ---> Apex resources server ---> Gym judge model server
```

The MCP servers run as stdio subprocesses inside the same outer sandbox as the harness. This matches the native
harness design and CVDP's initial trust boundary. Model-issued code is confined by Archipelago's `sandbox_fs.so`;
its default blocked paths include `/app`, where the MCP implementations, injected harness, Gym runner, and runner
configuration live. The sandbox never receives verifier metadata, rubrics, gold outputs, judge configuration, or
host model credentials.

An external `/mcp` service remains a possible later design for tools that require secrets. The initial integration
does not inject an FMP API key. EDGAR's live SEC requests require a contact User-Agent. Archipelago's public Apex
Agents example supplies `Mercor APEX-Agents apex@mercor.com`; Gym uses that same public, non-secret identity by
default and allows it to be overridden with `apex_edgar_user_agent`.

## Image and harness composition

Gym uses Archipelago's Dockerfile directly; it does not maintain a second container recipe or a separate manual
build script. When the configured local SIF is missing and `apex_agents_auto_build: true`, startup fetches the
pinned Archipelago commit, exports a clean build context, runs its Dockerfile, converts the resulting local OCI
image to SIF, and caches both outputs. The work runs under the agent's setup lock, so concurrent rollouts wait for
one setup and reuse it. All three image modes are supported:

1. Existing local SIF: set `apex_agents_image` to its path; no build occurs.
2. Prebuilt OCI image: set `apex_agents_image` to a registry reference; Apptainer pulls/caches it and no Docker build
   occurs.
3. Upstream Dockerfile: use the default missing local SIF path and keep `apex_agents_auto_build: true`; Gym fetches
   and builds the configured immutable Archipelago revision once.

No image or source path configuration is required for the standard flow. The generated SIF is cached under the
agent directory. The `apex_agents_*` and `apex_archipelago_*` overrides exist only for prebuilt images, shared
cluster paths, alternate upstream revisions, or a developer's existing source checkout; they do not belong in a
normal `env.yaml`.

```text
Mercor-Intelligence/archipelago@0cb5c476c219a9df637e0bd37fb86b2361f4ab89
```

The agent server fetches the repository using the trusted host's Git authentication at the immutable commit below,
then builds a small cached runtime *inside that same image*.

```text
Mercor-Intelligence/apex-agent-harness@1fd94befbb570eb6effe76b1895e5d599e820227
```

Gym uses `git archive` at that exact revision. Host credentials are used only for the trusted fetch and are never
forwarded into task sandboxes. The resulting runtime is archived once and injected into each task sandbox, following
the dependency-archive approach used by CVDP. Gym installs the clean pinned package without patching its source. This
keeps the world image and harness independently pinned and avoids rebuilding Archipelago just to change the agent
package. Interactive users can rely on their existing Git credential helper; CI can use a token or SSH key; and
developers can set `apex_harness_root` to an existing checkout.

This fetch is an agent-server startup preflight. On a fresh installation, the server does not become ready unless it
can fetch and verify the pinned commit. A successful source archive is cached, so later starts do not require network
access. The rollout path repeats the preflight check before image construction and task seeding as a safety net.

No tokenizer is downloaded or loaded by the agent. The upstream harness has a shared `set_client()` helper for its
token-ID/Tinker execution path, but this integration uses its native OpenAI-tools loop instead: structured messages
and tool schemas are sent directly to Gym's policy model server. Gym therefore installs the client fields without
calling that token-ID helper. The policy server owns chat templating, tokenization, and context-window enforcement.

Token counts in rollout output come from the OpenAI-compatible response's `usage` object. Because there is no local
tokenizer, the harness's optional client-side token-exact tool-output truncation is disabled. A request that exceeds
the serving limit fails at the policy server with its normal context-length error instead of being pre-counted by a
second tokenizer in the sandbox.

## Configuration

The policy model is Gym's global `policy_model_name`, normally set by `gym eval run --model` or in `env.yaml`; it is
not duplicated in the agent app. `ApexAgentConfig` is only the validated schema: every runtime value is supplied by
`configs/apex_agent.yaml`. The scalar rollout controls below read their `apex_agent_*` overrides from the root
`env.yaml` and retain documented YAML fallbacks:

- `max_turns`
- `max_output_tokens`
- `max_tool_calls_per_turn`
- `temperature`
- `top_p`
- `max_snapshot_bytes`
- `max_world_bytes`

The compressed-transfer byte limits are disabled by default because Apex and Archipelago do not specify values for
them. Optional caps can be enabled after real dataset/world and rollout artifact sizes are measured during
baselining. Independent safe-extraction protections remain enabled in the verifier.

For EDGAR worlds, the default matches Archipelago's public example. Override it only if required by the deployment:

```yaml
apex_edgar_user_agent: Mercor APEX-Agents apex@mercor.com
```

## Run

Compose the agent and resources configs, configure the policy and judge model servers, then run the example data:

```bash
gym env start \
  --config responses_api_agents/apex_agent/configs/apex_agent.yaml \
  --config resources_servers/apex_agents/configs/apex_agents.yaml \
  --model-type vllm_model

gym eval run --no-serve \
  --agent apex_agent \
  --input resources_servers/apex_agents/data/example.jsonl \
  --output results/apex_agents_example.jsonl \
  --num-repeats 1
```

Run the standalone script at the same upstream revisions first, then compare trajectories, criterion scores, and
all-criteria Pass@1 before setting `verified: true`.

# Licensing information

Code: Apache 2.0

External source and dataset licenses remain those of their respective upstream projects. The inspected
`apex-agent-harness` revision does not contain a `LICENSE` file or package-license metadata, so do not redistribute
the cached harness archive until upstream licensing terms are confirmed.
