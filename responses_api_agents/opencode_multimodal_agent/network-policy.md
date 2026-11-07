# Network access and per-sample tools

The adapter uses Gym's `SandboxNetworkAccess` values and defaults to `inherit`.

| Value | Allowed network access | Native WebFetch / WebSearch |
| --- | --- | --- |
| `inherit` | Existing sandbox network policy, unchanged | Independently configurable |
| `model_only` | Configured model API origin only | Disabled |
| `model_and_tools` | Model API and accepted MCP tool-grant origins | Disabled |

`model_and_tools` uses the effective MCP grants from the agent configuration and
session seed, as the shared OpenCode agent does. It has no built-in destination
allowance. With no grants, it has the same connectivity as `model_only`.
A required MCP grant under `model_only` is rejected before execution. Optional
non-MCP grants do not add network destinations.

Native WebFetch and WebSearch are not MCP grants. To use them, select `inherit`
and enable each tool explicitly. Tool permissions control advertisement and
execution; they do not prevent Bash or Python from using permitted networking.
`inherit` neither opens nor removes an outer sandbox restriction.

## Benchmark configuration

The shipped Agent-X pairing uses inherited networking with both web tools enabled:

```yaml
visual_agent:
  responses_api_agents:
    opencode_multimodal_agent:
      network_access: inherit
      opencode_config:
        tools:
          webfetch: true
          websearch: true
```

Other benchmark profiles and the offline training recipe explicitly use
`model_only` and disable both web tools. Use the appropriate agent instance name
when overriding a recipe.

## Per-sample training and evaluation

The run configuration sets the default and maximum access. A sample may narrow
it through `responses_create_params.metadata.network_access`:
`inherit` → `model_and_tools` → `model_only`. Invalid or excessive policies return
HTTP 422. Metadata is not added to the model prompt. Both execution paths use a
request-local configuration, so concurrent samples cannot change each other's
policy. The local training path does not provision session MCP grants; there,
`model_and_tools` has model-only connectivity.

For a mixed offline/online dataset, use this run configuration:

```yaml
opencode_multimodal_agent:
  responses_api_agents:
    opencode_multimodal_agent:
      network_access: inherit
      allowed_tools: [bash, edit, glob, grep, read, task, todowrite, write, websearch, webfetch]
```

Merge these metadata fields into each row's existing request:

```json
{"responses_create_params":{"metadata":{"network_access":"model_only","allowed_tools":"[\"bash\",\"edit\",\"glob\",\"grep\",\"read\",\"task\",\"todowrite\",\"write\"]"}}}
{"responses_create_params":{"metadata":{"network_access":"inherit","allowed_tools":"[\"bash\",\"edit\",\"glob\",\"grep\",\"read\",\"task\",\"todowrite\",\"write\",\"websearch\",\"webfetch\"]"}}}
```

Responses metadata values are strings, so `allowed_tools` is a JSON-encoded list.
Missing sample tool metadata uses the run's list; leaving both unset keeps native
tool selection. An empty list disables advertised tools. Samples cannot enable a
tool disabled by the run configuration or exceed its allowlist. Select `edit` and
`write` together; OpenCode shares their permission. For GPT models the native
registry substitutes `apply_patch`. Selecting `task` requires enabled subagents.
Existing command/path denies are preserved. Do not add a top-level wildcard
allow permission: in OpenCode 1.17.11 it can override specific denies.

## Enforcement and runtime requirements

OpenSandbox uses the shared `restrict_sandbox_network` helper with model and MCP
grant URLs. Do not combine a restricted mode with an explicit provider
`network_policy`; that conflicting configuration is rejected.

Enroot and local providers lack that egress API. The multimodal adapter retains
its namespace/proxy wrapper: OpenCode and its child tools run with loopback only
and a Unix-socket bridge to approved HTTP(S) origins. The outer process opens
connections; direct TCP/UDP connections cannot bypass the proxy. Namespace setup
failure stops execution without an unrestricted fallback. In owned sessions the
wrapper also enforces per-request narrowing of the seed policy.

The wrapper permits exact host/port origins, including all paths on those origins.
It is not an API-method filter. Code can contact any allowed origin, not only the
tool that introduced it. OpenSandbox's outer host policy may allow more ports;
the wrapper still enforces the configured ports. The model endpoint must not
provide an unrestricted forwarding service.

Owned Linux sessions need Bash, tar/gzip and util-linux `unshare`. Seed installs a
private, checksum-verified stdlib Python runtime before restriction. The local
training path uses its Bubblewrap filesystem sandbox. Both paths require nested
user, network, mount and PID namespaces. Restricted borrowed sessions and
unsupported providers fail closed. Runtime installation happens before restricted
execution; preinstalled OpenCode is preferred.

Filesystem isolation remains necessary: do not mount host service sockets or
credentials into the task sandbox. Outer provider restrictions always remain in
force. Other Gym agents and their sandbox policies are unchanged.
