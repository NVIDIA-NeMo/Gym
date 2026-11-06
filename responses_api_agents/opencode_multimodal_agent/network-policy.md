# Network access and per-sample tools

Set `network_access` on the multimodal agent configuration. It defaults to
`none` for both training and evaluation:

| Value | Harness and tool egress | OpenCode web tools |
| --- | --- | --- |
| `all` | Existing sandbox networking, unchanged | Existing configured permissions, unchanged |
| `none` (default) | Configured model API origin only | WebFetch and WebSearch disabled |
| `websearch` | Model API origin and `mcp.exa.ai:443` | WebSearch enabled with Exa; WebFetch disabled |

```yaml
visual_agent:
  responses_api_agents:
    opencode_multimodal_agent:
      network_access: none
```

Use the appropriate agent instance name in a benchmark overlay or a training
recipe. This setting is both the default and the maximum permitted access.
Separate instances can use different policies.

A dataset row can select a policy through structured request metadata:

```json
{"responses_create_params": {"metadata": {"network_access": "websearch"}}}
```

Merge that metadata with the row's existing request fields. Missing metadata uses
the configured default. Samples may restrict access (`all` → `websearch` → `none`),
but cannot exceed the run's configured policy. Invalid or excessive requests fail
with HTTP 422. The policy is not read from question text or model output and is not
added to the prompt. The producer of dataset metadata is trusted within the run's
allowed ceiling.

Both the local training path and owned evaluation sessions select request-local
configuration without changing the shared agent. Existing RL request forwarding
preserves this metadata; no RL change is required. A training run capped at
`websearch` can mix search-enabled and offline samples concurrently.

## Per-sample advertised tools

Set `allowed_tools` on the agent configuration to define the run's tool allowlist.
A sample can select a subset through a JSON-encoded list in Responses metadata:

```json
{"responses_create_params": {"metadata": {
  "network_access": "websearch",
  "allowed_tools": "[\"bash\",\"read\",\"websearch\"]"
}}}
```

Missing sample tool metadata uses the configured list. Leaving both unset keeps
native tool selection. Samples cannot exceed the run allowlist, enable explicitly
disabled tools, or request web tools prohibited by their network policy. Empty
lists disable advertised tools. Invalid lists are rejected before execution.

Supported names are `bash`, `edit`, `glob`, `grep`, `read`, `task`, `todowrite`,
`write`, `websearch`, and `webfetch`. OpenCode shares editing permissions: select
`edit` and `write` together. For GPT models, its native registry substitutes
`apply_patch` for those tools. Existing command/path restriction dictionaries
remain active, and subagent permissions are also restricted. Tool availability
does not restrict what an enabled shell can do within the sandbox's network and
filesystem permissions.

For a dataset mixing all three network policies, use `network_access: all` as the
run ceiling and include both web tools in the run's `allowed_tools`. Each row then
selects its own network policy and tool subset. These settings are request-local
and apply to both execution paths.

The agent wraps the existing launcher. An outer proxy opens connections to
approved origins. OpenCode and its tools run in fresh network and PID namespaces,
with only loopback and a Unix-socket proxy bridge. Proxy environment variables
route supported requests; removing those variables does not restore direct
network connectivity. Namespace setup failure stops execution without falling
back to unrestricted networking. The model API remains reachable so inference
can continue.

The session path currently supports agent-owned Linux sandboxes with Python 3
and util-linux `unshare`. Restricted borrowed sessions are rejected. The local
training path requires the existing Bubblewrap filesystem sandbox. The host must
permit nested user, network, mount and PID namespaces. Runtime installation takes
place before the restricted harness starts; preinstalled OpenCode is preferred.

`websearch` is a search-service egress allowlist, not an API-method filter. Python
can also call the permitted search service. Search queries and results pass
through that service, which may retrieve page content. Arbitrary destination
hosts remain blocked. The pinned OpenCode implementation is documented in
[its search transport](https://github.com/anomalyco/opencode/blob/v1.17.11/packages/opencode/src/tool/mcp-websearch.ts).

This wrapper supplements filesystem isolation. Do not mount host service sockets
or credentials into a restricted sandbox. The model origin is allowed on its
configured port, including other API paths hosted there. A model gateway must not
provide an unrestricted forwarding endpoint.

The wrapper adds two Python helpers and a namespace supervisor per restricted
rollout. Provider-level network restrictions remain in force: this agent setting
can restrict connectivity further, but cannot grant connectivity blocked by the
sandbox provider. The model API exception is specific to this agent's proxy and
does not change Gym's provider-level network policy.

## Complete configuration examples

For evaluation, apply an overlay after the benchmark and agent-session recipes:

```yaml
visual_agent:
  responses_api_agents:
    opencode_multimodal_agent:
      network_access: websearch
      opencode_config:
        tools:
          websearch: true
          webfetch: false
```

The Agent-X OpenCode pairing uses this search-only configuration. Other benchmark
pairings default to `none`. To enable arbitrary network access, select `all`
explicitly; WebFetch and WebSearch remain separately configured tools.

For a training run with three per-sample policies, set the run ceiling and tools:

```yaml
opencode_multimodal_agent:
  responses_api_agents:
    opencode_multimodal_agent:
      network_access: all
      allowed_tools: [bash, edit, glob, grep, read, task, todowrite, write, websearch, webfetch]
```

Then use these metadata values in each dataset row (all other task fields remain
unchanged):

```json
{"responses_create_params":{"metadata":{"network_access":"none","allowed_tools":"[\"bash\",\"edit\",\"glob\",\"grep\",\"read\",\"task\",\"todowrite\",\"write\"]"}}}
{"responses_create_params":{"metadata":{"network_access":"websearch","allowed_tools":"[\"bash\",\"edit\",\"glob\",\"grep\",\"read\",\"task\",\"todowrite\",\"write\",\"websearch\"]"}}}
{"responses_create_params":{"metadata":{"network_access":"all","allowed_tools":"[\"bash\",\"edit\",\"glob\",\"grep\",\"read\",\"task\",\"todowrite\",\"write\",\"websearch\",\"webfetch\"]"}}}
```

`allowed_tools` is a JSON-encoded string because Responses metadata values are
strings. The tool list controls advertisement and permissions; `network_access`
enforces the network boundary even for code launched through Bash.

## Migrating an existing configuration

Omitting `network_access` now selects `none`. A configuration that previously
relied on unrestricted networking must explicitly choose `all`. Restricted local
execution needs Bubblewrap, and restricted sessions need an owned Linux sandbox
with nested namespaces; unsupported runtimes fail rather than bypass isolation.
The model endpoint must be reachable through the proxy. Model-origin access
includes its configured host and port, not just the `/v1` path.
