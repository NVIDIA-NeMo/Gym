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

Restricted owned Linux sessions need Bash, tar/gzip and util-linux `unshare`. The private,
checksum-verified stdlib Python runtime is prepared on the host and uploaded at
seed. Restricted seed runs the actual wrapper, checking user, network, mount and
PID namespaces, private procfs, loopback and proxy listeners. Per-request narrowing
of an `inherit` session checks these before its first restricted activation.
An unchanged `inherit` session does not install this helper or require nested
namespaces; execution and isolation use the configured provider directly.
The local training path uses its Bubblewrap filesystem sandbox. Restricted
borrowed sessions and unsupported providers fail closed.

OpenSandbox's provider policy is already active during seed. Restricted
OpenSandbox sessions require pinned, preinstalled OpenCode, plugin dependencies
and ripgrep; use `opencode_sessions_opensandbox.yaml` and the shared agent's
[offline bundle instructions](../opencode_agent/README.md#pinned-offline-runtime-bundle).
There is no unrestricted installation phase followed by a provider policy switch.
Enroot/local apply the wrapper at execution, after seed. Do not loosen an
OpenSandbox policy to permit package downloads throughout an evaluation.

Filesystem isolation remains necessary: do not mount host service sockets or
credentials into the task sandbox. Outer provider restrictions always remain in
force. Other Gym agents and their sandbox policies are unchanged.

## Provider execution and host service isolation

The recommended `<benchmark>/opencode.yaml` recipes execute OpenCode inside a
Gym provider-managed sandbox. This follows the execution model of the merged
[BrowseComp OpenCode pairing](../../benchmarks/browsecomp/configs/browsecomp_opencode.yaml):
the provider supplies the process/filesystem boundary, and `inherit` retains its
network policy. The multimodal adapter uses the shared agent's session lifecycle;
it does not copy BrowseComp's Exa tools or change benchmark network settings.
Provider selection alone does not prove a deployment's network separation. The
following checks apply to any `inherit` benchmark, including Agent-X.

Agent-X needs public web access. Its `inherit` pairing must run inside a provider
boundary that isolates task processes from the Gym host. The local compatibility
Bubblewrap path currently shares host procfs and host networking; it is not a
credential-isolation boundary. A read-only procfs mount can expose host command
lines, and switching only that mount to private procfs still leaves host services
reachable through inherited networking.

For a remote sandbox deployment, keep the Gym head, agent control endpoints,
judge and verifier services on host loopback or a network inaccessible to task
sandboxes. Expose only the model gateway and explicitly granted task services.
The OpenSandbox overlay binds the head and agent to loopback; also configure
any judge/verifier instances and external firewall rules for the deployment.
Do not expose the head's unauthenticated `/global_config_dict_yaml` endpoint on
the sandbox-facing address. Do not mount host procfs, service sockets, runtime
configurations or credentials into the task image.

This arrangement preserves web tools and `inherit` policy. To qualify an
isolation claim for a deployment, use synthetic sentinels to check process/file
isolation, confirm public web and model access, and confirm head/judge/control
services are unreachable by hostname and IP. Local Enroot or Bubblewrap with
inherited host networking does not establish that separation. No remote
OpenSandbox qualification is claimed by the overlay alone.
