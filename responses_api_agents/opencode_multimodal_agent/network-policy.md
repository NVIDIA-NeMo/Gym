# Network access and per-sample tools

The adapter uses Gym's `SandboxNetworkAccess` values and defaults to `inherit`.

| Value | Allowed network access | Native WebFetch / WebSearch |
| --- | --- | --- |
| `inherit` | Public destinations plus model/MCP grants, excluding control services | Independently configurable |
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
`inherit` neither opens nor removes an outer sandbox restriction. In this adapter,
it additionally excludes local/private destinations and Gym control services;
it no longer launches directly on the provider's inherited host network.

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

All sandbox launches use the namespace/proxy wrapper, including `inherit`.
OpenCode and its child tools run with loopback only, private PID and mount
namespaces, and fresh procfs. A Unix-socket bridge reaches the outer proxy;
direct TCP/UDP cannot bypass it. The compatibility Bubblewrap path binds the
wrapper's private procfs, not the host's. Namespace failure stops execution;
there is no unrestricted fallback.

The proxy admits exact model/MCP host-and-port origins. In `inherit`, it also
admits public IP destinations after DNS resolution. It rejects loopback, private,
link-local, multicast, reserved and IPv6 transition addresses, local interface
addresses, and every registered Gym server except the selected policy model.
Head, agent, other model/judge, verifier/resources and environment server origins
are collected from Gym configuration automatically. Add deployment ingress aliases
not represented there using `protected_control_urls` on the agent configuration.

Every connection refreshes model/tool/control DNS and local interface addresses.
Previously resolved protected addresses remain blocked for the activation.
Unresolvable control origins fail closed. Mixed public/private DNS answers are
rejected; the proxy connects to the checked numeric address without resolving it
again. Explicit model/tool grants are rejected if they share a protected control
IP and port, including under another hostname. Separate tool-only MCP services
from verification/control routes: an MCP path on the same origin is insufficient.

The proxy allows all paths on an explicitly granted origin. It is not an
API-method filter, and CONNECT tunnels carry opaque bytes. The policy model and
tool services must not expose credentials, privileged control routes or an
unrestricted forwarding service. A shared reverse-proxy address may be rejected
because protecting control services takes precedence over granting web access.

All Linux sessions need Bash, tar/gzip and util-linux `unshare`. The private,
checksum-verified stdlib Python runtime is prepared on the host and uploaded at
seed. Seed runs the actual wrapper, checking user, network, mount and PID
namespaces, private procfs, loopback, proxy listeners and control-origin conflicts.
This applies to `inherit` too. Restricted borrowed sessions remain unsupported.
The local training path additionally uses its Bubblewrap filesystem sandbox.

OpenSandbox's provider policy is already active during seed. Restricted
OpenSandbox sessions require pinned, preinstalled OpenCode, plugin dependencies
and ripgrep; use `opencode_sessions_opensandbox.yaml` and the shared agent's
[offline bundle instructions](../opencode_agent/README.md#pinned-offline-runtime-bundle).
There is no unrestricted installation phase followed by a provider policy switch.
Enroot/local apply the wrapper at execution, after seed. Do not loosen an
OpenSandbox policy to permit package downloads throughout an evaluation.

Filesystem isolation remains necessary: do not mount host service sockets or
credentials into the task sandbox. Outer provider restrictions always remain in
force. The local subprocess provider is for trusted conformance tests; it does
not supply a filesystem isolation boundary. Use Enroot/OpenSandbox or the
Bubblewrap compatibility path for untrusted task code. Other Gym agents and
their sandbox policies are unchanged.

## Provider execution and deployment qualification

The recommended `<benchmark>/opencode.yaml` recipes use Gym provider-managed
sandboxes and the shared OpenCode session lifecycle. The multimodal wrapper
adds the process/network boundary described above to both provider execution
and compatibility Bubblewrap execution. This applies to every `inherit`
benchmark, including Agent-X, and preserves its native web-tool interface.
Other OpenCode agents, including BrowseComp's pairing, are unchanged.

Keep head, agent, judge and verifier control services off public ingress where
possible. Configure every additional ingress alias in `protected_control_urls`;
a wrapper cannot discover an unlisted public reverse proxy that forwards to a
control service. Do not expose the head's unauthenticated
`/global_config_dict_yaml` endpoint through a granted model/tool origin.

For each deployment, use synthetic sentinels to confirm host process entries and
control services are inaccessible by hostname, IP and redirects, while model
and public-web access work. Qualify the actual container and namespace policy:
Pyxis/Enroot configurations can differ. OpenSandbox still needs a live deployment
qualification; neither configuration resolution nor local Linux tests establish
its isolation or offline setup behavior.
