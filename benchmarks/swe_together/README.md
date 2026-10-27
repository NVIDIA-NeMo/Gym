# SWE-Together

Run the 109 pinned interactive repository tasks with OpenCode, a user simulator,
and a single-turn Claude Code judge. The environment owns both sandboxes and
calls the reusable agent servers through their session and Responses APIs.

## Prepare and run

From an installed Gym checkout, set `POLICY_BASE_URL`, `POLICY_MODEL`, and
`POLICY_API_KEY`; set `SIMULATOR_*`, `JUDGE_*`, and `INTERACTION_*` model endpoints,
models and keys as in [models.yaml](models.yaml). Credentials stay on the trusted
model-server host. The simulator and interaction scorer use Gemini 3.1 Pro Preview;
the judge uses Claude Opus 4.6 in the original recipe.

```bash
export SWE_TOGETHER_WORK_DIR="$PWD/outputs/swe_together"
python -m benchmarks.swe_together.prepare
python -m benchmarks.swe_together.prepare_runtime \
  --output "$SWE_TOGETHER_WORK_DIR/runtime"
```

Preparation verifies the [pinned upstream](https://github.com/Togetherbench/SWE-Together/tree/891d19eb4b3a64a47c3d49bbd066a311e0133254)
task assets and image inventory. `SWE_TOGETHER_SOURCE` may select an existing
checkout. Runtime preparation verifies OpenCode 1.15.13, Claude Code 2.1.108,
ripgrep and a complete Python runtime; `--seed-manifest` reuses a verified cache.

For [OpenSandbox](opensandbox.yaml), set `OPENSANDBOX_DOMAIN`,
`OPENSANDBOX_API_KEY`, and `OPENSANDBOX_SHARED_STORAGE_PATH`. The provider must
support private shared volumes and default-deny network policy. Set
`SWE_TOGETHER_POLICY_HOST` / `SWE_TOGETHER_JUDGE_HOST` to reachable listener
interfaces and `SWE_TOGETHER_POLICY_EGRESS_TARGET` /
`SWE_TOGETHER_JUDGE_EGRESS_TARGET` to their allowed addresses.

Use separate candidate and judge interfaces or fixed-role proxies. A shared host
IP allows both roles to reach every listener on that IP under the provider's
host-based policy. Keep simulator/interaction listeners private; proxies should
fix their model and reject provider-side tools and arbitrary extensions. Preserve
`X-Session-Id`, `X-Session-Affinity`, and `X-OpenCode-Assistant-Message-Id` for capture.

Start the benchmark's dependency proxy on a dedicated trusted host/interface
reachable from the task sandboxes. It needs Python 3.11+, OpenSSL and `packaging>=24.2`
(included with Gym; install it separately on a standalone proxy host). Use the same
random `SWE_TOGETHER_DEPENDENCY_ADMIN_KEY` (at least 32 characters) in that service
and the trusted Gym environment; keep it out of task environments. Configure
`SWE_TOGETHER_DEPENDENCY_PROXY_URL=http://<dependency-host>:3128` on the Gym host.
The endpoint must be separate from both model routes and expose no unauthenticated
management services. In particular, do not run it in a peer OpenSandbox task:
allowing that task's IP also exposes its exec daemon. The HTTP endpoint and admin
traffic belong on a trusted private network.

```bash
# Additional shell on the dependency host, from this checkout.
# Set SWE_TOGETHER_DEPENDENCY_ADMIN_KEY from your secret environment first.
python -m pip install 'packaging>=24.2'
python -m environment_servers.swe_together.egress_proxy \
  --state-dir /private/path/swet-dependency-ca
```

The environment registers an immutable package denylist before candidate work
and reuses it for judging. Sandboxes receive only a scoped lease token and public
CA trust. The proxy permits the pinned reference's Python, Node, Rust and apt
registries and model catalogs, while blocking registry searches and the task's
own packages. Direct registry access stays denied; HTTPS certificate validation
remains enabled. Python artifact checks support wheels, including the reference's
short `name-version.whl` spelling, and source archives (`.tar.gz`, `.tar.bz2`,
`.tar.xz`, `.tar.Z`, `.tgz`, `.tar`, `.zip`), plus their `.metadata` sidecars.
Unrecognized artifact names are denied.
Leases are revoked after cleanup and expire if the controller
is lost. A proxy restart invalidates its leases; restart affected episodes rather
than silently opening access. Stop the service when the campaign is finished.

```bash
# Shell 1
gym env start \
  --config benchmarks/swe_together/opencode.yaml \
  --config benchmarks/swe_together/opensandbox.yaml

# Shell 2: same node and environment, after startup
gym eval run --no-serve \
  --config benchmarks/swe_together/opencode.yaml \
  --config benchmarks/swe_together/opensandbox.yaml \
  --input benchmarks/swe_together/data/full109.jsonl \
  --output "$SWE_TOGETHER_WORK_DIR/results.jsonl" \
  --num-repeats 2 --concurrency 12
```

The Gym preparation/start/eval commands preserve the prototype recipe. The
additional dependency service is needed because OpenSandbox's supported
host-based policy cannot enforce the reference's package/path restrictions.
Role-isolated model routes are also required; the prototype's same-IP settings
do not enforce that isolation. Prepared inputs go directly to `--input`.

## Runtime and benchmark boundaries

The environment mounts Python and verified candidate executables read-only.
OpenCode copies its executable into its own runtime cache at seed and uses the
configured interpreter for supervision and transcript capture. The configured
PATH prefix adds the mounted ripgrep directory while retaining the image PATH;
image HOME/XDG variables are retained in task shells. The adapter's private
configuration/history directories are removed on close.

Each candidate request contains only the new user message, a one-based
`x-ng-agent-activation` header and `metadata.agent_timeout_sec`. Seed cookies bind
the conversation. A single 4,800-second interaction deadline starts after setup
and includes candidate work, repository inspection and simulator calls. Setup,
grading and finalization have separate budgets. Servers are shared across episodes.

The recipe selects the configured policy model as OpenCode's native model ID,
rather than the adapter's compatibility alias. Gateways with namespaced serving
IDs may override `swe_together_candidate.responses_api_agents.opencode_agent.sandbox_model_id`
with the model's native ID. Model identity affects OpenCode's family-specific
prompts, tools and sampling. This recipe sets explicit context/output limits;
reasoning effort is controlled by the chosen model route.

The recipe disables WebFetch/WebSearch globally, which is stricter than the
reference's tool restrictions on two tasks. Dependency access follows the
reference host and package/path rules, including task-specific host additions.
Public GitHub access remains blocked by the reference network policy.
The recipe overrides Gym’s default bash filters to allow commands, matching the
reference’s effective bash policy. The provider restricts destinations and the
dependency proxy enforces registry request restrictions outside task sandboxes.
Registry paths are URL-decoded before checking package names and search routes;
this closes an encoded-URL bypass in the pinned reference policy.
Project guidance is discovered explicitly and appended to the initial input;
the adapter keeps automatic project config disabled. These choices, supplied
ripgrep, image-default execution users, and the normalized Responses-based
simulator activity view require baseline qualification before score comparisons.

The grading boundary is the initial working directory and repositories discovered
at setup, including ignored/binary files, Git state, links and deletions. A trusted
helper seals a checksum-verified archive; the fresh judge sandbox restores it
from a separate read-only shared mount. Only manifests and receipts cross the
controller. OS/package changes outside those roots and live services are excluded.
The judge receives the withheld rubric/tests after replication and closes before
the verdict is read. See [environment details](../../environment_servers/swe_together/README.md).

## Smoke qualification

Use `--config benchmarks/swe_together/smoke.yaml` in both commands and
`--num-repeats 1`. This runs real models with a 600-second interaction budget and
grades even empty patches. A measured zero is healthy.

```bash
python -m benchmarks.swe_together.check_smoke \
  "$SWE_TOGETHER_WORK_DIR/results.jsonl" \
  --asset-root "$SWE_TOGETHER_WORK_DIR/source/tasks" \
  --output "$SWE_TOGETHER_WORK_DIR/smoke-health.json"
```

The checker requires each of the 109 tasks exactly once, valid frozen-goal grading,
interaction measurements, ordered activations, a stable native conversation,
submission replication and confirmed agent/sandbox cleanup. Retain failures and
all attempt provenance. Smoke health is distinct from benchmark accuracy and
Gym's generic trajectory health report.

The collector-facing `ng_trajectory` follows Gym's materialized task identity
convention. `candidate-agent-close.json` retains the candidate's original logical
task IDs and captured decisions, including when the episode fails.

Candidate swaps must support `resume_harness_session` and the requested observation
capabilities. Judge swaps need a borrowed single-turn session and confirmed close.
Change the inherited instances in a configuration overlay and remove unused base
instances. Mid-episode crash recovery requires coordinated restoration of native
history, filesystem, simulator state and time; this environment currently restarts.
