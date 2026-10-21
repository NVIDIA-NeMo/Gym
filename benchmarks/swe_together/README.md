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

The preparation and startup commands preserve the prototype recipe. The required
deployment correction is role-isolated model routes; its original same-IP settings
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

The reference's two WebFetch/WebSearch-restricted tasks are covered by this
recipe's global web-tool denial and restricted egress. This is intentionally
stricter for the remaining tasks, matching the attached deployment's network
contract. Some original tasks request public GitHub context or already block it
in their image. Public network context is not supplied or silently fetched.
The recipe overrides Gym’s default bash filters to allow commands, matching the
reference’s effective bash policy; network restrictions remain provider-enforced.
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

Candidate swaps must support `resume_harness_session` and the requested observation
capabilities. Judge swaps need a borrowed single-turn session and confirmed close.
Change the inherited instances in a configuration overlay and remove unused base
instances. Mid-episode crash recovery requires coordinated restoration of native
history, filesystem, simulator state and time; this environment currently restarts.
