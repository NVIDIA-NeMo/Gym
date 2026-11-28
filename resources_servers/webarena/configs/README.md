# WebArena timeout configuration

`webarena.yaml` contains deadlines at the boundaries where work can block a
browser actor or the rollout. All values are seconds.

## Timeout ownership

### Resources server

| Setting | Scope |
| --- | --- |
| `browser_startup_timeout_seconds` | One actor starting Xvfb, Chromium, and passing its health check. It does not include the startup stagger or total pool prewarming time. |
| `browser_pool_wait_timeout_seconds` | Waiting for an idle browser when the pool is full. |
| `browser_operation_timeout_seconds` | One actor reset or tool step. Reset includes login, initial navigation retries, snapshots, and the first screenshot. |
| `browser_evaluation_timeout_seconds` | Final Classic WebArena evaluation against the live browser. |
| `browser_session_max_seconds` | Absolute lifetime of one leased browser session. Active operations are allowed to finish. |
| `judge_timeout_seconds` | One evaluator HTTP call to the configured judge or helper service. |

Actor cleanup has an internal 30-second deadline. Initial task navigation uses
three attempts of at most 35 seconds with one- and two-second backoffs.

### Web agent

| Setting | Scope |
| --- | --- |
| `rollout_timeout_seconds` | Total rollout budget, including the interaction and finalization allocations. |
| `finalization_reserve_seconds` | Portion reserved for live-browser verification and cleanup. Interaction receives `rollout_timeout_seconds - finalization_reserve_seconds`. |
| `resource_request_timeout_seconds` | Generic client-side ceiling for seed, step, and verify HTTP requests. Server-side phase deadlines usually fire first. |

The verify client timeout is derived rather than configured:

```text
verify_client_timeout =
    min(resource_request_timeout_seconds,
        finalization_reserve_seconds - 30)
```

The `30` reserves time for cleanup.

## Recommended relationships

Use at least 30 seconds of transport and serialization margin:

```text
resource_request_timeout
    >= browser_pool_wait_timeout + browser_operation_timeout + 30

verify_client_timeout
    >= browser_evaluation_timeout + 30

finalization_reserve
    >= browser_evaluation_timeout + cleanup_timeout + 30

rollout_timeout > finalization_reserve

browser_session_max > rollout_timeout

judge_timeout + 30 <= browser_evaluation_timeout
```

The first relationship covers the worst seed path: wait for an actor and then
reset it. A normal step only needs the operation timeout.

For initial navigation, also check:

```text
browser_operation_timeout
    > (3 * initial_navigation_attempt_timeout)
      + retry_backoffs
      + login/snapshot margin
```

With the current 35-second attempts and 1+2-second backoffs, navigation can use
108 seconds. The current 180-second operation timeout leaves 72 seconds for
login, snapshots, and screenshot capture.

## Current settings

The checked-in configuration gives:

```text
pool wait                 = 300
browser operation         = 180
resource request          = 510
browser evaluation        = 300
judge call                = 120
finalization reserve      = 360
derived verify client     = min(510, 360 - 30) = 330
rollout total             = 1200
interaction budget        = 1200 - 360 = 840
browser session maximum   = 1800
```

These relationships are consistent:

- Seed has a 30-second client margin: `300 + 180 + 30 = 510`.
- Verification has a 30-second client margin: `300 + 30 = 330`.
- Finalization covers evaluation, cleanup, and 30 seconds of overhead:
  `300 + 30 + 30 = 360`.
- The 20-minute rollout allocates 14 minutes to seed/model/actions and six
  minutes to verification/cleanup.
- The browser session has ten minutes of margin beyond the rollout.
- One 120-second judge call fits within the 300-second evaluation deadline.

Tasks that make several slow judge/helper calls may require a larger evaluation
deadline.

## Quick consistency check

Run this from the Gym repository root in an activated NeMo Gym environment
after changing timeout values:

```bash
python - <<'PY'
from omegaconf import OmegaConf

cfg = OmegaConf.load("resources_servers/webarena/configs/webarena.yaml")
resource = cfg.webarena_resources_server.resources_servers.webarena
agent = cfg.webarena_agent.responses_api_agents.webarena_agent

cleanup = 30
margin = 30
pool_wait = float(resource.browser_pool_wait_timeout_seconds)
operation = float(resource.browser_operation_timeout_seconds)
evaluation = float(resource.browser_evaluation_timeout_seconds)
session_max = float(resource.browser_session_max_seconds)
judge = float(resource.judge_timeout_seconds)
request = float(agent.resource_request_timeout_seconds)
finalization = float(agent.finalization_reserve_seconds)
rollout = float(agent.rollout_timeout_seconds)
verify = min(
    request,
    finalization - cleanup,
)

checks = {
    "seed request covers pool wait + reset":
        request >= pool_wait + operation + margin,
    "verify client covers evaluation":
        verify >= evaluation + margin,
    "finalization covers evaluation + cleanup":
        finalization >= evaluation + cleanup + margin,
    "rollout leaves an interaction budget":
        rollout > finalization,
    "session outlives rollout":
        session_max > rollout,
    "judge fits inside evaluation":
        judge + margin <= evaluation,
}

for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")

if not all(checks.values()):
    raise SystemExit(1)
PY
```

This checks deadline ordering, not real service latency. Use startup, browser,
and judge latency metrics to increase values when operations regularly approach
their limits.
