# Token capture and usage (item 6)

Line numbers are at main `099832fa7e`.

- `B` = `nemo_gym/agents/base_responses_api_agent.py`
- `G` = `fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx`
- `S` = `.agents/skills/add-harness/SKILL.md`

## URL

- **Every model client uses the rollout-prefixed URL** from `base_url_for_run()` or `resolve_model_base_url()`. This includes summarization, compaction, title and subagent calls (B:1233, B:1259-1271; G:136, G:646-654; S:45).
- **Keep the path segments when forwarding:** `/ng-rollout/<key>[/training-token-capture]`.
- **Return the cookies** the Model Server sends.
- **A call sent straight to a provider is not captured,** and training masks that rollout.
- **If some calls cannot be routed,** mark the adapter evaluation-only and leave `token_id_capture` off.

## `x-session-id`

**Rule:** send `x-session-id` on every model request. Its value equals the `invocation_id` of an `AgentInvocation` the adapter returns at close (G:650).

**Without it:**

- Capture stores `client_session_id=None` (`nemo_gym/models/model_call_capture.py:276,916`).
- The join skips the call (`nemo_gym/trajectory/trajectory_record.py:843-847`).
- A failed attempt, which has no `response_id`, gets the gap `model_call_ownership_unavailable`.
- P0 `ownership.call_owner` and `ownership.gap` fail.
- `ng_perf` leaves the call out.

**Adapters with no header today:**

- Claude Code, Kilo, BrowseComp, Deep Agents (#5003), Pool and LangGraph;
- OpenCode outside continuation;
- the local and host paths of several adapters.

BrowseComp, LangGraph and Deep Agents also emit no `AgentInvocation`.

**Proposed (plan step 2):** any value equal to the root invocation id, with the capture key as the default. Plus a join rule so that sub-agent calls under one header do not conflict.

## Skip capture (pending #5711)

A row sets `_ng_skip_token_capture: true`, which drops only the `/training-token-capture` segment; the prefix stays. A harness that builds its own URL must not add the segment back from its config.

## Usage

**Rule:** report aggregate usage only when every model call has a usage record. Otherwise usage is unknown and an `ObservationGap` is added. Never report a partial sum or a 0 (G:672; S:55; conformance scenario `usage_omitted`).

**Seen in adapters:**

- partial sums with no gap: Stirrup, BrowseComp, Pi;
- zeros: Claude Code's cached and reasoning tokens; Codex after a timeout; NOOA when no call returned;
- nothing: Pool.

**Proposed (plan step 2):**

- HTTP rejections that return no body are not counted.
- Run totals the harness reports itself are accepted.
- Never overwrite native usage with the capture sum. That would blind `rollout_token_count_mismatch`.

## Who raises this

No human reviewer raised token capture on these PRs. The CI review bot raised usage-accounting risks on #5003, #5082, #5537, #5540 and #5877. Check it anyway: it decides whether a training run can use the rollout.
