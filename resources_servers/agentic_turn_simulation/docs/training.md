# Training

Practical considerations for RL on this environment, and what to monitor during training. The numbers in Gym configs
are defaults, not tuned values. For the scoring rules, see the [README](../README.md#scoring).

## Data selection

Profile the starting policy and keep rows with mixed outcomes that fit the training context, holding out whole
sessions; [data.md](data.md#picking-rl-rows) has the commands. Very long histories otherwise dominate masking. Beyond
that:

- **Prefer clean traces.** Rows with flattened tool history or a missing system prompt teach the wrong lessons: to
  invent tool output, or to distrust the system prompt.
- **Mixing harnesses is an option.** Rows pooled from several harnesses can share one run; evaluate on held-out turns
  of each harness you care about, since a mixture may under-serve one of them.

## Serving the simulator and the judge

- **They are the bottleneck.** Every tool call and every verdict re-sends a long prompt (the harness's system prompt,
  its tool definitions, the conversation), and a turn makes several simulator calls. Policy servers sit idle waiting
  on them, so adding policy capacity buys nothing. Give the simulator and judge dedicated replicas sized to the number
  of in-flight rollouts, and raise `tool_simulator_max_concurrency` and `judge_max_concurrency` to match.
- **Route by prompt prefix.** Consecutive calls of a rollout share almost all of their prompt, but the resources
  server's calls carry no session, so a list of replicas in `judge_base_url` spreads them per call (unless rollout
  observability or checkpointing is on, which pins each rollout's calls to one replica) and every call recomputes the
  shared prefix. Put a prefix-aware router in front of the replicas and pass its single URL; it
  restores prefix-cache hits and per-request decode speed. Make its key long enough that a system prompt shared by
  every row of a harness does not send them all to one engine.
- **Keep reasoning effort high** for both. A lower-effort judge is lenient (it accepts unverified claims and invented
  specifics) and a lower-effort simulator is optimistic (failed fixes come back as working); an RL policy can learn
  to exploit both. Fix speed with capacity, not with effort, output budgets or attempts.
- **Never change the judge mid-run.** The judge model, its prompts, its effort, `show_agent_system_prompt` and the note
  file together define the reward. A change is a new reward version: start a new run, re-profile, and re-judge old
  rollouts with `gym eval reverify` ([command](../README.md#reporting-an-eval)) to compare.
- **Expire abandoned sessions.** A rollout that fails before `/verify` leaves its session (the whole history) in the
  resources server's memory until `session_ttl_seconds`. For long runs, set it well above the turn time cap, for
  example twice the cap: a turn can overrun the cap by a model step and its tool calls, and a session dropped
  meanwhile masks the rollout.

By default the simulator and the judge share one model server. To serve them from separate models, define a second
model server in a config file that also points `tool_simulator_model_server` (or the `judges` list) at it:

```yaml
# simulator.yaml
agentic_turn_simulation_simulator_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: http://<simulator-host>:8000/v1
      api_key: EMPTY
      model: <simulator-served-name>
      return_token_id_information: false
      uses_reasoning_parser: true

agentic_turn_simulation_resources_server:
  resources_servers:
    agentic_turn_simulation:
      tool_simulator_model_server:
        name: agentic_turn_simulation_simulator_model
```

List it after the environment's config, so that it overrides the environment's setting. Use `--config` for both, not
`--resources-server`: configs named by `--resources-server` load after every `--config` file and would win.

```bash
gym env start \
  --config resources_servers/agentic_turn_simulation/configs/agentic_turn_simulation.yaml \
  --config simulator.yaml \
  --model-type vllm_model \
  --model-url http://<policy-host>:8000/v1 \
  --model <policy-model-name> \
  --model-api-key EMPTY \
  +judge_base_url=http://<judge-host>:8000/v1 \
  +judge_api_key=EMPTY \
  +judge_model_name=<judge-model-name>
```

## Wall-clock caps and masking

- **Training needs tighter caps than evaluation.** The agent's default time cap (two hours) suits evaluation; training
  steps wait on their slowest rollouts, so set `max_time_seconds` to tens of minutes. With a wall-clock cap, how many
  rollouts are masked depends on serving speed.
- **Expect masked rollouts from the start**, mostly time caps, which track simulator and judge load more than the
  policy. Rows too long for the policy are masked only when the policy server reports the overflow
  (`propagate_context_overflow_errors: true` on a `vllm_model` policy; Gym's default is false); without it they end
  as `max_output_tokens` and score 0. Filter them out either way ([Data selection](#data-selection)). Gym's
  `coverage/masked_rollouts` counts masked rollouts (it appears once at least one is masked); no metric breaks them
  down by reason, so count `failure_kind` in the rollouts file ([snippet](../README.md#reporting-an-eval)).
- **Step and output caps score 0.** A turn that ends on `max_steps`, `max_output_tokens`, a context overflow after
  its first step or an empty final reply (`empty_output`) scores 0 without a judge (`unfinished_turn`), so the reward
  pushes the policy to converge within the caps ([why](design.md#masking)). Keep `max_steps` high enough that correct
  work fits.
- **Decide how masked rollouts enter the advantage.** Masked rollouts get no gradient, but the trainer may still count
  them in the group baseline. If they count as failures, a fast stopping action (asking the user, handing back early)
  looks safe next to real work that might hit the cap; that is pressure toward the ask-the-user shortcut below.
  Excluding them from the baseline keeps a cap hit neutral.
- **Keep caps comparable.** Use the same caps for reward profiling and training where possible, and compare runs at
  the same training stage: masking and reward both drift as the policy changes.
- **Filtering by reason.** Trainers that filter masked rollouts by reason can match `failure_kind` values that start
  with `agentic_turn_simulation:` plus `judge_failed`; `mask_sample` alone covers all of them.
- **Masking or rescoring caps in the trainer.** To keep scored cap hits out of the gradient, or to give them a
  different reward, match their `stop_reason` in the trainer. To mask them in the environment for every consumer, list
  them in `masked_stop_reasons`.
- **Per-call costs.** The agent's response lists each model call's tokens and whether its output budget cut it off
  (`response.metadata.model_calls`, see the
  [agent README](../../../responses_api_agents/agentic_turn_simulation_agent/README.md#token-usage)), so a trainer can
  charge per call rather than per turn.

## Asynchronous rollouts

Turn lengths range from one model call to dozens of steps, so a synchronous batch waits on its longest turn.
Asynchronous rollouts with a small staleness bound (a few policy versions) keep generation busy. Expect step metrics to
wobble around restarts: the first steps after a resume come from fresh, short rollouts.

## Long contexts and output budgets

- Histories are long and never truncated. Train with a long context window and filter rows to fit it.
- Use large per-call output budgets. Hidden reasoning grows during RL, and a small cap turns longer reasoning into
  truncated steps. The agent doubles the budget on truncation up to `max_output_tokens_cap` (65,536 by default).

## Keep length shaping in the trainer

If generated length needs a penalty, put it in the trainer's reward or advantage computation, not in the environment
or the judge. The environment's reward should stay one meaning ("was the turn correct") across experiments. A length
penalty can also make the ask-the-user shortcut cheaper, so watch the stop-tool share when you add one.

## Agent settings for training

```yaml
agentic_turn_simulation_agent:
  responses_api_agents:
    agentic_turn_simulation_agent:
      token_id_capture: true      # required when the trainer captures token ids
      max_time_seconds: 1800      # tens of minutes, well below the evaluation default
```

## What to monitor

- **Asking the user too often, and giving up early.** A judge can pass a clear task ended with a focused question, or
  with "I can't access X", and RL finds such cheap stops. The rubric fails avoidable stops
  ([why](design.md#the-judge)); track the share of turns ending in a stop tool (`stop_reasons`) all the same. Per
  rollout, `ended_on_stop_tool` with `executed_calls` 0 marks a turn that stopped without acting, whatever the
  harness's stop tools are called.
- **Reasoning length.** Hidden reasoning can grow while visible answers shrink. Under small output caps, longer
  reasoning gets cut off and turns into failures.
- **Out-of-domain guardrails.** Track a few evals outside this environment (coding agents, instruction following,
  chat) during training.
