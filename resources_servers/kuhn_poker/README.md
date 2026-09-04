# Kuhn Poker

This environment implements seeded, two-player Kuhn Poker using NeMo Gym's
alternating-turn multi-agent API. The resources server owns the deal, legal
actions, turn order, invalid-move handling, and zero-sum rewards. Player 0 and
Player 1 are independent keyboard-agent servers, each running in its own
terminal. A dedicated rollout-orchestrator server owns `/run` and sends the
active seat's private observation to that seat's standard `/v1/responses`
endpoint. The participant agents make one policy decision per request and do
not implement the rollout loop.

The initial integration is interactive. One Gym rollout represents one hand and
stores both seat trajectories in `agent_trajectories`, both chip payoffs in
`agent_rewards`, and Player 0's payoff in the legacy scalar `reward`. Dataset
rows select this rollout protocol with `orchestrator_ref`.

## Run everything in one terminal

Omit `--no-serve` to start the head server, resources server, orchestrator, and
both keyboard agents; play one hand; and shut everything down from one command:

```bash
gym eval run \
  --config resources_servers/kuhn_poker/configs/kuhn_poker.yaml \
  --agent kuhn_poker_orchestrator \
  --split validation \
  --output results/kuhn_poker.jsonl \
  --concurrency 1 \
  +head_server.host=127.0.0.1 +head_server.port=11001
```

Both players' private prompts appear in this terminal. Use the workflow below
when each player must have a separate terminal.

## Run two LLM agents

`kuhn_poker_llm.yaml` replaces the keyboard servers with two independent
`simple_agent` instances. Both agents call the shared `policy_model` inference
provider configured by `policy_base_url`, `policy_api_key`, and
`policy_model_name` in `env.yaml`.

```bash
gym eval run \
  --config resources_servers/kuhn_poker/configs/kuhn_poker_llm.yaml \
  --agent kuhn_poker_orchestrator \
  --split validation \
  --output results/kuhn_poker_llm.jsonl \
  --concurrency 1 \
  +debug_mode=false
```

Each participant receives only its own observation and previous turns through
its `/v1/responses` endpoint. The participant agent then calls the shared model
server's `/v1/responses` endpoint. Keep `--concurrency 1` for this initial
alternating-turn integration.

## Run an interactive hand in separate terminals

The examples below use head-server port `11001`; use the default `11000` if it
is free. Every command must use the same config and head-server address. Stop
any previously running Kuhn Poker servers before starting.

1. Start the shared head server, environment, and orchestrator together in the
   coordinator terminal (Terminal 1):

   ```bash
   gym env head \
     --instance kuhn_poker \
     --instance kuhn_poker_orchestrator \
     --config resources_servers/kuhn_poker/configs/kuhn_poker.yaml \
     +head_server.host=127.0.0.1 +head_server.port=11001
   ```

2. Start Player 0 in Terminal 2. This terminal displays Player 0's private card
   and action prompts:

   ```bash
   gym env serve --instance kuhn_player0 \
     --config resources_servers/kuhn_poker/configs/kuhn_poker.yaml \
     +head_server.host=127.0.0.1 +head_server.port=11001
   ```

3. Start Player 1 in Terminal 3:

   ```bash
   gym env serve --instance kuhn_player1 \
     --config resources_servers/kuhn_poker/configs/kuhn_poker.yaml \
     +head_server.host=127.0.0.1 +head_server.port=11001
   ```

4. Trigger one hand from Terminal 4:

   ```bash
   gym eval run --no-serve \
     --agent kuhn_poker_orchestrator \
     --input resources_servers/kuhn_poker/data/example.jsonl \
     --output results/kuhn_poker.jsonl \
     --concurrency 1 \
     +head_server.host=127.0.0.1 +head_server.port=11001
   ```

The active player's prompt includes only that player's private card and the
public betting history. Invalid or ambiguous bracketed actions are retried once
by default, then the acting player forfeits. The participant communication uses
the standard `/v1/responses` endpoint; the orchestrator owns the episode-level
`/run` endpoint.

Prefetching beforehand is optional:

```bash
gym env prefetch \
  --config resources_servers/kuhn_poker/configs/kuhn_poker.yaml
```

It can reduce first-start latency, but virtual-environment setup is protected
by a cross-process lock and does not require prefetching for correctness.

## Upstream behavior

The rules, seeded task shape, strict bracket parser, retry semantics, and
payoffs follow Prime Intellect's MIT-licensed
[Kuhn Poker environment](https://github.com/PrimeIntellect-ai/verifiers/tree/8b292c9f1b14d9df6b98f4c03e42e416838662a2/environments/kuhn_poker)
at commit `8b292c9f1b14d9df6b98f4c03e42e416838662a2`. The implementation here is
native to NeMo Gym and does not depend on `verifiers`.
