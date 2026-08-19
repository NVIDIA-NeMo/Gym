# Kuhn Poker

This environment implements seeded, two-player Kuhn Poker using NeMo Gym's
alternating-turn multi-agent API. The resources server owns the deal, legal
actions, turn order, invalid-move handling, and zero-sum rewards. Player 0 and
Player 1 are independent keyboard-agent servers, each running in its own
terminal. A dedicated rollout-orchestrator server owns `/run` and sends the
active seat's private observation to that seat's `/act` endpoint. The
participant agents do not implement the rollout loop.

The initial integration is interactive. One Gym rollout represents one hand and
stores both seat trajectories in `agent_trajectories`, both chip payoffs in
`agent_rewards`, and Player 0's payoff in the legacy scalar `reward`. Dataset
rows select this rollout protocol with `orchestrator_ref`.

## Run an interactive hand in separate terminals

The examples below use head-server port `11001`; use the default `11000` if it
is free. Every command must use the same config and head-server address.

1. Pre-warm the shared server environments once before opening the terminals:

   ```bash
   gym env prefetch \
     --config resources_servers/kuhn_poker/configs/kuhn_poker.yaml
   ```

2. Start the shared head server, environment, and orchestrator together in the
   coordinator terminal:

   ```bash
   gym env head \
     --instance kuhn_poker \
     --instance kuhn_poker_orchestrator \
     --config resources_servers/kuhn_poker/configs/kuhn_poker.yaml \
     +head_server.host=127.0.0.1 +head_server.port=11001
   ```

3. Start each player in its own terminal. These are the terminals where cards
   and action prompts appear:

   ```bash
   gym env serve --instance kuhn_player0 \
     --config resources_servers/kuhn_poker/configs/kuhn_poker.yaml \
     +head_server.host=127.0.0.1 +head_server.port=11001
   ```

   ```bash
   gym env serve --instance kuhn_player1 \
     --config resources_servers/kuhn_poker/configs/kuhn_poker.yaml \
     +head_server.host=127.0.0.1 +head_server.port=11001
   ```

4. Trigger one hand from another terminal:

   ```bash
   gym eval run --no-serve \
     --agent kuhn_poker_orchestrator \
     --input resources_servers/kuhn_poker/data/example.jsonl \
     --output results/kuhn_poker.jsonl \
     +head_server.host=127.0.0.1 +head_server.port=11001
   ```

The active player's prompt includes only that player's private card and the
public betting history. Invalid or ambiguous bracketed actions are retried once
by default, then the acting player forfeits.

## Upstream behavior

The rules, seeded task shape, strict bracket parser, retry semantics, and
payoffs follow Prime Intellect's MIT-licensed
[Kuhn Poker environment](https://github.com/PrimeIntellect-ai/verifiers/tree/8b292c9f1b14d9df6b98f4c03e42e416838662a2/environments/kuhn_poker)
at commit `8b292c9f1b14d9df6b98f4c03e42e416838662a2`. The implementation here is
native to NeMo Gym and does not depend on `verifiers`.
