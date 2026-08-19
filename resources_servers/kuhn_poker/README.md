# Kuhn Poker

This environment implements seeded, two-player Kuhn Poker using NeMo Gym's
alternating-turn multi-agent API. The resources server owns the deal, legal
actions, turn order, invalid-move handling, and zero-sum rewards. Two keyboard
controllers receive isolated observations and enter actions such as `[check]`,
`[bet]`, `[fold]`, and `[call]`.

The initial integration is interactive. One Gym rollout represents one hand and
stores both seat trajectories in `agent_trajectories`, both chip payoffs in
`agent_rewards`, and Player 0's payoff in the legacy scalar `reward`.

## Run an interactive hand

Start the resources and agent servers:

```bash
gym env start \
  --config resources_servers/kuhn_poker/configs/kuhn_poker.yaml
```

In a second terminal, run the single example task:

```bash
gym eval run --no-serve \
  --agent kuhn_poker_keyboard_agent \
  --input resources_servers/kuhn_poker/data/example.jsonl \
  --output results/kuhn_poker.jsonl
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
