# Kuhn Poker

This environment implements seeded, two-player Kuhn Poker using NeMo Gym's
alternating-turn multi-agent API. The resources server owns the deal, legal
actions, turn order, invalid-move handling, and zero-sum rewards. Games can be
played in a pass-and-play browser or through the multi-agent harness.

One Gym rollout represents one hand and stores both seat trajectories in
`agent_trajectories`, both chip payoffs in `agent_rewards`, and Player 0's
payoff in the legacy scalar `reward`.

## Play in a browser

Start the resources server:

```bash
gym env start \
  --config resources_servers/kuhn_poker/configs/kuhn_poker.yaml
```

In a second terminal, open the web client:

```bash
python resources_servers/kuhn_poker/client.py
```

The **Play** tab runs one private pass-and-play hand in the browser session.
After every action, the screen is hidden before the next player's private card
is revealed. The **Spectate** tab watches the current or most recently completed
hand through a server-sent event stream. Spectators see public betting state
during play and both cards after the hand terminates.

This hackathon client intentionally assumes that only one hand is active on the
resources server at a time. A browser spectator has a different session cookie
from an agent, so the spectator stream is server-wide rather than scoped to the
browser's session.

## Run through the multi-agent harness

Start the same config, then run the single example task:

```bash
gym eval run --no-serve \
  --agent kuhn_poker_keyboard_agent \
  --input resources_servers/kuhn_poker/data/example.jsonl \
  --output results/kuhn_poker.jsonl
```

The active player's prompt includes only that player's private card and the
public betting history. Invalid or ambiguous bracketed actions are retried once
by default, then the acting player forfeits.

## Future work

- Add server tests for private/public view masking, terminal reveals, and SSE updates.
- Add browser and launcher smoke tests covering a complete pass-and-play hand.
- Add reconnect handling and event sequence checks.
- Add game IDs, bounded retention, and replay when concurrent games are supported.
- Add model-backed controllers for human-versus-agent and agent-versus-agent play.

## Upstream behavior

The rules, seeded task shape, strict bracket parser, retry semantics, and
payoffs follow Prime Intellect's MIT-licensed
[Kuhn Poker environment](https://github.com/PrimeIntellect-ai/verifiers/tree/8b292c9f1b14d9df6b98f4c03e42e416838662a2/environments/kuhn_poker)
at commit `8b292c9f1b14d9df6b98f4c03e42e416838662a2`. The implementation here is
native to NeMo Gym and does not depend on `verifiers`.
