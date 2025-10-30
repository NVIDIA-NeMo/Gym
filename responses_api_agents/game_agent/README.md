# Description
A specialized agent implementation designed for episodic, stateful environments like games (Sokoban, Tetris) and reinforcement learning tasks. Like `simple_agent`, it supports multi-turn agentic behavior with tool calling, but adds game-specific features for proper episode management.

## Key Differences from Simple Agent

Both `game_agent` and `simple_agent` support multi-step actions with tool calling. However, `game_agent` adds three critical features for game/RL environments:

### 1. **Environment-Controlled Termination (Done Flag)**
`game_agent` checks for a `"done"` flag in tool responses and terminates the episode early when detected:
```python
# In tool responses: {"observation": "...", "reward": 1.0, "done": true}
```

**Why**: Games can end before max_steps (e.g., Tetris game over, Sokoban puzzle solved). The `done` flag lets the environment signal episode completion so the agent doesn't keep trying to act in a finished game.

**Simple agent**: No done flag support - continues until the model stops calling tools or hits max_steps.

### 2. **Action-Based Step Limits**
`game_agent` counts **actual tool calls** (actions) toward `max_steps`, not loop iterations:
- If the model generates text without calling a tool, it doesn't count toward max_steps
- Useful for limiting episode length by number of actions taken

**Simple agent**: Counts every model invocation (loop iteration) toward max_steps, regardless of whether a tool is called.

### 3. **Tool-First Termination Check**
`game_agent` processes all tool calls in a turn before checking termination conditions.

**Simple agent**: Checks termination before processing tool calls (minor difference in control flow).

## When to Use

- **Use `game_agent`**: For episodic environments (games, RL tasks) that need environment-controlled termination and action-based limits
- **Use `simple_agent`**: For conversational assistants, general-purpose tool-use tasks, or non-episodic workflows


# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
