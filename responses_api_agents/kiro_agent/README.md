# Kiro Agent

Runs Kiro CLI through its Agent Client Protocol interface. Kiro executes tools internally and the
adapter converts ACP messages, tool calls, and tool outputs to Responses API items.

Kiro uses its hosted model service. It does not accept a Gym model server or an arbitrary local
model endpoint. Headless runs require `KIRO_API_KEY`.

```bash
export KIRO_API_KEY=ksk_...
```

`model` is optional and must be a model ID returned by `kiro-cli chat --list-models`. The adapter
uses Kiro's default model when `model` is null.

Each rollout uses an isolated workspace and `KIRO_HOME`. System prompts are written to a temporary
Kiro agent profile. Kiro CLI is installed from `https://cli.kiro.dev/install` when it is not found.
