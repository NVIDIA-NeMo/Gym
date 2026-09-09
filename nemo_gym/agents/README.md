# Agent harnesses

Each harness accepts `AgentHarnessConfig` and implements async
`run(NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse`.

```python
from nemo_gym.agents import AgentHarnessConfig, AgentModelConfig, CodexHarness
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

config = AgentHarnessConfig(
    model=AgentModelConfig(
        model="gpt-5-codex",
        provider="openai",
        api_key="...",
        base_url="https://api.openai.com/v1",
    ),
    system_prompt="Work carefully.",
    timeout_seconds=600,
    workspace="/work/repo",
)
harness = CodexHarness(config)
response = await harness.run(NeMoGymResponseCreateParamsNonStreaming(input="Fix the tests."))
```

Shared fields:

- `model`: `model`, `provider`, `api_key`, `base_url`, and provider-specific `settings`
- `system_prompt`
- `timeout_seconds`
- `max_turns` for Claude Code, Hermes, and Terminus-2
- `workspace`
- `settings` for harness-specific options

The caller installs the selected CLI. Terminus-2 requires
`pip install 'nemo-gym[terminus-2]'` and its OS tools. Agent servers still handle
server startup, model URL resolution, concurrency, and verification.
