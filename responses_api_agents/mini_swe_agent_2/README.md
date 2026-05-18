# Mini-SWE-Agent 2 Sandbox Agent

`mini_swe_agent_2` is the Gym integration for mini-swe-agent v2 using the
public `nemo_gym.sandbox` API. It intentionally does not carry over the older
Docker/Singularity mini-SWE path.

The verified path in this package is:

- mini-swe-agent `2.1.0`
- SWE-bench Verified task rows
- `env: sandbox`
- `responses_api_agents.mini_swe_agent_2.sandbox_environment.MiniSWESandboxEnvironment`
- OpenSandbox through `nemo_gym.sandbox.providers.opensandbox`
- OpenTelemetry sandbox observability

Use `configs/mini_swe_agent_opensandbox.yaml` as the Gym server config. For the
environment adapter details, see `SANDBOX_ENVIRONMENT.md`.
