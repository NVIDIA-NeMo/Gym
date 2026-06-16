# Custom Agent (manifest-driven, YAML-only onboarding)

Onboard a blackbox CLI agent **with config only** — no per-agent Python. It
subclasses nothing you write: the shared `SandboxCliAgent` lifecycle (sandbox +
capture proxy + in-box install/run + patch + gather + verify) does the work; you
declare the launch contract in the config:

```yaml
custom_agent:
  responses_api_agents:
    custom_agent:
      entrypoint: app.py
      model_base_url: ${policy_base_url}     # model root (no /v1)
      model: my-model
      model_api: chat                        # responses | chat | messages
      model_api_key_env: NEMO_GYM_MODEL_API_KEY
      sandbox: { ecs_fargate: { region: us-east-1 } }
      image: docker.io/org/my-agent:latest
      workdir: /workspace
      install_command: "pip install -U my-agent"
      run_template: "my-agent solve --task {prompt} --base-url {base_url}"
      model_base_url_env: OPENAI_BASE_URL
```

`run_template` is formatted with `{prompt}` (shell-quoted), `{base_url}`,
`{workdir}`, `{config_dir}`, `{model}`.

## Quick start

```bash
ng_run "+config_paths=[responses_api_agents/custom_agent/configs/custom_agent.yaml]"
ng_collect_rollouts +agent_name=custom_agent \
  +input_jsonl_fpath=responses_api_agents/custom_agent/data/example.jsonl \
  +output_jsonl_fpath=custom_rollouts.jsonl +limit=1
```
