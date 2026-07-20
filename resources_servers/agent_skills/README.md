# Agent Skills Verifier

This resources server verifies coding-agent patches in a fresh Gym sandbox. It is the hidden-check boundary for agent-skill A/B benchmarks:

1. the coding agent returns `workspace_patch` and `workspace_base_revision`;
2. the verifier starts the configured fixture image;
3. it requires a clean checkout at the same revision;
4. it validates and applies the patch;
5. it uploads hidden files outside the modified workspace;
6. it runs the server-defined check command from the hidden check directory.

## Check suite configuration

```yaml
agent_skills:
  resources_servers:
    agent_skills:
      entrypoint: app.py
      sandbox_provider: sandbox
      concurrency: 8
      cleanup_timeout: 30
      check_suites:
        create-environment-sql-v1:
          sandbox_spec:
            image: <immutable-fixture-image>
            workdir: /workspace/nemo-gym
            ttl_s: 1800
            resources:
              cpu: 4
              memory_mib: 8192
          workspace: /workspace/nemo-gym
          check_cwd: /tmp/nemo_gym_hidden_checks
          check_user: nobody
          hidden_files:
            run_checks.py: |
              import os
              from pathlib import Path

              workspace = Path(os.environ["NEMO_GYM_WORKSPACE"])
              assert (workspace / "resources_servers").is_dir()
          check_command: python run_checks.py
          timeout: 900
```

Include a sandbox provider config, such as `nemo_gym/sandbox/providers/docker/configs/docker.yaml`, when starting the environment.

Dataset rows select a server-side suite by identifier:

```json
{
  "verifier_metadata": {
    "task_id": "create-env-sql-001",
    "check_suite_id": "create-environment-sql-v1"
  }
}
```

Hidden test contents do not belong in the dataset. `hidden_files` are uploaded only after patch application and must use relative paths under `check_cwd`, which must be outside the agent workspace. They are made read-only before checks run. The check command runs as `check_user` (`nobody` by default) from the protected directory and receives the patched checkout path through `NEMO_GYM_WORKSPACE`.

Hidden checks should treat submitted code as untrusted. Run submitted programs as child processes and make the hidden runner—not submitted code—decide the final exit status. Avoid importing arbitrary submission modules into the hidden runner process.

## Result fields

The verifier returns:

- `reward` and `correctness`;
- `status`;
- task and check-suite identifiers;
- verifier base revision and elapsed time;
- bounded stdout, stderr, return code, and provider error type.

The primary `task_success` and `correctness` metrics are included in aggregate pass@k reporting.
