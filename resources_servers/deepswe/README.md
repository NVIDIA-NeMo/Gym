# DeepSWE

This resources server implements the DeepSWE v1.1 two-environment evaluation contract. The coding agent works in
one sandbox; its final workspace patch is graded by the canonical DeepSWE verifier in a fresh sandbox based on the
same task image.

## Prepare pinned task assets

```bash
python -m resources_servers.deepswe.prepare \
  --source-dir /path/to/deep-swe \
  --no-download
```

The preparation step copies each immutable, versioned upstream image reference from the pinned task definition into
the Gym JSONL alongside the instruction and task ID. The resources server rejects a row whose image differs from its
pinned task definition, and both the agent and fresh verifier consume that same image. Test assets and oracle patches
remain in the resources server's gitignored control-plane cache.

Task definitions are parsed with a narrow local adapter for the pinned Pier 0.3.1 task schema, without installing
Pier's unrelated agent/model-provider stack. Agent and verifier CPU, memory, and storage requests are resolved
independently from their corresponding `task.toml` environment sections. By default, Gym requests twice the task's
declared CPU and memory for each sandbox while preserving its storage request. The `task_cpu_multiplier` and
`task_memory_multiplier` resources-server settings can override that scaling; explicit `sandbox_config.resources`
values take final precedence.

## Oracle checkpoint

Start the resources server with golden-patch mode enabled:

```bash
gym env start \
  --config resources_servers/deepswe/configs/deepswe.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  +deepswe_resources_server.resources_servers.deepswe.is_verifying_golden_patch=true
```

Then run all 113 oracle patches from another terminal:

```bash
python resources_servers/deepswe/validate_golden.py +concurrency=113
```

## Run OpenCode rollouts

Use the benchmark config with a model and sandbox-provider config. Launch Gym with `+use_absolute_ip=true` when
OpenSandbox needs a host-routable model-server address.

```bash
gym env start \
  --config benchmarks/deepswe/opencode.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --config responses_api_models/<model>/configs/<model>.yaml
```

As in upstream DeepSWE v1.1, the agent must commit its work. At verification time the resources server executes the
task's pinned `[[verifier.collect]]` hook, which writes `git diff --binary <base_commit> HEAD` to
`/logs/artifacts/model.patch`. That patch is applied and graded in a fresh verifier sandbox. The untrusted agent
sandbox defaults to deny-all egress, and the OpenCode benchmark adds only the resolved Gym model-server host to that
policy. The trusted fresh verifier keeps the canonical container network stack because the current egress sidecar
disables IPv6 loopback required by upstream test suites; verifier assets and the submitted patch are supplied only
by the resources server.
