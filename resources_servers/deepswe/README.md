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
OpenSandbox needs a host-routable model-server address. Add the registry-interception overlay when OpenCode must
download its CLI in the agent sandbox.

```bash
gym env start \
  --config benchmarks/deepswe/opencode.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --config resources_servers/deepswe/configs/deepswe_registry_interception.yaml \
  --config responses_api_models/<model>/configs/<model>.yaml
```

The resources server snapshots the initial logical Git tree before the agent runs. At verification time it captures
committed, staged, unstaged, deleted, binary, and non-ignored untracked changes, then applies only those changes to
the task base commit in a fresh verifier sandbox. Harness-owned workspace paths can be excluded by resources-server
configuration; the OpenCode benchmark excludes its `export.json` session export, so the shared agent needs no
DeepSWE-specific behavior. The untrusted agent sandbox defaults to deny-all egress, and the OpenCode benchmark adds
only the resolved Gym model-server host to that policy. The trusted fresh verifier keeps the canonical container
network stack because the current egress sidecar disables IPv6 loopback required by upstream test suites; verifier
assets and the submitted patch are supplied only by the resources server.

## Optional registry interception

TODO: Remove this overlay once OpenCode is preinstalled in or mounted into the agent sandbox, then run with the
base deny-all agent egress policy instead of allow-all pass-through.

On a cluster with the registry-interception sidecar installed, add
`resources_servers/deepswe/configs/deepswe_registry_interception.yaml` after the base DeepSWE config. The overlay
sets both required activation controls: an OpenSandbox `network_policy` and
`OPENSANDBOX_EGRESS_MITMPROXY_TRANSPARENT=true`. Its allow-all policy preserves pass-through behavior for
unintercepted hosts in the agent sandbox; consequently, it intentionally replaces the agent's no-network policy
and must be used only when that egress policy is acceptable for the evaluation. The policy is not copied into the
fresh verifier sandbox.
