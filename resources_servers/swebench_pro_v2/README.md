# SWE-bench Pro v2

This resources server evaluates patches for the validated v2 split (642 tasks) of
[SWE-bench Pro](https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862/v2).
It reuses the [v1 server](../swebench_pro/README.md) and changes only what v2 changes:

- **Image:** the task image is the dataset's `docker_image` (`ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>`).
- **Verifier:** the candidate patch is applied in a fresh sandbox, then the task's own upstream `tests/test.sh` runs
  (reset test files, apply the test patch, run `run_script.sh`, parse, check fail-to-pass and pass-to-pass).
  `verification.py` only builds the entryscript that does this; the rest is v1's `run_verification`.
- **Prompt:** the upstream `instruction.md`, verbatim.
- **Limits:** 3000s per verification, as upstream. v1's container repairs, Go prefetch and anti-cheat setup are off,
  because v2 images ship a sanitised git history and each `test.sh` does its own repairs.

## Prepare data

```bash
uv run python benchmarks/swebench/pro_v2/prepare.py
```

The rows keep the dataset's `hard` column, so the HARD-51 subset is the rows with `hard == true`.

## Offline agent phase

Upstream runs the agent with no network except the model endpoint. With the OpenSandbox provider, set
`agent_network_policy` on the resources server (see `configs/swebench_pro_v2.yaml`); it applies to the agent sandbox
only, and the verifier sandbox keeps network access (a few Go tasks fetch modules at test time). The agent must
already have its tooling in the image or mounted, since it cannot download anything.

## Golden-patch smoke test

```bash
gym env start \
  --config resources_servers/swebench_pro_v2/configs/swebench_pro_v2.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  +swebench_pro_v2_resources_server.resources_servers.swebench_pro_v2.is_verifying_golden_patch=true
```

Then run `resources_servers/swebench_pro/apply_golden_patch.py` with
`+benchmark_jsonl=benchmarks/swebench/data/swebench_pro_v2_benchmark.jsonl` and
`+server_name=swebench_pro_v2_resources_server`; all 642 tasks should resolve.
