# Script-graded SWE tasks

`swe_external1` runs software-engineering tasks from prebuilt images. Each
JSONL row supplies the prompt, image reference, working directory, and complete
test/solution assets. The server does not build images or infer a test command
from the repository's language.

## Execution and grading

1. `/seed_session` creates the task sandbox and optionally runs `setup_script`.
   It returns a serialized sandbox descriptor, including the working directory.
2. The OpenCode agent connects to that **same sandbox** and edits the checkout.
   Neither test assets nor solution assets are uploaded by this server at this stage.
3. `/verify` uploads the held-out `test_files` to `/tests`, clears any previous
   `/logs/verifier` output, and runs `bash /tests/test.sh` in the task's `workdir`.
   It does **not** reset Git, extract a diff, or create a second grading sandbox.
4. The grader must write `0` or `1` to `/logs/verifier/reward.txt`. The server
   reads it and stops the sandbox. A passing reward also requires a zero script
   exit status. Missing/invalid rewards and infrastructure errors produce
   `reward: 0` with `evaluation_completed: false`, not a measured test failure.

Golden-solution mode creates a fresh sandbox, uploads all `solution_files` to
`/solution`, and runs `bash /solution/solve.sh` before the same verification flow.
It measures the supplied solution, not an agent's ability to solve the task.

Use a provider that supports serialization/reconnection, such as OpenSandbox,
for the agent flow. Configure the provider, endpoint, authentication, registry
access, and model endpoint privately; there are no embedded credentials or
dataset-specific registries. OpenSandbox is a separately deployed service, not
something this resources server deploys. The backend must support each image's
architecture and requested resources. Generic Docker-provider golden checks do
not validate OpenSandbox connectivity or the agent's reconnection path.

## Files and configuration

- `app.py`: session lifecycle, provider configuration, and seed/verify endpoints.
- `task_data.py`: portable row schema and asset validation.
- `verification.py`: asset upload, original script execution, and reward reading.
- `configs/swe_external1.yaml`: normal and golden resources-server instances.
- `configs/swe_external1_opencode.yaml`: OpenCode agent with a local training dataset.
- `configs/swe_external1_example.yaml`: five-example data validation and a
  simple-agent/golden-verifier pairing. Its reward evaluates the golden solution,
  regardless of the model's textual answer; use the OpenCode pairing for real
  agent evaluation.

Run normal agent evaluation after supplying private provider/model configuration:

```bash
gym env start \
  --config resources_servers/swe_external1/configs/swe_external1_opencode.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml \
  --config /path/to/private-runtime.yaml

gym eval run --no-serve \
  --agent swe_external1_opencode_sandboxed_agent \
  --input resources_servers/swe_external1/data/example.jsonl \
  --output resources_servers/swe_external1/data/example_rollouts.jsonl \
  --num-repeats 1
```

Golden checks instead target `/verify` on
`swe_external1_golden_patch_resources_server`, with the task row and an empty
Responses API `response`. They require sandbox access but no model inference.
Do not mix golden-solution and actual agent results in the same rollout artifact.

Keep `num_workers: 1`: session state is process-local. `max_concurrency` bounds
simultaneous sandbox creation/grading operations, not the number of agents already
running. Abandoned sessions have both a provider TTL and local expiry cleanup.
After verification the sandbox is gone; reverification is unsupported. Check
`cleanup_error` independently of the reward, and investigate provider cleanup
failures. Session cookies must be preserved between seeding and verification.

## Dataset contract

Each row contains `responses_create_params.input` and `verifier_metadata`:

- `task_id`, `image_ref`, `workdir`: task identity, pullable image, and absolute
  repository directory. Prefer immutable image digests for private datasets.
- `test_files`: required assets including `test.sh`.
- `solution_files`: optional assets; golden mode requires `solve.sh`.
- `setup_script`: optional trusted initialization on the fresh sandbox only.
- `verifier_timeout_s`, `solution_timeout_s`, `agent_timeout_s`, `cpu`,
  `memory_mib`, `disk_gib`: task budgets. `evaluation_timeout` caps each executed
  setup/solution/verifier command; it is not a total wall-clock deadline.

An asset has a normalized relative `path`, `content_b64`, `encoding` (`base64` or
`gzip+base64`), an uncompressed `sha256`, and POSIX `mode`. Complete directory
trees of regular files are embedded; no access to the source delivery is needed
at runtime. File bytes and permission bits are restored without rewriting the
supplied scripts. Symlinks, special files, duplicate/traversing paths, checksum
mismatches, and oversized assets are rejected. Limits are 64 MiB per decoded file
and 128 MiB of decoded assets per task.

Place the full, privately prepared dataset at `data/training.jsonl`. It is ignored,
not uploaded to a registry, and not shipped with this server. Transfer it only
through approved private channels. The public ignore rules also exclude other
JSONL data files; only the named public examples and inspected example rollouts
are allowlisted. Never put restricted data into either allowlisted file.

`verifier_metadata` is trusted evaluator input, not model input. Withholding test
files until verification reduces accidental exposure but **same-sandbox grading
is not a security boundary against an adversarial agent**: it can alter the
runtime or leave processes behind. Clearing stale reward files does not eliminate
that risk. Images may themselves contain tests; the server does not strip them.

## Public examples and provenance

The five examples are adapted from the [public SWE-rebench example rows in Gym
PR #3327](https://github.com/NVIDIA-NeMo/Gym/blob/5b02366a3039fbf0d0d3d8e0b2099bf213b97c9d/resources_servers/swe_rebench/data/example.jsonl):

| Task | Repository | Language |
| --- | --- | --- |
| `intel__rohd-458` | `intel/rohd` | Dart |
| `syuilo__aiscript-257` | `syuilo/aiscript` | TypeScript |
| `taiki-e__cargo-hack-70` | `taiki-e/cargo-hack` | Rust |
| `tox-dev__pipdeptree-279` | `tox-dev/pipdeptree` | Python |
| `cta-observatory__ctapipe-2397` | `cta-observatory/ctapipe` | Python |

These retain the original prompts, images, base commits, golden/test patches,
installation/test commands, and required `FAIL_TO_PASS`/`PASS_TO_PASS` test names.
The adaptation wraps them as `solve.sh`/`test.sh` plus embedded assets. All required
tests must be observed and pass; missing tests fail. Tests outside those required
lists do not determine reward. Installation commands retain the source adapter's
best-effort behavior. Patch application is deliberately strict (`git apply
--check` followed by `git apply`): a conflict is an incomplete evaluation, not a
silently accepted partial patch. The checkout is initialized to `base_commit`
before the agent/solution runs, never reset during verification.

The example assets include the upstream log parser and its MIT license, pinned to
SWE-rebench-V2 commit `c71902a8cf8d2b725f63d51f199f4d3e56f68d2d`. A Python 3.10+
interpreter must be available in these public example images to run that parser.
The source-example SHA-256 and parser commit are recorded in each public row.
The dataset is [nebius/SWE-rebench-V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2),
licensed CC-BY-4.0; task repositories retain their respective licenses.
Parser source: [SWE-rebench/SWE-rebench-V2](https://github.com/SWE-rebench/SWE-rebench-V2).
Adapter code and NeMo Gym are Apache-2.0. No private corpus is included or relicensed.

Independent source-row and historical test-output fixtures live in
[`tests/fixtures/`](tests/fixtures/README.md), with pinned provenance. They keep
the offline integrity/parser tests self-contained; no other SWE resource server
is required. Those historical logs are not live results from this adapter.

## Validation status

`verified: false` is intentional. Unit/schema/data-collation checks are separate
from live sandbox execution. Live golden and model-agent checks remain
required before treating this new adapter as runtime-validated. Existing results
from the source resource server are not new adapter rollouts.

```bash
pytest resources_servers/swe_external1/tests \
  --cov=resources_servers.swe_external1 --cov-report=term-missing
pytest responses_api_agents/opencode_sandboxed_agent/tests
gym dataset collate \
  --config resources_servers/swe_external1/configs/swe_external1_example.yaml \
  --output-dir data/swe_external1-example-check --mode example_validation
```

`data/example_metrics.json` is generated by collation, not an oracle score.
`data/example_rollouts.jsonl` must be generated by a real run, inspected for secrets,
and committed before Gym's example-rollout gate can pass. It is intentionally
absent while live validation is pending. Run the five golden solutions, then one
real OpenCode model rollout per public task through the normal server. No-change
controls are useful additional checks; keep their results separate from golden
and model runs. Do not set `verified:
true` until the required baselining/review is complete.
