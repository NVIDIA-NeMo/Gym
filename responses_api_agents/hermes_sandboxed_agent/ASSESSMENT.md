# Sandboxed Hermes: review guide

The reference is Gym's [OpenCode sandboxed agent](../opencode_sandboxed_agent/app.py#L885),
which uses OpenSandbox in its example configuration. We reuse its flow:
the resources server prepares a task container, the agent attaches and runs inside
it, then the resources server grades the patch. Our cluster uses Apptainer.

## Components and where they run

| Component | What was already present / what we changed | Runs where |
| --- | --- | --- |
| [Pro dataset preparation](../../benchmarks/swebench/pro/prepare.py#L153) | Existing. Downloads the pinned `ScaleAI/SWE-bench_Pro` test split and evaluator assets; resolves task image digests. Five prepared [example rows](../../resources_servers/swebench_pro/data/example.jsonl#L1) are already in Gym. | Before evaluation, outside task containers. |
| [Pro resources server](../../resources_servers/swebench_pro/app.py#L193) | Existing. We added the serialized sandbox handoff, explicit cleanup and local SIF provenance checks. | A Gym HTTP service outside task containers. |
| [Hermes agent server](app.py#L239) | Added. Coordinates seed → attach → run → verify → cleanup. | Another Gym HTTP service outside task containers. |
| [Hermes runner](runner.py#L95) | Added adapter around unchanged upstream Hermes, pinned to `2237be355906fbe6065ce1815711eee52b2d646e`. | Inside the task container. Tools work in `/app`; the runner works outside the repository to avoid Python import shadowing. |
| [Model proxy](../../responses_api_models/openai_model/app.py) | Existing. Holds provider credentials and forwards model requests. | Outside task containers; inference runs at the configured model endpoint. |
| [Pro verifier](../../resources_servers/swebench_pro/verification.py#L388) | Existing task scripts and grading. Completed parser output follows Pro's required-pass rule, including empty reports. Execution failures remain inconclusive; an explicit required-test failure is retained after a timeout. | Fresh verification containers, managed by the Pro server. |
| [Apptainer provider](../../nemo_gym/sandbox/providers/apptainer/provider.py#L560) | Existing provider; only `serialize_handle()` and `connect()` were added. | Library used by both Gym services on the same host and UID. |

The benchmark is data, task images and grading scripts. The **resources server**
is Gym's HTTP interface to those pieces. Slurm allocation, mounts and launch
settings live in the separate [Slurm evaluations repository](https://gitlab-master.nvidia.com/interactive-agents/slurm-evaluations/-/tree/jnolan/hermes-sandboxed-pro).

## Review order

1. [Agent `run()`](app.py#L239): the overall flow and failure handling.
2. [Runner](runner.py#L95), then [runtime preparation](prepare_runtime.sh#L1): how Hermes starts. The existing portable-Python helper is reused; GNU and musl builds support Debian/Ubuntu and Alpine task images.
3. [Pro `seed_session()`](../../resources_servers/swebench_pro/app.py#L305) and [verification](../../resources_servers/swebench_pro/verification.py#L388): inspect the diff against main to distinguish changes from the existing implementation.
4. [Apptainer handoff](../../nemo_gym/sandbox/providers/apptainer/provider.py#L560): the only core-library change. A bare instance ID cannot reconstruct the provider's staging directory, mount point and environment in the agent process.
5. Slurm [configuration](https://gitlab-master.nvidia.com/interactive-agents/slurm-evaluations/-/blob/jnolan/hermes-sandboxed-pro/configs/swebench_pro.yaml) and [existing launcher](https://gitlab-master.nvidia.com/interactive-agents/slurm-evaluations/-/blob/jnolan/hermes-sandboxed-pro/scripts/run_eval.sh).

## Why the remaining additions exist

- [Session cleanup](../../resources_servers/swebench_pro/app.py#L218): cleanup also has to work when execution fails before verification.
- [SIF provenance](../../resources_servers/swebench_pro/image_cache.py#L25): a cached local image must correspond to the dataset's pinned registry digest.
- [Runtime packaging](prepare_runtime.sh#L6): copied dependencies and a relocatable source path prevent imports from depending on inaccessible host caches. The libc selector is per image.
- [Partial trajectories and budget stops](runner.py#L21): wall-time, turn and output-token limits retain the patch's verifier score; genuine harness failures remain separately diagnosable.
- [Verifier timeout recovery](../../resources_servers/swebench_pro/verification.py#L420): parse saved test output after a timeout. An explicit failed required test is conclusive; otherwise verification remains incomplete.

Only SWE-bench Pro is integrated here. Verified compatibility is unproven. Rich
Hermes observation bundles are not implemented. The latest fixes still need a
complete production rerun; historical run reports are kept outside the source tree.
