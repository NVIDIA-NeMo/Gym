# CVDP v1.1.0

This benchmark evaluates the 302-task non-agentic, non-commercial code-generation
split with Gym's existing CVDP agent and response parser. Category instructions
and response schemas come from `ModelHelpers`: single-file tasks request plain
RTL, and multi-file tasks request JSON containing file names and code.

Preparation downloads the pinned public Hugging Face payload and builds the rows
locally. It checks the raw-file SHA-256, the 302-task count and the complete output
SHA-256 before replacing the cached dataset. Cached data is reused only when its
contents match the output checksum.

Every nonempty input context file is delivered to the simulator, including files
whose paths are generation targets. Preparation removes unsupported `PROCASSINIT`
lint directives from `.vlt` harness files. Output reference implementations are
excluded from model prompts and verifier metadata.

## Reproduce dataset preparation

After the standard Gym development setup, run from the repository root:

```bash
source .venv/bin/activate
python -m benchmarks.cvdp_v110.prepare
sha256sum benchmarks/cvdp_v110/data/cvdp_v110.jsonl
```

Expected output: 302 JSONL rows with SHA-256
`3fa9800ad8696277abd7338be225a9388d35b9aa3e1b9e42b21989299735b36d`.
The input is
[`nvidia/cvdp-benchmark-dataset@5b807d94`](https://huggingface.co/datasets/nvidia/cvdp-benchmark-dataset/tree/5b807d945f6a99aa645f7e43a64a2115e281b4bf),
file `cvdp_v1.1.0_nonagentic_code_generation_no_commercial.jsonl`, with SHA-256
`cbcd81295561ebb16e4d857e096f4d9908d042c33aff3b58abf236e868411857`.
The data retains its [upstream license](https://huggingface.co/datasets/nvidia/cvdp-benchmark-dataset/blob/5b807d945f6a99aa645f7e43a64a2115e281b4bf/LICENSE).

For offline preparation, supply that downloaded file; the same checks apply:

```bash
python -m benchmarks.cvdp_v110.prepare --raw /path/to/cvdp_v1.1.0_nonagentic_code_generation_no_commercial.jsonl
```

`--output-dir /path/to/output` selects the destination. Set `HF_HOME` to control
Hugging Face download/cache storage. The default output path is consumed by
`gym eval prepare --benchmark cvdp_v110` and the benchmark configuration.
The dataset default is 32 samples per task. For end-to-end `gym eval run`, omit
`--num-repeats` to retain that default, or set `--num-repeats 32` explicitly.
Use `--num-repeats 1` for a one-sample-per-task smoke run. With `--no-serve`,
use 32 collection repeats for the unexpanded preparation output, or one repeat
for an input file that already contains 32 copies per task.

## Verification runtime

The simulator base needs cocotb 2.0.1 and pytest 8.3.2 on the osvb image.
Task Dockerfiles that install cocotb-bus need cocotb 1.9.2. This benchmark sets
`cocotb_bus_version` for image preparation; the setting is included in each
prepared-image manifest key.

Configure `cvdp_sim_image`, `oss_pnr_image`, the prepared-image manifest and a
supported sandbox provider. Publish prepared images to a registry reachable by
that provider. See `resources_servers/cvdp/README.md` and
`resources_servers/cvdp/prepare.py` for sandbox configuration and image preparation.
Use the same simulator base images at preparation and runtime:

```bash
python resources_servers/cvdp/prepare.py benchmarks/cvdp_v110/data/cvdp_v110.jsonl \
  --image-prefix <registry/repository> --manifest <prepared-images.json> \
  --oss-sim-image <cvdp_sim_image> --oss-pnr-image <oss_pnr_image> \
  --cocotb-bus-version 1.9.2 --push
```

Model serving, sampling and output budgets are operator-configured.

The verifier retries an inconclusive infrastructure failure once on a fresh
sandbox with the same model answer. Persistent infrastructure failures are
masked with `evaluation_completed=false`. Test timeouts and control-plane-confirmed
OOM during test execution are task failures. Setup OOM and unconfirmed backend
failures are inconclusive. Sandbox lifetime covers setup, testing and file transfer.

Reference-answer simulator replay and representative model rollouts remain
required to qualify this recipe.
