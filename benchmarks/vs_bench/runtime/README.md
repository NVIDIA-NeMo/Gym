# VS-Bench evaluator runtime

The evaluator runs separately from Gym and the model server. The candidate
builder targets Linux ARM64 and Python 3.10, matching the HSG runs. It is based
on the hash-locked dependency inputs and build procedure used by those runs.
The extracted builder completed on an HSG ARM64 Gym runtime image, followed by
scorer and real-model smoke checks. Other platforms need their own lock and
validation.

This is the companion provisioning change for the
[VS-Bench Gym adapter (draft PR #5977)](https://github.com/NVIDIA-NeMo/Gym/pull/5977).
The provisioning branch is
[`vs-bench-runtime-provisioning-20261110`](https://github.com/aroshanghias-nvd/Gym/tree/vs-bench-runtime-provisioning-20261110).
Apply both changes to the same Gym checkout before using
the source-preparation commands below; this change alone does not provide
`benchmarks.vs_bench.prepare_patched_source`. The
[adapter instructions](../README.md) are supplied by the adapter change.

Inside a disposable build allocation with network access, install `uv`, a
C++17 compiler, `git`, `pkg-config`, SDL2 and zlib development headers. Choose
an absolute, persistent Python 3.10 executable and an unused absolute venv path:

```bash
bash benchmarks/vs_bench/runtime/build_linux_arm64.sh /absolute/new-venv /absolute/python3.10
```

The builder refuses an existing target and leaves failed builds for inspection.
It installs the locked Python dependencies, rebuilds OpenSpiel 1.4 with Hanabi
if necessary, verifies the SHA-256 of the ALE source archive from a pinned
upstream commit, builds ALE, and checks the Pong ROM against pinned wheel and
ROM hashes. `BUILD_COMPLETE` records only that the
build checks passed: it is not a benchmark or sandbox qualification. The ALE
archive hash was recorded from the pinned upstream download and is checked
separately from the Python dependency lock; a shared artifact still needs the
deployment's provenance review. Record the
container identity, build logs, `installed.txt`, and lock digest with releases.

The Python interpreter and venv paths must be visible inside the evaluation and
Bubblewrap environments. Separately prepare the pinned author source, dataset,
and KAZ scorer variant as described in the adapter instructions above. Perception also
requires a working Bubblewrap executable and user namespaces on the deployment.
Run real gameplay, scorer, and containment checks on the target image before
using this runtime for a reported result.

## Reference source and data

Starting with a local checkout of `https://github.com/zelaix/VS-Bench` that
contains commit `2c3531cdf91d1b620b4a9015126a242eb305a4d0`, export that commit
to an empty directory without copying local checkout changes. For example:

```bash
mkdir /absolute/clean-vs-bench-source
git -C /absolute/author-checkout archive 2c3531cdf91d1b620b4a9015126a242eb305a4d0 | \
  tar -x -C /absolute/clean-vs-bench-source
python -m benchmarks.vs_bench.prepare_patched_source \
  /absolute/clean-vs-bench-source /absolute/new-kaz-fixed-source
```

The last command verifies the pinned source-file manifest, creates a separate
destination, and records the patch identity. Use the clean export for upstream
scoring or the separate export for `kaz_numpy_labels_v1`; do not mix them.

With `huggingface_hub` installed in a separate provisioning environment,
retrieve the pinned released dataset and retain the returned snapshot path:

```python
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="VS-Bench/VS-Bench",
    repo_type="dataset",
    revision="30ef045024e1f4a778166f274767f6c8d354c676",
    allow_patterns=["reasoning/*.parquet", "perception/*.parquet"],
)
print(path)
```

Set `VS_BENCH_DATASET_ROOT` to that snapshot, not its parent cache directory.
If the files are copied elsewhere, provide the dataset provenance and checksum
receipts required by `benchmarks.vs_bench.prepare.validate_source`. The released
KAZ perception file has 300 examples; the other nine have 400 each.
