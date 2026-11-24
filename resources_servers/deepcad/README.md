# CAD reconstruction

Reconstruct a CAD solid from target images by writing Python with CadQuery.
The verifier executes the generated program in a separate disposable sandbox,
then scores its exported STEP geometry against a reference available only to
the trusted scorer. Programs must explicitly import CadQuery and assign a valid
solid to `result`.

## Setup

Install the locked resources-server dependencies from the repository root:

```bash
uv sync --locked --project resources_servers/deepcad
```

Build the execution image from the repository root:

```bash
docker build -t deepcad-cadquery:2.8.0 resources_servers/deepcad
```

`configs/deepcad.yaml` wires the verifier to Gym's single-turn agent. Set the
model endpoint using the normal Gym model-server configuration. Supply your own
prepared task JSONL and reference STEP files for training. The default config
includes five synthetic examples and their reference solids in `data/`, licensed
under Apache-2.0. No research corpus is bundled. To use external data, override
`reference_root` and pass your task JSONL to the rollout collector. Each task's
`reference_step` must be relative to `reference_root`; references outside that
directory are rejected.

The execution sandbox has no network and receives only the submitted program
and runner. It receives no reference geometry. Host scoring dependencies must
also be installed separately from the sandbox image.

## Rewards and failures

The default reward is B-rep Boolean volume IoU. Each solid is independently
centered and uniformly scaled by its longest bounding-box dimension. Orientation
and aspect ratio are preserved. `iou_backend: voxel` selects filled-grid IoU.

`verification_metric: mixed_iou` combines B-rep, octree, and boundary IoU with
default weights 0.3, 0.3, and 0.4. Set `verification_metric: mixed_iou`,
`mixed_iou: {}`, and `reward_weights: {voxel_iou: 0}` to enable it.
Geometry scoring requires a positive geometry reward weight. Octree depth and the boundary band
must fit `voxel_resolution`. `reference_cache_root` can cache reference geometry
preparation; use node-local storage for active caches on clusters.

`operation_bonus.enabled` optionally rewards recall of reference CadQuery
operation names found by static analysis of the submitted source in the verifier.
It adds at most `max_fraction` times the geometry reward. Failed or masked samples
receive no bonus. Task metadata supplies `reference_operations`; the response
reports `source_operations` instead of the previous `executed_operations` field.
The verifier does not read `operations.json` or execute source to extract names.
Comments and strings are ignored, and repeated operation names count once.
This initial heuristic can count dead code and unrelated methods with matching
names, and can miss dynamic calls. It does not prove execution or contribution
to the final geometry. The bonus remains disabled by default; scores with it
enabled are not directly comparable to the previous runtime-trace version.

A fresh isolated Python process fuses the exported STEP within the remaining
generated-program execution budget. It seals the result in a controller-owned
directory before download, and the scorer preserves that union during normalization.
This protection requires separate model and controller users, as supplied by the
Docker image; a provider that maps both to one UID does not enforce the same boundary. Invalid generated code or
geometry, including prediction normalization failures, earns zero. Missing/invalid references,
inconclusive pairwise scorer failures, and sandbox infrastructure failures set
`mask_sample: true`. Provider and scorer exceptions retry the same source in a
fresh sandbox up to `inconclusive_verification_retries` times (default 2).
Exhausted retries report `evaluation_completed: false` and a registered
`provider_unavailable` or `verifier_error` failure kind. Model failures do not retry.
The historical `voxel_iou` reward-component key denotes volume IoU from the
selected backend. Raw scores remain available independently of reward weights.

When native scoring libraries need external search paths, set
`geometry_library_dirs` to their directories. The verifier prepends these to
`LD_LIBRARY_PATH` for each scoring subprocess while preserving inherited paths.
Startup checks reject missing directories and failed scorer imports before
rollouts begin. A self-contained server environment normally needs no override.

## Inputs and tests

Regenerate the five synthetic examples (box, cylinder, sphere, tube, and drilled
plate) and their embedded axis-view images from the repository root:

```bash
python -m resources_servers.deepcad.dataset.prepare_examples
gym dataset collate --config resources_servers/deepcad/configs/deepcad.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml \
  +output_dirpath=/tmp/cad-example-validation +mode=example_validation
```

Use the normal model-server configuration for collection. With the services
running, collect five actual agent responses and verifier results:

```bash
gym eval run --no-serve --agent deepcad_simple_agent \
  --input resources_servers/deepcad/data/example.jsonl \
  --output resources_servers/deepcad/data/example_rollouts.jsonl --limit 5
```

The committed records are real single-turn responses from the Mixed76 step-76
checkpoint. These five CI fixtures omit optional training-token arrays; the full
validation captures retain token IDs and log probabilities. A Linux ARM64 smoke
ran all five tasks twice through the Environment Server with Enroot and
CadQuery 2.8.0 (`bypass_entrypoint: false`): mean B-rep IoU 0.933333, no masks or
failures. The sphere scored 0.666667 on both runs; the other tasks scored 1.0.
All ten records retained their prompts and three decodable target PNGs.
Reference-as-answer baselines scored 1.0 on every task; a valid, severely
mismatched thin plate scored between 0 and 1.91e-9.
See [OpenCode validation](OPENCODE.md#validation) for multi-turn results and the
remaining deployment validation gaps.

`dataset/prepare_gym.py` prepares images and task rows from DeepCAD construction
histories. `dataset/augment_inputs.py` supplies deterministic axis/diagonal view
selection and foreground colors. The renderer is also reused by the OpenCode
feedback tools. Reference assets and generated images remain outside the agent
workspace except for the supplied target PNGs.

```bash
python -m pytest resources_servers/deepcad/tests -q
```

Image-rendering tests require a functioning VTK/OpenGL runtime. The OpenCode
multi-turn integration is documented in
[`OpenCode workspace setting`](OPENCODE.md).

## Lazy inputs and source/operation sampling

CAD owns its rendering and sampling policy in Gym. Training frameworks only call
configured dataset hooks:

```yaml
data:
  shuffle: false
  gym_dataset_hooks:
    module: resources_servers.deepcad.dataset.training
    config:
      source_operation_sampling:
        sources: [zero-to-cad, cadevolve, op-cad, histcad]
        uniform_sources: [histcad]
        operation_fraction: 0.5
        tau: 1000
        seed: 42
```

The order is raw metadata selection, unique selected-row materialization, image
processing, then GRPO repetition. Indexing never renders or calls a processed
dataset item. The sampler has no PyTorch dependency; it implements the iterator
and checkpoint methods accepted by `StatefulDataLoader`.

Each prompt independently samples a source uniformly, so source counts can vary
within a batch. Within a source, a draw mixes uniform
rows with operation buckets weighted by `log(1 + bucket_size / tau)`. Duplicate
operation names count once. Sources in `uniform_sources` use uniform rows. Draws
use replacement. Epoch size must be a positive multiple of the positive batch
size. Do not combine this policy with
shuffling, static source balancing, or multiple dataloaders. A seed, batch index,
and epoch determine each batch; checkpoint fingerprints reject changed metadata
or settings and preserve resume across prefetching. Checkpoints from the previous
quota-based sampler require a fresh dataloader state.

`dataset/lazy_inputs.py` supports metadata-only `deepcad_lazy` manifests. Only
sampled rows invoke bounded renderer subprocesses. Per-key locks and atomic cache
writes share a cold render across repeated prompts and concurrent callers. A warm
cache needs no renderer. Cache keys include source identity and rendering settings;
use node-local `cache_root` and keep the reference catalog local when possible.
Rendering a batch must succeed before any input row is changed. Each caller keeps
its routing, sampling settings, and rollout identifiers.

Corpus reference programs require an explicit `reference_sandbox_image` and use
`reference_sandbox_provider` (default: network-disabled Docker). Source and STEP
checksums are checked before use. Reference operation hints use static source
analysis, consistently with the operation bonus. Invalid or quarantined references
can use bounded deterministic replacements from the pinned catalog. The default
requires the selected source and all requested operations. Set
`allow_partial_operation_replacements: true` to permit a same-source selected
reference with the largest number of distinct matching operations after the
exact-match candidates run out. The verifier metadata records missing operations
in `reference_operation_fallback`. An execution timeout can trigger immediate
bounded replacement. A render timeout, process crash or kill needs two matching
observations from fresh renderer attempts before quarantine; success clears the pending
observation. SIGKILL/exit 137 is a possible memory-limit kill, not proof of OOM.
Repeated failures are an operational heuristic; after repairing a shared runtime,
remove the affected cache's `.failure.json` and confirmed rejection/timeout markers
to retry the original references. Legacy single-failure markers are ignored.
Provider errors and unclassified OCC/export errors still stop the batch. Investigate
the logged asset first; if it is deterministically bad, exclude it from the input
manifest and replacement catalog, then restart with fresh dataloader state (the
sampler rejects changed manifests when restoring an old state).

### Warm a prefix of ordered draws

Use the same sampler JSON and prompt batch size as the consumer. Export to a
new directory. The example exports 1,601,024 lightweight descriptors and warms
the first 1,024. The optional overrides JSON supplies deployment settings such as
`cache_root`, `renderer_python`, and `allow_partial_operation_replacements`.

```sh
module=resources_servers.deepcad.dataset.sampled_lazy
python -m "$module" export --manifest train.jsonl --sampler-config sampler.json --overrides render.json --batch-size 64 --count 1601024 --output ordered
python -m "$module" warm --output ordered --start 0 --count 1024 --workers 8
```

Consume `ordered/draws.jsonl` in order with shuffling and runtime resampling
disabled. Repeated draws are intentional. Exported rows enable
`warm_start_cache`. Use a persistent cache directory shared by warmers and
consumers, a filesystem supporting interprocess locks and atomic rename, and the
same immutable catalog, dataset, and render environment in both. Entries use
two-character shards, render scratch follows `TMPDIR`, and resolved replacements
are pinned. Ordinary lazy manifests continue to use their existing cache layout.

The warm command publishes `warm-0-1024-ready.json` after a fresh process
reads every prompt and verifies its reference STEP checksum with rendering
disabled. Repeat `verify --output ordered --start 0 --count 1024` for a fresh
check. Later draws render on demand. Use a new cache directory when refreshing
the catalog or retrying a pinned replacement after repairing the renderer.

The export manifest also records sampler indices, source-code checksums, optional
catalog and override checksums, and the seed. Warming rejects source changes after
export. Each completed draw records prompt-cache, replacement and STEP checksums,
whether its cache was already warm, and preparation time. An immutable
`draw<start>-ready.json` receipt is published after the first draw passes a separate
process's cache check, before the rest of the requested range finishes. A full
0–1,023 warmup also publishes `warm1024-ready.json` for prefix consumers.

### Prepare ordered rows for offline rendering

Resolve a bounded range into ordinary Gym rows before evaluation or training.
Export descriptors with the same Gym source version used for preparation. If the
source checksums differ after an upgrade, regenerate the descriptor export using
the same source manifest, sampler settings, batch size and deployment overrides;
the existing cache keys and draw order can then be reused:

```sh
python -m "$module" prepare --output ordered --start 0 --count 1024 --destination prepared
# After an existing warm --workers run, require cache hits without rendering:
python -m "$module" prepare --output ordered --start 0 --count 1024 --destination prepared-cached --cache-only
```

Each destination must be new. Publication is atomic: a failed range leaves no
partial destination. The bundle contains only `rows.jsonl` and `assets/`; the latter
holds SHA-256-addressed, prefix-sharded image and reference STEP files plus
`manifest.json` with row and asset hashes. Duplicate draws remain duplicate rows
and share assets; render variants use their own image content hashes. Preparation
uses the existing resolver and pinned replacements, preserving prompt text,
request settings, routing, operations and replacement metadata. It changes only
asset references in copies of resolved rows; cached images remain embedded.
An unconfirmed reference failure stops preparation for diagnosis or a later retry.
For bulk preparation, use the existing bounded `warm --workers` command first.
This command supports the simple harness. It rejects the OpenCode harness and
`target_images` workspace seeds, whose runtime requires embedded images.

Consume `prepared/rows.jsonl` in order, with shuffling and runtime resampling
disabled, using the existing dataset and processor. No NeMo-RL code change is
needed. These rows have no `deepcad_lazy` field. Set the CAD resources server's
`reference_root` to the absolute `prepared/assets` directory: `reference_step`
is relative to that root. Image paths are absolute final paths supported by
NeMo-RL media normalization. Publish at the persistent path visible to every
worker and keep that path fixed; moving the bundle requires regenerating its
image paths. The bundle no longer needs the rendering cache or source dataset.

### Deployment overrides and staged catalogs

The `resources_servers.deepcad.dataset.training` hook accepts optional
`lazy_render_overrides` alongside `source_operation_sampling`:

```yaml
lazy_render_overrides:
  renderer_python: /opt/cad/bin/python
  cache_root: /scratch/cad-cache
  reference_catalog: /scratch/cad-inputs/catalog.sqlite
  reference_catalog_root: /scratch/cad-inputs
```

The overrides apply to a copy of each selected lazy descriptor; ordinary rows and
the caller's descriptors stay unchanged. Supported fields also include
`reference_sandbox_provider` and `allow_partial_operation_replacements`. With no
overrides, existing hook behavior is unchanged. An explicit
`reference_catalog_root` permits a staged catalog outside `dataset_root`, while
still rejecting paths that escape that explicit root. The default catalog root
remains `dataset_root`. Missing sandbox fonts propagate as infrastructure errors
and do not quarantine valid reference programs.

For a finite pool with all rendering completed before consumption, see
[Offline materialized CAD tasks](dataset/materialized-inputs.md). That utility
uses the same sampler and renderer to publish ordered JSONL rows and a checked
STEP archive, with resumable CPU render chunks and atomic release publication.

`prepare_opencode.py` accepts validated lazy rows without rendering. It marks the
harness in their configuration; materialization then adds the workspace prompt and
explicit `target_images` used by the resources seed. Standalone consumers can load
the same hooks with `from nemo_gym.data import load_dataset_hooks`. These hooks
require the corresponding generic adapter in the CAD NeMo-RL branch; older
`data.source_operation_sampling` configurations must move under the block above.

Optional API-cheatsheet and orientation-overlay features remain on the full Gym
branch. This PR accepts their legacy lazy-config flags only when disabled.
