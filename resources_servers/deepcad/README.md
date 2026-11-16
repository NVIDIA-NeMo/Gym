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

Invalid generated code or geometry earns zero. Missing/invalid references,
scorer failures, and sandbox infrastructure failures set `mask_sample: true`.
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

The committed records use the Mixed76 checkpoint through the single-turn
Environment Server on HSG, with Enroot and CadQuery 2.8.0. The Enroot deployment
sets `bypass_entrypoint: false`. Docker and OpenCode require separate live checks.

`dataset/prepare_gym.py` prepares images and task rows from DeepCAD construction
histories. `dataset/augment_inputs.py` supplies deterministic axis/diagonal view
selection and foreground colors. The renderer is also reused by the OpenCode
feedback tools. Reference assets and generated images remain outside the agent
workspace except for the supplied target PNGs.

```bash
python -m pytest resources_servers/deepcad/tests -q
```

Image-rendering tests require a functioning VTK/OpenGL runtime. The seven-agent
file-input recipe and runtime prerequisites are documented in
[`CAD workspaces with native agents`](OPENCODE.md).
