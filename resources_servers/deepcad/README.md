# CAD reconstruction

Reconstruct a CAD solid from target images by writing Python with CadQuery.
The verifier executes the generated program in a separate disposable sandbox,
then scores its exported STEP geometry against a reference available only to
the trusted scorer. Programs must explicitly import CadQuery and assign a valid
solid to `result`.

## Setup

Install `resources_servers/deepcad/requirements.txt` in the resources-server
environment. Build the execution image from the repository root:

```bash
docker build -t deepcad-cadquery:2.8.0 resources_servers/deepcad
```

`configs/deepcad.yaml` wires the verifier to Gym's single-turn agent. Set the
model endpoint using the normal Gym model-server configuration. Supply your own
prepared task JSONL and reference STEP files; this contribution bundles no data.
Set `reference_root` to the external reference directory (the config uses
`cad-data/references` as a placeholder) and pass your task JSONL to the rollout
collector. Each task's `reference_step` must be relative to `reference_root`;
references outside that directory are rejected.

The execution sandbox has no network and receives only the submitted program
and runner. It receives no reference geometry. Host scoring dependencies must
also be installed separately from the sandbox image.

## Rewards and failures

The default reward is B-rep Boolean volume IoU. Each solid is independently
centered and uniformly scaled by its longest bounding-box dimension. Orientation
and aspect ratio are preserved. `iou_backend: voxel` selects filled-grid IoU.

`verification_metric: mixed_iou` combines B-rep, octree, and boundary IoU with
default weights 0.3, 0.3, and 0.4. Configure `mixed_iou: {}` and
`reward_weights: {voxel_iou: 0}` to enable it. Octree depth and the boundary band
must fit `voxel_resolution`. `reference_cache_root` can cache reference geometry
preparation; use node-local storage for active caches on clusters.

`operation_bonus.enabled` optionally rewards recall of reference CadQuery
operations. It traces only the generated program in the execution sandbox and
adds at most `max_fraction` times the geometry reward. Failed or masked samples
receive no bonus. Task metadata supplies `reference_operations`.

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
[`responses_api_agents/deepcad_opencode`](../../responses_api_agents/deepcad_opencode/README.md).
