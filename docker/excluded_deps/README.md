# Optional container codecs

`bash docker/install_excluded_deps.sh` syncs this independent project into the
active Gym virtual environment using `uv sync --locked --inexact`. It preserves
base Gym packages and does not modify the root project's exclusion policy.
Activate the Gym virtual environment first; system Python is rejected.

The direct versions are unchanged from the previous installer, including
`torchvision==0.26.0` and `torchaudio==2.11.0`, whose locked dependencies require
`torch==2.11.0`. The root Gym project currently locks a different Torch version.
The installer rejects an existing incompatible Torch version before syncing,
rather than downgrading the model runtime. Use this opt-in layer only for a
compatible runtime; reconciling those existing codec/model-runtime versions is
a separate dependency change, not part of this lockfile migration. A subsequent root `uv sync` restores the root lock and
removes these deliberately excluded packages.

To change this layer, edit `pyproject.toml`, run `uv lock` in this directory,
review the dependency delta, and commit both files. No scheduled updates are
configured.
