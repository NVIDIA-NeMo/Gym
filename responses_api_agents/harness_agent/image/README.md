# Offline harness runtime

This layer adds Python 3.13.14 for Gym and Node 22.23.2 / Pi 0.85.1 to the
[scientific image](science/README.md). OpenCode and
the scientific Python remain in that base. `/opt/gym-runtime/bin/python` runs Gym;
`python3` on the tool's PATH continues to use the scientific package environment.

Build from the Gym repository root:

```bash
docker build --platform linux/amd64 \
  --build-arg SCIENCE_IMAGE=<registry/image@sha256:digest> \
  -f responses_api_agents/harness_agent/image/Dockerfile \
  -t gym-harness-science:<tag> .
```

The Dockerfile pins the Python image and Node archive digest. Gym dependencies
come from this checkout's `uv.lock`, exported during the build. Installation uses
that explicit package set, including Gym's existing dependency exclusions. Pi's
transitive dependencies are recorded in `package-lock.json`; `npm ci` disables
lifecycle scripts. OpenCode's matching plugin dependency is also locked and installed
in its config directory, avoiding an attempted runtime registry download. The Dockerfile-specific ignore file restricts the build context
to these inputs. No credentials or benchmark datasets belong in the image.

Gym source is uploaded by the host runner for each rollout, so the source and
runtime dependency lock must remain compatible. The selected adapter and its
files are included in that upload. Package installation happens during image
build, before sandbox network restrictions are applied.

Validate both adapters' imports and CLI versions offline, then run real tool-using
rollouts through the shared runner and assigned sandbox provider before promoting
an image digest. Registry publishing and validation scripts belong to the deployment
workflow, outside this runtime package.
