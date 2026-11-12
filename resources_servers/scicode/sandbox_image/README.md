# SciCode-Verified sandbox image

Build for `linux/amd64`, passing `--build-arg BASE_IMAGE=<runtime@sha256:digest>`
for the SciCode-Verified runtime being reproduced. Use this Dockerfile and the pinned release's
`test_data_cleaned.h5` in the build context. The build verifies the HDF5 checksum.
The base image preserves EFB's published 2024/2025 Python environments, including
`/opt/scicode/{2024,2025}-installed.txt`; dependencies are not reinstalled.

Publish to a registry the sandbox service can pull, then configure the resulting
immutable digest as `SCICODE_SANDBOX_IMAGE`. The image contains benchmark data
and scientific dependencies. Gym sends its current grader at execution time.
Validate a live sandbox pull, scientific imports, cleanup, and saved-generation
verdict parity before using a new image for benchmark reporting.
