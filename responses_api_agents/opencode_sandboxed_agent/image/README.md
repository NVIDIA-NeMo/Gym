# OpenCode STEM image

Build with `docker build --platform linux/amd64 -t <registry/repository>:<tag> .`.
The Dockerfile pins the base images, Debian snapshot and OpenCode binary checksum.
It downloads the Python lock from NeMo Skills revision
`bcf059af55c20a89f797724598f9908d126153e6` and verifies its SHA-256 before installing.
Installed package inventories and the input lock remain in `/opt/image-provenance/`.
No credentials, benchmark datasets or reference answers are included.
The sandbox service injects execd.

Push to your authorized registry and configure the sandbox with the resulting
image digest. Test a real sandbox pull and offline execution before adopting it.
Image validation and registry-specific receipts belong to the deployment workflow.

Packages may load lazily in Kubernetes. The first import can be much slower than
later imports in the same sandbox; this image does not perform a warm-up at startup.
