# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Grading runtime for critpt_custom_grader. The runner is invoked as `/usr/local/bin/python3 -I -B`, which ignores
# PYTHONPATH, so the pinned packages live in this interpreter's own site-packages. Every grader command runs as the
# non-root `grader` account; sandboxes are created with workdir /work, which must exist in the image.
# The build fails if the interpreter, started as the runner starts it, does not import the five pinned packages at their
# pinned versions (the table in that check must equal requirements.txt), or if `su` to the account does not give uid 1000.
# Build and publish for both architectures, then set sandbox_spec.image (or CRITPT_CUSTOM_GRADER_IMAGE) to the tag:
#   docker buildx build --platform linux/amd64,linux/arm64 \
#       -t <registry>/<path>/critpt-custom-grader-rt:<tag> --push resources_servers/critpt_custom_grader/runtime
FROM python:3.13-slim
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --root-user-action=ignore -r /tmp/requirements.txt && rm /tmp/requirements.txt
RUN /usr/local/bin/python3 -I -c "import pydantic, sympy, mpmath, numpy, scipy; got = {m.__name__: m.__version__ for m in (pydantic, sympy, mpmath, numpy, scipy)}; want = {'pydantic': '2.13.4', 'sympy': '1.14.0', 'mpmath': '1.3.0', 'numpy': '2.5.1', 'scipy': '1.18.0'}; assert got == want, (got, want)"
RUN useradd --uid 1000 --create-home --shell /bin/sh grader && mkdir -p /work && chown grader:grader /work
RUN test "$(su -s /bin/sh -c 'id -u' grader)" = 1000
WORKDIR /work
CMD ["sleep", "infinity"]
