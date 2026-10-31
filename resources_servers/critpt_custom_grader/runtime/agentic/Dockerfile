# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build before creating offline execution sandboxes; record and use the resulting immutable image digest.
# Override PYTHON_IMAGE with a registry digest to reproduce the exact base image as well.
ARG PYTHON_IMAGE=python:3.13.14-slim-bookworm
FROM ${PYTHON_IMAGE}

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore -r /tmp/requirements.txt \
    && mkdir -p /opt/critpt-agentic-runtime \
    && cp /tmp/requirements.txt /opt/critpt-agentic-runtime/requirements.txt \
    && python3 -m pip freeze --all > /opt/critpt-agentic-runtime/packages.txt \
    && python3 --version > /opt/critpt-agentic-runtime/python.txt \
    && rm /tmp/requirements.txt
RUN python3 -I -c "import sys, numpy, scipy, sympy, mpmath, matplotlib; assert sys.version_info[:2] == (3, 13) and sys.version_info[:3] >= (3, 13, 14); got = {m.__name__: m.__version__ for m in (numpy, scipy, sympy, mpmath, matplotlib)}; assert got == {'numpy': '2.5.1', 'scipy': '1.18.0', 'sympy': '1.14.0', 'mpmath': '1.3.0', 'matplotlib': '3.11.0'}, got"
RUN command -v bash && command -v timeout \
    && useradd --uid 1000 --create-home --shell /bin/bash agent \
    && mkdir -p /sandbox && chown agent:agent /sandbox

ENV MPLBACKEND=Agg OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
WORKDIR /sandbox
# Keep the provider's exec daemon as root so OpenSandbox can use su; the recipe's
# sandbox_spec.user: agent makes each model command run as the non-root account.
CMD ["sleep", "infinity"]
