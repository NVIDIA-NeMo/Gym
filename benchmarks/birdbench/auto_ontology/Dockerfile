# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The Auto Ontology agent's sandbox image for BIRD dev: Auto Ontology at the commit build_image.sh pins, the BIRD dev
# databases, and a catalog Postgres with every database ingested, described and compiled. Build it with
# build_image.sh, which passes the pin, the databases and the model keys; the keys reach the catalog build as a
# BuildKit secret only, never a layer.

FROM pgvector/pgvector:pg17 AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

# Auto Ontology from its own repository and lock file, as it ships: the package, its vendored engine and the catalog
# schema's Alembic migrations, which the package does not include.
ARG AUTO_ONTOLOGY_REPOSITORY
ARG AUTO_ONTOLOGY_COMMIT
RUN git init -q /opt/auto-ontology \
 && git -C /opt/auto-ontology fetch -q --depth 1 "$AUTO_ONTOLOGY_REPOSITORY" "$AUTO_ONTOLOGY_COMMIT" \
 && git -C /opt/auto-ontology checkout -q FETCH_HEAD
WORKDIR /opt/auto-ontology
RUN uv sync --frozen --no-dev

# The BIRD dev databases, each with its database_description CSVs.
COPY --from=bird . /opt/bird-catalog/databases/

# The catalog's Postgres. PGDATA is outside the base image's VOLUME, so the catalog built below stays in the image.
ENV PGDATA=/var/lib/postgresql/catalog \
    POSTGRES_HOST=localhost \
    POSTGRES_PORT=5432 \
    POSTGRES_USER=postgres \
    POSTGRES_PASSWORD=catalog \
    POSTGRES_DATABASE=auto_ontology
COPY benchmarks/birdbench/auto_ontology/build_catalog.py /opt/bird-catalog/

# The catalog, in its own stage: `--target runtime` builds the rest without the model keys.
FROM runtime AS catalog
RUN --mount=type=secret,id=models,required=true \
    /opt/auto-ontology/.venv/bin/python /opt/bird-catalog/build_catalog.py /run/secrets/models

# Last, so editing it does not rebuild the catalog.
COPY benchmarks/birdbench/auto_ontology/start-catalog /opt/bird-catalog/
