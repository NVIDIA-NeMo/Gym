# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The Auto Ontology agent's sandbox for a BIRD-Interact release: the release's database image (../Dockerfile), with
# pgvector, Auto Ontology at the commit build_image.sh pins, and Auto Ontology's catalog of the release's databases in
# the same PostgreSQL. Build it with build_image.sh, which passes the pin, the database image and the model keys; the
# keys reach the catalog build as a BuildKit secret only, never a layer.

ARG BIRD_INTERACT_DATABASE_IMAGE=benchmarks/bird_interact:lite
FROM ${BIRD_INTERACT_DATABASE_IMAGE} AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates curl git "postgresql-${PG_MAJOR}-pgvector" \
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

# The catalog, in the release's PostgreSQL, which runs as the release's superuser, root, with trust authentication. The
# base image holds each database twice: as <db>, which the catalog and the agent read, and as <db>_template, from which
# the bird_interact Environment Server copies a task's database.
ENV POSTGRES_HOST=localhost \
    POSTGRES_PORT=5432 \
    POSTGRES_USER=root \
    POSTGRES_PASSWORD=catalog \
    POSTGRES_DATABASE=auto_ontology
COPY benchmarks/bird_interact/auto_ontology/build_catalog.py benchmarks/bird_interact/auto_ontology/column_meanings.py \
     /opt/bird-catalog/

# The catalog, in its own stage: `--target runtime` builds the rest without the model keys.
FROM runtime AS catalog
# With COLUMN_MEANINGS=true, the catalog holds the release's column meanings as column descriptions, as
# auto-ontology-eval builds it. The build reads each database's schema and column meanings from a bind mount, so no
# layer holds the release's files, and the build context leaves out the knowledge base, part of which each task hides
# (Dockerfile.dockerignore).
ARG BIRD_INTERACT_RELEASE=lite
ARG COLUMN_MEANINGS=false
# build_image.sh's digest of the secret's settings: BuildKit leaves the secret's contents out of the cache key.
ARG MODELS_SETTINGS_DIGEST
RUN --mount=type=secret,id=models,required=true \
    --mount=type=bind,source=benchmarks/bird_interact/data/${BIRD_INTERACT_RELEASE}/databases,target=/opt/bird-catalog/databases \
    /opt/auto-ontology/.venv/bin/python /opt/bird-catalog/build_catalog.py /run/secrets/models \
      $([ "$COLUMN_MEANINGS" = true ] && echo --column-meanings /opt/bird-catalog/databases)

# Last, so editing it does not rebuild the catalog.
COPY benchmarks/bird_interact/auto_ontology/start-catalog /opt/bird-catalog/
