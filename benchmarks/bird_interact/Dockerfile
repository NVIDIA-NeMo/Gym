# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The episode sandbox for BIRD-Interact Lite and Full: PostgreSQL with every database of the release restored at build
# time, plus the Python that grading.py needs. Each episode starts its own sandbox from this image, so it gets its own
# copy of the databases without restoring them.
#
# The base is BIRD-Interact's official image, which holds the dumps and the scripts that restore them. On its own it
# restores them whenever a container first starts. Here they are restored once, into a data directory outside the
# base image's declared volume, so they stay in this image's layers.
#
#   Lite: docker build --platform linux/amd64 --tag benchmarks/bird_interact:lite benchmarks/bird_interact
#   Full: docker build --platform linux/amd64 --tag benchmarks/bird_interact:full \
#           --build-arg BIRD_INTERACT_IMAGE=docker.io/shawnxxh/bird-interact-postgresql-full@sha256:de4b88f6f3211238318a3dc265914f4e38122ec67ece642bbf3df0c3b781040c \
#           benchmarks/bird_interact
#
# Base images are digest-pinned: BIRD-Interact publishes only :latest.
ARG BIRD_INTERACT_IMAGE=docker.io/shawnxxh/bird-interact-postgresql@sha256:1ae45d7aa5d64dd8eb82e4058f56b4b9625d5035b9b9dc0d2afa0295d9d3053c
FROM ${BIRD_INTERACT_IMAGE}

# libpq reads PGHOST and PGUSER, so grading.py connects over the local socket as the release's superuser, root.
ENV PGDATA=/var/lib/postgresql/bird_interact \
    PGHOST=/var/run/postgresql \
    PGUSER=root

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-psycopg2 \
    && rm -rf /var/lib/apt/lists/*

# Restore the release's <db>_template databases with its own scripts, on a server that listens on the socket only.
# The scripts log failed imports to /tmp/error.log and go on, so the build prints that log and fails when no database
# was restored.
RUN set -eu; \
    install -d -o postgres -g postgres -m 0700 "$PGDATA"; \
    gosu postgres initdb --username=root --auth=trust --encoding=UTF8; \
    gosu postgres pg_ctl start --wait --log=/tmp/restore.log --options="-c listen_addresses="; \
    for script in /docker-entrypoint-initdb.d/*.sh; do gosu postgres bash "$script"; done; \
    restored="$(gosu postgres psql --dbname=postgres --tuples-only --no-align \
        --command="SELECT count(*) FROM pg_database WHERE datname LIKE '%\_template' AND datname <> 'sql_test_template'")"; \
    gosu postgres pg_ctl stop --wait --mode=fast; \
    if [ -s /tmp/error.log ]; then cat /tmp/error.log; fi; \
    echo "Restored ${restored} databases"; \
    test "${restored}" -gt 0
