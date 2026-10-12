# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM docker.io/library/python:3.13.7-slim-bookworm@sha256:adafcc17694d715c905b4c7bebd96907a1fd5cf183395f0ebc4d3428bd22d92d
# Freeze Debian packages and fonts to a snapshot, avoiding rolling apt sources.
RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    printf 'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20251001T000000Z bookworm main\n' > /etc/apt/sources.list && \
    apt-get -o APT::Sandbox::User=root update && apt-get -o APT::Sandbox::User=root install -y --no-install-recommends \
      util-linux=2.38.1-5+deb12u3 fonts-dejavu-core=2.37-6 fonts-liberation2=2.1.5-1 && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --no-deps -r /tmp/requirements.txt && \
    python -m pip check && rm /tmp/requirements.txt && mkdir /workspace
ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /workspace
CMD ["/bin/sh"]
