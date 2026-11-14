# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ui_generation_agentic agent image for the ui-static stack. Build it with build_images.sh in this directory (empty
# build context; the OpenCode plugin dependencies come from benchmarks/agentic_if/opencode as the named context
# `opencode_plugins`, and Codex's installer from responses_api_agents/codex_agent as `codex_installer`), push it to a
# registry your sandboxes can pull, and point UI_GEN_STATIC_IMAGE at its digest.
#
# Stage 1 is the stack's sample-runner project:
#   nginx on Debian serving one self-contained HTML page.
# The agent layer adds the pinned OpenCode runtime, ripgrep, OpenCode's plugin dependencies, a verifier user and a
# root-owned read-only copy of the project (/opt/ui-gen/template, checked against /opt/ui-gen/MANIFEST.sha256 at
# verify time).

# ---- Stage: the stack's sample-runner project ----
FROM nginx:1.27-bookworm@sha256:6784fb0834aa7dbbe12e3d7471e69c290df3e6ba810dc38b34ae33d3c1c05f7d AS stack
RUN rm -f /usr/share/nginx/html/*

COPY <<'EOF' /etc/nginx/conf.d/default.conf
server {
    listen 3000;
    root /usr/share/nginx/html;
    index index.html;
    location / { try_files $uri /index.html; }
}
EOF

COPY --chmod=755 <<'EOF' /usr/local/bin/run-sample
#!/bin/sh
# Modes: serve (default) | build (check the page is a complete HTML document)
set -eu
extract() {
  awk '{ l[NR]=$0 } END {
    s=0; for (i=1;i<=NR;i++) if (l[i] ~ /^```/) { s=i; break }
    if (!s) { for (i=1;i<=NR;i++) print l[i]; exit }
    for (i=s+1;i<=NR;i++) {
      # The closing fence is sometimes indented or glued to the last code line ("}```").
      if (l[i] ~ /```[ \t\r]*$/) { sub(/[ \t]*````*[ \t\r]*$/, "", l[i]); if (l[i] ~ /[^ \t\r]/) print l[i]; exit }
      print l[i]
    }
  }' "$1"
}
MODE="${1:-serve}"
SRC="${SAMPLE:-/sample/input}"
[ -f "$SRC" ] || { echo "run-sample: mount a sample at $SRC" >&2; exit 2; }
extract "$SRC" > /usr/share/nginx/html/index.html
grep -qi '</html>' /usr/share/nginx/html/index.html || { echo "run-sample: no </html> in sample" >&2; exit 1; }
case "$MODE" in
  build) echo "ok" ;;
  serve) exec nginx -g 'daemon off;' ;;
  *)     echo "run-sample: unknown mode $MODE" >&2; exit 2 ;;
esac
EOF

EXPOSE 3000
ENTRYPOINT ["run-sample"]
CMD ["serve"]

# ---- Stage: pinned OpenCode runtime and ripgrep ----
FROM debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc AS fetch
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
ARG OPENCODE_VERSION=1.17.11
ARG RG_VERSION=15.1.0
# SHA-256 of the OpenCode binary, the same pin as the compositions' sandbox_executable_sha256.
ARG OPENCODE_SHA256=0254a429cd0e6cf0ba53fc01672cf98e4a8dc728f7fa94be88f1e4b3645e6ded
# SHA-256 of the rg binary, as pinned in Gym benchmarks/agentic_if/README.md.
ARG RG_SHA256=ebeaf56f8a25e102e9419933423738b3a2a613a444fd749d695e15eba53f71f2
RUN set -eux; mkdir -p /out; \
  curl -fsSL --retry 3 "https://github.com/anomalyco/opencode/releases/download/v${OPENCODE_VERSION}/opencode-linux-x64.tar.gz" -o /tmp/oc.tgz; \
  tar -xzf /tmp/oc.tgz -C /out opencode; chmod 755 /out/opencode; \
  echo "${OPENCODE_SHA256}  /out/opencode" | sha256sum -c -; \
  test "$(/out/opencode --version)" = "${OPENCODE_VERSION}"; \
  curl -fsSL --retry 3 "https://github.com/BurntSushi/ripgrep/releases/download/${RG_VERSION}/ripgrep-${RG_VERSION}-x86_64-unknown-linux-musl.tar.gz" -o /tmp/rg.tgz; \
  tar -xzf /tmp/rg.tgz -C /tmp; cp "/tmp/ripgrep-${RG_VERSION}-x86_64-unknown-linux-musl/rg" /out/rg; chmod 755 /out/rg; \
  echo "${RG_SHA256}  /out/rg" | sha256sum -c -

# OpenCode's plugin dependencies (@opencode-ai/plugin@1.17.11), built reproducibly like
# benchmarks/agentic_if/README.md describes; the result matches that README's SHA-256.
FROM node:22-bookworm-slim@sha256:e9f565d32859f0cbfb52125eafe8be39390f24dac302dc3cdf566e8ba3ab21ee AS plugins
WORKDIR /build/opencode
COPY --from=opencode_plugins package.json package-lock.json /build/opencode/
RUN set -eux; npm ci --ignore-scripts --no-audit --no-fund --loglevel=error; \
  tar -C /build/opencode --sort=name --mtime=@1792627200 --owner=0 --group=0 --numeric-owner --mode=u+rwX,go+rX,go-w \
      --hard-dereference --format=gnu -cf - package.json package-lock.json node_modules \
  | gzip -n -9 > /build/opencode-1.17.11-plugin-dependencies.tar.gz

# ---- Stage: optional Codex runtime (--build-arg WITH_CODEX=1; off by default) ----
# Codex's own installer (responses_api_agents/codex_agent/install_codex_runtime.sh, named context `codex_installer`)
# fetches Node 22.19.0, checked against nodejs.org's SHASUMS256, and @openai/codex (Apache-2.0) into the runtime dir
# that the codex composition names as sandbox_runtime_dir, because the sandbox itself has no internet.
FROM debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc AS codex
ARG WITH_CODEX=0
ARG CODEX_VERSION=0.144.4
COPY --from=codex_installer install_codex_runtime.sh /tmp/install_codex_runtime.sh
RUN set -eux; mkdir -p /out; \
  if [ "$WITH_CODEX" = 1 ]; then \
    apt-get update; DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl \
      util-linux gawk; rm -rf /var/lib/apt/lists/*; \
    bash /tmp/install_codex_runtime.sh /opt/ui-gen/runtime/codex "$CODEX_VERSION"; \
    cd /opt/ui-gen/runtime/codex; rm -rf cache home install.lock node-v*.tar.gz SHASUMS256.txt; \
    chown -R root:root /opt/ui-gen/runtime/codex; chmod -R a-w,a+rX /opt/ui-gen/runtime/codex; \
    mkdir -p /out/opt/ui-gen/runtime; mv /opt/ui-gen/runtime/codex /out/opt/ui-gen/runtime/codex; \
    chmod 555 /out/opt/ui-gen /out/opt/ui-gen/runtime; \
  fi

# Final image for ui-static (Debian): agent user `ui`, verifier user `ui-verify`, web root owned by `ui`.
FROM stack
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends procps \
 && rm -rf /var/lib/apt/lists/*
COPY --from=fetch /out/opencode /opt/ui-gen/runtime/opencode
COPY --from=fetch /out/rg /opt/ui-gen/runtime/rg
COPY --from=plugins /build/opencode-1.17.11-plugin-dependencies.tar.gz /opt/ui-gen/runtime/opencode-1.17.11-plugin-dependencies.tar.gz
RUN set -eux; \
  useradd --uid 1000 --create-home --shell /bin/bash ui; \
  useradd --system --uid 1001 --no-create-home --shell /usr/sbin/nologin ui-verify; \
  mkdir -p /opt/ui-gen/template; cp /etc/nginx/conf.d/default.conf /opt/ui-gen/template/nginx-default.conf; \
  chown -R root:root /opt/ui-gen; chmod -R a-w,a+rX /opt/ui-gen; \
  chown -R ui:ui /usr/share/nginx/html; \
  cd /; { find /opt/ui-gen/template -type f -print | LC_ALL=C sort | xargs -d '\n' sha256sum; \
          sha256sum /usr/local/bin/run-sample /opt/ui-gen/runtime/opencode /opt/ui-gen/runtime/rg \
                    /opt/ui-gen/runtime/opencode-1.17.11-plugin-dependencies.tar.gz; } > /opt/ui-gen/MANIFEST.sha256; \
  chmod 444 /opt/ui-gen/MANIFEST.sha256; touch /opt/ui-gen/.built
# The Codex runtime when built WITH_CODEX=1 (nothing otherwise), root-owned and read-only as set in its stage. It is
# copied after the permission pass above (so its files are stored once) and stays out of MANIFEST.sha256, so the
# manifest pins are the same with or without it.
COPY --from=codex /out/ /
# Codex keeps its sessions under /tmp and refuses to create its helper binaries under the temp dir, so the temp dir
# is /var/tmp (harmless for the other harnesses and the build).
ENV TMPDIR=/var/tmp
WORKDIR /usr/share/nginx/html
USER ui
ENTRYPOINT []
CMD []
