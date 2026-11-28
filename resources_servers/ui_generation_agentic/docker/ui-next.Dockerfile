# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ui_generation_agentic agent image for the ui-next stack. Build it with build_images.sh in this directory (empty
# build context; the OpenCode plugin dependencies come from benchmarks/agentic_if/opencode as the named context
# `opencode_plugins`, and Codex's installer from responses_api_agents/codex_agent as `codex_installer`), push it to a
# registry your sandboxes can pull, and point UI_GEN_NEXT_IMAGE at its digest.
#
# Stage 1 is the stack's sample-runner project:
#   Next.js 15 App Router + React 19 + Tailwind 3 with the npm packages UI samples use.
# The agent layer adds the pinned OpenCode runtime, ripgrep, OpenCode's plugin dependencies, a verifier user and a
# root-owned read-only copy of the project (/opt/ui-gen/template, checked against /opt/ui-gen/MANIFEST.sha256 at
# verify time).

# ---- Stage: the stack's sample-runner project ----
FROM node:22-bookworm-slim@sha256:e9f565d32859f0cbfb52125eafe8be39390f24dac302dc3cdf566e8ba3ab21ee AS stack
ENV NEXT_TELEMETRY_DISABLED=1 NODE_OPTIONS=--max-old-space-size=4096
WORKDIR /app
RUN chown node:node /app
USER node

COPY --chown=node:node <<'EOF' /app/package.json
{
  "name": "ui-next",
  "private": true,
  "dependencies": {
    "@ant-design/icons": "6.0.2",
    "@ctrl/tinycolor": "4.1.0",
    "@dnd-kit/core": "6.3.1",
    "@dnd-kit/modifiers": "9.0.0",
    "@dnd-kit/sortable": "10.0.0",
    "@dnd-kit/utilities": "3.2.2",
    "@emotion/react": "11.14.0",
    "@emotion/styled": "11.14.1",
    "@faker-js/faker": "9.9.0",
    "@fontsource-variable/manrope": "5.2.8",
    "@fontsource-variable/montserrat": "5.2.8",
    "@fontsource/poppins": "5.2.7",
    "@fortawesome/fontawesome-svg-core": "7.0.1",
    "@fortawesome/free-brands-svg-icons": "7.0.1",
    "@fortawesome/free-solid-svg-icons": "7.0.1",
    "@fortawesome/react-fontawesome": "0.2.3",
    "@fullcalendar/core": "6.1.21",
    "@fullcalendar/daygrid": "6.1.21",
    "@fullcalendar/interaction": "6.1.21",
    "@fullcalendar/react": "6.1.21",
    "@fullcalendar/timegrid": "6.1.21",
    "@headlessui/react": "2.2.10",
    "@hello-pangea/dnd": "18.0.1",
    "@heroicons/react": "2.2.0",
    "@hookform/resolvers": "5.2.2",
    "@monaco-editor/react": "4.7.0",
    "@radix-ui/react-accordion": "1.2.21",
    "@radix-ui/react-avatar": "1.1.12",
    "@radix-ui/react-dialog": "1.1.23",
    "@radix-ui/react-dropdown-menu": "2.1.25",
    "@radix-ui/react-hover-card": "1.1.24",
    "@radix-ui/react-icons": "1.3.2",
    "@radix-ui/react-popover": "1.1.23",
    "@radix-ui/react-progress": "1.1.17",
    "@radix-ui/react-scroll-area": "1.2.18",
    "@radix-ui/react-select": "2.2.6",
    "@radix-ui/react-separator": "1.1.16",
    "@radix-ui/react-slider": "1.3.6",
    "@radix-ui/react-slot": "1.2.5",
    "@radix-ui/react-switch": "1.2.6",
    "@radix-ui/react-tabs": "1.1.22",
    "@radix-ui/react-toast": "1.2.24",
    "@radix-ui/react-toggle": "1.1.19",
    "@radix-ui/react-toggle-group": "1.1.20",
    "@radix-ui/react-tooltip": "1.2.16",
    "@radix-ui/themes": "3.2.1",
    "@react-spring/web": "10.0.4",
    "@react-three/drei": "10.6.1",
    "@react-three/fiber": "9.3.0",
    "@react-three/postprocessing": "3.0.5",
    "@tanstack/react-table": "8.21.3",
    "@tanstack/react-virtual": "3.13.26",
    "@tiptap/extension-color": "3.0.9",
    "@tiptap/extension-link": "3.0.9",
    "@tiptap/extension-placeholder": "3.0.9",
    "@tiptap/extension-text-style": "3.0.9",
    "@tiptap/extension-underline": "3.0.9",
    "@tiptap/react": "3.0.9",
    "@tiptap/starter-kit": "3.0.9",
    "@types/node": "24.1.0",
    "@types/react": "19.1.17",
    "@types/react-dom": "19.1.11",
    "@use-gesture/react": "10.3.1",
    "@xyflow/react": "12.8.6",
    "antd": "5.26.7",
    "aos": "2.3.4",
    "apexcharts": "5.3.6",
    "autoprefixer": "10.4.27",
    "axios": "1.11.0",
    "browser-image-compression": "2.0.2",
    "canvas-confetti": "1.9.4",
    "canvg": "4.0.3",
    "chart.js": "4.5.1",
    "chartjs-adapter-date-fns": "3.0.0",
    "class-variance-authority": "0.7.1",
    "classnames": "2.5.1",
    "clsx": "2.1.1",
    "color": "5.0.3",
    "colorthief": "2.6.0",
    "d3": "7.9.0",
    "d3-delaunay": "6.0.4",
    "d3-geo": "3.1.1",
    "date-fns": "4.1.0",
    "dayjs": "1.11.23",
    "diff": "8.0.4",
    "echarts": "6.0.0",
    "echarts-for-react": "3.0.6",
    "emoji-picker-react": "4.13.3",
    "file-saver": "2.0.5",
    "flubber": "0.4.2",
    "framer-motion": "12.23.28",
    "gantt-task-react": "0.3.9",
    "geojson": "0.5.0",
    "gif.js": "0.2.0",
    "gsap": "3.13.0",
    "howler": "2.2.4",
    "html-to-image": "1.11.13",
    "html2canvas": "1.4.1",
    "html2canvas-pro": "1.5.11",
    "jotai": "2.12.5",
    "jspdf": "3.0.4",
    "jspdf-autotable": "5.0.8",
    "jszip": "3.10.2",
    "leaflet": "1.9.4",
    "lodash": "4.17.23",
    "lorem-ipsum": "2.0.10",
    "lucide-react": "0.534.0",
    "marked": "16.1.2",
    "mathjs": "14.6.0",
    "matter-js": "0.20.0",
    "moment": "2.30.1",
    "monaco-editor": "0.52.2",
    "motion": "12.23.28",
    "motion-dom": "12.23.28",
    "nanoid": "5.1.16",
    "next": "15.4.11",
    "next-themes": "0.4.6",
    "ogl": "1.0.11",
    "openmeteo": "1.2.3",
    "papaparse": "5.5.5",
    "perlin-noise": "0.0.1",
    "postcss": "8.5.30",
    "postprocessing": "6.37.8",
    "qrcode": "1.5.4",
    "react": "19.1.9",
    "react-apexcharts": "1.7.0",
    "react-chartjs-2": "5.3.1",
    "react-circular-progressbar": "2.2.0",
    "react-colorful": "5.6.2",
    "react-confetti": "6.4.0",
    "react-countup": "6.5.3",
    "react-datepicker": "8.4.0",
    "react-day-picker": "9.8.1",
    "react-dnd": "16.0.1",
    "react-dnd-html5-backend": "16.0.1",
    "react-dnd-touch-backend": "16.0.1",
    "react-dom": "19.1.9",
    "react-dropzone": "14.3.8",
    "react-fast-marquee": "1.6.5",
    "react-feather": "2.0.10",
    "react-force-graph-2d": "1.28.0",
    "react-helmet": "6.1.0",
    "react-hook-form": "7.61.1",
    "react-hot-toast": "2.5.2",
    "react-icons": "5.5.0",
    "react-intersection-observer": "9.16.0",
    "react-markdown": "10.1.0",
    "react-modal": "3.16.3",
    "react-parallax-tilt": "1.7.346",
    "react-responsive": "10.0.1",
    "react-responsive-carousel": "3.2.23",
    "react-scroll": "1.9.3",
    "react-select": "5.10.2",
    "react-simple-maps": "3.0.0",
    "react-slick": "0.30.3",
    "react-sounds": "1.0.30",
    "react-spinners": "0.17.0",
    "react-swipeable": "7.0.2",
    "react-syntax-highlighter": "15.6.6",
    "react-toastify": "11.0.5",
    "react-tooltip": "5.29.1",
    "react-transition-group": "4.4.5",
    "react-type-animation": "3.2.0",
    "react-use": "17.6.1",
    "react-use-gesture": "9.1.3",
    "react-webcam": "7.2.0",
    "react-wrap-balancer": "1.1.1",
    "react-zoom-pan-pinch": "3.7.0",
    "reactflow": "11.11.4",
    "recharts": "3.1.2",
    "remark-gfm": "4.0.1",
    "rsuite": "5.83.4",
    "slick-carousel": "1.8.1",
    "sonner": "2.0.8",
    "styled-components": "6.1.19",
    "sweetalert2": "11.22.5",
    "swiper": "11.2.10",
    "swr": "2.3.8",
    "tailwind-merge": "3.3.1",
    "tailwindcss": "3.4.19",
    "three": "0.178.0",
    "three-stdlib": "2.36.1",
    "tone": "15.2.13",
    "topojson-client": "3.1.0",
    "typescript": "5.8.3",
    "use-sound": "5.0.0",
    "usehooks-ts": "3.1.1",
    "uuid": "11.1.1",
    "wavesurfer.js": "7.10.3",
    "xlsx": "0.18.5",
    "zod": "4.0.17",
    "zustand": "5.0.15"
  },
  "overrides": {
    "motion-dom": "12.23.28",
    "motion-utils": "12.23.28"
  }
}
EOF

# The committed lockfile (docker/<stack>/package-lock.json, named context `stack_lock`) pins every transitive
# dependency; npm ci installs exactly that tree.
COPY --chown=node:node --from=stack_lock package-lock.json /app/package-lock.json
RUN npm ci --legacy-peer-deps --no-audit --no-fund --loglevel=error \
    && npm cache clean --force

COPY --chown=node:node <<'EOF' /app/next.config.mjs
/** @type {import('next').NextConfig} */
export default {
  // Samples are judged on whether they run; set STRICT_TYPES=1 to fail on type errors.
  typescript: { ignoreBuildErrors: process.env.STRICT_TYPES !== "1" },
  eslint: { ignoreDuringBuilds: true },
  images: { unoptimized: true, remotePatterns: [{ protocol: "https", hostname: "**" }, { protocol: "http", hostname: "**" }] },
};
EOF

COPY --chown=node:node <<'EOF' /app/tsconfig.json
{
  "compilerOptions": {
    "target": "ES2017",
    "lib": [
      "dom",
      "dom.iterable",
      "esnext"
    ],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": false,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [
      {
        "name": "next"
      }
    ],
    "paths": {
      "@/*": [
        "./*"
      ]
    }
  },
  "include": [
    "next-env.d.ts",
    "**/*.ts",
    "**/*.tsx"
  ],
  "exclude": [
    "node_modules"
  ]
}
EOF

COPY --chown=node:node <<'EOF' /app/next-env.d.ts
/// <reference types="next" />
/// <reference types="next/image-types/global" />
EOF

COPY --chown=node:node <<'EOF' /app/tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,jsx,ts,tsx}"],
  darkMode: "class",
  theme: { extend: {} },
  plugins: [],
};
EOF

COPY --chown=node:node <<'EOF' /app/postcss.config.js
module.exports = { plugins: { tailwindcss: {}, autoprefixer: {} } };
EOF

COPY --chown=node:node <<'EOF' /app/app/globals.css
@tailwind base;
@tailwind components;
@tailwind utilities;
EOF

COPY --chown=node:node <<'EOF' /app/app/layout.tsx
import "./globals.css";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
EOF

COPY --chown=node:node <<'EOF' /app/app/page.tsx
"use client";
import dynamic from "next/dynamic";

const Sample = dynamic(() => import("./sample"), { ssr: false });

export default function Page() {
  return <Sample />;
}
EOF

COPY --chmod=755 <<'EOF' /usr/local/bin/run-sample
#!/bin/sh
# Modes: serve (build + start, default) | build (exit code = build result) | dev
# BUILD_TIMEOUT (seconds, default 600) caps the build; exit code 124 means it timed out.
# SSR=1 mounts the sample directly as app/page.tsx instead of the client-only wrapper.
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
cd /app
extract "$SRC" > app/sample.tsx
# PLAIN_STYLE=1 turns <style jsx> into a plain <style>: Next's styled-jsx transform never finishes on
# some large components with ${...} interpolations in their styles (Next 15.4-16.2 alike).
[ "${PLAIN_STYLE:-0}" = "1" ] && sed -i -E 's/<style jsx[^>]*>/<style>/g' app/sample.tsx
[ "${SSR:-0}" = "1" ] && mv app/sample.tsx app/page.tsx
case "$MODE" in
  build) exec timeout "${BUILD_TIMEOUT:-600}" ./node_modules/.bin/next build ;;
  serve) timeout "${BUILD_TIMEOUT:-600}" ./node_modules/.bin/next build && exec ./node_modules/.bin/next start -H 0.0.0.0 -p 3000 ;;
  dev)   exec ./node_modules/.bin/next dev -H 0.0.0.0 -p 3000 ;;
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

# Final image for ui-next / ui-vite: agent user `node`, verifier user `ui-verify`, root-owned read-only template.
FROM stack
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends procps \
 && rm -rf /var/lib/apt/lists/*
COPY --from=fetch /out/opencode /opt/ui-gen/runtime/opencode
COPY --from=fetch /out/rg /opt/ui-gen/runtime/rg
COPY --from=plugins /build/opencode-1.17.11-plugin-dependencies.tar.gz /opt/ui-gen/runtime/opencode-1.17.11-plugin-dependencies.tar.gz
RUN set -eux; \
  useradd --system --uid 1001 --no-create-home --shell /usr/sbin/nologin ui-verify; \
  mv /app /opt/ui-gen/template; \
  mkdir -p /opt/ui-gen/template/node_modules/.vite-temp; \
  chown -R root:root /opt/ui-gen; chmod -R a-w,a+rX /opt/ui-gen; \
  chmod 1777 /opt/ui-gen/template/node_modules/.vite-temp; \
  mkdir /app; cd /opt/ui-gen/template; \
  for f in $(ls -A); do [ "$f" = node_modules ] || cp -a "$f" /app/; done; \
  ln -s /opt/ui-gen/template/node_modules /app/node_modules; \
  chown -R node:node /app; chown -h node:node /app/node_modules; chmod -R u+w /app; \
  cd /; { find /opt/ui-gen/template -path /opt/ui-gen/template/node_modules -prune -o -type f -print | LC_ALL=C sort | xargs -d '\n' sha256sum; \
          sha256sum /usr/local/bin/run-sample /usr/local/bin/node /opt/ui-gen/runtime/opencode /opt/ui-gen/runtime/rg \
                    /opt/ui-gen/runtime/opencode-1.17.11-plugin-dependencies.tar.gz; } > /opt/ui-gen/MANIFEST.sha256; \
  chmod 444 /opt/ui-gen/MANIFEST.sha256; touch /opt/ui-gen/.built
# The Codex runtime when built WITH_CODEX=1 (nothing otherwise), root-owned and read-only as set in its stage. It is
# copied after the permission pass above (so its files are stored once) and stays out of MANIFEST.sha256, so the
# manifest pins are the same with or without it.
COPY --from=codex /out/ /
# Codex keeps its sessions under /tmp and refuses to create its helper binaries under the temp dir, so the temp dir
# is /var/tmp (harmless for the other harnesses and the build).
ENV TMPDIR=/var/tmp
WORKDIR /app
USER node
ENTRYPOINT []
CMD []
