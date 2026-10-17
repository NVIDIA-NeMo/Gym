# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is derived from utopia-group/VeriSoftBench `Dockerfile`
# (https://github.com/utopia-group/VeriSoftBench) at commit
# d066523f2b087f28b793c05eacb29bf2e42c0504 and remains under its original MIT
# terms (reproduced below). Modifications Copyright (c) 2026 NVIDIA CORPORATION &
# AFFILIATES and contributors, licensed under the Apache License 2.0 (SPDX header
# above). See THIRD_PARTY_NOTICES.txt in the build context for the licences of
# the 23 Lean repositories the image contains.
#
# MIT License
#
# Copyright (c) 2026 VeriSoftBench Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# VeriSoftBench Lean proof environment for NeMo Gym.
# NVIDIA modifications — three changes, all to the build tooling and none to what
# gets built:
#   1. One RUN layer per repository instead of a single layer for all 23, so a
#      failure late in the ~1 h build does not discard the repositories already built.
#   2. `scripts/build_repo.sh` is the patched copy from this directory, which
#      pre-seeds Lake packages at their pinned revisions and verifies the final
#      revisions after the build (see the comments in that file).
#   3. `build_config/` carries per-repo `nemo-gym-build.json` overrides from this
#      directory recording repairs a repository needs beyond upstream's protocol
#      (currently lean-mlir: `lake build --try-cache` fallback plus three extra
#      targets); `build_repo.sh` reads them.
# Build with `build_image.sh`, which assembles the context from a pinned upstream
# checkout and overlays the patched script and overrides.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git ca-certificates python3 g++ make \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | bash -s -- -y --default-toolchain none
ENV PATH="/root/.elan/bin:${PATH}"

WORKDIR /workspace

COPY repos.json /workspace/repos.json
COPY scripts/build_repo.sh /workspace/scripts/build_repo.sh
RUN chmod +x /workspace/scripts/build_repo.sh
COPY build_config/ /workspace/build_config/
COPY THIRD_PARTY_NOTICES.txt /workspace/THIRD_PARTY_NOTICES.txt

ENV B=/workspace/scripts/build_repo.sh CFG=/workspace/build_config OUT=/workspace/lean_repos

# Non-mathlib repos
RUN $B IntroEffects --clone $CFG $OUT
RUN $B iris-lean --clone $CFG $OUT
RUN $B lean-hoare --clone $CFG $OUT
RUN $B LeanExprEvaluator --clone $CFG $OUT
RUN $B Lentil --clone $CFG $OUT
RUN $B LeroyCompilerVerificationCourse --clone $CFG $OUT
RUN $B verified-compiler --clone $CFG $OUT

# Mathlib-dependent repos. The `|| echo WARNING` tolerances reproduce upstream's
# Dockerfile exactly: only the repositories whose partial failure upstream's own
# protocol tolerates get one, and no others.
RUN $B CvxLean --clone $CFG $OUT
RUN $B pcf-lean --clone $CFG $OUT
RUN $B lean-formal-reasoning-program --clone $CFG $OUT
RUN $B wadray_verification --clone $CFG $OUT
RUN $B formal-snarks-project --clone $CFG $OUT || echo "WARNING: formal-snarks-project build failed (known issues)"
RUN $B splean --clone $CFG $OUT
RUN $B TTBFL --clone $CFG $OUT
RUN $B FVIntmax --clone $CFG $OUT
RUN $B juvix-lean --clone $CFG $OUT
RUN $B capless-lean --clone $CFG $OUT
RUN $B ArkLib --clone $CFG $OUT
RUN $B loom --clone $CFG $OUT || echo "WARNING: loom build had partial failures (known issue)"
RUN $B clean --clone $CFG $OUT || echo "WARNING: clean build had partial failures"
RUN $B veil --clone $CFG $OUT || echo "WARNING: veil build had partial failures"
RUN $B VCV-io --clone $CFG $OUT || echo "WARNING: VCV-io build had partial failures"
RUN $B lean-mlir --clone $CFG $OUT || echo "WARNING: lean-mlir build had partial failures"

RUN mkdir -p /workspace/THIRD_PARTY_LICENSES; \
    for repo_dir in $OUT/*/; do \
        repo_name=$(basename "$repo_dir"); \
        for lf in LICENSE LICENSE.md LICENSE.txt COPYING COPYING.md; do \
            if [ -f "$repo_dir/$lf" ]; then \
                cp "$repo_dir/$lf" "/workspace/THIRD_PARTY_LICENSES/${repo_name}_${lf}"; \
            fi; \
        done; \
    done

COPY scripts/docker_repo_patches.sh /workspace/scripts/docker_repo_patches.sh
RUN chmod +x /workspace/scripts/docker_repo_patches.sh && \
    /workspace/scripts/docker_repo_patches.sh --repos-dir /workspace/lean_repos --config-dir /workspace/build_config

COPY scripts/generate_repo_index.py /workspace/scripts/generate_repo_index.py
COPY scripts/chunking/ /workspace/scripts/chunking/
RUN python3 /workspace/scripts/generate_repo_index.py --repos-dir /workspace/lean_repos --output-dir /workspace/repo_index

WORKDIR /workspace/lean_repos
CMD ["sleep", "infinity"]
